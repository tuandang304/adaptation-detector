"""Architecture and training changes applied to a COCO-initialised YOLO26n, selected by name in configs/node.yaml.

Every change leaves the network equal to the pretrained one at step 0 (new branches end in a zero-initialised layer).

  rep   neck C3k2 (L13, L16, L19, L22): each 3x3 Conv gains a parallel 1x1 branch, merged back into the 3x3 at inference
  lk    backbone C3k2 (L2, L4, L6, L8): residual depthwise 5x5 after the block; stays at inference
  film  shift-conditioned scale/shift on L0, L1, L3: a small network reads photometric statistics of the image and
        the deviation of stem feature statistics from their training-time reference, one forward pass, no gradient
  mix   training only: per-sample feature statistics of L1 and L3 are mixed with another sample of the batch
  cons  training only: a photometrically jittered second view must give the same Detect input features

Candidate clusters stacked on film + mix + cons (each: an inference module + a generator applied only to the cons view;
the detection-loss images stay clean and cons stays the single agreement point unless stated):
  glaregate  local-contrast gate on L2, L4, L6 outputs; generator: glare blobs + wet-road reflections
  glow       residual depthwise 7x7 + 1x1 on L8, dilated depthwise 3x3 on L6; generator: glare
  freqcal    dual-band (low/high-pass) recalibration on L13, L16, L19, L22 outputs; generator: motion blur + veil
  headanchor per-level affine on L16, L19, L22 + per-image class-logit bias in the Detect heads (L23); extra objective:
             clean top-50 one2one anchors must keep their class probabilities and boxes on the jittered view
  specaa     input-conditioned anti-aliasing blur before the stride-2 convs L5, L7, L17, L20; generator: rain + sensor noise
  darkattn   residual soft-shrinkage on L8, L9 outputs + per-image query temperature in the L10 attention;
             generator: gamma drop + shot noise
Every gate is 0.2*tanh(...) of a zero-initialised layer, so the network equals the pretrained one at step 0.

Hooks are module-level callables so that checkpoints (pickled models) and the EMA deep copy keep them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2, Conv, RepConv
from ultralytics.nn.tasks import DetectionModel

NECK, BACKBONE, FILM_LAYERS, MIX_LAYERS = (13, 16, 19, 22), (2, 4, 6, 8), (0, 1, 3), (1, 3)


# --- rep ---------------------------------------------------------------------------------------------------------
def add_rep(model: nn.Module) -> None:
    for i in NECK:
        for parent in list(model.model[i].modules()):
            for name, c in list(parent.named_children()):
                if type(c) is Conv and c.conv.kernel_size == (3, 3) and c.conv.groups == 1 and c.conv.stride == (1, 1):
                    r = RepConv(c.conv.in_channels, c.conv.out_channels, 3, 1, act=c.act)
                    r.conv1.conv.load_state_dict(c.conv.state_dict())
                    r.conv1.bn.load_state_dict(c.bn.state_dict())
                    nn.init.zeros_(r.conv2.bn.weight)  # branch output is 0 at step 0
                    setattr(parent, name, r)


# --- lk ----------------------------------------------------------------------------------------------------------
class C3k2LK(C3k2):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y + self.lk(y)


def add_lk(model: nn.Module) -> None:
    for i in BACKBONE:
        m = model.model[i]
        c = m.cv2.conv.out_channels
        m.__class__ = C3k2LK
        m.lk = Conv(c, c, 5, g=c, act=False)
        nn.init.zeros_(m.lk.bn.weight)


# --- film --------------------------------------------------------------------------------------------------------
class ShiftFiLM(nn.Module):
    """Predicts per-channel (scale, shift) for the early layers from how far this image is from the training domain."""

    def __init__(self, chs: list[int], hidden: int = 32):
        super().__init__()
        self.chs = chs
        self.register_buffer("ref_mu", torch.zeros(chs[0]))
        self.register_buffer("ref_sd", torch.ones(chs[0]))
        self.register_buffer("seen", torch.zeros(()))
        self.net = nn.Sequential(nn.Linear(9 + 2 * chs[0], hidden), nn.SiLU(), nn.Linear(hidden, 2 * sum(chs)))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.ab = None  # per-forward state, cleared by the last hooked layer

    def descriptor(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        img, feat = img.float(), feat.float()
        hf = (img - F.avg_pool2d(img, 3, 1, 1)).abs().mean((2, 3))
        mu, sd = feat.mean((2, 3)), feat.std((2, 3))
        if self.training:  # reference = running mean of the stem statistics on training images
            with torch.no_grad():
                m = 1.0 if self.seen == 0 else 0.01
                self.ref_mu.lerp_(mu.mean(0), m)
                self.ref_sd.lerp_(sd.mean(0), m)
                self.seen.fill_(1)
        dev = (mu - self.ref_mu) / (self.ref_sd + 1e-3)
        ratio = torch.log((sd + 1e-3) / (self.ref_sd + 1e-3))
        return torch.cat([img.mean((2, 3)), img.std((2, 3)), hf, dev, ratio], 1)

    def modulate(self, k: int, img: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if k == 0:
            d = self.descriptor(img, out).to(self.net[0].weight.dtype)
            self.ab = self.net(d).split([2 * c for c in self.chs], 1)
        a, b = self.ab[k].to(out.dtype).chunk(2, 1)
        if k == len(self.chs) - 1:
            self.ab = None
        return out * (1 + a[:, :, None, None]) + b[:, :, None, None]


class FilmHook:
    def __init__(self, film: ShiftFiLM, k: int):
        self.film, self.k = film, k

    def __call__(self, module, inputs, output):
        return self.film.modulate(self.k, inputs[0], output)


def add_film(model: nn.Module) -> None:
    model.film = ShiftFiLM([model.model[i].conv.out_channels for i in FILM_LAYERS])
    for k, i in enumerate(FILM_LAYERS):
        model.model[i].register_forward_hook(FilmHook(model.film, k))


# --- mix ---------------------------------------------------------------------------------------------------------
class MixStat:
    """MixStyle-type mixing of per-sample channel statistics; identity in eval mode."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1):
        self.p, self.beta = p, torch.distributions.Beta(alpha, alpha)

    def __call__(self, module, inputs, x):
        if not module.training or x.shape[0] < 2 or torch.rand(()) > self.p:
            return None
        mu, sd = x.mean((2, 3), keepdim=True).detach(), (x.var((2, 3), keepdim=True) + 1e-6).sqrt().detach()
        lam = self.beta.sample((x.shape[0], 1, 1, 1)).to(x)
        perm = torch.randperm(x.shape[0], device=x.device)
        return (x - mu) / sd * (lam * sd + (1 - lam) * sd[perm]) + lam * mu + (1 - lam) * mu[perm]


def add_mix(model: nn.Module) -> None:
    for i in MIX_LAYERS:
        model.model[i].register_forward_hook(MixStat())


# --- cons --------------------------------------------------------------------------------------------------------
class DetectTap:
    def __init__(self):
        self.feats = None

    def __call__(self, module, inputs):
        if module.training:
            self.feats = list(inputs[0])


def style_jitter(img: torch.Tensor) -> torch.Tensor:
    """Per-sample gamma, contrast, brightness and channel gain on a [0, 1] image batch."""
    u = lambda lo, hi, c=1: torch.empty(img.shape[0], c, 1, 1, device=img.device).uniform_(lo, hi)
    x = img.float().clamp(1e-4, 1) ** torch.exp(u(-0.7, 0.7))
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.5, 1.5) + m + u(-0.2, 0.2)
    return (x * u(0.8, 1.2, 3)).clamp(0, 1).to(img.dtype)


class ConsistModel(DetectionModel):
    cons_weight = 1.0

    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        if not self.training or preds is not None:
            return loss, items
        f1, self.tap.feats = self.tap.feats, None
        grabs = self.__dict__.get("grabs", {})
        g1 = {k: g.val for k, g in grabs.items()}
        view = style_jitter(batch["img"])
        for g in self.__dict__.get("gens", []):  # candidate generators act on the cons view only, half of the images
            view = AUGS[g](view, 0.5)
        self.forward(view)
        f2, self.tap.feats = self.tap.feats, None
        c = sum(1 - F.cosine_similarity(a.detach().float(), b.float(), dim=1).mean() for a, b in zip(f1, f2)) / len(f1)
        for k, g in grabs.items():
            c = c + anchor_gap(g1[k], g.val)
            g.val = None
        # the trainer sums the loss vector, which is already scaled by the batch size
        return loss + self.cons_weight * c * batch["img"].shape[0] / loss.numel(), items


def add_cons(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    model.tap = DetectTap()
    model.model[-1].register_forward_pre_hook(model.tap)


# --- candidate clusters ---------------------------------------------------------------------------------------------
def _u(b: int, lo: float, hi: float, dev, c: int = 1) -> torch.Tensor:
    return torch.empty(b, c, 1, 1, device=dev).uniform_(lo, hi)


def _keep(x0: torch.Tensor, x: torch.Tensor, p: float) -> torch.Tensor:
    keep = torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) >= p
    return torch.where(keep, x0.float(), x).clamp(0, 1).to(x0.dtype)


def _line_kernels(b: int, k: int, dev, angle_lo: float, angle_hi: float) -> torch.Tensor:
    """(b, 1, k, k) normalised line kernels of random direction (radians, 0 = vertical)."""
    th = torch.empty(b, 1, 1, device=dev).uniform_(angle_lo, angle_hi)
    r = torch.arange(k, device=dev, dtype=torch.float32) - (k - 1) / 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    dist = (xx * torch.cos(th) - yy * torch.sin(th)).abs()  # distance to a line through the centre
    w = torch.exp(-dist ** 2 / 0.25)
    return (w / w.sum((1, 2), keepdim=True))[:, None]


def glare_aug(img: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Up to 8 tinted Gaussian glows with a vertical wet-road reflection under each, added to the image."""
    b, _, h, w = img.shape
    dev, gh, gw, k = img.device, max(h // 4, 8), max(w // 4, 8), 8
    yy = torch.linspace(0, 1, gh, device=dev)[None, None, :, None]
    xx = torch.linspace(0, 1, gw, device=dev)[None, None, None, :]
    on = (torch.rand(b, k, 1, 1, device=dev) < 0.7).float()
    cx, cy = torch.rand(b, k, 1, 1, device=dev), torch.rand(b, k, 1, 1, device=dev) * 0.7
    sg = torch.empty(b, k, 1, 1, device=dev).uniform_(0.01, 0.06)
    amp = torch.empty(b, k, 1, 1, device=dev).uniform_(0.3, 1.0) * on
    glow = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))
    ry, stretch = cy + torch.empty_like(cy).uniform_(0.05, 0.3), torch.empty_like(sg).uniform_(3, 8)
    refl = 0.35 * amp * torch.exp(-(xx - cx) ** 2 / (2 * sg ** 2) - (yy - ry) ** 2 / (2 * (sg * stretch) ** 2))
    tint = torch.empty(b, k, 3, device=dev).uniform_(0.6, 1.0)
    g = torch.einsum("bkhw,bkc->bchw", glow + refl, tint)
    g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
    return _keep(img, img.float() + g, p)


def blurveil_aug(img: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Directional motion blur (5x5) then an atmospheric veil t*I + (1-t)*A."""
    b, c, h, w = img.shape
    kern = _line_kernels(b, 5, img.device, 0, 3.1416).repeat_interleave(c, 0)
    x = F.conv2d(img.float().reshape(1, b * c, h, w), kern, padding=2, groups=b * c).reshape(b, c, h, w)
    t = _u(b, 0.6, 1.0, img.device)
    return _keep(img, t * x + (1 - t) * _u(b, 0.6, 1.0, img.device), p)


def rain_aug(img: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Oriented rain streaks (near-vertical lines from sparse seeds) plus Poisson-Gaussian sensor noise."""
    b, c, h, w = img.shape
    seeds = (torch.rand(b, 1, h, w, device=img.device) < _u(b, 0.002, 0.01, img.device)).float()
    kern = _line_kernels(b, 15, img.device, -0.35, 0.35)
    st = F.conv2d(seeds.reshape(1, b, h, w), kern, padding=7, groups=b).reshape(b, 1, h, w) * 15
    x = img.float() + _u(b, 0.2, 0.6, img.device) * st.clamp(0, 1)
    x = x + torch.randn_like(x) * (x.clamp(0, 1) * _u(b, 1e-3, 1e-2, img.device) + _u(b, 1e-3, 2e-2, img.device) ** 2).sqrt()
    return _keep(img, x, p)


def lowsnr_aug(img: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Gamma drop (x ** 1.5-2.2) and strong shot noise."""
    b = img.shape[0]
    x = img.float().clamp(0, 1) ** _u(b, 1.5, 2.2, img.device)
    x = x + torch.randn_like(x) * (x * _u(b, 5e-3, 2e-2, img.device) + _u(b, 1e-3, 2e-2, img.device) ** 2).sqrt()
    return _keep(img, x, p)


AUGS = {"glare": glare_aug, "blurveil": blurveil_aug, "rain": rain_aug, "lowsnr": lowsnr_aug}


def _mlp(c_in: int, c_out: int, hidden: int) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(c_in, hidden), nn.SiLU(), nn.Linear(hidden, c_out))
    nn.init.zeros_(m[2].weight)
    nn.init.zeros_(m[2].bias)
    return m


def _run(m: nn.Module, x: torch.Tensor) -> torch.Tensor:  # float32 result whatever the weight dtype
    return m(x.to(next(m.parameters()).dtype)).float()


def _prep(model: nn.Module) -> None:
    if "gens" not in model.__dict__:
        model.gens, model.grabs = [], {}


class GlareGate(nn.Module):
    """F * (1 + 0.2 tanh(W2 SiLU(W1 |F - avgpool5(F)|))): a per-pixel gate driven by local contrast; W2 = 0 at step 0."""

    def __init__(self, c: int):
        super().__init__()
        self.w1, self.w2 = nn.Conv2d(c, max(c // 8, 8), 1), nn.Conv2d(max(c // 8, 8), c, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, module, inputs, out):
        x = out.float()
        d = (x - F.avg_pool2d(x, 5, 1, 2, count_include_pad=False)).abs()
        s = F.conv2d(F.silu(F.conv2d(d, self.w1.weight.float(), self.w1.bias.float())), self.w2.weight.float(),
                     self.w2.bias.float())
        return (x * (1 + 0.2 * torch.tanh(s))).to(out.dtype)


def add_glaregate(model: nn.Module) -> None:
    _prep(model)
    model.glaregate = nn.ModuleList()
    for i in (2, 4, 6):
        g = GlareGate(model.model[i].cv2.conv.out_channels)
        model.glaregate.append(g)
        model.model[i].register_forward_hook(g)
    model.gens.append("glare")


class Residual(nn.Module):
    """out + branch(out); the branch ends in a BatchNorm whose weight is 0 at step 0."""

    def __init__(self, branch: nn.Module):
        super().__init__()
        self.branch = branch

    def forward(self, module, inputs, out):
        return out + self.branch(out)


def add_glow(model: nn.Module) -> None:
    _prep(model)
    c8, c6 = model.model[8].cv2.conv.out_channels, model.model[6].cv2.conv.out_channels
    b8 = nn.Sequential(Conv(c8, c8, 7, g=c8, act=False), Conv(c8, c8, 1, act=False))
    b6 = Conv(c6, c6, 3, g=c6, d=2, act=False)
    nn.init.zeros_(b8[1].bn.weight)
    nn.init.zeros_(b6.bn.weight)
    model.glow = nn.ModuleList([Residual(b6), Residual(b8)])
    model.model[6].register_forward_hook(model.glow[0])
    model.model[8].register_forward_hook(model.glow[1])
    model.gens.append("glare")


class FreqCal(nn.Module):
    """F + 0.2 tanh(MLP[mean(low), std(high)]) * high, with low = avgpool3(F), high = F - low."""

    def __init__(self, c: int):
        super().__init__()
        self.mlp = _mlp(2 * c, c, max(c // 8, 8))

    def forward(self, module, inputs, out):
        x = out.float()
        hi = x - F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        g = 0.2 * torch.tanh(_run(self.mlp, torch.cat([(x - hi).mean((2, 3)), hi.std((2, 3))], 1)))
        return (x + g[:, :, None, None] * hi).to(out.dtype)


def add_freqcal(model: nn.Module) -> None:
    _prep(model)
    model.freqcal = nn.ModuleList()
    for i in (13, 16, 19, 22):
        f = FreqCal(model.model[i].cv2.conv.out_channels)
        model.freqcal.append(f)
        model.model[i].register_forward_hook(f)
    model.gens.append("blurveil")


class LevelAffine(nn.Module):
    """Per-level channel affine from the level's own pooled statistics: F * (1 + 0.2 tanh a) + 0.2 tanh b."""

    def __init__(self, c: int):
        super().__init__()
        self.mlp = _mlp(2 * c, 2 * c, max(c // 8, 8))

    def forward(self, module, inputs, out):
        x = out.float()
        a, b = (0.2 * torch.tanh(_run(self.mlp, torch.cat([x.mean((2, 3)), x.std((2, 3))], 1)))).chunk(2, 1)
        return (x * (1 + a[:, :, None, None]) + b[:, :, None, None]).to(out.dtype)


class ClassBias(nn.Module):
    """Per-image additive class-logit bias (bounded by tanh) from the pooled statistics of the three Detect inputs."""

    def __init__(self, chs: list[int], nc: int):
        super().__init__()
        self.mlp = _mlp(2 * sum(chs), nc, 32)
        self.db = None  # per-forward state, cleared after Detect

    def read(self, module, inputs):  # forward pre-hook on Detect
        fs = [f.float() for f in inputs[0]]
        self.db = torch.tanh(_run(self.mlp, torch.cat([t for f in fs for t in (f.mean((2, 3)), f.std((2, 3)))], 1)))


class BiasRead:
    def __init__(self, cb: ClassBias):
        self.cb = cb

    def __call__(self, module, inputs):
        self.cb.read(module, inputs)


class BiasAdd:
    def __init__(self, cb: ClassBias):
        self.cb = cb

    def __call__(self, module, inputs, out):
        return out + self.cb.db.to(out.dtype)[:, :, None, None]


class BiasClear:
    def __init__(self, cb: ClassBias):
        self.cb = cb

    def __call__(self, module, inputs, out):
        self.cb.db = None


def anchor_gap(p1: dict, p2: dict, k: int = 50, thr: float = 0.25) -> torch.Tensor:
    """Clean top-k one2one anchors (confidence > thr) keep their class probabilities and boxes on the jittered view."""
    s1, s2 = p1["scores"].detach().float(), p2["scores"].float()
    conf, _ = torch.sigmoid(s1).max(1)  # (B, A)
    v, idx = conf.topk(min(k, conf.shape[1]), 1)
    m = (v > thr).float()
    if m.sum() == 0:
        return s2.sum() * 0
    g = lambda t: t.gather(2, idx[:, None].expand(-1, t.shape[1], -1))
    cls = F.binary_cross_entropy_with_logits(g(s2), torch.sigmoid(g(s1)), reduction="none").mean(1)
    box = F.smooth_l1_loss(g(p2["boxes"].float()), g(p1["boxes"].detach().float()), reduction="none").mean(1)
    return ((cls + box) * m).sum() / m.sum()


class Grab:
    def __init__(self, key: str | None = None):
        self.key, self.val = key, None

    def __call__(self, module, inputs, out):
        if module.training:
            self.val = out[self.key] if self.key else out


def add_headanchor(model: nn.Module) -> None:
    _prep(model)
    det, chs = model.model[-1], [model.model[i].cv2.conv.out_channels for i in (16, 19, 22)]
    model.levelaff = nn.ModuleList()
    for i, c in zip((16, 19, 22), chs):
        a = LevelAffine(c)
        model.levelaff.append(a)
        model.model[i].register_forward_hook(a)
    model.clsbias = ClassBias(chs, det.nc)
    det.register_forward_pre_hook(BiasRead(model.clsbias))
    for heads in (det.cv3, getattr(det, "one2one_cv3", None) or []):
        for h in heads:
            h.register_forward_hook(BiasAdd(model.clsbias))
    det.register_forward_hook(BiasClear(model.clsbias))
    model.grabs["anchor"] = Grab("one2one")
    det.register_forward_hook(model.grabs["anchor"])


class SpecAA(nn.Module):
    """Before a stride-2 conv: x + beta * (blur3(x) - x), beta = 0.2 tanh(lam * sigmoid(MLP(high-frequency energy)))."""

    def __init__(self, c: int):
        super().__init__()
        self.mlp = _mlp(c, c, max(c // 8, 8))
        nn.init.normal_(self.mlp[2].weight, std=0.01)  # lam carries the zero init, so the MLP may start non-zero
        self.lam = nn.Parameter(torch.zeros(c))
        k = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("k", (k[:, None] * k[None]) / 16)

    def forward(self, module, inputs):
        x = inputs[0].float()
        c = x.shape[1]
        bl = F.conv2d(x, self.k.float().expand(c, 1, 3, 3), padding=1, groups=c)
        e = (x - bl).abs().mean((2, 3))
        beta = 0.2 * torch.tanh(self.lam.float() * torch.sigmoid(_run(self.mlp, e)))
        return ((x + beta[:, :, None, None] * (bl - x)).to(inputs[0].dtype),)


def add_specaa(model: nn.Module) -> None:
    _prep(model)
    model.specaa = nn.ModuleList()
    for i in (5, 7, 17, 20):
        a = SpecAA(model.model[i].conv.in_channels)
        model.specaa.append(a)
        model.model[i].register_forward_pre_hook(a)
    model.gens.append("rain")


class Shrink(nn.Module):
    """x - 0.2 tanh(eta) * tau * tanh(x / tau), tau = softplus(MLP(GAP|x|)) + 1e-3; eta = 0 at step 0. On L9 it also
    predicts the query temperature of the L10 attention."""

    def __init__(self, c: int, temp: bool = False):
        super().__init__()
        self.mlp = _mlp(c, c, max(c // 8, 8))
        self.eta = nn.Parameter(torch.zeros(c))
        self.temp = nn.Linear(c, 1) if temp else None
        if temp:
            nn.init.zeros_(self.temp.weight)
            nn.init.zeros_(self.temp.bias)
        self.f = None  # per-forward query scale for L10, cleared by the L10 hook

    def forward(self, module, inputs, out):
        x = out.float()
        tau = F.softplus(_run(self.mlp, x.abs().mean((2, 3))))[:, :, None, None] + 1e-3
        y = x - (0.2 * torch.tanh(self.eta.float()))[None, :, None, None] * tau * torch.tanh(x / tau)
        if self.temp is not None:
            self.f = 1 + 0.2 * torch.tanh(_run(self.temp, y.mean((2, 3))))
        return y.to(out.dtype)


class QueryScale(nn.Module):
    """Forward hook on the qkv conv of an L10 attention block: multiplies the query channels by the per-image factor."""

    def __init__(self, src: Shrink, attn: nn.Module):
        super().__init__()
        self.src = src
        per = 2 * attn.key_dim + attn.head_dim
        mask = torch.zeros(attn.num_heads, per)
        mask[:, : attn.key_dim] = 1
        self.register_buffer("mask", mask.reshape(1, -1, 1, 1))

    def forward(self, module, inputs, out):
        f = self.src.f
        return (out.float() * (1 + self.mask.float() * (f[:, :, None, None] - 1))).to(out.dtype)


class ClearF:
    def __init__(self, src: Shrink):
        self.src = src

    def __call__(self, module, inputs, out):
        self.src.f = None


def add_darkattn(model: nn.Module) -> None:
    _prep(model)
    model.shrink = nn.ModuleList([Shrink(model.model[8].cv2.conv.out_channels),
                                  Shrink(model.model[9].cv2.conv.out_channels, temp=True)])
    model.model[8].register_forward_hook(model.shrink[0])
    model.model[9].register_forward_hook(model.shrink[1])
    model.qscale = nn.ModuleList()
    for blk in model.model[10].m:
        q = QueryScale(model.shrink[1], blk.attn)
        model.qscale.append(q)
        blk.attn.qkv.register_forward_hook(q)
    model.model[10].register_forward_hook(ClearF(model.shrink[1]))
    model.gens.append("lowsnr")


ADD = {"rep": add_rep, "lk": add_lk, "film": add_film, "mix": add_mix, "cons": add_cons, "glaregate": add_glaregate,
       "glow": add_glow, "freqcal": add_freqcal, "headanchor": add_headanchor, "specaa": add_specaa,
       "darkattn": add_darkattn}


def apply(model: nn.Module, mods: list[str]) -> nn.Module:
    """Apply the named changes to a model whose pretrained weights are already loaded."""
    alive = list(model.parameters())  # held so that a freed tensor's id cannot be reused by a new one
    before = {id(p) for p in alive}
    n0 = sum(p.numel() for p in model.parameters())
    for name in mods:
        ADD[name](model)
    assert "cons" in mods or not model.__dict__.get("gens") and not model.__dict__.get("grabs"), \
        "candidate clusters need cons (their generators and objective live in its second view)"
    for m in model.modules():  # same BatchNorm settings as Ultralytics gives the rest of the network
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    total = sum(p.numel() for p in model.parameters())
    # pretrained = tensors that were there before, plus the 3x3 weights that rep copied into fresh tensors
    kept = sum(p.numel() for p in model.parameters() if id(p) in before) + sum(
        p.numel() for m in model.modules() if isinstance(m, RepConv) for p in m.conv1.parameters())
    print(f"[mods] mods={','.join(mods) or 'none'} train_params={total} stock_params={n0} "
          f"new_params={total - kept} pretrained_share={100 * kept / total:.2f}%", flush=True)
    return model


class ModTrainer(DetectionTrainer):
    mods: list[str] = []

    def get_model(self, cfg=None, weights=None, verbose=True):
        return apply(super().get_model(cfg, weights, verbose), self.mods)


if __name__ == "__main__":  # self-check: identity at step 0, fuse, checkpoint round trip, one training step
    import copy
    import io

    from ultralytics import YOLO

    torch.manual_seed(0)
    base = YOLO("yolo26n.pt").model.float().eval()
    x = torch.rand(2, 3, 256, 256)
    ref = base(x)[0]
    m = apply(copy.deepcopy(base), list(ADD))
    assert torch.allclose(m.eval()(x)[0], ref, atol=1e-4), "not identical to the pretrained model at step 0"
    for p in m.parameters():  # move off the zero init so that fusing is actually tested
        if p.abs().sum() == 0:
            nn.init.normal_(p, std=0.05)
    out = m(x)[0]
    assert not torch.allclose(out, ref, atol=1e-3)
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False).eval()
    assert torch.allclose(m2(x)[0], out, atol=1e-5), "checkpoint round trip changed the output"
    fused = copy.deepcopy(m).fuse(verbose=False)
    assert not any(isinstance(k, RepConv) and hasattr(k, "conv1") for k in fused.modules())
    assert torch.allclose(fused(x)[0], out, atol=1e-3), "fused model differs"
    from types import SimpleNamespace
    from ultralytics.cfg import get_cfg
    m.args = get_cfg()
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    batch = {"img": x, "cls": torch.zeros(3, 1), "bboxes": torch.rand(3, 4) * 0.3 + 0.3,
             "batch_idx": torch.tensor([0.0, 0.0, 1.0])}
    loss, _ = m.loss(batch)
    loss.sum().backward()
    assert m.tap.feats is None and m.film.ab is None
    assert all(p.grad is not None for p in m.film.parameters()), "film received no gradient"
    new = {"glaregate": ["glaregate"], "glow": ["glow"], "freqcal": ["freqcal"], "headanchor": ["levelaff", "clsbias"],
           "specaa": ["specaa"], "darkattn": ["shrink"]}
    for name, attrs in new.items():  # each candidate on top of film + mix + cons
        mods = ["film", "mix", "cons", name]
        r = apply(copy.deepcopy(base), mods).eval()
        assert torch.allclose(r(x)[0], ref, atol=1e-4), f"{name}: not identical at step 0"
        for q in r.parameters():
            if q.abs().sum() == 0:
                nn.init.normal_(q, std=0.05)
        out = r(x)[0]
        assert not torch.allclose(out, ref, atol=1e-3), name
        buf = io.BytesIO()
        torch.save(r, buf)
        buf.seek(0)
        assert torch.allclose(torch.load(buf, weights_only=False).eval()(x)[0], out, atol=1e-5), f"{name}: checkpoint"
        assert torch.allclose(copy.deepcopy(r).fuse(verbose=False)(x)[0], out, atol=1e-3), f"{name}: fuse"
        h = copy.deepcopy(r).half()  # final eval and evaluate.py load best.pt in half precision
        assert torch.allclose(h(x.half())[0].float(), out, atol=5e-2, rtol=5e-2), f"{name}: half"
        assert torch.allclose(copy.deepcopy(h).fuse(verbose=False)(x.half())[0].float(), out, atol=5e-2, rtol=5e-2)
        r.args = get_cfg()
        r.train()
        for q in r.parameters():
            q.requires_grad_(True)
        loss, _ = r.loss(batch)
        loss.sum().backward()
        for a_ in attrs:
            assert all(q.grad is not None and q.grad.abs().sum() > 0 for q in getattr(r, a_).parameters()), (name, a_)
        assert r.tap.feats is None and all(g.val is None for g in r.grabs.values())
        assert getattr(r, "clsbias", SimpleNamespace(db=None)).db is None and r.film.ab is None
        print("candidate ok", name, round(float(loss.sum()), 3))
    for g in AUGS.values():
        y = g(x, 1.0)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and not torch.allclose(y, x)
        assert torch.equal(g(x, 0.0), x)
    print("mods self-check ok", SimpleNamespace(loss=float(loss.sum())))
