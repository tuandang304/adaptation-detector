"""Architecture and training changes applied to a COCO-initialised YOLO26n, selected by name in configs/node.yaml.

Every change leaves the network equal to the pretrained one at step 0 (new branches end in a zero-initialised layer).

  rep   neck C3k2 (L13, L16, L19, L22): each 3x3 Conv gains a parallel 1x1 branch, merged back into the 3x3 at inference
  lk    backbone C3k2 (L2, L4, L6, L8): residual depthwise 5x5 after the block; stays at inference
  film  shift-conditioned scale/shift on L0, L1, L3: a small network reads photometric statistics of the image and
        the deviation of stem feature statistics from their training-time reference, one forward pass, no gradient
  mix   training only: per-sample feature statistics of L1 and L3 are mixed with another sample of the batch
  cons  training only: a photometrically jittered second view must give the same Detect input features
  align first-layer features are re-aligned per image to their training-time reference through a per-channel gate
  lowpass  before the stride-2 stem convs (L0, L1, L3) the input is blended with a 3x3 low-pass copy; the strength
        grows with the image's excess high-frequency energy over the training reference (3 parameters)
  restore  training only, the signal for film / align / lowpass: a perturbed copy goes through the stem (L0-L3) with
        the stem weights detached and BatchNorm in eval mode, and must give the clean stem output; only the new
        modules receive this gradient. Perturbation: Gaussian noise if lowpass is on, photometric jitter otherwise

Three more clusters, each = a shift generator + an agreement objective + an input-responsive module:
  illum  input module (YOLA-style, arXiv 2410.18398): zero-sum 3x3 kernels on log(image), zero-initialised 1x1 back to
         RGB and added to the image before L0; generator: physics low-light synthesis on 30% of the detection-loss
         images and on the second view; objective: L4 outputs of the two views agree
  style  generator: Normalization Perturbation (arXiv 2211.04393) at L2, L4 (training only); module: SNR-style
         restitution (arXiv 2101.00588) on L4, L6, L8; objective: L8 outputs of the two views agree
  head   module: a small network reads the L10 statistics and predicts per-channel scale/shift for L16, L19, L22 and a
         per-image temperature on the class logits of the Detect heads; generator: PhysAug-style random-kernel
         illumination + planar-wave occlusion (arXiv 2412.11807) on 30% of the detection-loss images and on the second
         view; objective: DivAlign-style (arXiv 2405.14497) agreement of class probabilities and boxes, weighted by the
         clean view's confidence
All active clusters share one second forward whose view composes every active image generator.

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


# --- align ------------------------------------------------------------------------------------------------------
class Align(nn.Module):
    """Forward hook on L0: x + g * (x re-standardised to the training reference - x), g = 0 at step 0."""

    def __init__(self, c: int):
        super().__init__()
        self.register_buffer("ref_mu", torch.zeros(1, c, 1, 1))
        self.register_buffer("ref_sd", torch.ones(1, c, 1, 1))
        self.register_buffer("seen", torch.zeros(()))
        self.g = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, module, inputs, out):
        x = out.float()
        mu, sd = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-3
        if self.training:
            with torch.no_grad():
                m = 1.0 if self.seen == 0 else 0.01
                self.ref_mu.lerp_(mu.mean(0, keepdim=True), m)
                self.ref_sd.lerp_(sd.mean(0, keepdim=True), m)
                self.seen.fill_(1)
        # clones: the next training forward updates the references in place
        return (x + self.g.float() * ((x - mu) / sd * self.ref_sd.clone() + self.ref_mu.clone() - x)).to(out.dtype)


def add_align(model: nn.Module) -> None:
    model.align = Align(model.model[0].conv.out_channels)
    model.model[0].register_forward_hook(model.align)


# --- lowpass ----------------------------------------------------------------------------------------------------
class LowPass(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("ref", torch.zeros(()))
        self.w = nn.Parameter(torch.zeros(n))
        self.dev = None  # per-forward state, cleared by the last hooked layer

    def blend(self, k: int, x: torch.Tensor) -> torch.Tensor:
        if k == 0:
            xf = x.float()
            hf = (xf - F.avg_pool2d(xf, 3, 1, 1, count_include_pad=False)).abs().mean((1, 2, 3))
            if self.training:
                with torch.no_grad():
                    self.ref.lerp_(hf.mean(), 1.0 if self.ref == 0 else 0.01)
            self.dev = F.relu(hf / (self.ref + 1e-6) - 1)
        a = torch.tanh(self.w[k] * self.dev)[:, None, None, None].to(x.dtype)
        if k == len(self.w) - 1:
            self.dev = None
        return x + a * (F.avg_pool2d(x, 3, 1, 1, count_include_pad=False) - x)


class LowHook:
    def __init__(self, lp: LowPass, k: int):
        self.lp, self.k = lp, k

    def __call__(self, module, inputs):
        return (self.lp.blend(self.k, inputs[0]),)


def add_lowpass(model: nn.Module) -> None:
    model.lowpass = LowPass(len(FILM_LAYERS))
    for k, i in enumerate(FILM_LAYERS):
        model.model[i].register_forward_pre_hook(LowHook(model.lowpass, k))


# --- illum -----------------------------------------------------------------------------------------------------
class IIM(nn.Module):
    """Illumination-invariant input module: zero-sum kernels on log(image) cancel a locally constant illumination."""

    def __init__(self, k: int = 8):
        super().__init__()
        self.k = nn.Parameter(torch.randn(k, 3, 3, 3) * 0.1)
        self.out = nn.Conv2d(k, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, module, inputs):  # forward pre-hook on L0
        img = inputs[0]
        w = self.k - self.k.mean((1, 2, 3), keepdim=True)
        f = F.conv2d(torch.log(img.float().clamp(1e-3, 1)), w, padding=1)
        return ((img.float() + self.out(f)).to(img.dtype),)


def dark_aug(img: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Physics low-light synthesis: linearise, darken, shot + read noise, colour-temperature gain, re-gamma."""
    b = img.shape[0]
    u = lambda lo, hi, c=1: torch.empty(b, c, 1, 1, device=img.device).uniform_(lo, hi)
    x = img.float().clamp(0, 1) ** 2.2 * u(0.03, 0.4)
    x = x + torch.randn_like(x) * (x * u(1e-3, 1e-2) + u(1e-3, 2e-2) ** 2).sqrt()
    x = (x * u(0.7, 1.3, 3)).clamp(0, 1) ** (1 / 2.2)
    keep = torch.rand(b, 1, 1, 1, device=img.device) >= p
    return torch.where(keep, img.float(), x).to(img.dtype)


def add_illum(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    model.iim = IIM()
    model.model[0].register_forward_pre_hook(model.iim)
    model.grabs["illum"] = Grab()
    model.model[4].register_forward_hook(model.grabs["illum"])
    model.gens.append("dark")


# --- style ------------------------------------------------------------------------------------------------------
class NormPerturb:
    """Normalization Perturbation: y = a*x + (b - a)*mu with a, b ~ N(1, 0.75) per sample and channel; training only."""

    def __init__(self, p: float = 0.5, std: float = 0.75):
        self.p, self.std = p, std

    def __call__(self, module, inputs, x):
        if not module.training or torch.rand(()) > self.p:
            return None
        mu = x.mean((2, 3), keepdim=True).detach()
        a, b = (1 + self.std * torch.randn(2, *mu.shape, device=x.device)).to(x.dtype)
        return a * x + (b - a) * mu


class SNR(nn.Module):
    """Style restitution: out = F - lam * (1 - g) * (F - IN(F)), g from an SE on the style residual; lam = 0 at step 0."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.se = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c))
        self.lam = nn.Parameter(torch.zeros(c))

    def forward(self, module, inputs, out):
        x = out.float()
        mu, sd = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-3
        res = x - (x - mu) / sd
        g = torch.sigmoid(self.se(res.mean((2, 3))))[:, :, None, None]
        return (x - self.lam[None, :, None, None] * (1 - g) * res).to(out.dtype)


def add_style(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    for i in (2, 4):
        model.model[i].register_forward_hook(NormPerturb())
    model.snr = nn.ModuleList()
    for i in (4, 6, 8):
        m = SNR(model.model[i].cv2.conv.out_channels)
        model.snr.append(m)
        model.model[i].register_forward_hook(m)
    model.grabs["style"] = Grab()  # registered after the SNR hooks, so it sees the restituted L8 output
    model.model[8].register_forward_hook(model.grabs["style"])


# --- head -------------------------------------------------------------------------------------------------------
NECK_OUT = (16, 19, 22)


class HeadCond(nn.Module):
    """Reads the pooled L10 statistics; predicts scale/shift for L16, L19, L22 and a per-image class-logit temperature."""

    def __init__(self, c_in: int, chs: list[int], hidden: int = 32):
        super().__init__()
        self.chs = chs
        self.net = nn.Sequential(nn.Linear(2 * c_in, hidden), nn.SiLU(), nn.Linear(hidden, 2 * sum(chs) + 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.ab = self.t = None  # per-forward state, cleared by the hook after Detect

    def read(self, module, inputs, out):  # forward hook on L10
        x = out.float()
        y = self.net(torch.cat([x.mean((2, 3)), x.std((2, 3))], 1))
        *ab, self.t = y.split([2 * c for c in self.chs] + [1], 1)
        self.ab = ab


class NeckHook:
    def __init__(self, cond: HeadCond, k: int):
        self.cond, self.k = cond, k

    def __call__(self, module, inputs, out):
        a, b = self.cond.ab[self.k].to(out.dtype).chunk(2, 1)
        return out * (1 + a[:, :, None, None]) + b[:, :, None, None]


class TempHook:
    def __init__(self, cond: HeadCond):
        self.cond = cond

    def __call__(self, module, inputs, out):
        return out * torch.exp(self.cond.t).to(out.dtype)[:, :, None, None]


class CondClear:
    def __init__(self, cond: HeadCond):
        self.cond = cond

    def __call__(self, module, inputs, out):
        self.cond.ab = self.cond.t = None


class ReadHook:
    def __init__(self, cond: HeadCond):
        self.cond = cond

    def __call__(self, module, inputs, out):
        self.cond.read(module, inputs, out)


def phys_aug(img: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """PhysAug-style: random 3x3 kernel (non-uniform illumination, rescaled to the image's statistics) blended in, then
    a planar wave of random frequency, direction and phase (particle occlusion)."""
    b, c, h, w = img.shape
    x0 = img.float()
    k = torch.randn(b * c, 1, 3, 3, device=img.device) * 2.0
    y = F.conv2d(x0.reshape(1, b * c, h, w), k, padding=1, groups=b * c).reshape(b, c, h, w)
    y = (y - y.mean((2, 3), keepdim=True)) / (y.std((2, 3), keepdim=True) + 1e-4) * x0.std((2, 3), keepdim=True) \
        + x0.mean((2, 3), keepdim=True)
    u = lambda lo, hi: torch.empty(b, 1, 1, 1, device=img.device).uniform_(lo, hi)
    a = u(0, 1)
    x = a * y + (1 - a) * x0
    yy, xx = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
    om, f, ph = u(0, 3.1416), u(2, 40) / max(h, w), u(0, 6.2832)
    wave = torch.cos(2 * 3.1416 * f * (xx * torch.cos(om) + yy * torch.sin(om)) + ph)
    x = (x + u(0, 0.3) * wave).clamp(0, 1)
    keep = torch.rand(b, 1, 1, 1, device=img.device) >= p
    return torch.where(keep, x0, x).to(img.dtype)


def add_head(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    det = model.model[-1]
    model.cond = HeadCond(model.model[10].cv2.conv.out_channels,
                          [model.model[i].cv2.conv.out_channels for i in NECK_OUT])
    model.model[10].register_forward_hook(ReadHook(model.cond))
    for k, i in enumerate(NECK_OUT):
        model.model[i].register_forward_hook(NeckHook(model.cond, k))
    for heads in (det.cv3, getattr(det, "one2one_cv3", None) or []):
        for h in heads:
            h.register_forward_hook(TempHook(model.cond))
    det.register_forward_hook(CondClear(model.cond))
    model.grabs["head"] = Grab()
    det.register_forward_hook(model.grabs["head"])
    model.gens.append("phys")


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
class Grab:
    """Keeps a layer's training-time output (Detect: its one2many predictions) for an agreement loss."""

    def __init__(self):
        self.val = None

    def __call__(self, module, inputs, out):
        if module.training:
            self.val = out["one2many"] if isinstance(out, dict) and "one2many" in out else out


def cos_gap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - F.cosine_similarity(a.detach().float(), b.float(), dim=1).mean()


def pred_gap(p1: dict, p2: dict) -> torch.Tensor:
    """DivAlign-style: augmented-view class probabilities and boxes follow the clean view, weighted by its confidence."""
    s1, s2 = p1["scores"].detach().float(), p2["scores"].float()
    t = torch.sigmoid(s1)
    w = t.amax(1)  # (B, anchors)
    cls = F.binary_cross_entropy_with_logits(s2, t, reduction="none").mean(1)
    b1, b2 = p1["boxes"].detach().float(), p2["boxes"].float()
    box = F.smooth_l1_loss(b2, b1, reduction="none").mean(1)
    return ((cls + box) * w).sum() / (w.sum() + 1e-6)


AUGS = {"dark": dark_aug, "phys": phys_aug}


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
    cons_weight = restore_weight = 1.0
    grabs: dict = {}
    gens: list = []

    def restore_loss(self, img: torch.Tensor) -> torch.Tensor:
        stem = self.model[:4]
        parts = [stem] + [getattr(self, n) for n in ("film", "align", "lowpass") if hasattr(self, n)]
        for m in parts:  # eval: BatchNorm as at inference, no mixing, references not updated
            m.eval()
        try:
            with torch.no_grad():
                target = stem(img).float()
            view = img.float()
            if hasattr(self, "lowpass"):
                view = view + torch.randn_like(view) * torch.empty(len(view), 1, 1, 1, device=view.device).uniform_(0, 0.1)
            else:
                view = style_jitter(view)
            frozen = {k: v.detach() for k, v in stem.named_parameters()}
            out = torch.func.functional_call(stem, frozen, (view.clamp(0, 1).to(img.dtype),)).float()
        finally:
            for m in parts:
                m.train()
        return F.mse_loss(out, target) / (target.var() + 1e-6)

    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        if not self.training or preds is not None:
            return loss, items
        scale = batch["img"].shape[0] / loss.numel()  # the trainer sums the loss vector, already scaled by the batch size
        grabs = getattr(self, "grabs", {})
        if hasattr(self, "tap") or grabs:
            f1 = None
            if hasattr(self, "tap"):
                f1, self.tap.feats = self.tap.feats, None
            g1 = {k: g.val for k, g in grabs.items()}
            view = style_jitter(batch["img"]) if hasattr(self, "tap") else batch["img"]
            p = 1.0 if len(self.gens) == 1 else 0.5  # several generators: each is applied to half of the view
            for g in self.gens:
                view = AUGS[g](view, p)
            self.forward(view)
            for k, g in grabs.items():
                if k == "head":
                    c = pred_gap(g1[k], g.val)
                else:
                    c = cos_gap(g1[k], g.val)
                loss = loss + self.cons_weight * c * scale
                g.val = None
            if hasattr(self, "tap"):
                f2, self.tap.feats = self.tap.feats, None
                c = sum(cos_gap(a, b) for a, b in zip(f1, f2)) / len(f1)
                loss = loss + self.cons_weight * c * scale
        if getattr(self, "restore", False):
            loss = loss + self.restore_weight * self.restore_loss(batch["img"]) * scale
        return loss, items


def add_cons(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    model.tap = DetectTap()
    model.model[-1].register_forward_pre_hook(model.tap)


def add_restore(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    model.restore = True


def _prep(model: nn.Module) -> None:  # per-instance containers before any cluster adds to them
    if "grabs" not in model.__dict__:
        model.grabs, model.gens = {}, []


ADD = {"rep": add_rep, "lk": add_lk, "film": add_film, "mix": add_mix, "cons": add_cons, "align": add_align,
       "lowpass": add_lowpass, "restore": add_restore, "illum": add_illum, "style": add_style, "head": add_head}


def apply(model: nn.Module, mods: list[str]) -> nn.Module:
    """Apply the named changes to a model whose pretrained weights are already loaded."""
    alive = list(model.parameters())  # held so that a freed tensor's id cannot be reused by a new one
    before = {id(p) for p in alive}
    n0 = sum(p.numel() for p in model.parameters())
    _prep(model)
    for name in mods:
        ADD[name](model)
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

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        for g in getattr(self.model, "gens", []):  # generators also feed the detection loss, on 30% of the images
            batch["img"] = AUGS[g](batch["img"], 0.3)
        return batch


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
    for mods, new in ((["film", "restore"], "film"), (["align", "restore"], "align"), (["lowpass", "restore"], "lowpass")):
        r = apply(copy.deepcopy(base), mods)  # restore alone must reach the new module and nothing in the stem
        assert torch.allclose(r.eval()(x)[0], ref, atol=1e-4), f"{mods} not identical at step 0"
        r.train()
        for p in r.parameters():
            p.requires_grad_(True)
        if new == "film":  # the last film layer is zero, so nothing upstream of it has a gradient yet
            nn.init.normal_(r.film.net[2].weight, std=0.05)
        r.restore_loss(x).backward()
        got = {n for n, p in r.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        assert got and all(n.startswith(new) for n in got), (mods, sorted(got)[:5])
        assert getattr(r, "film", r).training and r.model[0].training
    # the three clusters: identity at step 0, every new module reached by a gradient, fuse, checkpoint, generators
    full = ["film", "mix", "cons", "illum", "style", "head"]
    for mods in (["illum"], ["style"], ["head"], full):
        r = apply(copy.deepcopy(base), mods).eval()
        assert torch.allclose(r(x)[0], ref, atol=1e-4), f"{mods} not identical at step 0"
        assert r.cond.t is None if hasattr(r, "cond") else True
        for q in r.parameters():
            if q.abs().sum() == 0:
                nn.init.normal_(q, std=0.05)
        out = r(x)[0]
        assert not torch.allclose(out, ref, atol=1e-3), mods
        buf = io.BytesIO()
        torch.save(r, buf)
        buf.seek(0)
        assert torch.allclose(torch.load(buf, weights_only=False).eval()(x)[0], out, atol=1e-5), f"{mods} checkpoint"
        assert torch.allclose(copy.deepcopy(r).fuse(verbose=False)(x)[0], out, atol=1e-3), f"{mods} fuse"
        r.args = get_cfg()
        r.train()
        for q in r.parameters():
            q.requires_grad_(True)
        loss, _ = r.loss(batch)
        loss.sum().backward()
        new = [n for n in ("iim", "snr", "cond", "film") if hasattr(r, n)]
        for n in new:
            assert all(q.grad is not None and q.grad.abs().sum() > 0 for q in getattr(r, n).parameters()), (mods, n)
        assert all(g.val is None for g in r.grabs.values()) and (not hasattr(r, "cond") or r.cond.t is None)
        print("cluster ok", mods, round(float(loss.sum()), 3))
    for g in AUGS.values():
        y = g(x, 1.0)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and not torch.allclose(y, x)
        assert torch.equal(g(x, 0.0), x)
    print("mods self-check ok", SimpleNamespace(loss=float(loss.sum())))
