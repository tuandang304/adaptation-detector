"""Architecture and training changes applied to a COCO-initialised YOLO26n, selected by name in configs/node.yaml.

Every change leaves the network equal to the pretrained one at step 0 (new branches end in a zero-initialised layer).

  rep   neck C3k2 (L13, L16, L19, L22): each 3x3 Conv gains a parallel 1x1 branch, merged back into the 3x3 at inference
  lk    backbone C3k2 (L2, L4, L6, L8): residual depthwise 5x5 after the block; stays at inference
  film  shift-conditioned scale/shift on L0, L1, L3: a small network reads photometric statistics of the image and
        the deviation of stem feature statistics from their training-time reference, one forward pass, no gradient
  mix   training only: per-sample feature statistics of L1 and L3 are mixed with another sample of the batch
  cons  training only: a photometrically jittered second view must give the same Detect input features

  dbb   every plain 3x3 conv (groups 1) inside L2, L4, L6, L8, L13, L16, L19, L22 and the stride-2 convs L5, L7, L17,
        L20 gain three Diverse-Branch-Block branches (1x1; 1x1 -> 3x3; 1x1 -> avgpool3), each ending in a BatchNorm with
        weight 0; fuse() folds them into the single 3x3 conv, so the deployed network has the stock layers and cost

Small clusters (each 2 layers, on top of film + mix + cons):
  dbbp3      DBB on the 3x3 convs inside L2, L4 (folded at inference)
  spddetail  L5 output += Conv1x1(space_to_depth(L5 input)), BN weight 0 at step 0 (stays) + DBB inside L6 (folded)
  coordatt   coordinate-attention gates on the L16, L19 outputs: x*(1+0.2tanh(s_h))*(1+0.2tanh(s_w)), zero-init (stays)
  gcctx      global-context block on the L9 output: x + T(softmax-pooled context), last conv zero-init (stays)
             + DBB inside L22 (folded)

Round 3 (each on top of film + mix + cons + coordatt; the new modules get a training signal tied to shift):
  fusegate    SANF: L10-conditioned per-group gates 1 + 0.5 tanh on both inputs of Concat L12, L15, L18, L21
  crossalign  spatial gates from the discrepancy of the two inputs of Concat L12, L15, L18, L21; lateral input noised in
              the cons pass
  ndn         x + gamma * s * high-pass(x) on L4, L6, L8; s from an estimator on L4 regressed to synthetic noise/veil
  sorc        residual bottleneck adapters on L2, L4, L6, detached in the clean detection pass
  fusegate, ndn and sorc also put night (dark + shot/read noise) or rain (veil + streaks) on half of the cons view;
  fusegate, crossalign and sorc are detached in the clean detection pass, so they learn only from the cons view.

Round 4 (stronger structural changes, each on top of film + mix + cons + coordatt, no synthetic generator):
  dynconv   every plain 3x3 Conv in L6, L8 and the stride-2 Conv L7: conv(x) + A(pi(x) * B(x)), input-routed low-rank
            kernel delta (K=3 experts of rank 8), A zero-init, BN folded into conv and A at fuse
  p2        slim stride-4 detection level from L2 and the P3 input, light heads in both Detect branches; class
            logits offset -30 fading to 0 over the first 1500 training forwards
  dysample  learned point-sampling upsampling at L11, L14 as a zero-gated residual on nearest

Hooks are module-level callables so that checkpoints (pickled models) and the EMA deep copy keep them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2, Conv, Conv2, RepConv
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


class Phase:
    """Shared by the round-3 modules: which view the forward is on, extra loss terms, degradation targets."""

    def __init__(self):
        self.shift, self.aux, self.target, self.degrade = False, [], None, None


def _phase(model: nn.Module) -> Phase:
    if not hasattr(model, "phase"):
        model.phase = Phase()
    return model.phase


def _streak_kernel(n: int, theta: float, device) -> torch.Tensor:
    r = torch.arange(n, device=device) - n // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    return ((xx * math.cos(theta) - yy * math.sin(theta)).abs() < 0.5).float()[None, None]


def degrade(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Training only, on the cons view: per sample none (p 1/2), night = dark + shot/read noise (1/4),
    rain = veil + streaks (1/4). Returns the view and per-sample targets (log1p(noise std / 0.01), 5 * veil)."""
    b, dev = img.shape[0], img.device
    x = img.float()
    u = lambda lo, hi: torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)
    kind = torch.randint(0, 4, (b, 1, 1, 1), device=dev)
    night, rain = (kind == 2).float(), (kind == 3).float()
    dark, shot, read = x * u(0.1, 0.5), u(0.002, 0.02), u(0.01, 0.05)
    noisy = dark + (dark * shot).sqrt() * torch.randn_like(x) + read * torch.randn_like(x)
    drops = (torch.rand(b, 1, *x.shape[2:], device=dev) < u(0.001, 0.005)).float()
    k = _streak_kernel(15, float(torch.empty(()).uniform_(-0.35, 0.35)), dev)
    veil = u(0.05, 0.3)
    wet = x * (1 - veil) + 0.8 * veil + u(0.2, 0.5) * F.conv2d(drops, k, padding=7).clamp(0, 1)
    out = x * (1 - night - rain) + noisy * night + wet * rain
    sigma = (read + (dark.mean((1, 2, 3), keepdim=True) * shot).sqrt()) * night
    target = torch.cat([torch.log1p(sigma / 0.01), 5 * veil * rain], 1).flatten(1)
    return out.clamp(0, 1).to(img.dtype), target


class ConsistModel(DetectionModel):
    cons_weight = 1.0

    def loss(self, batch, preds=None):
        ph = getattr(self, "phase", None)
        if ph is not None:
            ph.aux, ph.target = [], None
        loss, items = super().loss(batch, preds)
        if not self.training or preds is not None:
            return loss, items
        f1, self.tap.feats = self.tap.feats, None
        view = style_jitter(batch["img"])
        if ph is not None:
            if ph.degrade is not None:
                view, ph.target = ph.degrade(view)
            ph.shift = True
        try:
            self.forward(view)
        finally:
            if ph is not None:
                ph.shift = False
        f2, self.tap.feats = self.tap.feats, None
        c = sum(1 - F.cosine_similarity(a.detach().float(), b.float(), dim=1).mean() for a, b in zip(f1, f2)) / len(f1)
        if ph is not None:
            c = c + sum(ph.aux, torch.zeros((), device=c.device))
            ph.aux, ph.target = [], None
        # the trainer sums the loss vector, which is already scaled by the batch size
        return loss + self.cons_weight * c * batch["img"].shape[0] / loss.numel(), items


def add_cons(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    model.tap = DetectTap()
    model.model[-1].register_forward_pre_hook(model.tap)


# --- dbb ---------------------------------------------------------------------------------------------------------
DBB_INNER, DBB_DOWN = (2, 4, 6, 8, 13, 16, 19, 22), (5, 7, 17, 20)


def _fold(w: torch.Tensor, bn: nn.BatchNorm2d, b: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    t = bn.weight.float() / (bn.running_var.float() + bn.eps).sqrt()
    b = torch.zeros_like(bn.bias, dtype=torch.float32) if b is None else b.float()
    return w.float() * t[:, None, None, None], bn.bias.float() + (b - bn.running_mean.float()) * t


class DBB(Conv2):
    """A pretrained 3x3 Conv plus three zero-gated training branches; a Conv2 so that BaseModel.fuse() calls fuse_convs."""

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + self.bn1(self.b1(x)) + self.bn2(self.b2b(self.b2a(x)))
                        + self.bn3(self.pool(self.b3(x))))

    def fuse_convs(self):
        """Fold every branch (and the main BatchNorm) into one 3x3 conv with bias; leave an identity BatchNorm behind,
        which BaseModel.fuse() then folds away with Conv.forward_fuse."""
        w0, b0 = _fold(self.conv.weight, self.bn, self.conv.bias)
        w1, b1 = _fold(F.pad(self.b1.weight, [1, 1, 1, 1]), self.bn1)
        w2, b2 = _fold(torch.einsum("omhw,mi->oihw", self.b2b.weight.float(), self.b2a.weight[:, :, 0, 0].float()), self.bn2)
        w3, b3 = _fold(self.b3.weight.float().expand(-1, -1, 3, 3) / 9, self.bn3)
        c = self.conv
        conv = nn.Conv2d(c.in_channels, c.out_channels, 3, c.stride, 1, bias=True).to(c.weight.device)
        conv.weight.data, conv.bias.data = (w0 + w1 + w2 + w3).to(c.weight.dtype), (b0 + b1 + b2 + b3).to(c.weight.dtype)
        self.conv = conv
        bn = self.bn
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1 - bn.eps)
        for n in ("b1", "bn1", "b2a", "b2b", "bn2", "b3", "pool", "bn3"):
            delattr(self, n)
        self.forward = super(Conv2, self).forward


def to_dbb(m: Conv) -> None:
    ci, co, s = m.conv.in_channels, m.conv.out_channels, m.conv.stride
    dev = m.conv.weight.device
    m.__class__ = DBB
    m.b1 = nn.Conv2d(ci, co, 1, s, 0, bias=False)
    m.b2a, m.b2b = nn.Conv2d(ci, ci, 1, bias=False), nn.Conv2d(ci, co, 3, s, 1, bias=False)
    nn.init.dirac_(m.b2a.weight)  # the 1x1 before the 3x3 starts as identity (as in DBB)
    m.b3, m.pool = nn.Conv2d(ci, co, 1, bias=False), nn.AvgPool2d(3, s, 1, count_include_pad=True)
    m.bn1, m.bn2, m.bn3 = nn.BatchNorm2d(co), nn.BatchNorm2d(co), nn.BatchNorm2d(co)
    for bn in (m.bn1, m.bn2, m.bn3):
        nn.init.zeros_(bn.weight)  # every branch adds 0 at step 0
    for k in ("b1", "b2a", "b2b", "b3", "bn1", "bn2", "bn3"):
        getattr(m, k).to(dev)


def dbb_on(model: nn.Module, inner=(), down=()) -> None:
    assert not any(isinstance(k, RepConv) for k in model.modules()), "dbb and rep rewrite the same convs; use one"
    is3 = lambda c: type(c) is Conv and c.conv.kernel_size == (3, 3) and c.conv.groups == 1
    for i in inner:
        for c in [c for c in model.model[i].modules() if is3(c)]:
            to_dbb(c)
    for i in down:
        assert is3(model.model[i]) and model.model[i].conv.stride == (2, 2), i
        to_dbb(model.model[i])


def add_dbb(model: nn.Module) -> None:
    dbb_on(model, DBB_INNER, DBB_DOWN)


def add_dbbp3(model: nn.Module) -> None:
    dbb_on(model, (2, 4))


class SPDBranch(nn.Module):
    """Forward hook on a stride-2 Conv: out + Conv1x1(space_to_depth(input)); the branch's BN weight is 0 at step 0."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = Conv(4 * c_in, c_out, 1, act=False)
        nn.init.zeros_(self.proj.bn.weight)

    def forward(self, module, inputs, out):
        x = inputs[0]
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return out + self.proj(x)


def add_spddetail(model: nn.Module) -> None:
    m5 = model.model[5]
    model.spd = SPDBranch(m5.conv.in_channels, m5.conv.out_channels)
    m5.register_forward_hook(model.spd)
    dbb_on(model, (6,))


class CoordAtt(nn.Module):
    """Coordinate attention as a residual gate: x * (1 + 0.2 tanh(s_h)) * (1 + 0.2 tanh(s_w)); s = 0 at step 0."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(c // r, 8)
        self.shared = Conv(c, mid, 1)  # 1x1 + BN + SiLU, folded by fuse()
        self.ch, self.cw = nn.Conv2d(mid, c, 1), nn.Conv2d(mid, c, 1)
        for k in (self.ch, self.cw):
            nn.init.zeros_(k.weight)
            nn.init.zeros_(k.bias)

    def forward(self, module, inputs, out):
        h, w = out.shape[2:]
        y = self.shared(torch.cat([out.mean(3, keepdim=True), out.mean(2, keepdim=True).transpose(2, 3)], 2))
        sh, sw = y.split([h, w], 2)
        gh = 1 + 0.2 * torch.tanh(self.ch(sh))
        gw = 1 + 0.2 * torch.tanh(self.cw(sw.transpose(2, 3)))
        return out * gh * gw


def add_coordatt(model: nn.Module) -> None:
    model.coordatt = nn.ModuleList()
    for i in (16, 19):
        a = CoordAtt(model.model[i].cv2.conv.out_channels)
        model.coordatt.append(a)
        model.model[i].register_forward_hook(a)


class GCBlock(nn.Module):
    """GCNet global context: x + T(sum softmax(Wk x) * x), T = 1x1 -> LayerNorm -> ReLU -> 1x1 (last one zero-init)."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(c // r, 8)
        self.key = nn.Conv2d(c, 1, 1)
        self.t1, self.ln, self.t2 = nn.Conv2d(c, mid, 1), nn.LayerNorm([mid, 1, 1]), nn.Conv2d(mid, c, 1)
        nn.init.zeros_(self.t2.weight)
        nn.init.zeros_(self.t2.bias)

    def forward(self, module, inputs, out):
        b, c = out.shape[:2]
        a = self.key(out).flatten(2).float().softmax(-1).to(out.dtype)  # (b, 1, hw)
        ctx = torch.bmm(out.flatten(2), a.transpose(1, 2)).view(b, c, 1, 1)
        return out + self.t2(F.relu(self.ln(self.t1(ctx))))


def add_gcctx(model: nn.Module) -> None:
    model.gc = GCBlock(model.model[9].cv2.conv.out_channels)
    model.model[9].register_forward_hook(model.gc)
    dbb_on(model, (22,))


# --- round 3 (on top of film + mix + cons + coordatt; all need cons for their training signal) -------------------
CONCATS = (12, 15, 18, 21)  # inputs: [path feature, lateral/skip feature]


class _Hook:
    """Module-level callable that forwards a hook to owner.<name>(k, ...), so checkpoints can pickle it."""

    def __init__(self, owner: nn.Module, name: str, k: int):
        self.owner, self.name, self.k = owner, name, k

    def __call__(self, *args):
        return getattr(self.owner, self.name)(self.k, *args)


class FuseGate(nn.Module):
    """SANF: GAP(L10) -> 16 -> per-group gates 1 + 0.5 tanh on both inputs of each neck Concat; last layer zero-init.
    Detached in the clean detection pass, so the gates learn only from cons on the degraded view."""
    G = 8

    def __init__(self, c_ctx: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_ctx, 16), nn.SiLU(), nn.Linear(16, 2 * len(CONCATS) * self.G))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.phase, self.g = None, None

    def ctx(self, k, module, inputs, out):
        g = 1 + 0.5 * torch.tanh(self.mlp(out.mean((2, 3))))
        if module.training and not self.phase.shift:
            g = g.detach()
        self.g = g.view(g.shape[0], -1, self.G)

    def gate(self, k, module, inputs):
        xs = list(inputs[0])
        for j in range(2):
            c = xs[j].shape[1]
            xs[j] = xs[j] * self.g[:, 2 * k + j].repeat_interleave(c // self.G, 1)[:, :, None, None]
        if k == len(CONCATS) - 1:
            self.g = None  # holds the graph; cleared so that deep copies of the model work
        return (xs,)


def add_fusegate(model: nn.Module) -> None:
    model.fusegate = FuseGate(model.model[10].cv2.conv.out_channels)
    model.fusegate.phase = ph = _phase(model)
    ph.degrade = degrade
    model.model[10].register_forward_hook(_Hook(model.fusegate, "ctx", 0))
    for k, i in enumerate(CONCATS):
        model.model[i].register_forward_pre_hook(_Hook(model.fusegate, "gate", k))


class CrossAlign(nn.Module):
    """Spatial gates from the discrepancy of the two Concat inputs: |1x1(a) - 1x1(b)| -> dw3 -> zero-init 1x1 -> 2 masks,
    inputs scaled by 1 + 0.2 tanh(mask). Detached in the clean pass; in the cons pass the lateral input gets noise/blur."""

    def __init__(self, ca: int, cb: int, mid: int = 16):
        super().__init__()
        self.pa, self.pb, self.dw = Conv(ca, mid, 1), Conv(cb, mid, 1), Conv(mid, mid, 3, 1, None, mid)
        self.m = nn.Conv2d(mid, 2, 1)
        nn.init.zeros_(self.m.weight)
        nn.init.zeros_(self.m.bias)
        self.phase = None

    def forward(self, module, inputs):
        a, b = inputs[0]
        if module.training and self.phase.shift:
            sd = b.detach().std((2, 3), keepdim=True)
            b = b + 0.5 * torch.rand(b.shape[0], 1, 1, 1, device=b.device, dtype=b.dtype) * sd * torch.randn_like(b)
            if torch.rand(()) < 0.5:
                b = F.avg_pool2d(b, 3, 1, 1, count_include_pad=False)
        m = 0.2 * torch.tanh(self.m(self.dw((self.pa(a) - self.pb(b)).abs())))
        if module.training and not self.phase.shift:
            m = m.detach()
        return ([a * (1 + m[:, :1]), b * (1 + m[:, 1:])],)


def add_crossalign(model: nn.Module) -> None:
    shapes = {}
    hs = [model.model[i].register_forward_pre_hook(lambda mod, inp, i=i: shapes.__setitem__(i, [t.shape[1] for t in inp[0]]))
          for i in CONCATS]
    was = model.training
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, 64, 64))
    model.train(was)
    for h in hs:
        h.remove()
    model.crossalign = nn.ModuleList()
    ph = _phase(model)
    for i in CONCATS:
        g = CrossAlign(*shapes[i])
        g.phase = ph
        model.crossalign.append(g)
        model.model[i].register_forward_pre_hook(g)


class NDN(nn.Module):
    """out = x + gamma * s * (x - avg3(x)) on L4, L6, L8; gamma zero-init. s = 1 + 0.5 tanh(a . p) per image, p from a
    zero-init estimator on GAP|high-pass of L4|, regressed on the cons view to its synthetic noise level and veil."""

    def __init__(self, chs: list[int]):
        super().__init__()
        self.est = nn.Sequential(nn.Linear(chs[0], 16), nn.SiLU(), nn.Linear(16, 2))
        nn.init.zeros_(self.est[2].weight)
        nn.init.zeros_(self.est[2].bias)
        self.a = nn.Parameter(torch.ones(2))
        self.gamma = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in chs])
        self.phase, self.s = None, None

    def modulate(self, k, module, inputs, x):
        h = x - F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        if k == 0:
            p = self.est(h.abs().mean((2, 3)))
            if module.training and self.phase.shift and self.phase.target is not None:
                self.phase.aux.append(0.1 * F.mse_loss(p.float(), self.phase.target.float()))
            self.s = 1 + 0.5 * torch.tanh(p @ self.a)
        out = x + self.gamma[k].view(1, -1, 1, 1) * self.s.view(-1, 1, 1, 1) * h
        if k == len(self.gamma) - 1:
            self.s = None
        return out


def add_ndn(model: nn.Module) -> None:
    layers = (4, 6, 8)
    model.ndn = NDN([model.model[i].cv2.conv.out_channels for i in layers])
    model.ndn.phase = ph = _phase(model)
    ph.degrade = degrade
    for k, i in enumerate(layers):
        model.model[i].register_forward_hook(_Hook(model.ndn, "modulate", k))


class Adapter(nn.Module):
    """SORC: x + up(dw3(down(x))), up zero-init. Detached in the clean detection pass (plus a 1e-3 penalty on its
    output there), so it learns only from cons on the degraded view."""

    def __init__(self, c: int):
        super().__init__()
        mid = c // 4
        self.down, self.dw, self.up = Conv(c, mid, 1), Conv(mid, mid, 3, 1, None, mid), nn.Conv2d(mid, c, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.phase = None

    def forward(self, module, inputs, x):
        d = self.up(self.dw(self.down(x)))
        if module.training and not self.phase.shift:
            self.phase.aux.append(1e-3 * d.float().pow(2).mean())
            return x + d.detach()
        return x + d


def add_sorc(model: nn.Module) -> None:
    model.sorc = nn.ModuleList()
    ph = _phase(model)
    ph.degrade = degrade
    for i in (2, 4, 6):
        a = Adapter(model.model[i].cv2.conv.out_channels)
        a.phase = ph
        model.sorc.append(a)
        model.model[i].register_forward_hook(a)


# --- round 4: stronger structural changes (on top of film + mix + cons + coordatt; no synthetic generator) ---------
class DynConv(Conv2):
    """Pretrained Conv plus an input-routed low-rank kernel delta: conv(x) + A(pi(x) * B(x)). B = K*r 3x3 filters,
    A = zero-init 1x1, pi = softmax of a zero-init router on GAP(x). A Conv2 so that BaseModel.fuse() calls fuse_convs,
    which folds the BatchNorm into conv and into A; the delta stays (static convs + per-image channel scaling)."""

    def _delta(self, x):
        p = torch.softmax(self.route(x.mean((2, 3))), 1)
        return self.dA(self.dB(x) * p.repeat_interleave(self.dB.out_channels // p.shape[1], 1)[:, :, None, None])

    def forward(self, x):
        return self.act(self.bn(self.conv(x) + self._delta(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x) + self._delta(x))

    def fuse_convs(self):
        bn, c = self.bn, self.conv
        t = bn.weight.float() / (bn.running_var.float() + bn.eps).sqrt()
        w, b = _fold(c.weight, bn, c.bias)
        conv = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=True).to(c.weight.device)
        conv.weight.data, conv.bias.data = w.to(c.weight.dtype), b.to(c.weight.dtype)
        self.conv = conv
        self.dA.weight.data = (self.dA.weight.float() * t[:, None, None, None]).to(self.dA.weight.dtype)
        bn.weight.data.fill_(1)  # identity BatchNorm, folded away by BaseModel.fuse()
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1 - bn.eps)


def to_dyn(m: Conv, k: int = 3, r: int = 8) -> DynConv:
    c = m.conv
    m.__class__ = DynConv
    m.route = nn.Linear(c.in_channels, k)
    m.dB = nn.Conv2d(c.in_channels, k * r, c.kernel_size, c.stride, c.padding, bias=False)
    m.dA = nn.Conv2d(k * r, c.out_channels, 1, bias=False)
    for z in (m.route.weight, m.route.bias, m.dA.weight):
        nn.init.zeros_(z)
    for n in ("route", "dB", "dA"):
        getattr(m, n).to(c.weight.device)
    return m


def add_dynconv(model: nn.Module) -> None:
    """D1: every plain 3x3 Conv inside L6 and L8, and the stride-2 Conv L7."""
    assert not any(isinstance(q, (DBB, RepConv)) for q in model.modules()), "dynconv rewrites the same convs as dbb/rep"
    convs = [q for i in (6, 8) for q in model.model[i].modules()
             if type(q) is Conv and q.conv.kernel_size == (3, 3) and q.conv.groups == 1] + [model.model[7]]
    model.dyn = nn.ModuleList(nn.ModuleList([d.route, d.dB, d.dA]) for d in map(to_dyn, convs))


class P2Level(nn.Module):
    """D2: slim stride-4 detection level. Feature = DW3(1x1(L2 out) + up2(1x1(P3 input))), 16 channels; light DW3 + 1x1
    box and class heads in both Detect branches. Class logits get an offset that fades from -30 to 0 over the first
    `warm` training forwards (a float buffer, so the EMA copy follows), so the output equals the pretrained one at step 0."""

    def __init__(self, c2: int, c3: int, c: int = 16, warm: int = 1500):
        super().__init__()
        self.a, self.b, self.mix = Conv(c2, c, 1), Conv(c3, c, 1), Conv(c, c, 3, 1, None, c)
        self.register_buffer("seen", torch.zeros(()))
        self.warm, self.src = warm, None

    def offset(self):
        return -30.0 * (1 - self.seen.float() / self.warm).clamp(0, 1)

    def grab(self, k, module, inputs, out):
        self.src = out

    def expand(self, k, module, inputs):
        x = list(inputs[0])
        if module.training:
            self.seen += 1
        p2 = self.mix(self.a(self.src) + F.interpolate(self.b(x[0]), scale_factor=2.0, mode="nearest"))
        self.src = None  # holds the graph; cleared so that deep copies of the model work
        return ([p2] + x,)

    def cls_offset(self, k, module, inputs, out):
        return out + self.offset().to(out.dtype)


def add_p2(model: nn.Module) -> None:
    det = model.model[-1]
    assert det.nl == 3 and det.f == [16, 19, 22]
    c = 16
    model.p2 = p2 = P2Level(model.model[2].cv2.conv.out_channels, model.model[16].cv2.conv.out_channels, c)
    model.model[2].register_forward_hook(_Hook(p2, "grab", 0))
    det.register_forward_pre_hook(_Hook(p2, "expand", 0))
    branches = [("cv2", "cv3")] + ([("one2one_cv2", "one2one_cv3")] if getattr(det, "one2one_cv2", None) is not None else [])
    for box_name, cls_name in branches:
        box = nn.Sequential(Conv(c, c, 3, 1, None, c), nn.Conv2d(c, 4 * det.reg_max, 1))
        cls = nn.Sequential(Conv(c, c, 3, 1, None, c), nn.Conv2d(c, det.nc, 1))
        box[-1].bias.data[:] = 2.0  # as Detect.bias_init
        cls[-1].bias.data[:] = math.log(5 / det.nc / (640 / 4) ** 2)
        cls[-1].register_forward_hook(_Hook(p2, "cls_offset", 0))
        getattr(det, box_name).insert(0, box)
        getattr(det, cls_name).insert(0, cls)
    det.nl, det.shape = 4, None
    det.stride = torch.cat([torch.tensor([4.0]), det.stride])
    model.stride = det.stride
    model.to(next(det.parameters()).device)


class DySample(nn.Module):
    """D3: learned point-sampling 2x upsampling as a residual on nearest:
    out = nearest(x) + g * (grid_sample(x, base + 0.25 * pixel_shuffle(offset(x))) - nearest(x)), g zero-init."""

    def __init__(self, c: int):
        super().__init__()
        self.off = nn.Conv2d(c, 8, 1)
        nn.init.normal_(self.off.weight, std=0.001)
        nn.init.zeros_(self.off.bias)
        self.g = nn.Parameter(torch.zeros(c))

    def forward(self, module, inputs, out):
        x = inputs[0]
        h, w = x.shape[2:]
        o = F.pixel_shuffle(self.off(x) * 0.25, 2).float()  # (b, 2, 2h, 2w): x, y offsets in input pixels
        ys = (torch.arange(2 * h, device=x.device, dtype=torch.float32) + 0.5) / h - 1
        xs = (torch.arange(2 * w, device=x.device, dtype=torch.float32) + 0.5) / w - 1
        base = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)  # (2h, 2w, 2) as (x, y)
        grid = base + o.permute(0, 2, 3, 1) * torch.tensor([2.0 / w, 2.0 / h], device=x.device)
        s = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(out.dtype)
        return out + self.g.view(1, -1, 1, 1) * (s - out)


def add_dysample(model: nn.Module) -> None:
    assert all(isinstance(model.model[i], nn.Upsample) for i in (11, 14))
    model.dysample = nn.ModuleList()
    for i, src in ((11, 10), (14, 13)):
        d = DySample(model.model[src].cv2.conv.out_channels)
        model.dysample.append(d)
        model.model[i].register_forward_hook(d)


ROUND3 = ("fusegate", "crossalign", "ndn", "sorc")
ROUND4 = {"dynconv": "dyn", "p2": "p2", "dysample": "dysample"}  # mod -> attribute holding its modules
ADD = {"rep": add_rep, "lk": add_lk, "film": add_film, "mix": add_mix, "cons": add_cons, "dbb": add_dbb,
       "dbbp3": add_dbbp3, "spddetail": add_spddetail, "coordatt": add_coordatt, "gcctx": add_gcctx,
       "fusegate": add_fusegate, "crossalign": add_crossalign, "ndn": add_ndn, "sorc": add_sorc,
       "dynconv": add_dynconv, "p2": add_p2, "dysample": add_dysample}


def apply(model: nn.Module, mods: list[str]) -> nn.Module:
    """Apply the named changes to a model whose pretrained weights are already loaded."""
    alive = list(model.parameters())  # held so that a freed tensor's id cannot be reused by a new one
    before = {id(p) for p in alive}
    n0 = sum(p.numel() for p in model.parameters())
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


if __name__ == "__main__":  # self-check: identity at step 0, fuse, checkpoint round trip, one training step
    import copy
    import io

    from ultralytics import YOLO

    torch.manual_seed(0)
    base = YOLO("yolo26n.pt").model.float().eval()
    x = torch.rand(2, 3, 256, 256)
    ref = base(x)[0]
    m = apply(copy.deepcopy(base), [k for k in ADD if k not in ("dbb", "dbbp3", "spddetail", "gcctx") + ROUND3 + tuple(ROUND4)])  # dbb and rep both rewrite the neck 3x3 convs
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
    # dbb on top of film + mix + cons: identity at step 0; fused = unfused; fused cost = stock + film; half; checkpoint
    r = apply(copy.deepcopy(base), ["film", "mix", "cons", "dbb"]).eval()
    n_dbb = sum(isinstance(k, DBB) for k in r.modules())
    assert torch.allclose(r(x)[0], ref, atol=1e-4), "dbb: not identical at step 0"
    r.train()
    with torch.no_grad():  # move BatchNorm statistics off their initial values so that folding them is tested
        for _ in range(3):
            r(torch.rand(2, 3, 256, 256))
    r.tap.feats = None
    r.eval()
    for q in r.parameters():
        if q.abs().sum() == 0:
            nn.init.normal_(q, std=0.05)
    out = r(x)[0]
    assert not torch.allclose(out, ref, atol=1e-3)
    buf = io.BytesIO()
    torch.save(r, buf)
    buf.seek(0)
    assert torch.allclose(torch.load(buf, weights_only=False).eval()(x)[0], out, atol=1e-5), "dbb: checkpoint"
    fz = copy.deepcopy(r).fuse(verbose=False)
    assert torch.allclose(fz(x)[0], out, atol=1e-3, rtol=1e-3), (fz(x)[0] - out).abs().max()
    assert not any(hasattr(k, "b1") or hasattr(k, "bn") and isinstance(k, DBB) for k in fz.modules())
    ref_fused = apply(copy.deepcopy(base), ["film"]).fuse(verbose=False)
    assert sum(q.numel() for q in fz.parameters()) == sum(q.numel() for q in ref_fused.parameters()), "dbb: deployed size"
    hf = copy.deepcopy(r).half().fuse(verbose=False)
    assert torch.allclose(hf(x.half())[0].float(), out, atol=5e-2, rtol=5e-2), "dbb: half + fuse"
    r.args = get_cfg()
    r.train()
    for q in r.parameters():
        q.requires_grad_(True)
    loss, _ = r.loss(batch)
    loss.sum().backward()
    assert all(q.grad is not None for k in r.modules() if isinstance(k, DBB) for q in k.bn1.parameters())
    print("dbb ok: branched convs", n_dbb, "deployed params", sum(q.numel() for q in fz.parameters()))
    film_fused = sum(q.numel() for q in apply(copy.deepcopy(base), ["film"]).fuse(verbose=False).parameters())
    for name, kept in (("dbbp3", ()), ("spddetail", ("spd",)), ("coordatt", ("coordatt",)), ("gcctx", ("gc",)),
                       *((k, ("coordatt", k)) for k in ROUND3), *((k, ("coordatt", a)) for k, a in ROUND4.items())):
        on_s4 = name in ROUND3 or name in ROUND4
        r = apply(copy.deepcopy(base), ["film", "mix", "cons"] + (["coordatt"] if on_s4 else []) + [name]).eval()
        n_ref = ref.shape[-1]
        cut = (lambda t: t[..., -n_ref:]) if name == "p2" else (lambda t: t)  # p2 prepends its anchors
        assert torch.allclose(cut(r(x)[0]), ref, atol=1e-4), f"{name}: not identical at step 0"
        if name == "p2":  # and its class scores cannot reach any confidence threshold
            assert r(x)[0][:, 4:, :-n_ref].max() < 1e-9
        r.train()
        with torch.no_grad():
            for _ in range(3):
                r(torch.rand(2, 3, 256, 256))
        r.tap.feats = None
        if hasattr(r, "phase"):
            r.phase.aux = []
        r.eval()
        for q in r.parameters():
            if q.abs().sum() == 0:
                nn.init.normal_(q, std=0.05)
        out = r(x)[0]
        assert not torch.allclose(cut(out), ref, atol=1e-3), name
        buf = io.BytesIO()
        torch.save(r, buf)
        buf.seek(0)
        assert torch.allclose(torch.load(buf, weights_only=False).eval()(x)[0], out, atol=1e-5), f"{name}: checkpoint"
        fz = copy.deepcopy(r).fuse(verbose=False)
        assert torch.allclose(fz(x)[0], out, atol=1e-3, rtol=1e-3), (name, (fz(x)[0] - out).abs().max())
        extra = sum(q.numel() for k in kept for q in getattr(fz, k).parameters())
        if name == "p2":  # the level-0 heads live inside Detect (one2one branch after fuse)
            extra += sum(q.numel() for h in (fz.model[-1].cv2[0], fz.model[-1].cv3[0]) for q in h.parameters())
        assert sum(q.numel() for q in fz.parameters()) == film_fused + extra, f"{name}: deployed size"
        hf = copy.deepcopy(r).half().fuse(verbose=False)
        assert torch.allclose(hf(x.half())[0].float(), out, atol=5e-2, rtol=5e-2), f"{name}: half + fuse"
        r.args = get_cfg()
        r.train()
        if name == "p2":  # past the warm-up the class offset is 0, so the level actually trains
            assert float(r.p2.offset()) < -29.9
            r.p2.seen.fill_(r.p2.warm)
            assert float(r.p2.offset()) == 0.0
        for q in r.parameters():
            q.requires_grad_(True)
        loss, _ = r.loss(batch)
        loss.sum().backward()
        for k in kept:
            assert all(q.grad is not None and q.grad.abs().sum() > 0 for q in getattr(r, k).parameters()), (name, k)
        assert r.tap.feats is None and (not hasattr(r, "phase") or (r.phase.aux == [] and not r.phase.shift)), name
        print(f"{name} ok: deployed params {sum(q.numel() for q in fz.parameters())} (+{extra} kept over film)")
    print("mods self-check ok", SimpleNamespace(loss=float(loss.sum())))
