"""Architecture and training changes applied to a COCO-initialised YOLO26n, selected by name in configs/node.yaml.

Every change leaves the network equal to the pretrained one at step 0 (new branches end in a zero-initialised layer).

  rep         neck C3k2 (L13, L16, L19, L22): each 3x3 Conv gains a parallel 1x1 branch, merged back at inference
  lk          backbone C3k2 (L2, L4, L6, L8): residual depthwise 5x5 after the block; stays at inference
  film        shift-conditioned scale/shift on L0, L1, L3 from image photometric stats + stem deviation
  mix         training only: per-sample feature statistics of L1 and L3 are mixed with another sample of the batch
  cons        training only: a photometrically jittered second view must give the same Detect input features

Round 9 (stacked directly on Cluster 1: film + mix + cons):
  cleancons   training only (0 inference cost): freezes BatchNorm running stats and ShiftFiLM ref_mu/ref_sd on the
              shifted view, and keeps MixStat off on the clean view (active only on the shifted view) so the clean
              detection pass and consistency target f1/p1 are uncorrupted.
  spatialctx  Cluster A (L2, L4, L6, L8): SpatialRetinexGate on L2/L4 (local brightness+contrast -> spatial scale/shift)
              + WideContextRes on L6 (dilated dw3x3) and L8 (dw7x7 + 1x1), paired with local glare+shadow aug on view 2.
  headalign   Cluster B (L16, L19, L22, L23): L10-context + RMS magnitude calibrator on L16/L19/L22 + foreground-gated
              logit calibration on L23 (both one2many and one2one), paired with feature-RMS loss + dual-head cls/box
              output alignment on diversified view 2 (sensor noise + mild blur).

Hooks are module-level callables so that checkpoints (pickled models) and the EMA deep copy keep them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2, Conv, RepConv
from ultralytics.nn.tasks import DetectionModel

NECK, BACKBONE, FILM_LAYERS, MIX_LAYERS = (13, 16, 19, 22), (2, 4, 6, 8), (0, 1, 3), (1, 3)


# --- shared per-forward state & hook wrapper ---------------------------------------------------------------------
class CleanState:
    """Shared lightweight state across mods; holds no graph tensors."""

    def __init__(self):
        self.clean_cons = False  # enabled by cleancons
        self.shift = False       # True only during the shifted view-2 forward pass
        self.augs: list[str] = []


def _state(model: nn.Module) -> CleanState:
    if not hasattr(model, "clean_state"):
        model.clean_state = CleanState()
    return model.clean_state


class _Hook:
    """Module-level callable forwarding a hook to owner.<name>(k, ...), pickle-safe for checkpoints/EMA."""

    def __init__(self, owner: nn.Module, name: str, k: int = 0):
        self.owner, self.name, self.k = owner, name, k

    def __call__(self, *args):
        return getattr(self.owner, self.name)(self.k, *args)


def _run_mod(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a small module in its own weight dtype (works in fp32 and fp16/half eval) and return float32."""
    return m(x.to(next(m.parameters()).dtype)).float()


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
        self.ab = None     # per-forward state, cleared by the last hooked layer
        self.state = None  # CleanState attached by apply()

    def descriptor(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        img, feat = img.float(), feat.float()
        hf = (img - F.avg_pool2d(img, 3, 1, 1)).abs().mean((2, 3))
        mu, sd = feat.mean((2, 3)), feat.std((2, 3))
        freeze_ref = self.state is not None and self.state.clean_cons and self.state.shift
        if self.training and not freeze_ref:  # reference = clean training-domain stem statistics
            with torch.no_grad():
                m = 1.0 if self.seen == 0 else 0.01
                self.ref_mu.lerp_(mu.mean(0), m)
                self.ref_sd.lerp_(sd.mean(0), m)
                self.seen.fill_(1)
        dev = (mu - self.ref_mu.float()) / (self.ref_sd.float() + 1e-3)
        ratio = torch.log((sd + 1e-3) / (self.ref_sd.float() + 1e-3))
        return torch.cat([img.mean((2, 3)), img.std((2, 3)), hf, dev, ratio], 1)

    def modulate(self, k: int, img: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if k == 0:
            d = self.descriptor(img, out)
            self.ab = _run_mod(self.net, d).split([2 * c for c in self.chs], 1)
        a, b = self.ab[k].chunk(2, 1)
        if k == len(self.chs) - 1:
            self.ab = None
        return (out.float() * (1 + a[:, :, None, None]) + b[:, :, None, None]).to(out.dtype)


class FilmHook:
    def __init__(self, film: ShiftFiLM, k: int):
        self.film, self.k = film, k

    def __call__(self, module, inputs, output):
        return self.film.modulate(self.k, inputs[0], output)


def add_film(model: nn.Module) -> None:
    model.film = ShiftFiLM([model.model[i].conv.out_channels for i in FILM_LAYERS])
    model.film.state = _state(model)
    for k, i in enumerate(FILM_LAYERS):
        model.model[i].register_forward_hook(FilmHook(model.film, k))


# --- mix ---------------------------------------------------------------------------------------------------------
class MixStat:
    """MixStyle-type mixing of per-sample channel statistics; identity in eval mode (and on view 1 with cleancons)."""

    def __init__(self, state: CleanState, p: float = 0.5, alpha: float = 0.1):
        self.state, self.p, self.beta = state, p, torch.distributions.Beta(alpha, alpha)

    def __call__(self, module, inputs, x):
        if not module.training or x.shape[0] < 2:
            return None
        if self.state.clean_cons and not self.state.shift:
            return None
        if torch.rand(()) > self.p:
            return None
        mu, sd = x.mean((2, 3), keepdim=True).detach(), (x.var((2, 3), keepdim=True) + 1e-6).sqrt().detach()
        lam = self.beta.sample((x.shape[0], 1, 1, 1)).to(x)
        perm = torch.randperm(x.shape[0], device=x.device)
        return (x - mu) / sd * (lam * sd + (1 - lam) * sd[perm]) + lam * mu + (1 - lam) * mu[perm]


def add_mix(model: nn.Module) -> None:
    st = _state(model)
    for i in MIX_LAYERS:
        model.model[i].register_forward_hook(MixStat(st))


# --- view-2 shift perturbations (general photometric / spatial lighting / noise+blur; no weather synthesis) ------
def style_jitter(img: torch.Tensor) -> torch.Tensor:
    """Per-sample global gamma, contrast, brightness and channel gain on a [0, 1] image batch."""
    u = lambda lo, hi, c=1: torch.empty(img.shape[0], c, 1, 1, device=img.device).uniform_(lo, hi)
    x = img.float().clamp(1e-4, 1) ** torch.exp(u(-0.7, 0.7))
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.5, 1.5) + m + u(-0.2, 0.2)
    return (x * u(0.8, 1.2, 3)).clamp(0, 1).to(img.dtype)


def spatial_illum_aug(img: torch.Tensor) -> torch.Tensor:
    """Spatially non-uniform illumination (smooth shadow/vignette field) + localized tinted glows & reflections."""
    b, _, h, w = img.shape
    dev = img.device
    x = img.float()
    # 1. Smooth low-frequency multiplicative illumination field (dark regions next to bright ones in the same frame)
    field = torch.empty(b, 1, 4, 4, device=dev).uniform_(-0.9, 0.35).exp()
    field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)
    x = x * field
    # 2. Up to 6 localized Gaussian glows with soft vertical reflection (glare / streetlight blooms)
    gh, gw, k = max(h // 4, 8), max(w // 4, 8), 6
    yy = torch.linspace(0, 1, gh, device=dev)[None, None, :, None]
    xx = torch.linspace(0, 1, gw, device=dev)[None, None, None, :]
    on = (torch.rand(b, k, 1, 1, device=dev) < 0.7).float()
    cx, cy = torch.rand(b, k, 1, 1, device=dev), torch.rand(b, k, 1, 1, device=dev) * 0.7
    sg = torch.empty(b, k, 1, 1, device=dev).uniform_(0.015, 0.06)
    amp = torch.empty(b, k, 1, 1, device=dev).uniform_(0.25, 0.9) * on
    glow = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))
    ry = cy + torch.empty_like(cy).uniform_(0.05, 0.25)
    stretch = torch.empty_like(sg).uniform_(3.0, 7.0)
    refl = 0.3 * amp * torch.exp(-(xx - cx) ** 2 / (2 * sg ** 2) - (yy - ry) ** 2 / (2 * (sg * stretch) ** 2))
    tint = torch.empty(b, k, 3, device=dev).uniform_(0.65, 1.0)
    g = torch.einsum("bkhw,bkc->bchw", glow + refl, tint)
    g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
    return (x + g).clamp(0, 1).to(img.dtype)


def div_noise_blur_aug(img: torch.Tensor) -> torch.Tensor:
    """General non-weather sensor degradation: mild directional/Gaussian blur and Poisson-Gaussian shot+read noise."""
    b, c, h, w = img.shape
    dev = img.device
    x = img.float()
    # Mild 3x3 Gaussian blur on half of the selected samples
    k1 = torch.tensor([1.0, 2.0, 1.0], device=dev)
    kern = ((k1[:, None] * k1[None]) / 16.0).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    blurred = F.conv2d(x, kern, padding=1, groups=c)
    do_blur = (torch.rand(b, 1, 1, 1, device=dev) < 0.5).float()
    x = x * (1 - do_blur) + blurred * do_blur
    # Signal-dependent shot noise + additive read noise
    shot = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.001, 0.015)
    read = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.005, 0.03)
    noise = torch.randn_like(x) * (x.clamp(0, 1) * shot + read ** 2).sqrt()
    return (x + noise).clamp(0, 1).to(img.dtype)


SHIFT_AUGS = {"spatial_illum": spatial_illum_aug, "div_noise_blur": div_noise_blur_aug}


def apply_shift_augs(img: torch.Tensor, aug_names: list[str], total_p: float = 0.5) -> torch.Tensor:
    """Mutually exclusive per-sample routing on view 2: each image gets at most ONE extra aug (probability total_p)."""
    if not aug_names:
        return img
    b, dev = img.shape[0], img.device
    u = torch.rand(b, device=dev)
    out = img.clone()
    step = total_p / len(aug_names)
    for idx, name in enumerate(aug_names):
        mask = (u >= idx * step) & (u < (idx + 1) * step)
        if mask.any():
            out[mask] = SHIFT_AUGS[name](img[mask])
    return out


# --- cons + cleancons --------------------------------------------------------------------------------------------
class DetectTap:
    def __init__(self):
        self.feats = None

    def __call__(self, module, inputs):
        if module.training:
            self.feats = list(inputs[0])


class PredGrab:
    """Captures Detect raw prediction dict during training so view-1 outputs need no extra forward pass."""

    def __init__(self):
        self.preds = None

    def __call__(self, module, inputs, output):
        if module.training:
            self.preds = output


def _dual_head_align(p1: dict, p2: dict, k: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """DivAlign-style output alignment on both one2many (trains backbone/neck) and one2one (calibrates inference head).
    Selects top-k anchors by max(clean_conf, shift_conf) to preserve recall and suppress shifted false positives."""
    cls_losses, box_losses = [], []
    for branch in ("one2many", "one2one"):
        if branch not in p1 or branch not in p2:
            continue
        s1, s2 = p1[branch]["scores"].detach().float(), p2[branch]["scores"].float()
        b1, b2 = p1[branch]["boxes"].detach().float(), p2[branch]["boxes"].float()
        prob1 = torch.sigmoid(s1)
        conf1 = prob1.max(1).values                        # (B, A)
        conf2 = torch.sigmoid(s2.detach()).max(1).values   # (B, A)
        _, idx = torch.maximum(conf1, conf2).topk(min(k, conf1.shape[1]), dim=1)  # (B, K)
        g = lambda t: t.gather(2, idx[:, None].expand(-1, t.shape[1], -1))
        s1_k, s2_k = g(s1), g(s2)
        b1_k, b2_k = g(b1), g(b2)
        c1_k = conf1.gather(1, idx)                        # (B, K)
        # Classification probability alignment (foreground weighted higher, background FPs still penalized)
        bce = F.binary_cross_entropy_with_logits(s2_k, torch.sigmoid(s1_k), reduction="none").mean(1)
        w_cls = 0.25 + c1_k
        cls_losses.append((bce * w_cls).sum() / w_cls.sum().clamp_min(1e-6))
        # Box regression alignment on confident clean foreground anchors only
        fg = (c1_k > 0.15).float() * c1_k
        if fg.sum() > 0:
            bl = F.smooth_l1_loss(b2_k, b1_k, beta=0.5, reduction="none").mean(1)
            box_losses.append((bl * fg).sum() / fg.sum().clamp_min(1e-6))
        else:
            box_losses.append(b2_k.sum() * 0.0)
    z = torch.zeros((), device=next(iter(p2.values()))["scores"].device)
    return (sum(cls_losses) / len(cls_losses) if cls_losses else z,
            sum(box_losses) / len(box_losses) if box_losses else z)


class ConsistModel(DetectionModel):
    cons_weight = 1.0

    def loss(self, batch, preds=None):
        st = getattr(self, "clean_state", None)
        if st is not None:
            st.shift = False
        loss, items = super().loss(batch, preds)
        if not self.training or preds is not None:
            return loss, items
        f1, self.tap.feats = self.tap.feats, None
        pgrab = getattr(self, "pred_grab", None)
        p1 = None
        if pgrab is not None:
            p1, pgrab.preds = pgrab.preds, None

        view = style_jitter(batch["img"])
        if st is not None and st.augs:
            view = apply_shift_augs(view, st.augs, total_p=0.5)

        saved_bn = []
        if st is not None and st.clean_cons:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    saved_bn.append((m, m.track_running_stats))
                    m.track_running_stats = False
        if st is not None:
            st.shift = True
        try:
            p2 = self.forward(view)
        finally:
            if st is not None:
                st.shift = False
            for m, flag in saved_bn:
                m.track_running_stats = flag
            if pgrab is not None:
                pgrab.preds = None
            ha = getattr(self, "headalign", None)
            if ha is not None:
                ha.ctx = None

        f2, self.tap.feats = self.tap.feats, None
        c = sum(1 - F.cosine_similarity(a.detach().float(), b.float(), dim=1).mean() for a, b in zip(f1, f2)) / len(f1)

        if getattr(self, "headalign", None) is not None and p1 is not None and isinstance(p2, dict):
            # Feature-RMS magnitude alignment (closes cosine's amplitude blind spot under low light)
            mag = sum(
                F.smooth_l1_loss(
                    (b.float().pow(2).mean((2, 3)) + 1e-4).sqrt().log(),
                    (a.detach().float().pow(2).mean((2, 3)) + 1e-4).sqrt().log(),
                    beta=0.2,
                )
                for a, b in zip(f1, f2)
            ) / len(f1)
            l_cls, l_box = _dual_head_align(p1, p2, k=64)
            c = c + 0.25 * mag + 0.5 * l_cls + 0.1 * l_box

        # the trainer sums the loss vector, which is already scaled by the batch size
        return loss + self.cons_weight * c * batch["img"].shape[0] / loss.numel(), items


def add_cons(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    model.tap = DetectTap()
    model.model[-1].register_forward_pre_hook(model.tap)


def add_cleancons(model: nn.Module) -> None:
    assert isinstance(model, ConsistModel), "cleancons requires cons"
    _state(model).clean_cons = True


# --- Cluster A: spatialctx (L2, L4, L6, L8) ----------------------------------------------------------------------
class SpatialRetinexGate(nn.Module):
    """Local spatial illumination & contrast modulation on early backbone C3k2 outputs (L2, L4):
    reads [local_mean, |x - local_mean|] -> 1x1 -> dw3x3 -> zero-init 1x1 -> x*(1 + 0.25 tanh s) + 0.25 tanh b."""

    def __init__(self, c: int):
        super().__init__()
        mid = max(c // 8, 8)
        self.reduce = Conv(2 * c, mid, 1)
        self.smooth = Conv(mid, mid, 3, 1, None, mid)
        self.out = nn.Conv2d(mid, 2 * c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, module, inputs, out):
        x = out.float()
        loc = F.avg_pool2d(x, 7, 1, 3, count_include_pad=False)
        inp = torch.cat([loc, (x - loc).abs()], 1).to(self.out.weight.dtype)
        ab = self.out(self.smooth(self.reduce(inp))).float()
        s, b = ab.chunk(2, 1)
        return (x * (1 + 0.25 * torch.tanh(s)) + 0.25 * torch.tanh(b)).to(out.dtype)


class ResidualBranch(nn.Module):
    """out + branch(out); branch ends in a zero-init BatchNorm so output equals pretrained at step 0."""

    def __init__(self, branch: nn.Module):
        super().__init__()
        self.branch = branch

    def forward(self, module, inputs, out):
        return out + self.branch(out)


def add_spatialctx(model: nn.Module) -> None:
    c2 = model.model[2].cv2.conv.out_channels
    c4 = model.model[4].cv2.conv.out_channels
    c6 = model.model[6].cv2.conv.out_channels
    c8 = model.model[8].cv2.conv.out_channels
    b6 = Conv(c6, c6, 3, g=c6, d=2, act=False)
    b8 = nn.Sequential(Conv(c8, c8, 7, g=c8, act=False), Conv(c8, c8, 1, act=False))
    nn.init.zeros_(b6.bn.weight)
    nn.init.zeros_(b8[1].bn.weight)
    model.spatialctx = nn.ModuleList([
        SpatialRetinexGate(c2),
        SpatialRetinexGate(c4),
        ResidualBranch(b6),
        ResidualBranch(b8),
    ])
    for mod, idx in zip(model.spatialctx, (2, 4, 6, 8)):
        model.model[idx].register_forward_hook(mod)
    st = _state(model)
    if "spatial_illum" not in st.augs:
        st.augs.append("spatial_illum")


# --- Cluster B: headalign (L16, L19, L22, L23) -------------------------------------------------------------------
class HeadAlignNet(nn.Module):
    """L10-context + feature-RMS magnitude calibrator on L16/L19/L22 + foreground-gated logit calibrator on L23."""

    def __init__(self, c_ctx: int, chs: list[int], nc: int, ctx_dim: int = 32):
        super().__init__()
        self.ctx_enc = nn.Sequential(nn.Linear(c_ctx, ctx_dim), nn.SiLU())
        self.neck = nn.ModuleList()
        for c in chs:
            mid = max(c // 8, 16)
            m = nn.Sequential(nn.Linear(ctx_dim + 2 * c, mid), nn.SiLU(), nn.Linear(mid, 2 * c))
            nn.init.zeros_(m[2].weight)
            nn.init.zeros_(m[2].bias)
            self.neck.append(m)
        # Separate zero-init logit calibrators for one2many (k=0) and one2one (k=1)
        self.head = nn.ModuleList([nn.Linear(ctx_dim, 2 * nc), nn.Linear(ctx_dim, 2 * nc)])
        for h in self.head:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.ctx = None  # per-forward context vector, cleared after Detect

    def read_ctx(self, k, module, inputs, out):
        self.ctx = _run_mod(self.ctx_enc, out.float().mean((2, 3)))

    def calib_neck(self, k, module, inputs, out):
        x = out.float()
        mu = x.mean((2, 3))
        log_rms = (x.pow(2).mean((2, 3)) + 1e-4).sqrt().log()
        d = torch.cat([self.ctx, mu, log_rms], 1)
        a, b = _run_mod(self.neck[k], d).chunk(2, 1)
        return (x * (1 + 0.3 * torch.tanh(a[:, :, None, None])) + 0.3 * torch.tanh(b[:, :, None, None])).to(out.dtype)

    def calib_logit(self, k, module, inputs, out):
        # k=0 -> one2many (flows back to L10); k=1 -> one2one (detached from backbone during training, matching YOLO26n)
        ctx = self.ctx.detach() if (k == 1 and module.training) else self.ctx
        g, b = _run_mod(self.head[k], ctx).chunk(2, 1)
        z = out.float()
        # Foreground-gated additive bias prevents background false positives (sigmoid(z) ~ 0 when z << 0)
        z_cal = z * (1 + 0.25 * torch.tanh(g[:, :, None, None])) + 0.25 * torch.tanh(b[:, :, None, None]) * torch.sigmoid(z)
        return z_cal.to(out.dtype)

    def clear(self, k, module, inputs, out):
        self.ctx = None


def add_headalign(model: nn.Module) -> None:
    assert isinstance(model, ConsistModel), "headalign requires cons"
    det = model.model[-1]
    c_ctx = model.model[10].cv2.conv.out_channels
    chs = [model.model[i].cv2.conv.out_channels for i in (16, 19, 22)]
    model.headalign = ha = HeadAlignNet(c_ctx, chs, det.nc)
    model.model[10].register_forward_hook(_Hook(ha, "read_ctx", 0))
    for k, i in enumerate((16, 19, 22)):
        model.model[i].register_forward_hook(_Hook(ha, "calib_neck", k))
    for h in det.cv3:
        h[-1].register_forward_hook(_Hook(ha, "calib_logit", 0))
    if getattr(det, "one2one_cv3", None) is not None:
        for h in det.one2one_cv3:
            h[-1].register_forward_hook(_Hook(ha, "calib_logit", 1))
    det.register_forward_hook(_Hook(ha, "clear", 0))
    model.pred_grab = PredGrab()
    det.register_forward_hook(model.pred_grab)
    st = _state(model)
    if "div_noise_blur" not in st.augs:
        st.augs.append("div_noise_blur")


ADD = {
    "rep": add_rep,
    "lk": add_lk,
    "film": add_film,
    "mix": add_mix,
    "cons": add_cons,
    "cleancons": add_cleancons,
    "spatialctx": add_spatialctx,
    "headalign": add_headalign,
}


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


if __name__ == "__main__":  # exhaustive self-check for all 4 Round-9 configs
    import copy
    import io
    from types import SimpleNamespace

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    torch.manual_seed(0)
    base = YOLO("yolo26n.pt").model.float().eval()
    x = torch.rand(2, 3, 256, 256)
    ref = base(x)[0]
    batch = {
        "img": x,
        "cls": torch.zeros(3, 1),
        "bboxes": torch.rand(3, 4) * 0.3 + 0.3,
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    }

    # Check all 4 Round-9 configs end-to-end: step-0 identity, checkpoint, fuse, half+fuse, EMA deepcopy,
    # clean BN & FiLM ref invariance on shifted pass, zero double-registered tensors, and full gradient flow.
    configs = [
        ("R1:C1-Clean", ["film", "mix", "cons", "cleancons"], ("film",)),
        ("R2:+SpatialCtx", ["film", "mix", "cons", "cleancons", "spatialctx"], ("film", "spatialctx")),
        ("R3:+HeadAlign", ["film", "mix", "cons", "cleancons", "headalign"], ("film", "headalign")),
        ("R4:+Both", ["film", "mix", "cons", "cleancons", "spatialctx", "headalign"], ("film", "spatialctx", "headalign")),
    ]
    base_fused_params = sum(p.numel() for p in copy.deepcopy(base).fuse(verbose=False).parameters())
    for label, mod_list, attr_names in configs:
        r = apply(copy.deepcopy(base), mod_list).eval()
        # 1. No tensor registered twice in state_dict (prevents ModelEMA double updates)
        assert len(r.state_dict()) == len({id(v) for v in r.state_dict().values()}), f"{label}: duplicate tensor"
        # 2. Exact step-0 identity with stock pretrained YOLO26n
        assert torch.allclose(r(x)[0], ref, atol=1e-4), f"{label}: not identical at step 0"
        # 3. Verify BN & ShiftFiLM ref invariance during view 2 (shift=True) and MixStat clean pass
        r.train()
        with torch.no_grad():
            r.clean_state.shift = False
            r(x)
            r.tap.feats = None
            if hasattr(r, "pred_grab"):
                r.pred_grab.preds = None
            bn_snap = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                       for m in r.modules() if isinstance(m, nn.BatchNorm2d)]
            ref_mu_snap, ref_sd_snap = r.film.ref_mu.clone(), r.film.ref_sd.clone()
            # Run view 2 through ConsistModel's exact BN-freeze mechanism
            saved_bn = []
            for m in r.modules():
                if isinstance(m, nn.BatchNorm2d):
                    saved_bn.append((m, m.track_running_stats))
                    m.track_running_stats = False
            r.clean_state.shift = True
            r(style_jitter(x))
            r.clean_state.shift = False
            for m, fl in saved_bn:
                m.track_running_stats = fl
            r.tap.feats = None
            if hasattr(r, "pred_grab"):
                r.pred_grab.preds = None
            assert torch.equal(r.film.ref_mu, ref_mu_snap) and torch.equal(r.film.ref_sd, ref_sd_snap), f"{label}: ref_mu drifted"
            for (rm, rv, nb), m in zip(bn_snap, [m for m in r.modules() if isinstance(m, nn.BatchNorm2d)]):
                assert torch.equal(m.running_mean, rm) and torch.equal(m.running_var, rv) and torch.equal(m.num_batches_tracked, nb), f"{label}: BN drifted"
        # 4. Perturb zero-init params off zero to test checkpoint, fuse, half+fuse, EMA deepcopy
        r.eval()
        for q in r.parameters():
            if q.abs().sum() == 0:
                nn.init.normal_(q, std=0.03)
        out = r(x)[0]
        assert not torch.allclose(out, ref, atol=1e-3), f"{label}: perturbed output unchanged"
        buf = io.BytesIO()
        torch.save(r, buf)
        buf.seek(0)
        r_load = torch.load(buf, weights_only=False).eval()
        assert torch.allclose(r_load(x)[0], out, atol=1e-5), f"{label}: checkpoint round-trip mismatch"
        fz = copy.deepcopy(r).fuse(verbose=False)
        assert torch.allclose(fz(x)[0], out, atol=1e-3, rtol=1e-3), f"{label}: fused mismatch"
        hf = copy.deepcopy(r).half().fuse(verbose=False)
        assert torch.allclose(hf(x.half())[0].float(), out, atol=5e-2, rtol=5e-2), f"{label}: half+fuse mismatch"
        hf2 = copy.deepcopy(r).fuse(verbose=False).half()
        assert torch.allclose(hf2(x.half())[0].float(), out, atol=5e-2, rtol=5e-2), f"{label}: fuse+half mismatch"
        # 5. Full training step + backward + gradient coverage + clean state + EMA deepcopy
        r.args = get_cfg()
        r.train()
        for q in r.parameters():
            q.requires_grad_(True)
        loss, _ = r.loss(batch)
        assert torch.isfinite(loss).all(), f"{label}: non-finite loss"
        loss.sum().backward()
        assert r.tap.feats is None and r.film.ab is None and not r.clean_state.shift
        assert getattr(r, "pred_grab", SimpleNamespace(preds=None)).preds is None
        assert getattr(r, "headalign", SimpleNamespace(ctx=None)).ctx is None
        for an in attr_names:
            for pname, p in getattr(r, an).named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{label}: zero/missing grad on {an}.{pname}"
        # EMA deepcopy must succeed right after training step
        _ = copy.deepcopy(r)
        fz_params = sum(p.numel() for p in fz.parameters())
        pct = 100.0 * (fz_params - base_fused_params) / base_fused_params
        print(f"[check OK] {label:16s} | fused_params={fz_params:,} (+{fz_params - base_fused_params:,}, +{pct:.2f}%) | loss={float(loss.sum()):.3f}")

    for name, fn in SHIFT_AUGS.items():
        y = fn(x)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and not torch.allclose(y, x), name
    print("ALL ROUND-9 SELF-CHECKS PASSED")
