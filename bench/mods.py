"""Architecture and training changes applied to a COCO-initialised YOLO26n, selected by name in configs/node.yaml.

Every architecture change leaves the network equal to the pretrained one at step 0 (zero-initialised residual/BN).

  rep          neck C3k2 (L13, L16, L19, L22): parallel 1x1 branch merged at inference
  lk           backbone C3k2 (L2, L4, L6, L8): residual depthwise 5x5
  film         shift-conditioned scale/shift on L0, L1, L3 from image photometric stats + stem deviation
  mix          training only: MixStyle on L1 and L3
  cons         training only: photometric consistency on Detect inputs

Round 10 (identical Div-Ldet + paired dual-head alignment across matched controls & adaptive architectures):
  divdet       training only (0 inference cost): 0.5 * L_det(clean) + 0.5 * L_det(div) + paired feature cosine +
               dual-head (one2many + one2one) cls/box output alignment. Non-weather diversification: global photometric
               + one per-sample corruption among {local illumination+glare, noise+blur, Fourier amplitude mixing}.
  spatialctx   SpatialRetinexGate on L2/L4 + WideContextRes on L6/L8 (7 layers total with film).
  condroute    Input-routed K=2 weight-blended CondConv2 (L4, L6, L13, L16, L19; ~+0% batch-1 conv FLOPs) +
               SoftDualNorm (L4, L6, L8, L16, L19, L22) driven by a shared CondRouter reading 12 physical image cues
               (10 layers total with film).
  staticroute  Causal control for condroute + film: severs input conditioning in ShiftFiLM and CondRouter, replacing
               image descriptors with learned constant vectors (identical architecture, conv capacity, and FLOPs).

Round 11 (deep in-graph edge-native SOTA clusters on YOLO26n + Div-Ldet; zero Python hooks, pure FP16 TensorRT ops):
  detailpath   Cluster 2' (YOLO26-RD P2/4 feature-only injection + FRFDet nonlinear lossless downsampling):
               L2 (64ch, 160x160) -> DW3x3 s2 + PW 64->32 -> zero-init cv1 channel expansion on L16 (+32ch, no P2 head);
               plus nonlinear lossless PixelUnshuffle(2) -> PW -> BN -> SiLU -> PW -> BN(0) residual inside L17 & L20.
  hyperpad     Cluster 3' (YOLOv13 causal low-res Edge-HyperACE + FullPAD):
               pools L4/L6/L10 to 20x20 (C_h=64), runs O(N*M) M=8 scene-adaptive hypergraph in native FP16, projects
               low-res 1x1 tunnels (64->16) before bilinear resize into L13, L16, L19, L22 via zero-init cv1 expansion.

Hooks are module-level callables so that checkpoints (pickled models) and ModelEMA deepcopy keep them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2, Conv, Conv2, RepConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import fuse_conv_and_bn

NECK, BACKBONE, FILM_LAYERS, MIX_LAYERS = (13, 16, 19, 22), (2, 4, 6, 8), (0, 1, 3), (1, 3)
COND_CONV_LAYERS = (4, 6, 13, 16, 19)       # 18 plain 3x3 convs total (~129k delta params = +5.4%, ~0% b=1 FLOPs)
DUAL_NORM_LAYERS = (4, 6, 8, 16, 19, 22)    # 6 backbone + neck feature taps
COND_STAGES = (4, 6, 8, 13, 16, 19, 22)     # 7 stage-specific routing weights in [0, 1]


class RouteBox:
    """Plain non-Module container for per-forward routing weights; avoids duplicate nn.Module registration in EMA."""

    def __init__(self):
        self.alpha: torch.Tensor | None = None  # (B, len(COND_STAGES)) in [0, 1]


class _Hook:
    """Module-level callable forwarding a hook to owner.<name>(k, ...), pickle-safe for checkpoints/EMA."""

    def __init__(self, owner: nn.Module, name: str, k: int = 0):
        self.owner, self.name, self.k = owner, name, k

    def __call__(self, *args):
        return getattr(self.owner, self.name)(self.k, *args)


def _run_mod(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a small module in its own weight dtype (fp32 train or fp16 eval) and return float32."""
    return m(x.to(next(m.parameters()).dtype)).float()


def _fold(w: torch.Tensor, bn: nn.BatchNorm2d, b: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    t = bn.weight.float() / (bn.running_var.float() + bn.eps).sqrt()
    b = torch.zeros_like(bn.bias, dtype=torch.float32) if b is None else b.float()
    return w.float() * t[:, None, None, None], bn.bias.float() + (b - bn.running_mean.float()) * t


# --- rep & lk (retained from M0/M1) ------------------------------------------------------------------------------
def add_rep(model: nn.Module) -> None:
    for i in NECK:
        for parent in list(model.model[i].modules()):
            for name, c in list(parent.named_children()):
                if type(c) is Conv and c.conv.kernel_size == (3, 3) and c.conv.groups == 1 and c.conv.stride == (1, 1):
                    r = RepConv(c.conv.in_channels, c.conv.out_channels, 3, 1, act=c.act)
                    r.conv1.conv.load_state_dict(c.conv.state_dict())
                    r.conv1.bn.load_state_dict(c.bn.state_dict())
                    nn.init.zeros_(r.conv2.bn.weight)
                    setattr(parent, name, r)


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
    """Predicts per-channel (scale, shift) for L0, L1, L3 from image photometric stats + stem deviation."""

    def __init__(self, chs: list[int], hidden: int = 32):
        super().__init__()
        self.chs = chs
        self.in_dim = 9 + 2 * chs[0]
        self.register_buffer("ref_mu", torch.zeros(chs[0]))
        self.register_buffer("ref_sd", torch.ones(chs[0]))
        self.register_buffer("seen", torch.zeros(()))
        self.net = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * sum(chs)))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.ab = None
        self.static_mode = False
        self.const_desc = None  # created only if staticroute is applied

    def descriptor(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.static_mode and self.const_desc is not None:
            return torch.zeros(img.shape[0], self.in_dim, device=img.device, dtype=torch.float32) + self.const_desc.float()
        img, feat = img.float(), feat.float()
        hf = (img - F.avg_pool2d(img, 3, 1, 1)).abs().mean((2, 3))
        mu, sd = feat.mean((2, 3)), feat.std((2, 3))
        if self.training:
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
    for k, i in enumerate(FILM_LAYERS):
        model.model[i].register_forward_hook(FilmHook(model.film, k))


# --- mix & legacy cons -------------------------------------------------------------------------------------------
class MixStat:
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


class DetectTap:
    def __init__(self):
        self.feats = None

    def __call__(self, module, inputs):
        if module.training:
            self.feats = list(inputs[0])


class PredGrab:
    def __init__(self):
        self.preds = None

    def __call__(self, module, inputs, output):
        if module.training:
            self.preds = output


# --- Non-weather domain diversification (DivAlign-style: photometric + local illum + noise/blur + Fourier amp) ---
def style_jitter(img: torch.Tensor) -> torch.Tensor:
    """Global gamma, contrast, brightness and RGB gain on a [0, 1] image batch."""
    u = lambda lo, hi, c=1: torch.empty(img.shape[0], c, 1, 1, device=img.device).uniform_(lo, hi)
    x = img.float().clamp(1e-4, 1) ** torch.exp(u(-0.85, 0.85))
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.45, 1.55) + m + u(-0.22, 0.22)
    return (x * u(0.8, 1.2, 3)).clamp(0, 1).to(img.dtype)


def spatial_illum_aug(img: torch.Tensor) -> torch.Tensor:
    """Fast GPU spatially non-uniform illumination field + up to 5 localized Gaussian light blooms."""
    b, _, h, w = img.shape
    dev = img.device
    x = img.float()
    field = torch.empty(b, 1, 4, 4, device=dev).uniform_(-1.0, 0.35).exp()
    field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)
    x = x * field
    # Compute glow at 1/8 resolution (80x80 for 640) then bilinear upsample -> ~4x faster than 1/4
    gh, gw, k = max(h // 8, 8), max(w // 8, 8), 5
    yy = torch.linspace(0, 1, gh, device=dev)[None, None, :, None]
    xx = torch.linspace(0, 1, gw, device=dev)[None, None, None, :]
    on = (torch.rand(b, k, 1, 1, device=dev) < 0.7).float()
    cx, cy = torch.rand(b, k, 1, 1, device=dev), torch.rand(b, k, 1, 1, device=dev) * 0.7
    sg = torch.empty(b, k, 1, 1, device=dev).uniform_(0.02, 0.065)
    amp = torch.empty(b, k, 1, 1, device=dev).uniform_(0.25, 0.9) * on
    glow = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))
    ry = cy + torch.empty_like(cy).uniform_(0.05, 0.25)
    stretch = torch.empty_like(sg).uniform_(3.0, 7.0)
    refl = 0.3 * amp * torch.exp(-(xx - cx) ** 2 / (2 * sg ** 2) - (yy - ry) ** 2 / (2 * (sg * stretch) ** 2))
    tint = torch.empty(b, k, 3, device=dev).uniform_(0.65, 1.0)
    g = torch.einsum("bkhw,bkc->bchw", glow + refl, tint)
    g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
    return (x + g).clamp(0, 1).to(img.dtype)


def noise_blur_aug(img: torch.Tensor) -> torch.Tensor:
    """General non-weather sensor degradation: 3x3 Gaussian blur (p=0.5) + Poisson-Gaussian shot/read noise."""
    b, c, h, w = img.shape
    dev = img.device
    x = img.float()
    k1 = torch.tensor([1.0, 2.0, 1.0], device=dev)
    kern = ((k1[:, None] * k1[None]) / 16.0).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    blurred = F.conv2d(x, kern, padding=1, groups=c)
    do_blur = (torch.rand(b, 1, 1, 1, device=dev) < 0.5).float()
    x = x * (1 - do_blur) + blurred * do_blur
    shot = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.002, 0.02)
    read = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.005, 0.035)
    noise = torch.randn_like(x) * (x.clamp(0, 1) * shot + read ** 2).sqrt()
    return (x + noise).clamp(0, 1).to(img.dtype)


def fourier_amp_aug(img: torch.Tensor) -> torch.Tensor:
    """Fourier amplitude perturbation (preserves exact phase = exact object boundaries & boxes)."""
    b, _, h, w = img.shape
    dev = img.device
    x = img.float()
    fft = torch.fft.rfft2(x, norm="ortho")
    amp = fft.abs()
    pha = torch.angle(fft)
    if b >= 2:
        perm = torch.randperm(b, device=dev)
        lam = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.5, 1.0)
        amp_mix = lam * amp + (1 - lam) * amp[perm]
    else:
        amp_mix = amp
    # Mild smooth random frequency-gain modulation (low/mid/high bands)
    gain = torch.empty(b, 1, 4, 4, device=dev).uniform_(-0.3, 0.3).exp()
    gain = F.interpolate(gain, size=amp.shape[-2:], mode="bilinear", align_corners=False)
    fft_new = torch.polar(amp_mix * gain, pha)
    out = torch.fft.irfft2(fft_new, s=(h, w), norm="ortho")
    return out.clamp(0, 1).to(img.dtype)


def div_augment(img: torch.Tensor) -> torch.Tensor:
    """Non-weather domain diversification: global photometric + at most ONE per-sample structural perturbation."""
    x = style_jitter(img)
    b, dev = x.shape[0], x.device
    u = torch.rand(b, device=dev)
    out = x.clone()
    for idx, fn in enumerate((spatial_illum_aug, noise_blur_aug, fourier_amp_aug)):
        mask = (u >= idx * 0.25) & (u < (idx + 1) * 0.25)
        if mask.any():
            out[mask] = fn(x[mask])
    return out


def _dual_head_align(p1: dict, p2: dict, k: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Paired cls/box output alignment on both one2many (trains backbone/neck) and one2one (inference head)."""
    cls_losses, box_losses = [], []
    for branch in ("one2many", "one2one"):
        if branch not in p1 or branch not in p2:
            continue
        s1, s2 = p1[branch]["scores"].detach().float(), p2[branch]["scores"].float()
        b1, b2 = p1[branch]["boxes"].detach().float(), p2[branch]["boxes"].float()
        conf1 = torch.sigmoid(s1).max(1).values
        conf2 = torch.sigmoid(s2.detach()).max(1).values
        _, idx = torch.maximum(conf1, conf2).topk(min(k, conf1.shape[1]), dim=1)
        g = lambda t: t.gather(2, idx[:, None].expand(-1, t.shape[1], -1))
        s1_k, s2_k = g(s1), g(s2)
        b1_k, b2_k = g(b1), g(b2)
        c1_k = conf1.gather(1, idx)
        bce = F.binary_cross_entropy_with_logits(s2_k, torch.sigmoid(s1_k), reduction="none").mean(1)
        w_cls = 0.25 + c1_k
        cls_losses.append((bce * w_cls).sum() / w_cls.sum().clamp_min(1e-6))
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
        loss, items = super().loss(batch, preds)
        if not self.training or preds is not None:
            return loss, items
        f1, self.tap.feats = self.tap.feats, None
        self.forward(style_jitter(batch["img"]))
        f2, self.tap.feats = self.tap.feats, None
        c = sum(1 - F.cosine_similarity(a.detach().float(), b.float(), dim=1).mean() for a, b in zip(f1, f2)) / len(f1)
        return loss + self.cons_weight * c * batch["img"].shape[0] / loss.numel(), items


def add_cons(model: nn.Module) -> None:
    model.__class__ = ConsistModel
    model.tap = DetectTap()
    model.model[-1].register_forward_pre_hook(model.tap)


class EdgeArchModel(DetectionModel):
    """In-graph YOLO26n extension for Round-11 edge clusters (DetailPath + Causal Edge-HyperPAD): zero Python hooks,
    native FP16 TensorRT-fusable ops, exact step-0 COCO equivalence, and built-in liveness/causal checks."""

    def _predict_once(self, x, profile=False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        p2_proj = getattr(self, "p2_proj", None)
        hyperpad = getattr(self, "hyperpad", None)
        extra_order = getattr(self, "layer_extra_order", None)
        hp_taps = None
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if extra_order and m.i in extra_order:
                parts = [x]
                for src in extra_order[m.i]:
                    if src == "dp":
                        feat = p2_proj(y[2])
                    else:  # "hp"
                        feat = hp_taps[m.i]
                    if feat.shape[-2:] != x.shape[-2:]:
                        feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    parts.append(feat)
                x = torch.cat(parts, 1)
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if hyperpad is not None and m.i == 10:
                hp_taps = hyperpad(y[4], y[6], y[10])
            if m.i in embed:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def fuse(self, verbose=True, imgsz=640):
        was_fused = self.is_fused()
        super().fuse(verbose=verbose, imgsz=imgsz)
        if not was_fused:
            for attr in ("p2_proj", "hyperpad"):
                sub = getattr(self, attr, None)
                if sub is not None:
                    for m in sub.modules():
                        if isinstance(m, Conv) and hasattr(m, "bn"):
                            m.conv = fuse_conv_and_bn(m.conv, m.bn)
                            delattr(m, "bn")
                            m.forward = m.forward_fuse
        return self

    def liveness_report(self) -> dict[str, float]:
        """RMS/abs-mean magnitude of zero-initialized architectural injection weights (must stay >= 1e-3)."""
        out: dict[str, float] = {}
        for idx, s, e, tag in getattr(self, "cv1_expansions", ()):
            w = self.model[idx].cv1.conv.weight[:, s:e]
            out[tag] = float(w.detach().float().pow(2).mean().sqrt())
        for idx in (17, 20):
            m = self.model[idx]
            if isinstance(m, NLDownConv):
                if hasattr(m.nl_pw2, "bn"):
                    out[f"dp_nl{idx}"] = float(m.nl_pw2.bn.weight.detach().float().abs().mean())
                else:
                    out[f"dp_nl{idx}"] = float(m.nl_pw2.conv.weight.detach().float().pow(2).mean().sqrt())
        return out

    def zero_new_weights(self) -> None:
        """Causal ablation helper: zeros all newly injected slices/branches to verify measured gain is architectural."""
        with torch.no_grad():
            for idx, s, e, _ in getattr(self, "cv1_expansions", ()):
                self.model[idx].cv1.conv.weight[:, s:e].zero_()
            for idx in (17, 20):
                m = self.model[idx]
                if isinstance(m, NLDownConv):
                    if hasattr(m.nl_pw2, "bn"):
                        m.nl_pw2.bn.weight.zero_()
                        m.nl_pw2.bn.bias.zero_()
                    else:
                        m.nl_pw2.conv.weight.zero_()
                        if m.nl_pw2.conv.bias is not None:
                            m.nl_pw2.conv.bias.zero_()


class DivDetModel(EdgeArchModel):
    """Supervised detection on BOTH clean and non-weather diversified views (0.5/0.5) + paired feature & output alignment."""

    def loss(self, batch, preds=None):
        loss_clean, items = super().loss(batch, preds)
        if not self.training or preds is not None:
            return loss_clean, items
        f1, self.tap.feats = self.tap.feats, None
        p1, self.pred_grab.preds = self.pred_grab.preds, None
        view = div_augment(batch["img"])
        try:
            loss_div, _ = super().loss({**batch, "img": view})
            f2, self.tap.feats = self.tap.feats, None
            p2, self.pred_grab.preds = self.pred_grab.preds, None
        finally:
            self.tap.feats = None
            self.pred_grab.preds = None
            rbox = getattr(self, "rbox", None)
            if rbox is not None:
                rbox.alpha = None
        c_cos = sum(1 - F.cosine_similarity(a.detach().float(), b.float(), dim=1).mean() for a, b in zip(f1, f2)) / len(f1)
        l_cls, l_box = _dual_head_align(p1, p2, k=64) if (p1 is not None and isinstance(p2, dict)) else (0.0, 0.0)
        align = 0.5 * c_cos + 0.25 * l_cls + 0.05 * l_box
        total = 0.5 * loss_clean + 0.5 * loss_div + align * batch["img"].shape[0] / loss_clean.numel()
        return total, items


def add_divdet(model: nn.Module) -> None:
    model.__class__ = DivDetModel
    model.tap = DetectTap()
    model.pred_grab = PredGrab()
    model.model[-1].register_forward_pre_hook(model.tap)
    model.model[-1].register_forward_hook(model.pred_grab)


# --- Candidate B: spatialctx (L2, L4, L6, L8; from Round 9) ------------------------------------------------------
class SpatialRetinexGate(nn.Module):
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


# --- Candidate C: condroute (CondRouter + K=2 weight-blended CondConv2 + SoftDualNorm; 10 layers w/ film) --------
class CondRouter(nn.Module):
    """Reads 12 physical image cues at L0 -> per-stage soft routing gates alpha in [0, 1] for 7 stages."""

    def __init__(self, n_stages: int, rbox: RouteBox, hidden: int = 16):
        super().__init__()
        self.rbox = rbox
        self.net = nn.Sequential(nn.Linear(12, hidden), nn.SiLU(), nn.Linear(hidden, n_stages))
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.zeros_(self.net[2].bias)  # sigmoid(0) = 0.5 at step 0 so dconv/dualnorm get gradients at step 1
        self.static_mode = False
        self.const_cues = None

    def extract_cues(self, img: torch.Tensor) -> torch.Tensor:
        if self.static_mode and self.const_cues is not None:
            return torch.zeros(img.shape[0], 12, device=img.device, dtype=torch.float32) + self.const_cues.float()
        x = img.float()
        rgb_mu = x.mean((2, 3))                                    # (B, 3)
        rgb_sd = x.std((2, 3))                                     # (B, 3)
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        hf = (lum - F.avg_pool2d(lum, 3, 1, 1)).abs()
        hf_mu = hf.mean((2, 3))                                    # (B, 1) noise/detail energy
        hf_sd = hf.std((2, 3))                                     # (B, 1)
        dark_frac = torch.sigmoid((0.15 - lum) * 20.0).mean((2, 3))   # (B, 1) dark pixel share
        glare_frac = torch.sigmoid((lum - 0.85) * 20.0).mean((2, 3))  # (B, 1) saturated/glare share
        loc_c = (lum - F.avg_pool2d(lum, 7, 1, 3)).abs().mean((2, 3)) / (lum.std((2, 3)) + 0.05)  # (B, 1)
        log_lum = torch.log(lum.mean((2, 3)) + 0.05)               # (B, 1)
        return torch.cat([rgb_mu, rgb_sd, hf_mu, hf_sd, dark_frac, glare_frac, loc_c, log_lum], 1)

    def route(self, k, module, inputs):
        cues = self.extract_cues(inputs[0])
        self.rbox.alpha = torch.sigmoid(_run_mod(self.net, cues))

    def clear(self, k, module, inputs, out):
        self.rbox.alpha = None


class CondConv2(Conv2):
    """Pretrained 3x3 Conv + input-routed K=2 expert delta. Zero-init dbn -> exact step-0 COCO identity.
    After fuse(), batch=1 inference blends W = W0 + alpha * dW in a single F.conv2d (+0% extra conv FLOPs)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.rbox.alpha[:, self.stage_idx].view(-1, 1, 1, 1).to(x.dtype)
        return self.act(self.bn(self.conv(x)) + a * self.dbn(self.dconv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        a = self.rbox.alpha[:, self.stage_idx]
        if x.shape[0] == 1:
            s = a[0].to(self.conv.weight.dtype)
            w = self.conv.weight + s * self.dconv.weight
            b = self.conv.bias + s * self.dconv.bias
            return self.act(F.conv2d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups))
        return self.act(self.conv(x) + a.view(-1, 1, 1, 1).to(x.dtype) * self.dconv(x))

    def fuse_convs(self):
        c, d = self.conv, self.dconv
        w0, b0 = _fold(c.weight, self.bn, c.bias)
        dw, db = _fold(d.weight, self.dbn, d.bias)
        c_new = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=True).to(c.weight.device)
        d_new = nn.Conv2d(d.in_channels, d.out_channels, d.kernel_size, d.stride, d.padding, bias=True).to(d.weight.device)
        c_new.weight.data, c_new.bias.data = w0.to(c.weight.dtype), b0.to(c.weight.dtype)
        d_new.weight.data, d_new.bias.data = dw.to(d.weight.dtype), db.to(d.weight.dtype)
        self.conv, self.dconv = c_new, d_new
        delattr(self, "dbn")
        bn = self.bn
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1 - bn.eps)


def to_condconv(m: Conv, rbox: RouteBox, stage_idx: int) -> None:
    c = m.conv
    m.__class__ = CondConv2
    m.rbox = rbox
    m.stage_idx = stage_idx
    m.dconv = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False).to(c.weight.device)
    m.dbn = nn.BatchNorm2d(c.out_channels).to(c.weight.device)
    nn.init.zeros_(m.dbn.weight)
    nn.init.zeros_(m.dbn.bias)


class SoftDualNorm(nn.Module):
    """Input-routed InstanceNorm residual: y = x + alpha * (0.5 tanh gamma) * ((1 + a) * IN(x) + b - x).
    Zero-init gamma -> exact step-0 identity; cleanly separates shifted statistics from static BN."""

    def __init__(self, c: int, rbox: RouteBox, stage_idx: int):
        super().__init__()
        self.rbox, self.stage_idx = rbox, stage_idx
        self.gamma = nn.Parameter(torch.zeros(c))
        self.a = nn.Parameter(torch.zeros(c))
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self, module, inputs, out):
        x = out.float()
        mu = x.mean((2, 3), keepdim=True)
        sd = (x.var((2, 3), keepdim=True, unbiased=False) + 1e-4).sqrt()
        x_in = (x - mu) / sd
        x_in = x_in * (1.0 + self.a.float().view(1, -1, 1, 1)) + self.b.float().view(1, -1, 1, 1)
        alpha = self.rbox.alpha[:, self.stage_idx].view(-1, 1, 1, 1)
        g = 0.5 * torch.tanh(self.gamma.float()).view(1, -1, 1, 1)
        return (x + alpha * g * (x_in - x)).to(out.dtype)


def add_condroute(model: nn.Module) -> None:
    assert not any(isinstance(q, RepConv) for q in model.modules()), "condroute conflicts with rep"
    rbox = RouteBox()
    model.rbox = rbox
    model.condrouter = CondRouter(len(COND_STAGES), rbox)
    model.model[0].register_forward_pre_hook(_Hook(model.condrouter, "route", 0))
    model.model[-1].register_forward_hook(_Hook(model.condrouter, "clear", 0))
    stage_of = {layer: idx for idx, layer in enumerate(COND_STAGES)}
    for i in COND_CONV_LAYERS:
        convs = [q for q in model.model[i].modules() if type(q) is Conv and q.conv.kernel_size == (3, 3) and q.conv.groups == 1]
        for c in convs:
            to_condconv(c, rbox, stage_of[i])
    model.dualnorm = nn.ModuleList()
    for i in DUAL_NORM_LAYERS:
        dn = SoftDualNorm(model.model[i].cv2.conv.out_channels, rbox, stage_of[i])
        model.dualnorm.append(dn)
        model.model[i].register_forward_hook(dn)


def add_staticroute(model: nn.Module) -> None:
    """Severs input conditioning in ShiftFiLM and CondRouter (replaces image cues with learned constants)."""
    if hasattr(model, "film"):
        model.film.static_mode = True
        model.film.const_desc = nn.Parameter(torch.zeros(1, model.film.in_dim))
    if hasattr(model, "condrouter"):
        model.condrouter.static_mode = True
        model.condrouter.const_cues = nn.Parameter(torch.zeros(1, 12))


# --- Round 11: Cluster 2' (detailpath) & Cluster 3' (hyperpad) ---------------------------------------------------
def _ensure_edge_arch(model: nn.Module) -> None:
    if not isinstance(model, EdgeArchModel):
        model.__class__ = EdgeArchModel
    if not hasattr(model, "layer_extra_order"):
        model.layer_extra_order = {}
    if not hasattr(model, "cv1_expansions"):
        model.cv1_expansions = []
    if not hasattr(model, "_kept_extra_params"):
        model._kept_extra_params = 0


def _expand_c3k2_cv1(model: nn.Module, layer_idx: int, extra_c: int, src: str, tag: str) -> None:
    """Expands C3k2.cv1 input channels with zero-init weights on the new slice (exact step-0 identity, O(1) grad at step 1)."""
    c3k2 = model.model[layer_idx]
    old = c3k2.cv1.conv
    c_in, c_out = old.in_channels, old.out_channels
    new = nn.Conv2d(
        c_in + extra_c, c_out, old.kernel_size, old.stride, old.padding,
        old.dilation, old.groups, old.bias is not None,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight[:, :c_in].copy_(old.weight)
        new.weight[:, c_in:].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)
    c3k2.cv1.conv = new
    # Track preserved pretrained weights of the original cv1 (excluding any earlier expansion on this same layer)
    if not any(idx == layer_idx for idx, _, _, _ in model.cv1_expansions):
        model._kept_extra_params += old.weight.numel() + (old.bias.numel() if old.bias is not None else 0)
    model.cv1_expansions.append((layer_idx, c_in, c_in + extra_c, tag))
    model.layer_extra_order.setdefault(layer_idx, []).append(src)


class NLDownConv(Conv):
    """Neck strided Conv + parallel nonlinear lossless PixelUnshuffle detail branch (FRFDet IBS-inspired).
    SiLU between the two 1x1 convs prevents collapse to linear re-parameterisation; zero-init nl_pw2.bn gives
    exact step-0 COCO identity and O(1) gradient into the no-decay BN group at step 1."""

    def _detail(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if (h & 1) or (w & 1):
            x = F.pad(x, (0, w & 1, 0, h & 1))
        return self.nl_pw2(self.nl_pw1(self.unshuffle(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)) + self._detail(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self._detail(x))


def add_detailpath(model: nn.Module) -> None:
    """Cluster 2' (DetailPath): P2/4 feature-only injection (L2->L16) + nonlinear lossless downsampling (L17, L20)."""
    _ensure_edge_arch(model)
    dev = next(model.parameters()).device
    c2 = model.model[2].cv2.conv.out_channels
    model.save = sorted(set(model.save) | {2})
    model.p2_proj = nn.Sequential(
        Conv(c2, c2, 3, s=2, g=c2),
        Conv(c2, 32, 1, 1),
    ).to(dev)
    _expand_c3k2_cv1(model, 16, 32, src="dp", tag="dp_l16")
    for idx in (17, 20):
        m = model.model[idx]
        c_in, c_out = m.conv.in_channels, m.conv.out_channels
        c_mid = max(c_out // 2, 16)
        m.__class__ = NLDownConv
        m.unshuffle = nn.PixelUnshuffle(2)
        m.nl_pw1 = Conv(4 * c_in, c_mid, 1, 1).to(dev)
        m.nl_pw2 = Conv(c_mid, c_out, 1, 1, act=False).to(dev)
        nn.init.zeros_(m.nl_pw2.bn.weight)
        nn.init.zeros_(m.nl_pw2.bn.bias)


class CausalEdgeHyperPAD(nn.Module):
    """Causal low-res (20x20) linear O(N*M) scene-adaptive hypergraph + FullPAD low-res 1x1 tunnels (YOLOv13-inspired).
    Native FP16 throughout (zero .float() casts), convex-normalized message passing prevents FP16 overflow."""

    def __init__(self, c4: int, c6: int, c10: int, c_h: int = 64, n_edges: int = 8, out_c: int = 16,
                 targets: tuple[int, ...] = (13, 16, 19, 22)):
        super().__init__()
        self.c_h = c_h
        self.n_edges = n_edges
        self.scale = c_h ** -0.5
        self.targets = targets
        self.agg = Conv(c4 + c6 + c10, c_h, 1, 1)
        self.base_proto = nn.Parameter(torch.randn(1, n_edges, c_h) * (c_h ** -0.5))
        self.ctx_fc1 = nn.Linear(2 * c_h, 32)
        self.ctx_fc2 = nn.Linear(32, n_edges * c_h)
        nn.init.normal_(self.ctx_fc2.weight, std=0.01)
        nn.init.zeros_(self.ctx_fc2.bias)
        self.q_proj = nn.Linear(c_h, c_h, bias=False)
        self.e_proj = nn.Linear(c_h, c_h, bias=False)
        self.v_proj = nn.Linear(c_h, c_h, bias=False)
        self.local_dw = Conv(c_h, c_h, 3, 1, g=c_h)
        self.tunnels = nn.ModuleDict({str(t): Conv(c_h, out_c, 1, 1) for t in targets})

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> dict[int, torch.Tensor]:
        h5, w5 = p5.shape[-2:]
        a3 = F.adaptive_avg_pool2d(p3, (h5, w5))
        a4 = F.adaptive_avg_pool2d(p4, (h5, w5))
        x = self.agg(torch.cat([a3, a4, p5], 1))
        b, c, _, _ = x.shape
        v = x.flatten(2).transpose(1, 2)  # (B, N, C_h)
        ctx = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)  # (B, 2*C_h)
        delta_p = self.ctx_fc2(F.silu(self.ctx_fc1(ctx))).view(b, self.n_edges, c)
        proto = delta_p + self.base_proto.to(x.dtype)
        q = self.q_proj(v)
        logits = torch.bmm(q, proto.transpose(1, 2)) * self.scale
        a = torch.softmax(logits, dim=-1)  # (B, N, M)
        a_t = a.transpose(1, 2)            # (B, M, N)
        norm_e = a_t.sum(dim=-1, keepdim=True).clamp_min(1e-2)
        e = F.silu(self.e_proj(torch.bmm(a_t, v) / norm_e))
        h_glob = self.v_proj(torch.bmm(a, e)).transpose(1, 2).view(b, c, h5, w5)
        y = x + h_glob + self.local_dw(x)
        return {t: self.tunnels[str(t)](y) for t in self.targets}


def add_hyperpad(model: nn.Module) -> None:
    """Cluster 3' (Causal Edge-HyperPAD): aggregates L4/L6/L10 at 20x20 -> distributes to neck L13/L16/L19/L22."""
    _ensure_edge_arch(model)
    dev = next(model.parameters()).device
    c4 = model.model[4].cv2.conv.out_channels
    c6 = model.model[6].cv2.conv.out_channels
    c10 = model.model[10].cv2.conv.out_channels
    targets = (13, 16, 19, 22)
    out_c = 16
    model.save = sorted(set(model.save) | {4, 6, 10})
    model.hyperpad = CausalEdgeHyperPAD(c4, c6, c10, c_h=64, n_edges=8, out_c=out_c, targets=targets).to(dev)
    for t in targets:
        _expand_c3k2_cv1(model, t, out_c, src="hp", tag=f"hp_l{t}")


ADD = {
    "rep": add_rep,
    "lk": add_lk,
    "film": add_film,
    "mix": add_mix,
    "cons": add_cons,
    "divdet": add_divdet,
    "spatialctx": add_spatialctx,
    "condroute": add_condroute,
    "staticroute": add_staticroute,
    "detailpath": add_detailpath,
    "hyperpad": add_hyperpad,
}


def apply(model: nn.Module, mods: list[str]) -> nn.Module:
    alive = list(model.parameters())
    before = {id(p) for p in alive}
    n0 = sum(p.numel() for p in model.parameters())
    for name in mods:
        ADD[name](model)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    total = sum(p.numel() for p in model.parameters())
    kept = (
        sum(p.numel() for p in model.parameters() if id(p) in before)
        + sum(p.numel() for m in model.modules() if isinstance(m, RepConv) for p in m.conv1.parameters())
        + getattr(model, "_kept_extra_params", 0)
    )
    print(f"[mods] mods={','.join(mods) or 'none'} train_params={total} stock_params={n0} "
          f"new_params={total - kept} pretrained_share={100 * kept / total:.2f}%", flush=True)
    return model


class ModTrainer(DetectionTrainer):
    mods: list[str] = []

    def get_model(self, cfg=None, weights=None, verbose=True):
        return apply(super().get_model(cfg, weights, verbose), self.mods)


if __name__ == "__main__":  # exhaustive self-check for all 4 Round-11 configs
    import copy
    import io

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    from bench import efficiency

    torch.manual_seed(0)
    base = YOLO("yolo26n.pt").model.float().eval()
    base_eff = efficiency.params_flops(base, 640)
    x_sq = torch.rand(2, 3, 256, 256)
    x_rect = torch.rand(2, 3, 192, 256)  # tests rect=True letterbox aspect ratios
    ref_sq = base(x_sq)[0]
    ref_rect = base(x_rect)[0]
    batch = {
        "img": x_sq,
        "cls": torch.zeros(3, 1),
        "bboxes": torch.rand(3, 4) * 0.3 + 0.3,
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    }

    configs = [
        ("R11-A1:DivDet  ", ["divdet"]),
        ("R11-DP:DetailP ", ["divdet", "detailpath"]),
        ("R11-HP:HyperPAD", ["divdet", "hyperpad"]),
        ("R11-DPHP:Both  ", ["divdet", "detailpath", "hyperpad"]),
        ("R11-HPDP:RevOrd", ["divdet", "hyperpad", "detailpath"]),  # verify order independence!
    ]
    for label, mod_list in configs:
        r = apply(copy.deepcopy(base), mod_list).eval()
        # 1. Zero duplicate tensors in state_dict (protects ModelEMA)
        assert len(r.state_dict()) == len({id(v) for v in r.state_dict().values()}), f"{label}: duplicate tensor"
        # 2. Exact step-0 identity with pretrained COCO YOLO26n on BOTH square & rectangular inputs
        assert torch.allclose(r(x_sq)[0], ref_sq, atol=1e-5), f"{label}: not identical at step 0 (sq)"
        assert torch.allclose(r(x_rect)[0], ref_rect, atol=1e-5), f"{label}: not identical at step 0 (rect)"
        # 3. Warm BN running stats then perturb zero-init slices/BN off zero to verify fuse, checkpoint, half, causal zero-out
        r.train()
        with torch.no_grad():
            for _ in range(2):
                r(torch.rand(2, 3, 256, 256))
            r.tap.feats = None
            r.pred_grab.preds = None
        r.eval()
        ref_warm = r(x_sq)[0]
        if mod_list != ["divdet"]:
            with torch.no_grad():
                for idx, s, e, _ in getattr(r, "cv1_expansions", ()):
                    nn.init.normal_(r.model[idx].cv1.conv.weight[:, s:e], std=0.05)
                for idx in (17, 20):
                    if isinstance(r.model[idx], NLDownConv):
                        nn.init.normal_(r.model[idx].nl_pw2.bn.weight, std=0.05)
        out_sq = r(x_sq)[0]
        out_rect = r(x_rect)[0]
        if mod_list != ["divdet"]:
            assert not torch.allclose(out_sq, ref_warm, atol=1e-3), f"{label}: perturbed output unchanged"
            # Causal zero-out check on a copy: zeroing new weights must restore exact base output!
            rz = copy.deepcopy(r)
            rz.zero_new_weights()
            assert torch.allclose(rz(x_sq)[0], ref_warm, atol=1e-5), f"{label}: zero_new_weights failed to restore base!"
        # 4. Checkpoint round-trip
        buf = io.BytesIO()
        torch.save(r, buf)
        buf.seek(0)
        assert torch.allclose(torch.load(buf, weights_only=False).eval()(x_sq)[0], out_sq, atol=1e-5), f"{label}: checkpoint"
        # 5. Fuse check (all Conv+BN including p2_proj, hyperpad, NLDownConv must fold!)
        fz = copy.deepcopy(r).fuse(verbose=False)
        for sub_name in ("p2_proj", "hyperpad"):
            sub = getattr(fz, sub_name, None)
            if sub is not None:
                assert not any(isinstance(k, Conv) and hasattr(k, "bn") for k in sub.modules()), f"{label}: {sub_name} BN not fused"
        for idx in (17, 20):
            if isinstance(fz.model[idx], NLDownConv):
                assert not hasattr(fz.model[idx], "bn") and not hasattr(fz.model[idx].nl_pw1, "bn") and not hasattr(fz.model[idx].nl_pw2, "bn")
        assert torch.allclose(fz(x_sq)[0], out_sq, atol=1e-3, rtol=1e-3), f"{label}: fused sq mismatch"
        assert torch.allclose(fz(x_rect)[0], out_rect, atol=1e-3, rtol=1e-3), f"{label}: fused rect mismatch"
        # 6. Native FP16 half+fuse and fuse+half checks (ZERO NaNs, both square & rect!)
        hf1 = copy.deepcopy(r).half().fuse(verbose=False)
        o_hf1_sq = hf1(x_sq.half())[0].float()
        o_hf1_rect = hf1(x_rect.half())[0].float()
        assert torch.isfinite(o_hf1_sq).all() and torch.isfinite(o_hf1_rect).all(), f"{label}: NaN in half+fuse"
        assert torch.allclose(o_hf1_sq, out_sq, atol=5e-2, rtol=5e-2), f"{label}: half+fuse sq mismatch"
        assert torch.allclose(o_hf1_rect, out_rect, atol=5e-2, rtol=5e-2), f"{label}: half+fuse rect mismatch"
        hf2 = copy.deepcopy(r).fuse(verbose=False).half()
        o_hf2_sq = hf2(x_sq[:1].half())[0].float()
        assert torch.isfinite(o_hf2_sq).all() and torch.allclose(o_hf2_sq, out_sq[:1], atol=5e-2, rtol=5e-2), f"{label}: fuse+half b=1"
        # 7. Two full training steps from EXACT step-0 zero init -> verify ALL new params wake up with finite non-zero grad!
        rt = apply(copy.deepcopy(base), mod_list)
        rt.args = get_cfg()
        rt.train()
        for q in rt.parameters():
            q.requires_grad_(True)
        opt = torch.optim.SGD(rt.parameters(), lr=1e-2)
        for step in range(2):
            opt.zero_grad()
            loss, _ = rt.loss(batch)
            assert torch.isfinite(loss).all(), f"{label}: non-finite loss step {step}"
            loss.sum().backward()
            if step == 0:
                opt.step()
        assert rt.tap.feats is None and rt.pred_grab.preds is None
        for idx, s, e, tag in getattr(rt, "cv1_expansions", ()):
            g = rt.model[idx].cv1.conv.weight.grad[:, s:e]
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{label}: zero grad on {tag}"
        for sub_name in ("p2_proj", "hyperpad"):
            sub = getattr(rt, sub_name, None)
            if sub is not None:
                for pname, p in sub.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{label}: zero grad {sub_name}.{pname}"
        for idx in (17, 20):
            if isinstance(rt.model[idx], NLDownConv):
                for sub_attr in ("nl_pw1", "nl_pw2"):
                    for pname, p in getattr(rt.model[idx], sub_attr).named_parameters():
                        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{label}: zero grad L{idx}.{sub_attr}.{pname}"
        liveness = rt.liveness_report()
        _ = copy.deepcopy(rt)
        # 8. Exact deployed params & GFLOPs@640 after fuse() within <= +10% budget
        eff = efficiency.params_flops(r, 640)
        dp = 100.0 * (eff["params_M"] - base_eff["params_M"]) / base_eff["params_M"]
        df = 100.0 * (eff["GFLOPs@640"] - base_eff["GFLOPs@640"]) / base_eff["GFLOPs@640"]
        assert dp <= 10.0 and df <= 10.0, f"{label}: exceeds +10% budget ({dp:.2f}%, {df:.2f}%)"
        print(f"[check OK] {label} | {eff} | dParams={dp:+.2f}% dFLOPs={df:+.2f}% | live={len(liveness)} | loss={float(loss.detach().sum()):.3f}")

    print("ALL ROUND-11 SELF-CHECKS PASSED")
