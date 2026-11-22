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

Hooks are module-level callables so that checkpoints (pickled models) and ModelEMA deepcopy keep them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2, Conv, Conv2, Detect, RepConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import fuse_conv_and_bn, unwrap_model

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


class DivDetModel(DetectionModel):
    """Supervised detection on BOTH clean and non-weather diversified views (0.5/0.5) + paired feature & output alignment,
    plus optional Round-12 cls-only StyleNullProj covariance updates and TAL-aligned PAMR-Lite prototype losses."""

    @torch.no_grad()
    def _tal_targets(self, preds_o2m: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        from ultralytics.utils.tal import make_anchors

        crit = self.criterion.one2many
        pred_distri = preds_o2m["boxes"].detach().permute(0, 2, 1).contiguous()
        pred_scores = preds_o2m["scores"].detach().permute(0, 2, 1).contiguous()
        feats = preds_o2m["feats"]
        anchor_points, stride_tensor = make_anchors(feats, crit.stride, 0.5)
        bs = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=crit.device, dtype=pred_scores.dtype) * crit.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = crit.preprocess(targets.to(crit.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = crit.bbox_decode(anchor_points, pred_distri)
        target_labels, _, _, fg_mask, _ = crit.assigner(
            pred_scores.sigmoid(),
            (pred_bboxes * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        return fg_mask, target_labels.long()

    def zero_new_weights(self) -> None:
        """Causal ablation helper: disables newly added cls-only inference ops (alpha=0 / out_projs=0)."""
        det = self.model[-1]
        sn = getattr(det, "stylenull", None)
        if sn is not None:
            sn.zero_new_weights()
        pm = getattr(det, "pamr", None)
        if pm is not None:
            pm.zero_new_weights()

    def diag_report(self) -> dict[str, float]:
        out: dict[str, float] = {}
        det = self.model[-1]
        sn = getattr(det, "stylenull", None)
        if sn is not None:
            out.update(sn.diag_report())
        pm = getattr(det, "pamr", None)
        if pm is not None:
            out.update(pm.diag_report())
        return out

    def fuse(self, verbose=True, imgsz=640):
        was_fused = self.is_fused()
        super().fuse(verbose=verbose, imgsz=imgsz)
        if not was_fused:
            pm = getattr(self.model[-1], "pamr", None)
            if pm is not None:
                for m in pm.modules():
                    if isinstance(m, Conv) and hasattr(m, "bn"):
                        m.conv = fuse_conv_and_bn(m.conv, m.bn)
                        delattr(m, "bn")
                        m.forward = m.forward_fuse
        return self

    def loss(self, batch, preds=None):
        det = self.model[-1]
        pm = getattr(det, "pamr", None)
        sn = getattr(det, "stylenull", None)
        loss_clean, items = super().loss(batch, preds)
        if not self.training or preds is not None:
            if pm is not None:
                pm._last_z_hats = None
            return loss_clean, items
        f1, self.tap.feats = self.tap.feats, None
        p1, self.pred_grab.preds = self.pred_grab.preds, None
        z1 = None
        if pm is not None:
            z1, pm._last_z_hats = pm._last_z_hats, None
        view = div_augment(batch["img"])
        z2 = None
        try:
            loss_div, _ = super().loss({**batch, "img": view})
            f2, self.tap.feats = self.tap.feats, None
            p2, self.pred_grab.preds = self.pred_grab.preds, None
            if pm is not None:
                z2, pm._last_z_hats = pm._last_z_hats, None
        finally:
            self.tap.feats = None
            self.pred_grab.preds = None
            if pm is not None:
                pm._last_z_hats = None
            rbox = getattr(self, "rbox", None)
            if rbox is not None:
                rbox.alpha = None
        if sn is not None and f1 is not None and f2 is not None:
            sn.update_cov(f1, f2)
        c_cos = sum(1 - F.cosine_similarity(a.detach().float(), b.float(), dim=1).mean() for a, b in zip(f1, f2)) / len(f1)
        l_cls, l_box = _dual_head_align(p1, p2, k=64) if (p1 is not None and isinstance(p2, dict)) else (0.0, 0.0)
        align = 0.5 * c_cos + 0.25 * l_cls + 0.05 * l_box
        if pm is not None and z1 is not None and z2 is not None and isinstance(p1, dict) and "one2many" in p1:
            fg_mask, target_labels = self._tal_targets(p1["one2many"], batch)
            align = align + pm.compute_loss(z1, z2, fg_mask, target_labels)
        total = 0.5 * loss_clean + 0.5 * loss_div + align * batch["img"].shape[0] / loss_clean.numel()
        return total, items


def add_divdet(model: nn.Module) -> None:
    model.__class__ = DivDetModel
    model.tap = DetectTap()
    model.pred_grab = PredGrab()
    model.model[-1].register_forward_pre_hook(model.tap)
    model.model[-1].register_forward_hook(model.pred_grab)


# --- Round 12 (Method B): Cls-only edge-native domain-invariant clusters (StyleNullProj + PAMR-Lite) -------------
class ClsModDetect(Detect):
    """Detect head extension applying cls-only invariance ops (stylenull / pamr) BEFORE cv3 and one2one_cv3.
    Leaves box regression (cv2 / one2one_cv2) 100% untouched, computes x_cls ONCE per forward (preserving YOLO26n's
    one2one detach boundary during training and halving inference overhead), and keeps extra params outside cv3
    so they do not inherit MuSGD's 3x classification-head LR boost."""

    def _cls_transform(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        xc = x
        sn = getattr(self, "stylenull", None)
        if sn is not None:
            xc = sn(xc)
        pm = getattr(self, "pamr", None)
        if pm is not None:
            xc = pm(xc)
        return xc

    def _forward_split(
        self,
        x_box: list[torch.Tensor],
        x_cls: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return {}
        bs = x_box[0].shape[0]
        boxes = torch.cat([box_head[i](x_box[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x_cls[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x_box}

    def forward(self, x: list[torch.Tensor]):
        x_cls = self._cls_transform(x)
        preds = self._forward_split(x, x_cls, **self.one2many)
        if getattr(self, "one2one_cv2", None) is not None:
            x_det = [xi.detach() for xi in x] if self.training else x
            xc_det = [ci.detach() for ci in x_cls] if self.training else x_cls
            one2one = self._forward_split(x_det, xc_det, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one" if self.end2end else "one2many"] if "one2one" in preds else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


def _ensure_cls_mod_detect(model: nn.Module) -> Detect:
    det = model.model[-1]
    if not isinstance(det, ClsModDetect):
        det.__class__ = ClsModDetect
    return det


class StyleNullProj(nn.Module):
    """Cluster 2'' (DG-DETR DAQS faithful edge port): cls-only paired-view rank-k orthogonal style-nullspace projection.
    Zero learned parameters (pure buffers, immune to Muon), exact step-0 COCO identity (alpha=0 during 3-epoch warmup,
    linearly ramped 0->1 over epochs 3..10), pure FP16 TensorRT 1x1 conv projections, trivial causal check (alpha=0)."""

    def __init__(self, chs: list[int], k: int = 4, warmup_epochs: int = 3, ramp_epochs: int = 8):
        super().__init__()
        self.chs = list(chs)
        self.ks = [min(k, max(1, c // 8)) for c in self.chs]
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.register_buffer("alpha", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("seen", torch.tensor(0, dtype=torch.long))
        for i, (c, ki) in enumerate(zip(self.chs, self.ks)):
            self.register_buffer(f"cov_{i}", torch.zeros(c, c, dtype=torch.float32))
            self.register_buffer(f"u_{i}", torch.zeros(c, ki, dtype=torch.float32))

    @torch.no_grad()
    def update_cov(self, f_clean: list[torch.Tensor], f_div: list[torch.Tensor]) -> None:
        m = 1.0 if int(self.seen.item()) == 0 else 0.02
        with torch.amp.autocast(device_type=f_clean[0].device.type, enabled=False):
            for i, (fc, fd) in enumerate(zip(f_clean, f_div)):
                d = fd.detach().float().mean((2, 3)) - fc.detach().float().mean((2, 3))  # (B, C)
                cov_b = ((d.t() @ d) / max(d.shape[0], 1)).float()
                getattr(self, f"cov_{i}").lerp_(cov_b, m)
        self.seen.add_(1)

    @torch.no_grad()
    def update_basis(self) -> None:
        if int(self.seen.item()) == 0:
            return
        with torch.amp.autocast(device_type=self.alpha.device.type, enabled=False):
            for i, ki in enumerate(self.ks):
                cov = getattr(self, f"cov_{i}").float()
                cov = 0.5 * (cov + cov.t())
                evals, evecs = torch.linalg.eigh(cov)
                u_new = evecs[:, -ki:].contiguous()
                u_buf = getattr(self, f"u_{i}")
                if u_buf.abs().sum() > 0:
                    flip = (u_buf.float() * u_new).sum(dim=0) < 0
                    if flip.any():
                        u_new[:, flip] *= -1.0
                u_buf.copy_(u_new.to(u_buf.dtype))

    @torch.no_grad()
    def set_epoch(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            val = 0.0
        else:
            val = min(1.0, float(epoch - self.warmup_epochs + 1) / float(self.ramp_epochs))
        self.alpha.fill_(val)

    @torch.no_grad()
    def zero_new_weights(self) -> None:
        self.alpha.zero_()

    @torch.no_grad()
    def sync_to(self, other: "StyleNullProj") -> None:
        other.alpha.copy_(self.alpha)
        other.seen.copy_(self.seen)
        for i in range(len(self.chs)):
            getattr(other, f"cov_{i}").copy_(getattr(self, f"cov_{i}"))
            getattr(other, f"u_{i}").copy_(getattr(self, f"u_{i}"))

    @torch.no_grad()
    def diag_report(self) -> dict[str, float]:
        out = {"sn_alpha": float(self.alpha.item()), "sn_seen": float(self.seen.item())}
        max_err, var_ratios = 0.0, []
        for i, ki in enumerate(self.ks):
            u = getattr(self, f"u_{i}").float()
            if u.abs().sum() > 0:
                eye = torch.eye(ki, device=u.device, dtype=torch.float32)
                max_err = max(max_err, float((u.t() @ u - eye).abs().max().item()))
            cov = getattr(self, f"cov_{i}").float()
            tr = float(torch.trace(cov).item())
            if tr > 1e-8:
                evals = torch.linalg.eigvalsh(0.5 * (cov + cov.t()))
                var_ratios.append(float(evals[-ki:].clamp_min(0).sum().item() / tr))
        out["sn_ortho_err"] = max_err
        if var_ratios:
            out["sn_top4_var"] = sum(var_ratios) / len(var_ratios)
        return out

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        a = self.alpha.to(feats[0].dtype)
        out = []
        for i, x in enumerate(feats):
            u = getattr(self, f"u_{i}").to(x.dtype)  # (C, k)
            proj = F.conv2d(F.conv2d(x, u.t()[:, :, None, None]), u[:, :, None, None])
            out.append(x - a * proj)
        return out


def add_stylenull(model: nn.Module) -> None:
    """Cluster 2'': cls-only paired-view rank-4 orthogonal style-nullspace projection on Detect P3/P4/P5."""
    det = _ensure_cls_mod_detect(model)
    dev = next(det.parameters()).device
    chs = [det.cv2[i][0].conv.in_channels for i in range(det.nl)]
    det.stylenull = StyleNullProj(chs, k=4, warmup_epochs=3, ramp_epochs=8).to(dev)


class PAMRLite(nn.Module):
    """Cluster 3'' (MR-DCoT edge port): cls-only magnitude-invariant EMA-prototype manifold rectifier.
    Uses EMA prototype buffers (immune to Muon), exact TAL one2many foreground/background targets, post-BN L2 norm
    (FP16-safe), constant tau=10 (no 0-D SGD decay), and zero-init 4-D out_proj (exact step-0 COCO identity).
    With loss_only=True, out_projs stay frozen at 0 (stock inference graph) as a mandatory causal loss-only control."""

    def __init__(self, chs: list[int], nc: int, d_p: int = 64, tau: float = 10.0, loss_only: bool = False):
        super().__init__()
        self.chs = list(chs)
        self.nc = int(nc)
        self.k = self.nc + 1  # nc foreground classes + 1 background anchor
        self.d_p = int(d_p)
        self.tau = float(tau)
        self.loss_only = bool(loss_only)
        self.in_projs = nn.ModuleList([Conv(c, self.d_p, 1, 1, act=False) for c in self.chs])
        self.out_projs = nn.ModuleList([nn.Conv2d(self.d_p, c, 1, 1, bias=False) for c in self.chs])
        for op in self.out_projs:
            nn.init.zeros_(op.weight)
            if self.loss_only:
                op.weight.requires_grad_(False)
        self.register_buffer("protos", torch.zeros(self.k, self.d_p, dtype=torch.float32))
        self.register_buffer("proto_mask", torch.full((self.k,), -1e4, dtype=torch.float32))
        self.register_buffer("proto_seen", torch.zeros(self.k, dtype=torch.long))
        self._last_z_hats: list[torch.Tensor] | None = None

    @torch.no_grad()
    def zero_new_weights(self) -> None:
        for op in self.out_projs:
            op.weight.zero_()

    @torch.no_grad()
    def sync_to(self, other: "PAMRLite") -> None:
        other.protos.copy_(self.protos)
        other.proto_mask.copy_(self.proto_mask)
        other.proto_seen.copy_(self.proto_seen)

    @torch.no_grad()
    def diag_report(self) -> dict[str, float]:
        active = self.proto_seen > 0
        out = {
            "pm_active": float(active.sum().item()),
            "pm_out_rms": sum(float(op.weight.detach().float().pow(2).mean().sqrt()) for op in self.out_projs) / len(self.out_projs),
        }
        if active.sum() >= 2:
            p = self.protos[active].float()
            gram = p @ p.t()
            eye = torch.eye(gram.shape[0], device=gram.device)
            off = (gram - eye).abs()
            out["pm_proto_offcos"] = float(off.sum() / max(gram.shape[0] * (gram.shape[0] - 1), 1))
        return out

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.loss_only and not self.training:
            return feats
        z_hats = []
        outs = []
        p_hat = (self.protos.clone() if self.training else self.protos).to(feats[0].dtype)
        mask = (self.proto_mask.clone() if self.training else self.proto_mask).to(feats[0].dtype).view(1, self.k, 1, 1)
        for i, x in enumerate(feats):
            z = self.in_projs[i](x)
            z_hat = z * torch.rsqrt((z * z).sum(dim=1, keepdim=True) + 1e-4)
            if self.training:
                z_hats.append(z_hat)
            if self.loss_only:
                outs.append(x)
            else:
                logits = F.conv2d(z_hat, p_hat[:, :, None, None]) * self.tau + mask
                theta = torch.softmax(logits, dim=1)
                m_proto = F.conv2d(theta, p_hat.t()[:, :, None, None])
                disp = m_proto - z_hat
                outs.append(x + self.out_projs[i](disp))
        self._last_z_hats = z_hats if self.training else None
        return outs

    def compute_loss(
        self,
        z1_list: list[torch.Tensor],
        z2_list: list[torch.Tensor],
        fg_mask: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=z1_list[0].device.type, enabled=False):
            return self._compute_loss_fp32(z1_list, z2_list, fg_mask, target_labels)

    def _compute_loss_fp32(
        self,
        z1_list: list[torch.Tensor],
        z2_list: list[torch.Tensor],
        fg_mask: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        z1 = torch.cat([z.flatten(2) for z in z1_list], dim=2).transpose(1, 2).float()  # (B, N, d_p)
        z2 = torch.cat([z.flatten(2) for z in z2_list], dim=2).transpose(1, 2).float()  # (B, N, d_p)
        dev = z1.device
        zero = z1.sum() * 0.0

        # Sample up to 256 background anchors per batch for balanced EMA update & bg loss
        bg_coords = torch.nonzero(~fg_mask, as_tuple=False)
        if bg_coords.shape[0] > 256:
            perm = torch.randperm(bg_coords.shape[0], device=dev)[:256]
            bg_coords = bg_coords[perm]

        # 1. EMA prototype updates strictly from clean view (detached, immune to Muon)
        with torch.no_grad():
            z1_det = z1.detach()
            if fg_mask.any():
                fg_z = z1_det[fg_mask]
                fg_y = target_labels[fg_mask]
                for c in torch.unique(fg_y).tolist():
                    if 0 <= c < self.nc:
                        mean_c = F.normalize(fg_z[fg_y == c].mean(dim=0, keepdim=True), dim=1, eps=1e-6).squeeze(0)
                        mom = 1.0 if int(self.proto_seen[c].item()) == 0 else 0.05
                        updated = F.normalize((1.0 - mom) * self.protos[c] + mom * mean_c, dim=0, eps=1e-6)
                        self.protos[c].copy_(updated)
                        self.proto_mask[c] = 0.0
                        self.proto_seen[c] += 1
            if bg_coords.shape[0] > 0:
                bg_z = z1_det[bg_coords[:, 0], bg_coords[:, 1]]
                mean_bg = F.normalize(bg_z.mean(dim=0, keepdim=True), dim=1, eps=1e-6).squeeze(0)
                mom = 1.0 if int(self.proto_seen[self.nc].item()) == 0 else 0.05
                updated = F.normalize((1.0 - mom) * self.protos[self.nc] + mom * mean_bg, dim=0, eps=1e-6)
                self.protos[self.nc].copy_(updated)
                self.proto_mask[self.nc] = 0.0
                self.proto_seen[self.nc] += 1

        # 2. Prototype-anchored manifold regression & inter-class separation losses
        p_all = self.protos.detach()
        seen_mask = self.proto_seen > 0
        valid_fg = fg_mask & (self.proto_seen[target_labels.clamp(0, self.nc - 1)] > 0)
        l_anchor_fg, l_rect_fg, l_sep = zero, zero, zero
        if valid_fg.any():
            z1_fg = z1[valid_fg]
            z2_fg = z2[valid_fg]
            y_fg = target_labels[valid_fg]
            p_y = p_all[y_fg]
            l_anchor_fg = F.relu(1.0 - (z1_fg * p_y).sum(dim=-1) - 0.15).mean()
            l_rect_fg = F.relu(1.0 - (z2_fg * p_y).sum(dim=-1) - 0.15).mean()
            if int(seen_mask.sum().item()) >= 2:
                cos_all = z1_fg @ p_all.t()  # (M, K)
                invalid = (~seen_mask).unsqueeze(0).expand_as(cos_all).clone()
                invalid.scatter_(1, y_fg.unsqueeze(1), True)
                cos_other = cos_all.masked_fill(invalid, -2.0)
                l_sep = F.relu(cos_other.max(dim=-1).values - 0.30).mean()

        l_bg = zero
        if bg_coords.shape[0] > 0 and bool(seen_mask[self.nc].item()):
            z1_bg = z1[bg_coords[:, 0], bg_coords[:, 1]]
            z2_bg = z2[bg_coords[:, 0], bg_coords[:, 1]]
            p_bg = p_all[self.nc]
            l_bg = (
                F.relu(1.0 - (z1_bg * p_bg).sum(dim=-1) - 0.25).mean()
                + 0.5 * F.relu(1.0 - (z2_bg * p_bg).sum(dim=-1) - 0.25).mean()
            )

        return 0.20 * (0.5 * l_anchor_fg + 0.5 * l_rect_fg + 0.25 * l_bg + 0.25 * l_sep)


def add_pamrlite(model: nn.Module) -> None:
    """Cluster 3'': cls-only magnitude-invariant EMA-prototype manifold rectifier on Detect P3/P4/P5."""
    det = _ensure_cls_mod_detect(model)
    dev = next(det.parameters()).device
    chs = [det.cv2[i][0].conv.in_channels for i in range(det.nl)]
    det.pamr = PAMRLite(chs, nc=det.nc, d_p=64, tau=10.0, loss_only=False).to(dev)


def add_pamrloss(model: nn.Module) -> None:
    """Causal loss-only control for PAMR-Lite: identical TAL prototype losses, out_proj frozen at 0 (stock inference)."""
    det = _ensure_cls_mod_detect(model)
    dev = next(det.parameters()).device
    chs = [det.cv2[i][0].conv.in_channels for i in range(det.nl)]
    det.pamr = PAMRLite(chs, nc=det.nc, d_p=64, tau=10.0, loss_only=True).to(dev)


def on_r12_epoch_start(tr) -> None:
    """Updates StyleNullProj alpha schedule on both model and EMA at the start of every training epoch."""
    m = unwrap_model(tr.model)
    sn = getattr(m.model[-1], "stylenull", None)
    if sn is not None:
        sn.set_epoch(tr.epoch)
        if getattr(tr, "ema", None) is not None and tr.ema.enabled:
            sn_ema = getattr(tr.ema.ema.model[-1], "stylenull", None)
            if sn_ema is not None:
                sn_ema.set_epoch(tr.epoch)


def on_r12_epoch_end(tr) -> None:
    """Updates StyleNullProj eigenvectors and exact-copies non-EMA buffers (U, alpha, unit protos, masks) to ModelEMA
    RIGHT BEFORE Ultralytics validates tr.ema.ema and saves best.pt."""
    m = unwrap_model(tr.model)
    det = m.model[-1]
    ema_det = tr.ema.ema.model[-1] if (getattr(tr, "ema", None) is not None and tr.ema.enabled) else None
    sn = getattr(det, "stylenull", None)
    if sn is not None:
        sn.update_basis()
        if ema_det is not None and hasattr(ema_det, "stylenull"):
            sn.sync_to(ema_det.stylenull)
    pm = getattr(det, "pamr", None)
    if pm is not None and ema_det is not None and hasattr(ema_det, "pamr"):
        pm.sync_to(ema_det.pamr)


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
    "stylenull": add_stylenull,
    "pamrlite": add_pamrlite,
    "pamrloss": add_pamrloss,
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
    kept = sum(p.numel() for p in model.parameters() if id(p) in before) + sum(
        p.numel() for m in model.modules() if isinstance(m, RepConv) for p in m.conv1.parameters())
    print(f"[mods] mods={','.join(mods) or 'none'} train_params={total} stock_params={n0} "
          f"new_params={total - kept} pretrained_share={100 * kept / total:.2f}%", flush=True)
    return model


class ModTrainer(DetectionTrainer):
    mods: list[str] = []

    def get_model(self, cfg=None, weights=None, verbose=True):
        return apply(super().get_model(cfg, weights, verbose), self.mods)


if __name__ == "__main__":  # exhaustive self-check for all 4 Round-12 configs (incl. odd 224x256, EMA, MuSGD, causal)
    import copy
    import io
    from types import SimpleNamespace

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import ModelEMA

    from bench import efficiency

    torch.manual_seed(0)
    base = YOLO("yolo26n.pt").model.float().eval()
    base_eff = efficiency.params_flops(base, 640)
    x_sq = torch.rand(2, 3, 256, 256)
    x_rect = torch.rand(2, 3, 192, 256)
    x_odd = torch.rand(2, 3, 224, 256)  # P5 = 7x8 (odd height!)
    ref_sq = base(x_sq)[0]
    ref_rect = base(x_rect)[0]
    ref_odd = base(x_odd)[0]
    batch = {
        "img": x_sq,
        "cls": torch.tensor([[0.0], [1.0], [2.0]]),
        "bboxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.25, 0.25], [0.5, 0.5, 0.3, 0.3]]),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
    }

    configs = [
        ("R12-Ctrl :DivDet   ", ["divdet"]),
        ("R12-SN   :StyleNull", ["divdet", "stylenull"]),
        ("R12-PL   :PAMRLite ", ["divdet", "pamrlite"]),
        ("R12-PLoss:PAMRLoss ", ["divdet", "pamrloss"]),
    ]
    for label, mod_list in configs:
        torch.manual_seed(0)
        r = apply(copy.deepcopy(base), mod_list).eval()
        # 1. Zero duplicate tensors in state_dict (protects ModelEMA)
        assert len(r.state_dict()) == len({id(v) for v in r.state_dict().values()}), f"{label}: duplicate tensor"
        # 2. Exact step-0 identity with pretrained COCO YOLO26n on square, rect, AND odd-P5 rect!
        assert torch.allclose(r(x_sq)[0], ref_sq, atol=1e-5), f"{label}: step-0 mismatch (sq)"
        assert torch.allclose(r(x_rect)[0], ref_rect, atol=1e-5), f"{label}: step-0 mismatch (rect)"
        assert torch.allclose(r(x_odd)[0], ref_odd, atol=1e-5), f"{label}: step-0 mismatch (odd 224x256)"

        # 3. Verify optimizer parameter groups (MuSGD): stylenull=0 params, pamr NOT in 3x lr group, protos not a Param
        det = r.model[-1]
        if hasattr(det, "stylenull"):
            assert sum(p.numel() for p in det.stylenull.parameters()) == 0, f"{label}: stylenull must have 0 params!"
        if hasattr(det, "pamr"):
            boosted = {id(p) for m in (det.cv3, det.one2one_cv3) for p in m.parameters()}
            assert not any(id(p) in boosted for p in det.pamr.parameters()), f"{label}: pamr leaked into 3x lr group!"
            assert "protos" not in dict(det.pamr.named_parameters()), f"{label}: protos must be buffer, not Parameter!"

        # 4. Run 2 training steps + ModelEMA + epoch start/end callbacks (tests TAL targets, EMA buffer sync, grads)
        rt = apply(copy.deepcopy(base), mod_list)
        rt.args = get_cfg()
        rt.train()
        for q in rt.parameters():
            q.requires_grad_(True)
        if "pamrloss" in mod_list:
            for op in rt.model[-1].pamr.out_projs:
                op.weight.requires_grad_(False)
        ema = ModelEMA(rt)
        fake_tr = SimpleNamespace(model=rt, ema=ema, epoch=0)
        opt = torch.optim.SGD([p for p in rt.parameters() if p.requires_grad], lr=1e-2)
        for step in range(2):
            fake_tr.epoch = 4 if step == 1 else 0  # test alpha ramp on step 1
            on_r12_epoch_start(fake_tr)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss, _ = rt.loss(batch)
            assert torch.isfinite(loss).all(), f"{label}: non-finite loss step {step}"
            loss.sum().backward()
            opt.step()
            ema.update(rt)
            on_r12_epoch_end(fake_tr)
        assert rt.tap.feats is None and rt.pred_grab.preds is None
        if hasattr(rt.model[-1], "pamr"):
            assert rt.model[-1].pamr._last_z_hats is None
            pm = rt.model[-1].pamr
            pm_ema = ema.ema.model[-1].pamr
            assert int((pm.proto_seen > 0).sum().item()) >= 2, f"{label}: PAMR protos did not update from TAL!"
            assert torch.equal(pm.proto_seen, pm_ema.proto_seen), f"{label}: EMA proto_seen out of sync!"
            assert torch.allclose(pm.protos, pm_ema.protos, atol=1e-6), f"{label}: EMA protos out of sync!"
            for ip in pm.in_projs:
                for pname, p in ip.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{label}: zero grad in_proj.{pname}"
            if "pamrloss" in mod_list:
                assert all(op.weight.abs().sum() == 0 for op in pm.out_projs), f"{label}: pamrloss out_projs moved!"
            else:
                assert all(op.weight.grad is not None and op.weight.grad.abs().sum() > 0 for op in pm.out_projs), f"{label}: pamrlite zero grad out_projs!"
        if hasattr(rt.model[-1], "stylenull"):
            sn = rt.model[-1].stylenull
            sn_ema = ema.ema.model[-1].stylenull
            assert float(sn_ema.alpha.item()) > 0.0, f"{label}: EMA alpha stayed 0!"
            diag = sn_ema.diag_report()
            assert diag["sn_ortho_err"] < 1e-4, f"{label}: EMA U_l not orthonormal ({diag})!"

        # 5. Eval model (from trained EMA): verify cls-only isolation (boxes unchanged vs zero_new_weights), fuse, FP16, ckpt
        ev = copy.deepcopy(ema.ema).float().eval()
        if "pamrlite" in mod_list:
            with torch.no_grad():
                for op in ev.model[-1].pamr.out_projs:
                    nn.init.normal_(op.weight, std=0.05)
        out_sq, raw_sq = ev(x_sq)
        out_odd = ev(x_odd)[0]
        # Causal zero-out must restore exact base/unprojected outputs AND box logits must be 100% identical!
        ez = copy.deepcopy(ev)
        ez.zero_new_weights()
        out_z, raw_z = ez(x_sq)
        assert torch.allclose(raw_sq["one2one"]["boxes"], raw_z["one2one"]["boxes"], atol=1e-5), f"{label}: box head was touched!"
        if mod_list in (["divdet", "stylenull"], ["divdet", "pamrlite"]):
            assert not torch.allclose(out_sq, out_z, atol=1e-4), f"{label}: active module had zero effect on output!"
        else:
            assert torch.allclose(out_sq, out_z, atol=1e-5), f"{label}: control/loss-only output changed!"

        # Checkpoint round-trip
        buf = io.BytesIO()
        torch.save(ev, buf)
        buf.seek(0)
        assert torch.allclose(torch.load(buf, weights_only=False).eval()(x_sq)[0], out_sq, atol=1e-5), f"{label}: ckpt"

        # Fuse check (no BN left inside pamr.in_projs!)
        fz = copy.deepcopy(ev).fuse(verbose=False)
        if hasattr(fz.model[-1], "pamr"):
            assert not any(isinstance(k, Conv) and hasattr(k, "bn") for k in fz.model[-1].pamr.modules()), f"{label}: pamr BN unfused"
        assert torch.allclose(fz(x_sq)[0], out_sq, atol=1e-3, rtol=1e-3), f"{label}: fused sq mismatch"
        assert torch.allclose(fz(x_odd)[0], out_odd, atol=1e-3, rtol=1e-3), f"{label}: fused odd mismatch"

        # Native FP16 half+fuse and fuse+half checks on square & odd 224x256 (plus direct module-level FP16 precision!)
        if hasattr(ev.model[-1], "stylenull") or hasattr(ev.model[-1], "pamr"):
            sample_feats = [torch.randn(2, c, h, w) * 0.5 for c, h, w in ((64, 32, 32), (128, 16, 16), (256, 7, 8))]
            det_fp32 = copy.deepcopy(ev.model[-1]).float().eval()
            det_fp16 = copy.deepcopy(ev.model[-1]).half().eval()
            y32 = det_fp32._cls_transform(sample_feats)
            y16 = det_fp16._cls_transform([t.half() for t in sample_feats])
            for a32, a16 in zip(y32, y16):
                assert torch.isfinite(a16).all() and torch.allclose(a16.float(), a32, atol=5e-3, rtol=5e-3), f"{label}: module FP16 drift"
        hf1 = copy.deepcopy(ev).half().fuse(verbose=False)
        o_hf1_sq, r_hf1_sq = hf1(x_sq.half())
        o_hf1_odd = hf1(x_odd.half())[0].float()
        o_hf1_sq = o_hf1_sq.float()
        assert torch.isfinite(o_hf1_sq).all() and torch.isfinite(o_hf1_odd).all(), f"{label}: NaN in half+fuse"
        assert torch.allclose(o_hf1_sq[:, 4:], out_sq[:, 4:], atol=1e-2, rtol=5e-2), f"{label}: half+fuse cls sq mismatch"
        assert torch.allclose(r_hf1_sq["boxes"].float(), raw_sq["one2many"]["boxes"], atol=0.25, rtol=5e-2), f"{label}: half+fuse raw box sq mismatch"
        assert torch.allclose(o_hf1_odd[:, 4:], out_odd[:, 4:], atol=1e-2, rtol=5e-2), f"{label}: half+fuse cls odd mismatch"
        hf2 = copy.deepcopy(ev).fuse(verbose=False).half()
        o_hf2_b1, r_hf2_b1 = hf2(x_sq[:1].half())
        o_hf2_b1 = o_hf2_b1.float()
        assert torch.isfinite(o_hf2_b1).all() and torch.allclose(o_hf2_b1[:, 4:], out_sq[:1, 4:], atol=1e-2, rtol=5e-2), f"{label}: fuse+half b=1 cls"
        assert torch.allclose(r_hf2_b1["boxes"].float(), raw_sq["one2many"]["boxes"][:1], atol=0.2, rtol=5e-2), f"{label}: fuse+half b=1 box"

        # Exact deployed params & GFLOPs@640 after fuse() within <= +10% budget
        eff = efficiency.params_flops(ev, 640)
        dp = 100.0 * (eff["params_M"] - base_eff["params_M"]) / base_eff["params_M"]
        df = 100.0 * (eff["GFLOPs@640"] - base_eff["GFLOPs@640"]) / base_eff["GFLOPs@640"]
        assert dp <= 10.0 and df <= 10.0, f"{label}: exceeds +10% budget ({dp:.2f}%, {df:.2f}%)"
        print(f"[check OK] {label} | {eff} | dParams={dp:+.2f}% dFLOPs={df:+.2f}% | diag={ev.diag_report()} | loss={float(loss.detach().sum()):.3f}")

    # Verify _pool_eval in bench.main produces valid pooled COCO metrics without ID collisions
    from bench.main import _pool_eval
    cats = [{"id": 0, "name": "car"}]
    gt1 = {"images": [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
           "categories": cats}
    gt2 = {"images": [{"id": 1, "width": 100, "height": 100, "file_name": "b.jpg"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 0}],
           "categories": cats}
    d1 = [{"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9}]
    d2 = [{"image_id": 1, "category_id": 0, "bbox": [30, 30, 20, 20], "score": 0.8}]
    mp, mip = _pool_eval([("night", gt1, {}, d1), ("dayrain", gt2, {}, d2)])
    assert abs(mip["mAP50_95"] - 100.0) < 1e-3, f"_pool_eval failed: {mip}"
    print("[check OK] _pool_eval exact disjoint ID merge verified (mAP=100.0)")
    print("ALL ROUND-12 SELF-CHECKS PASSED")
