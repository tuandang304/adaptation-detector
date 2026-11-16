"""Budget-aware continual test-time adaptation (CTTA) for YOLO detectors on a COCO-C cyclic stream.

One run = one stream, many method configs (config.yaml). Every method starts from the same source model,
sees the same frames in the same order, and is scored with the same COCOeval procedure.
"""
import argparse, contextlib, copy, hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

T0 = time.time()


def log(tag, **kw):
    print(f"[{tag}] " + json.dumps(kw, default=lambda o: float(o) if np.isscalar(o) else str(o)), flush=True)


# ----------------------------------------------------------------------------- data
def fetch_coco(root):
    ann = f"{root}/annotations/instances_val2017.json"
    os.makedirs(root, exist_ok=True)
    for url, check in [("http://images.cocodataset.org/zips/val2017.zip", f"{root}/val2017"),
                       ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann)]:
        if os.path.exists(check):
            continue
        z = f"{root}/{os.path.basename(url)}"
        t = time.time()
        subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", z, url], check=True)
        with zipfile.ZipFile(z) as f:
            names = [n for n in f.namelist() if "annotations" not in n or n.endswith("instances_val2017.json")]
            f.extractall(root, members=names)
        os.remove(z)
        log("data", fetched=url, secs=round(time.time() - t))
    return ann


def split_ids(coco, n_source, seed=0):
    ids = sorted(i for i in coco.getImgIds() if coco.getAnnIds(imgIds=i, iscrowd=False))
    perm = np.random.RandomState(seed).permutation(ids).tolist()
    n_val = (len(perm) - n_source) // 2
    sp = {"source": perm[:n_source], "val": perm[n_source:n_source + n_val], "test": perm[n_source + n_val:]}
    log("split", **{k: len(v) for k, v in sp.items()},
        sha1={k: hashlib.sha1(json.dumps(v).encode()).hexdigest()[:12] for k, v in sp.items()})
    return sp


KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]  # DontCare dropped


def kitti_root():
    import glob
    for pat in ("/kaggle/input/*/data_object_label_2", "/kaggle/input/*/*/data_object_label_2", "/kaggle/input/*/*/*/data_object_label_2"):
        hit = glob.glob(pat)
        if hit:
            return os.path.dirname(hit[0])
    if os.environ.get("KITTI_DIR"):
        return os.environ["KITTI_DIR"]
    raise FileNotFoundError("KITTI not mounted: attach Kaggle dataset klemenko/kitti-dataset or set KITTI_DIR")


def kitti_labels(root, i):
    out = []
    for line in open(f"{root}/data_object_label_2/training/label_2/{i}.txt"):
        f = line.split()
        if f[0] in KITTI_CLASSES:
            out.append((KITTI_CLASSES.index(f[0]), *map(float, f[4:8])))  # cls, x1, y1, x2, y2 (pixels)
    return out


def kitti_coco(root, ids):
    """COCO-format ground truth for KITTI images, so KITTI and COCO share one COCOeval procedure."""
    from PIL import Image
    from pycocotools.coco import COCO
    d = {"images": [], "annotations": [], "categories": [{"id": k + 1, "name": n} for k, n in enumerate(KITTI_CLASSES)]}
    for i in ids:
        p = f"{root}/data_object_image_2/training/image_2/{i}.png"
        w, h = Image.open(p).size
        d["images"].append({"id": int(i), "file_name": f"{i}.png", "path": p, "width": w, "height": h})
        for c, x1, y1, x2, y2 in kitti_labels(root, i):
            d["annotations"].append({"id": len(d["annotations"]) + 1, "image_id": int(i), "category_id": c + 1, "iscrowd": 0,
                                     "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)})
    coco = COCO()
    coco.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def finetune_kitti(root, split, ft, base_weights, outdir):
    """Fine-tune COCO-pretrained weights on the KITTI 3DOP train split. Model selection never sees the eval pool:
    Ultralytics validates on `holdout` images carved from *train*, and we keep last.pt."""
    from ultralytics import YOLO
    work = "/tmp/kitti_yolo"
    tr = list(split["train"])
    np.random.RandomState(0).shuffle(tr)
    parts = {"trainval": tr[:ft.get("holdout", 300)], "train": tr[ft.get("holdout", 300):]}
    for name, ids in parts.items():
        os.makedirs(f"{work}/images/{name}", exist_ok=True); os.makedirs(f"{work}/labels/{name}", exist_ok=True)
        for i in ids:
            dst = f"{work}/images/{name}/{i}.png"
            if not os.path.exists(dst):
                os.symlink(f"{root}/data_object_image_2/training/image_2/{i}.png", dst)
            from PIL import Image
            w, h = Image.open(dst).size
            with open(f"{work}/labels/{name}/{i}.txt", "w") as f:
                for c, x1, y1, x2, y2 in kitti_labels(root, i):
                    f.write(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    yaml.safe_dump({"path": work, "train": "images/train", "val": "images/trainval", "names": dict(enumerate(KITTI_CLASSES))},
                   open(f"{work}/kitti.yaml", "w"))
    m = YOLO(base_weights)
    t0 = time.time()
    m.add_callback("on_fit_epoch_end", lambda tr_: log(
        f"epoch {tr_.epoch + 1}/{tr_.epochs}", loss={k.split("/")[-1]: round(float(v), 4) for k, v in tr_.label_loss_items(tr_.tloss).items()},
        map50=round(float(tr_.metrics.get("metrics/mAP50(B)", 0)), 4), map=round(float(tr_.metrics.get("metrics/mAP50-95(B)", 0)), 4),
        lr=round(float(tr_.optimizer.param_groups[0]["lr"]), 6), secs=round(time.time() - t0)))
    args = dict(data=f"{work}/kitti.yaml", epochs=ft.get("epochs", 30), imgsz=ft.get("imgsz", 640), batch=ft.get("batch", 32),
                device=0 if torch.cuda.is_available() else "cpu", workers=4, seed=0, project=f"{outdir}/ft", name="kitti", exist_ok=True, plots=False, verbose=False,
                freeze=ft.get("freeze", None), close_mosaic=ft.get("close_mosaic", 10), patience=0)
    log("finetune", **{k: v for k, v in args.items() if k != "data"}, n_train=len(parts["train"]), n_holdout=len(parts["trainval"]))
    m.train(**args)
    w = f"{outdir}/ft/kitti/weights/last.pt"
    log("finetune_done", weights=w, secs=round(time.time() - t0))
    return w


def finetune_robust(ft, base_weights, outdir):
    """Robust source model: fine-tune on COCO train2017, optionally with photometric AugMix whose ops (posterize, solarize,
    autocontrast, equalize) are disjoint from the test corruptions. augmix=0 is the matched plain fine-tune control."""
    import subprocess
    from torchvision.transforms import v2
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    work = "/tmp/rob/coco"
    src = os.environ.get("COCO_TRAIN_DIR")  # an already-present image folder (Kaggle dataset mount / local smoke)
    if not src:  # a mounted COCO dataset saves the 19 GB download
        src = next((d for d, _, fs in os.walk("/kaggle/input") if os.path.basename(d) == "train2017" and len(fs) > 100000), None)
    log("train_images", src=src or "download")
    split = os.path.basename(src.rstrip("/")) if src else "train2017"
    os.makedirs(f"{work}/images", exist_ok=True)
    if not os.path.isdir(f"{work}/labels/{split}"):
        torch.hub.download_url_to_file("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip", "/tmp/rob/l.zip", progress=False)
        subprocess.run(["unzip", "-q", "-o", "/tmp/rob/l.zip", f"coco/labels/{split}/*", "-d", "/tmp/rob"], check=True)
    if not os.path.exists(f"{work}/images/{split}"):
        if src:
            os.symlink(src, f"{work}/images/{split}")
        else:
            t = time.time()
            subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", "/tmp/rob/t.zip", "http://images.cocodataset.org/zips/train2017.zip"], check=True)
            log("download", what="train2017", secs=round(time.time() - t))
            subprocess.run(["unzip", "-q", "-o", "/tmp/rob/t.zip", "-d", f"{work}/images"], check=True)
            os.remove("/tmp/rob/t.zip")
    m = YOLO(base_weights)
    with open(f"{work}/list.txt", "w") as f:  # a list keeps the symlinked path, so labels resolve under {work}/labels
        f.write("\n".join(f"{work}/images/{split}/{n}" for n in sorted(os.listdir(f"{work}/images/{split}"))))
    yaml.safe_dump({"path": work, "train": "list.txt", "val": "list.txt", "names": m.names}, open("/tmp/rob/data.yaml", "w"))
    p_aug = float(os.environ.get("AUGMIX", ft.get("augmix", 0.0)))

    class PhotoAugMix(v2.AugMix):
        _PARTIAL_AUGMENTATION_SPACE = {k: v2.AugMix._PARTIAL_AUGMENTATION_SPACE[k] for k in ("Posterize", "Solarize", "AutoContrast", "Equalize")}
    aug = PhotoAugMix(severity=ft.get("severity", 3))

    class Trainer(DetectionTrainer):
        def preprocess_batch(self, batch):
            img = batch["img"].to(self.device)  # uint8; chunks of 8 share one sampled op chain
            for k in range(0, len(img), 8):
                if np.random.rand() < p_aug:
                    img[k:k + 8] = aug(img[k:k + 8])
            batch["img"] = img
            return super().preprocess_batch(batch)

    t0 = time.time()
    m.add_callback("on_fit_epoch_end", lambda tr_: log(
        f"epoch {tr_.epoch + 1}/{tr_.epochs}", loss={k.split("/")[-1]: round(float(v), 4) for k, v in tr_.label_loss_items(tr_.tloss).items()},
        lr=round(float(tr_.optimizer.param_groups[0]["lr"]), 6), secs=round(time.time() - t0)))
    smoke = bool(os.environ.get("SMOKE"))
    args = dict(data="/tmp/rob/data.yaml", epochs=1 if smoke else ft.get("epochs", 8), imgsz=ft.get("imgsz", 640), batch=8 if smoke else ft.get("batch", 64),
                fraction=0.004 if smoke else 1.0, device=0 if torch.cuda.is_available() else "cpu", workers=4, seed=0, project=f"{outdir}/ft",
                name="robust", exist_ok=True, plots=False, verbose=False, val=False, patience=0, optimizer="SGD", lr0=ft.get("lr0", 0.002),
                lrf=0.1, warmup_epochs=0.5, close_mosaic=ft.get("close_mosaic", 2))
    log("finetune", **{k: v for k, v in args.items() if k != "data"}, augmix=p_aug, ops=list(PhotoAugMix._PARTIAL_AUGMENTATION_SPACE))
    m.train(trainer=Trainer, **args)
    w = f"{outdir}/robust_augmix{p_aug:g}.pt"
    shutil.copy(f"{outdir}/ft/robust/weights/last.pt", w)
    log("finetune_done", weights=w, secs=round(time.time() - t0))
    return w


def _patch_imagecorruptions():
    np.float_ = np.float64  # imagecorruptions 1.1.2 predates NumPy 2
    import imagecorruptions.corruptions as C
    from skimage.filters import gaussian
    C.gaussian = lambda img, sigma, multichannel=None, **kw: gaussian(img, sigma=sigma, channel_axis=-1 if multichannel else None, **kw)
    return C


def letterbox(img, size):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    dh, dw = (size - nh) // 2, (size - nw) // 2
    out[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (r, dw, dh)


def _make_frame(args):
    path, corruption, severity, seed, size = args
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if corruption != "clean":
        from imagecorruptions import corrupt
        np.random.seed(seed)
        img = np.ascontiguousarray(corrupt(img, corruption_name=corruption, severity=severity))
    return letterbox(img, size)


def _init_worker():
    _patch_imagecorruptions()
    cv2.setNumThreads(1)


def build_frames(coco, root, ids_by_phase, phases, severity, size, seed):
    jobs, meta = [], []
    for p, (name, ids) in enumerate(zip(phases, ids_by_phase)):
        for i in ids:
            jobs.append((coco.imgs[i]["path"], name, severity, seed * 100000 + len(jobs), size))
            meta.append((p, i))
    imgs, lb = np.empty((len(jobs), size, size, 3), np.uint8), []
    t, last = time.time(), time.time()
    with Pool(os.cpu_count(), initializer=_init_worker) as pool:
        for k, (im, l) in enumerate(pool.imap(_make_frame, jobs, chunksize=8)):
            imgs[k] = im
            lb.append(l)
            if time.time() - last > 120:
                last = time.time()
                log("frames", done=k + 1, total=len(jobs), eta_s=round((time.time() - t) / (k + 1) * (len(jobs) - k - 1)))
    log("frames", done=len(jobs), secs=round(time.time() - t))
    return imgs, lb, meta


# ----------------------------------------------------------------------------- model
class AdaBN(nn.Module):
    """Drop-in BatchNorm2d: normalises with running stats (eval semantics) that may be EMA-updated from test batches."""

    def __init__(self, bn):
        super().__init__()
        self.weight, self.bias, self.eps = bn.weight, bn.bias, bn.eps
        self.register_buffer("running_mean", bn.running_mean.clone())
        self.register_buffer("running_var", bn.running_var.clone())
        self.register_buffer("src_mean", bn.running_mean.clone())
        self.register_buffer("src_var", bn.running_var.clone())
        self.momentum, self.anchor, self.update, self.use_src = 0.0, 0.0, False, False
        self.mode, self.prior, self.gamma, self.beta = "ema", 0.8, 10.0, 0.0  # baseline re-implementations, see forward

    def forward(self, x):
        if self.use_src:  # routed to the untouched source statistics for this batch
            return F.batch_norm(x, self.src_mean, self.src_var, self.weight, self.bias, False, 0.0, self.eps)
        if self.mode != "ema" and self.update:  # forward-only baselines that need this batch's statistics
            with torch.no_grad():
                bv, bm = torch.var_mean(x.detach(), dim=(0, 2, 3), correction=0)
                skl = lambda m1, v1, m2, v2: float(0.5 * (kl(m1, v1, m2, v2) + kl(m2, v2, m1, v1)))
                if self.mode == "norm":      # Schneider et al. 2020: source prior N vs test batch n -> fixed source weight
                    w = self.prior
                elif self.mode == "artta":   # AR-TTA-style: source/batch mixing weight from their symmetric KL, EMA-smoothed
                    self.beta = 0.95 * self.beta + 0.05 * (1 - float(np.exp(-self.gamma * skl(self.src_mean, self.src_var, bm, bv))))
                    w = 1 - self.beta
                else:                        # MECTA-norm-style: forget gate from KL(running, batch) on the running stats
                    g = 1 - float(np.exp(-skl(self.running_mean, self.running_var, bm, bv)))
                    self.running_mean.mul_(1 - g).add_(bm, alpha=g); self.running_var.mul_(1 - g).add_(bv, alpha=g)
                    return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
                mean, var = w * self.src_mean + (1 - w) * bm, w * self.src_var + (1 - w) * bv
            return F.batch_norm(x, mean, var, self.weight, self.bias, False, 0.0, self.eps)
        if self.update and self.momentum > 0:
            with torch.no_grad():
                m, a = self.momentum, self.anchor
                # fused cuDNN pass that only updates the running buffers (its normalised output is discarded)
                F.batch_norm(x.detach(), self.running_mean, self.running_var, None, None, True, m, self.eps)
                if a > 0:
                    self.running_mean.mul_(1 - a).add_(self.src_mean, alpha=a)
                    self.running_var.mul_(1 - a).add_(self.src_var, alpha=a)
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)


def swap_bn(module, layer_idx=None, out=None):
    out = [] if out is None else out
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new = AdaBN(child)
            new.layer = layer_idx
            setattr(module, name, new)
            out.append(new)
        else:
            swap_bn(child, layer_idx, out)
    return out


def recalibrate_bn(det, paths, size, device, bs=16):
    """Source BN recalibration (no labels, no gradients): replace every BN's running stats by the cumulative average of
    batch stats over clean source images at deployment preprocessing. Fixes stats inherited from mosaic training."""
    bns = [m for m in det.modules() if isinstance(m, AdaBN)]
    with Pool(os.cpu_count(), initializer=_init_worker) as pool:
        imgs = np.stack([im for im, _ in pool.imap(_make_frame, [(p, "clean", 1, 0, size) for p in paths], chunksize=8)])
    with torch.no_grad():
        for k in range(0, len(imgs), bs):
            for b in bns:
                b.update, b.momentum, b.anchor, b.use_src = True, 1.0 / (k // bs + 1), 0.0, False
            det(torch.from_numpy(imgs[k:k + bs]).to(device).permute(0, 3, 1, 2).float().div_(255))
    for b in bns:
        b.update, b.momentum = False, 0.0
        b.src_mean.copy_(b.running_mean); b.src_var.copy_(b.running_var)
    log("recal_bn", images=len(imgs), batch=bs)


def load_model(weights, device):
    from ultralytics import YOLO
    det = YOLO(weights).model.float()
    bns = []
    for i, layer in enumerate(det.model):
        bns += swap_bn(layer, i)
    for p in det.parameters():
        p.requires_grad_(False)
    det.to(device).eval()
    return det, bns


class Taps:
    """Per-channel mean/var of chosen submodule outputs (paths like "4" or "0.conv"), captured on every forward."""

    def __init__(self, det, paths):
        self.out, self.handles = {}, []
        for p in paths:
            m = det.model.get_submodule(str(p))
            self.handles.append(m.register_forward_hook(lambda m, inp, o, p=str(p): self.out.__setitem__(p, o)))

    def stats(self, keys=None):
        return {k: torch.var_mean(o, dim=(0, 2, 3), correction=0)[::-1] for k, o in self.out.items() if keys is None or k in keys}

    def remove(self):
        for h in self.handles:
            h.remove()


def kl(mt, vt, ms, vs, eps=1e-4):
    vt, vs = vt + eps, vs + eps
    return 0.5 * (torch.log(vs / vt) + (vt + (mt - ms) ** 2) / vs - 1).mean()


def divergence(stats, src):
    """Symmetric Gaussian KL between batch feature stats and source stats, averaged over taps."""
    return float(sum(0.5 * (kl(m, v, *src[i]) + kl(*src[i], m, v)) for i, (m, v) in stats.items()) / len(stats))


# ----------------------------------------------------------------------------- energy
class Energy:
    """Cumulative GPU energy (J) from the NVML hardware counter; time-proportional proxy when no NVML (CPU smoke)."""

    def __init__(self, device):
        self.h = None
        if device.type == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                self.h = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
                self.nv = pynvml
                self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)
            except Exception as e:  # noqa: BLE001 - report and fall back, never silently
                log("warn", energy="nvml unavailable, using time proxy", error=repr(e))
                self.h = None
        self.source = "nvml" if self.h else "proxy_70W"

    def joules(self):
        if self.h:
            return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0
        return time.perf_counter() * 70.0


# ----------------------------------------------------------------------------- evaluation
def evaluate(coco, dets_by_phase, ids_by_phase):
    from pycocotools.cocoeval import COCOeval
    res = []
    for p, (dets, ids) in enumerate(zip(dets_by_phase, ids_by_phase)):
        if not dets:
            res.append((0.0, 0.0))
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(coco, coco.loadRes(dets), "bbox")
            E.params.imgIds = list(ids)
            E.evaluate(); E.accumulate(); E.summarize()
        res.append((float(E.stats[0]), float(E.stats[1])))
    return res


# ----------------------------------------------------------------------------- one method on the stream
def snapshot(params):
    return [p.detach().clone() for p in params]


def restore_source(params, src_params, bns, opt):
    with torch.no_grad():
        for p, s0 in zip(params, src_params):
            p.copy_(s0)
        for b in bns:
            b.running_mean.copy_(b.src_mean); b.running_var.copy_(b.src_var)
    if opt is not None:
        opt.state.clear()


def calibrate_energy(det0, mcfg, cfg, device, n=30):
    """Measured per-batch energy of a skip (forward only) vs an adapt batch (forward+backward+step) on a throwaway copy."""
    det = copy.deepcopy(det0)
    bns = [m for m in det.modules() if isinstance(m, AdaBN)]
    sel = bns if mcfg.get("params", "all") == "all" else [b for b in bns if b.layer < mcfg.get("stem_end", 5)]
    params = [p for b in sel for p in (b.weight, b.bias)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=1e-12)
    taps = Taps(det, [str(k) for k in cfg["taps"]])
    x = torch.rand(cfg["batch"], 3, cfg["imgsz"], cfg["imgsz"], device=device)
    en, out = Energy(device), {}
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    reads = set()
    for mode in ("skip", "adapt", "skip", "adapt"):  # second pass is the one kept (first warms clocks)
        sync(); e0 = en.joules()
        for _ in range(n):
            reads.add(en.joules())
            with torch.set_grad_enabled(mode == "adapt"):
                det(x)
                if mode == "adapt":
                    loss = sum(v.mean() for _, v in taps.stats().values())
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(); out[mode] = (en.joules() - e0) / n
    taps.remove()
    del det
    log("calib", skip_j=round(out["skip"], 4), adapt_j=round(out["adapt"], 4), distinct_counter_reads=len(reads), of=4 * n)
    return out["skip"], max(out["adapt"] - out["skip"], 1e-6)


def calibrate_bn_energy(det0, mcfg, cfg, device, n=100, reps=3):
    """Measured per-batch extra energy of a forward with BN-EMA updates in every layer vs a plain forward."""
    det = copy.deepcopy(det0)
    bns = [m for m in det.modules() if isinstance(m, AdaBN)]
    x = torch.rand(cfg["batch"], 3, cfg["imgsz"], cfg["imgsz"], device=device)
    en, out = Energy(device), {}
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    diffs = []
    with torch.no_grad():
        for r in range(reps + 1):  # rep 0 warms clocks and is discarded
            for mode in ("skip", "upd"):
                for b in bns:
                    b.momentum, b.update = 0.05, mode == "upd"
                sync(); e0 = en.joules()
                for _ in range(n):
                    det(x)
                sync(); out[mode] = (en.joules() - e0) / n
            if r:
                diffs.append(out["upd"] - out["skip"])
    del det
    log("calib_bn", extra_j_reps=[round(d_, 4) for d_ in diffs], skip_j=round(out["skip"], 4))
    return max(float(np.median(diffs)), 1e-6)


def run_method(mcfg, cfg, det0, src_stats, src_div, frames, coco, ids_by_phase, device, outdir, cat_map, replay=None):
    """replay=None: accuracy pass (controller decides, eval-grade NMS, COCOeval).
    replay=trace: cost pass that re-executes the recorded decisions with deployment post-processing (conf 0.25 NMS)
    so latency/energy reflect the adaptation pipeline rather than the evaluation harness."""
    from ultralytics.utils.nms import non_max_suppression
    name = mcfg["name"]
    det = copy.deepcopy(det0)
    bns = [m for m in det.modules() if isinstance(m, AdaBN)]
    align_keys = [str(k) for k in cfg["taps"]]
    sig_keys = [str(k) for k in cfg.get("signal_taps", cfg["taps"])]
    taps = Taps(det, sorted(set(align_keys + sig_keys)))
    stem_end = mcfg.get("stem_end", 5)
    adapt_bns = bns if mcfg.get("params", "all") == "all" else [b for b in bns if b.layer < stem_end]
    mom = mcfg.get("bn_momentum", 0.0)
    for b in bns:
        b.momentum, b.anchor = mom, mcfg.get("bn_anchor", 0.0)
        b.mode, b.prior, b.gamma, b.beta = mcfg.get("bn_mode", "ema"), mcfg.get("bn_prior", 0.8), mcfg.get("bn_gamma", 10.0), 0.0
    mom_decay, mom_min = mcfg.get("bn_mom_decay", 0.0), mcfg.get("bn_mom_min", 0.005)   # DUA momentum schedule
    reset_every = mcfg.get("reset_every", 0)     # RDumb: restore source every K batches
    oracle = mcfg.get("oracle", False)           # upper bound: knows phase boundaries and which phases are clean
    since_reset, zhist = 0, []
    if mcfg.get("bn_mode", "ema") != "ema":
        mom = 1.0  # these modes update on every batch they are allowed to; `mom` only gates the update flag
    grad = mcfg.get("adapt", "none") == "grad"
    params = [p for b in adapt_bns for p in (b.weight, b.bias)] if grad else []
    for p in params:
        p.requires_grad_(True)
    src_params = snapshot(params)
    opt = torch.optim.Adam(params, lr=mcfg.get("lr", 1e-4)) if grad else None
    w_align, w_ent = mcfg.get("loss", {}).get("align", 1.0), mcfg.get("loss", {}).get("ent", 0.0)
    alpha = mcfg.get("ema_alpha", 0.1)
    sched = mcfg.get("schedule", "always")       # always | periodic | trigger | budget
    K = mcfg.get("period", 10)
    bn_gate = mcfg.get("bn_gate", "always")      # always | ood : BN-EMA only while the signal says out-of-distribution
    tau_hi = src_div["q99"] * mcfg.get("thr_mult", 1.0)
    tau_lo = src_div["q90"] * mcfg.get("thr_mult", 1.0)
    cp_delta = mcfg.get("cp_delta", 0.0)         # |log(fast/slow divergence EMA)| that declares a new domain
    cp_burst = mcfg.get("cp_burst", 0)           # batches of adaptation granted after a detected shift
    cp_cool = mcfg.get("cp_cool", 15)            # batches before another shift can be declared
    reset_on_shift = mcfg.get("reset_on_shift", False)
    steady = mcfg.get("steady", True)            # adapt while OOD even outside bursts (budget permitting)
    reset_after = mcfg.get("reset_after", 0)     # consecutive in-domain batches before restoring source state
    rho = mcfg.get("budget_frac", 0.2)           # energy budget per batch, as a fraction of one adapt step's measured extra energy
    depth = mcfg.get("burst", 8)                 # bucket depth, in adapt steps
    route = mcfg.get("route", False)             # zero-lag per-batch routing: source vs adapted BN stats
    tau_route = src_div.get("rq99", src_div["q99"]) * mcfg.get("route_mult", 1.0)
    tau_low = src_div.get(mcfg.get("route_low_q", "rq90"), 0.0) if mcfg.get("route_hyst") else float("inf")
    cp_mode = mcfg.get("cp_mode", "ratio")       # ratio (R2) | dom: direction vs current-domain reference
    cp_k, cp_z, cp_persist, cp_mom = mcfg.get("cp_k", 1.0), mcfg.get("cp_z", 3.0), mcfg.get("cp_persist", 2), mcfg.get("cp_mom", 0.1)
    bn_budget = mcfg.get("bn_budget")            # None: update whenever routed adapted; else J budget per batch as a fraction of one update
    dense = mcfg.get("dense", 0)                 # batches of unconditional BN updates right after a detected shift

    imgs, lbs, meta = frames
    B = cfg["batch"]
    n_batches = (len(imgs) + B - 1) // B
    ema_stats, d_fast, d_slow = None, None, None
    dets_by_phase = [[] for _ in ids_by_phase]
    trace = {k: np.zeros(n_batches, np.float32) for k in ("div", "adapt", "want", "event", "bn_upd", "restore", "lat_ms", "energy_j")}
    need_sig = (cp_mode == "zpeak" or sched in ("trigger", "budget") or bn_gate == "ood" or bool(reset_after) or bool(cp_delta) or cp_mode == "dom") and not route
    need_stats = need_sig or (grad and bool(w_align))
    d = 0.0
    calm, burst_left, cool = 0, 0, 0
    budget = {}
    state = {"allow": True, "d": 0.0, "adapted": False, "upd": False, "stats": None}
    sig_key = sig_keys[0]
    if route:
        def _route(m_, i_, o):
            v_, m2 = torch.var_mean(o.detach(), dim=(0, 2, 3), correction=0)
            state["stats"] = (m2, v_)
            state["d"] = divergence({sig_key: (m2, v_)}, src_stats)
            state["adapted"] = state["d"] > tau_route or (state.get("domain_ood", False) and state["d"] > tau_low)
            state["upd"] = state["adapted"] and state["allow"] and mom > 0
            for b_ in bns:
                b_.use_src, b_.update = not state["adapted"], state["upd"]
        route_handle = det.model.get_submodule(sig_key).register_forward_hook(_route)
    route2_handle = None
    if route and mcfg.get("route2"):  # deeper tap for shifts invisible at conv0 (jpeg, pixelate); only valid while the stem runs on source stats
        k2, sd2 = str(cfg["signal2"]), src_div["tap2"]
        L2 = int(k2.split(".")[0])

        def _route2(m_, i_, o):
            if state["adapted"]:
                return
            v_, m2 = torch.var_mean(o.detach(), dim=(0, 2, 3), correction=0)
            d2 = divergence({k2: (m2, v_)}, src_stats)
            state["deep"] = d2 > sd2["rq99"] or (state.get("deep", False) and d2 > sd2["rq90"])
            if state["deep"]:
                state["upd"] = state["allow"] and mom > 0
                for b_ in bns:
                    if b_.layer >= L2:
                        b_.use_src, b_.update = False, state["upd"]
        route2_handle = det.model.get_submodule(k2).register_forward_hook(_route2)
    cp_ref, cp_noise, cp_run, cp_buf = src_stats[sig_key], None, 0, []
    if bn_budget is not None and replay is None:
        bn_cost = calibrate_bn_energy(det0, mcfg, cfg, device)
        bn_bucket, bn_cap, bn_refill = depth * bn_cost, depth * bn_cost, bn_budget * bn_cost
        budget.update({"cal_bn_extra_j": bn_cost, "bn_budget_j_per_batch": bn_refill})
    if sched == "budget" and replay is None:
        e_skip, e_extra = calibrate_energy(det0, mcfg, cfg, device)
        bucket, cap, refill = depth * e_extra, depth * e_extra, rho * e_extra
        base_est, extra_est = e_skip, e_extra
        win_e, win_n, win_debit, win_adapts = 0.0, 0, 0.0, 0
        budget = {"cal_skip_j": e_skip, "cal_extra_j": e_extra, "budget_j_per_batch": refill}
    energy = Energy(device)

    for i in range(5):  # warm-up (incl. backward for grad methods) so autotune/allocator do not land mid-stream
        with torch.set_grad_enabled(grad and i >= 3):
            det(torch.zeros(B, 3, cfg["imgsz"], cfg["imgsz"], device=device))
            if grad and i >= 3:
                opt.zero_grad(set_to_none=True)
                sum(v.mean() for _, v in taps.stats().values()).backward()
    restore_source(params, src_params, bns, opt)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()
    e_start, t_start = energy.joules(), time.time()
    e_prev = e_start
    for t in range(n_batches):
        sl = slice(t * B, min((t + 1) * B, len(imgs)))
        t0 = time.perf_counter()
        x = torch.from_numpy(imgs[sl]).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)

        # ---- decide from previous batches' signal only
        ood = d_fast is not None and d_fast > tau_hi
        want = (burst_left > 0 or (steady and ood)) if cp_burst else ood
        if sched == "always":
            do = grad
        elif sched == "periodic":
            do = grad and t % K == 0
        elif sched == "trigger":
            do = grad and want
        elif sched == "budget":
            do = grad and want and (replay is not None or bucket >= extra_est)
        else:
            raise ValueError(sched)
        upd = mom > 0 and (bn_gate == "always" or ood)
        if mom_decay:
            for b in bns:
                b.momentum = max(mom * mom_decay ** since_reset, mom_min)
        if oracle:
            ph = meta[sl.start][0]
            if t and meta[sl.start - 1][0] != ph:
                restore_source(params, src_params, bns, opt)
            upd = cfg["stream"]["phases"][ph] != "clean"
            for b in bns:
                b.use_src = not upd
        if replay is not None:
            do, upd = bool(replay["adapt"][t]), bool(replay["bn_upd"][t])
        if route:  # the hook decides routing/update inside the forward; here only whether an update is affordable
            state["allow"] = upd if replay is not None else (bn_budget is None or burst_left > 0 or bn_bucket >= bn_cost)
        else:
            for b in bns:
                b.update = upd

        with torch.set_grad_enabled(do):
            y, raw = det(x)
            st = taps.stats() if need_stats else {}
            if route:
                d, upd = state["d"], state["upd"]
            elif need_sig:
                d = divergence({k: (m.detach(), v.detach()) for k, (m, v) in st.items() if k in sig_keys}, src_stats)
            ast = {k: st[k] for k in align_keys} if st else {}
            if do:
                loss = 0.0
                if w_align:
                    if ema_stats is None:
                        ema_stats = {k: (m.detach(), v.detach()) for k, (m, v) in ast.items()}
                    cur = {k: ((1 - alpha) * ema_stats[k][0] + alpha * m, (1 - alpha) * ema_stats[k][1] + alpha * v)
                           for k, (m, v) in ast.items()}
                    loss = loss + w_align * sum(kl(m, v, *src_stats[k]) for k, (m, v) in cur.items()) / len(cur)
                    ema_stats = {k: (m.detach(), v.detach()) for k, (m, v) in cur.items()}
                if w_ent:
                    s = torch.cat([z.flatten(2) for z in raw["scores"]], 2) if isinstance(raw["scores"], list) else raw["scores"]
                    p = s.sigmoid().amax(1)  # [B, anchors] max class prob
                    pe = p[p > mcfg.get("ent_conf", 0.3)].clamp(1e-6, 1 - 1e-6)
                    if pe.numel():
                        loss = loss + w_ent * -(pe * pe.log() + (1 - pe) * (1 - pe).log()).mean()
                if torch.is_tensor(loss):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            elif ema_stats is not None and ast:  # keep the test-stat EMA current even when not adapting
                ema_stats = {k: ((1 - alpha) * ema_stats[k][0] + alpha * m.detach(), (1 - alpha) * ema_stats[k][1] + alpha * v.detach())
                             for k, (m, v) in ast.items()}

        # ---- signal bookkeeping: fast/slow EMAs, change-point events, clean-return restoration
        d_fast = d if d_fast is None else 0.7 * d_fast + 0.3 * d
        d_slow = d if d_slow is None else 0.95 * d_slow + 0.05 * d
        event, restored = False, False
        burst_left, cool = max(burst_left - 1, 0), max(cool - 1, 0)
        if replay is not None:
            if replay["restore"][t]:
                restore_source(params, src_params, bns, opt)
                ema_stats = None
        elif cp_mode == "zpeak":  # ERSK-style: z-score peak of the KL signal over a sliding window -> reset to source
            if len(zhist) >= 10 and (d - np.mean(zhist)) / (np.std(zhist) + 1e-8) > cp_z and cool == 0:
                event, cool, zhist = True, cp_cool, []
                restore_source(params, src_params, bns, opt)
                ema_stats, restored = None, True
            else:
                zhist = (zhist + [d])[-30:]
        elif cp_mode == "dom":
            cs = state["stats"] if route else tuple(x_.detach() for x_ in st[sig_key])
            dd = 0.5 * (float(kl(*cs, *cp_ref)) + float(kl(*cp_ref, *cs)))
            thr = max(cp_k * src_div["rq99"], cp_z * cp_noise) if cp_noise is not None else cp_k * src_div["rq99"]
            cp_run = cp_run + 1 if dd > thr else 0
            cp_buf = (cp_buf + [cs])[-cp_persist:]
            if cp_run >= cp_persist and cool == 0:
                event, burst_left, cool, cp_run, cp_noise = True, max(cp_burst, dense), cp_cool, 0, None
                cp_ref = (torch.stack([a_ for a_, _ in cp_buf]).mean(0), torch.stack([b_ for _, b_ in cp_buf]).mean(0))
                if reset_on_shift:
                    restore_source(params, src_params, bns, opt)
                    ema_stats, restored = None, True
            elif cp_run == 0:
                cp_ref = ((1 - cp_mom) * cp_ref[0] + cp_mom * cs[0], (1 - cp_mom) * cp_ref[1] + cp_mom * cs[1])
                cp_noise = dd if cp_noise is None else 0.9 * cp_noise + 0.1 * dd
            # domain state for routing hysteresis: is the current domain (its running reference) out-of-distribution?
            state["domain_ood"] = 0.5 * (float(kl(*cp_ref, *src_stats[sig_key])) + float(kl(*src_stats[sig_key], *cp_ref))) > tau_route
        elif cp_delta and t >= 3 and cool == 0 and abs(np.log((d_fast + 1e-8) / (d_slow + 1e-8))) > cp_delta:
            event, burst_left, cool, d_slow = True, cp_burst, cp_cool, d_fast
            if reset_on_shift:
                restore_source(params, src_params, bns, opt)
                ema_stats, restored = None, True
        if reset_every and replay is None and (t + 1) % reset_every == 0:
            restore_source(params, src_params, bns, opt)
            restored = True
        since_reset = 0 if (restored or (replay is not None and replay["restore"][t])) else since_reset + 1
        if reset_after and replay is None:
            calm = calm + 1 if d_fast < tau_lo else 0
            if calm == reset_after:
                restore_source(params, src_params, bns, opt)
                ema_stats, restored = None, True

        if replay is not None:  # deployment post-processing; boxes handed to the consumer on CPU
            out = [o.cpu() for o in non_max_suppression(y.detach(), conf_thres=0.25, iou_thres=0.7, max_det=100)]
            out = []
        else:
            out = non_max_suppression(y.detach(), conf_thres=0.001, iou_thres=0.7, max_det=100)
        for k, o in enumerate(out):
            fi = sl.start + k
            p, img_id = meta[fi]
            r, dw, dh = lbs[fi]
            for x1, y1, x2, y2, sc, c in o.cpu().numpy():
                x1, x2 = (x1 - dw) / r, (x2 - dw) / r
                y1, y2 = (y1 - dh) / r, (y2 - dh) / r
                dets_by_phase[p].append({"image_id": int(img_id), "category_id": cat_map[int(c)],
                                         "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                         "score": round(float(sc), 5)})
        sync()
        lat = (time.perf_counter() - t0) * 1000
        e_now = energy.joules()
        e_batch, e_prev = e_now - e_prev, e_now

        # ---- energy token bucket in measured Joules: debit the expected cost now, then every 25 batches reconcile
        # against the cumulative NVML counter (per-batch reads are coarse; windowed sums are not)
        if bn_budget is not None and replay is None:
            if upd and burst_left == 0:  # dense updates right after a detected shift are not charged
                bn_bucket -= bn_cost
            bn_bucket = min(bn_bucket + bn_refill, bn_cap)
        if sched == "budget" and replay is None:
            win_e, win_n = win_e + e_batch, win_n + 1
            if do:
                bucket -= extra_est
                win_debit, win_adapts = win_debit + extra_est, win_adapts + 1
            if win_n == 25:
                if win_adapts == 0:
                    base_est = 0.8 * base_est + 0.2 * win_e / win_n
                else:
                    actual = win_e - win_n * base_est
                    bucket -= actual - win_debit
                    extra_est = 0.8 * extra_est + 0.2 * max(actual / win_adapts, 0.1 * e_extra)
                win_e, win_n, win_debit, win_adapts = 0.0, 0, 0.0, 0
            bucket = min(bucket + refill, cap)

        for k_, v_ in (("div", d), ("adapt", do), ("want", want), ("event", event), ("bn_upd", upd), ("restore", restored), ("lat_ms", lat), ("energy_j", e_batch)):
            trace[k_][t] = float(v_)
        if not np.isfinite(d):
            log("warn", method=name, batch=t, issue="non-finite divergence")
    sync()
    e_total, wall = energy.joules() - e_start, time.time() - t_start
    taps.remove()
    if route:
        route_handle.remove()
        if route2_handle:
            route2_handle.remove()
    for p in params:
        p.requires_grad_(False)
    lat = trace["lat_ms"][5:]
    cost = {"j_per_frame": e_total / len(imgs), "energy_source": energy.source, "fps": len(imgs) / wall,
            "lat_p50": float(np.percentile(lat, 50)), "lat_p95": float(np.percentile(lat, 95)), "lat_p99": float(np.percentile(lat, 99))}
    if replay is not None:
        np.savez_compressed(f"{outdir}/costtrace_{name}.npz", lat_ms=trace["lat_ms"], energy_j=trace["energy_j"])
        return cost

    ev = evaluate(coco, dets_by_phase, ids_by_phase)
    phases = cfg["stream"]["phases"]
    shift = [i for i, n in enumerate(phases) if n != "clean"]
    res = {"method": name, "map50_shift": np.mean([ev[i][1] for i in shift]), "map_shift": np.mean([ev[i][0] for i in shift]),
           "map50_clean_first": ev[0][1], "map50_clean_last": ev[-1][1],
           "map50_clean_mean": np.mean([ev[i][1] for i, n in enumerate(phases) if n == "clean"]),
           "map50_by_phase": {f"{i}:{n}": round(ev[i][1], 4) for i, n in enumerate(phases)},
           "adapt_steps": int(trace["adapt"].sum()), "adapt_frac": float(trace["adapt"].mean()), "bn_upd_frac": float(trace["bn_upd"].mean()),
           "eval_pass": {k: round(v, 3) if isinstance(v, float) else v for k, v in cost.items()}, "secs": round(wall, 1), **budget}
    if cp_delta or cp_mode in ("dom", "zpeak"):
        res.update(trigger_metrics(trace, meta, B, phases, cfg.get("trigger_window", 15)))
    np.savez_compressed(f"{outdir}/trace_{name}.npz", **trace)
    with open(f"{outdir}/dets_{name}.json", "w") as f:
        json.dump(dets_by_phase, f)
    res["_trace"] = trace
    return res


def trigger_metrics(trace, meta, B, phases, W):
    """Change-point reliability (EmbodiTTA-style counts, per batch): every phase boundary, incl. the return to clean,
    is a domain change; a declared event within W batches after it is a hit, any other event is a false trigger."""
    phase_of_batch = np.array([meta[min(t * B, len(meta) - 1)][0] for t in range(len(trace["event"]))])
    bounds = [int(np.argmax(phase_of_batch == p)) for p in range(1, len(phases))]
    onsets = [int(t) for t in np.flatnonzero(trace["event"] > 0)]
    hit, delays = 0, []
    for b in bounds:
        o = [t for t in onsets if b <= t <= b + W]
        if o:
            hit += 1
            delays.append(o[0] - b)
    false = sum(1 for t in onsets if not any(b <= t <= b + W for b in bounds))
    return {"shifts_detected": hit, "shifts_total": len(bounds), "false_triggers": false,
            "delay_batches_mean": float(np.mean(delays)) if delays else None,
            "adapt_in_clean0": int((trace["adapt"] > 0)[phase_of_batch == 0].sum())}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if os.environ.get("STREAM"):  # shard-level stream choice from cfg["streams"]
        cfg["stream"].update(cfg["streams"][os.environ["STREAM"]])
        cfg["stream_name"] = os.environ["STREAM"]
    if os.environ.get("BATCH"):
        cfg["batch"] = int(os.environ["BATCH"])
    if os.environ.get("RECAL_BN") is not None:  # shard-level override so one commit can run with/without recalibration
        cfg["recal_bn"] = int(os.environ["RECAL_BN"])
    if os.environ.get("SMOKE"):
        cfg["stream"]["per_phase"], cfg["source_images"] = int(os.environ["SMOKE"]), 16
    outdir = os.environ.get("ORX_OUTPUT_DIR", "out")
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    import ultralytics
    log("env", commit=os.environ.get("ORX_COMMIT", "local"), python=sys.version.split()[0], torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        n_gpu=torch.cuda.device_count(), ultralytics=ultralytics.__version__, numpy=np.__version__)
    log("config", **cfg)

    from pycocotools.coco import COCO
    from ultralytics.data.converter import coco80_to_coco91_class
    weights = cfg["model"]
    if cfg.get("dataset", "coco") == "kitti":
        root = kitti_root()
        split = json.load(open("kitti_split.json"))
        split = {k: [i for i in split[k] if os.path.exists(f"{root}/data_object_image_2/training/image_2/{i}.png")] for k in ("train", "val")}
        log("kitti", root=root, train=len(split["train"]), val=len(split["val"]))
        coco = kitti_coco(root, split["val"])
        cat_map = list(range(1, len(KITTI_CLASSES) + 1))
        if cfg.get("finetune") and not os.path.exists(cfg.get("weights", "")):
            weights = finetune_kitti(root, split, cfg["finetune"], cfg["model"], outdir)
        elif cfg.get("weights"):
            weights = cfg["weights"]
    else:
        root = os.environ.get("DATA_DIR", "/tmp/coco")
        ann = fetch_coco(root)
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(ann)
        for im in coco.imgs.values():
            im["path"] = f"{root}/val2017/{im['file_name']}"
        cat_map = coco80_to_coco91_class()
        if os.environ.get("WEIGHTS") or cfg.get("weights"):
            weights = os.environ.get("WEIGHTS") or cfg["weights"]
        elif cfg.get("robust_ft") and os.environ.get("ROBUST_FT"):
            weights = finetune_robust(cfg["robust_ft"], cfg["model"], outdir)
    sp = split_ids(coco, cfg["source_images"], seed=0)
    st = cfg["stream"]
    rng = np.random.RandomState(cfg["seed"])
    pool = sp[st["split"]]
    ids_by_phase = [rng.choice(pool, st["per_phase"], replace=False).tolist() for _ in st["phases"]]
    if st.get("same_clean_ids"):
        first = ids_by_phase[st["phases"].index("clean")]
        ids_by_phase = [first if n == "clean" else ids for n, ids in zip(st["phases"], ids_by_phase)]
    frames = build_frames(coco, root, ids_by_phase, st["phases"], st["severity"], cfg["imgsz"], cfg["seed"])

    det0, bns = load_model(weights, device)
    if cfg.get("recal_bn"):
        if cfg.get("dataset", "coco") == "kitti":
            pick = np.random.RandomState(0).choice(split["train"], cfg["recal_bn"], replace=False)
            paths = [f"{root}/data_object_image_2/training/image_2/{i}.png" for i in pick]
        else:
            paths = [coco.imgs[i]["path"] for i in sp["source"][:cfg["recal_bn"]]]
        recalibrate_bn(det0, paths, cfg["imgsz"], device)
    log("model", weights=weights, params=sum(p.numel() for p in det0.parameters()), bn_layers=len(bns),
        bn_affine=sum(b.weight.numel() * 2 for b in bns), stem_bn_affine=sum(b.weight.numel() * 2 for b in bns if b.layer < 5))

    # source feature statistics + per-batch divergence distribution on the clean source split
    src_imgs, _, _ = build_frames(coco, root, [sp["source"]], ["clean"], 1, cfg["imgsz"], cfg["seed"])
    align_keys = [str(k) for k in cfg["taps"]]
    sig_keys = [str(k) for k in cfg.get("signal_taps", cfg["taps"])]
    taps = Taps(det0, sorted(set(align_keys + sig_keys + ([str(cfg["signal2"])] if cfg.get("signal2") else []))))
    B = cfg["batch"]
    per_img = {}  # key -> list of (per-channel mean, per-channel E[x^2]) per source image
    with torch.no_grad():
        for k in range(0, len(src_imgs), B):
            det0(torch.from_numpy(src_imgs[k:k + B]).to(device).permute(0, 3, 1, 2).float().div_(255))
            for key, o in taps.out.items():
                per_img.setdefault(key, []).append(torch.stack([o.mean((2, 3)), (o ** 2).mean((2, 3))], 1).double().cpu())
    taps.remove()
    per_img = {k: torch.cat(v) for k, v in per_img.items()}  # [N, 2, C]
    g = np.random.RandomState(1)

    def batch_stats(idx):
        out = {}
        for k, v in per_img.items():
            m = v[idx, 0].mean(0); out[k] = (m.float().to(device), (v[idx, 1].mean(0) - m ** 2).clamp_min(0).float().to(device))
        return out

    def expected_stats(pool, n=2000):  # E over random B-image batches of the within-batch mean/var (what test batches see)
        acc = None
        for _ in range(n):
            st_ = batch_stats(g.choice(pool, B, replace=False))
            acc = st_ if acc is None else {k: (acc[k][0] + st_[k][0], acc[k][1] + st_[k][1]) for k in st_}
        return {k: (a / n, b / n) for k, (a, b) in acc.items()}

    N = len(src_imgs)
    src_stats = expected_stats(np.arange(N))
    def oof_div(keys):  # out-of-fold: stats from one half, signal on random batches of the other, smoothed like d_fast
        divs, raw = [], []
        halves = np.array_split(g.permutation(N), 2)
        for f in range(2):
            ref = expected_stats(halves[1 - f], 1000)
            df = None
            for t in range(400):
                st_ = batch_stats(g.choice(halves[f], B, replace=False))
                d = divergence({k: st_[k] for k in keys}, ref)
                df = d if df is None else 0.7 * df + 0.3 * d
                raw.append(d)
                if t >= 3:
                    divs.append(df)
        out = {f"q{q}": float(np.percentile(divs, q)) for q in (50, 90, 95, 99)}
        out.update({f"rq{q}": float(np.percentile(raw, q)) for q in (50, 90, 99)})
        return out

    src_div = oof_div(sig_keys)
    if cfg.get("signal2"):
        src_div["tap2"] = oof_div([str(cfg["signal2"])])
    log("source", images=len(src_imgs), **{k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()}) for k, v in src_div.items()})
    del src_imgs
    torch.save({"src_stats": src_stats, "src_div": src_div}, f"{outdir}/source_profile.pt")

    en = Energy(device)
    e0, t0 = en.joules(), time.time()
    time.sleep(5)
    log("idle", source=en.source, watts=round((en.joules() - e0) / (time.time() - t0), 2))
    results = []
    only = os.environ.get("METHODS")
    for mcfg in cfg["methods"]:
        if only and mcfg["name"] not in only.split(","):
            continue
        try:
            r = run_method(mcfg, cfg, det0, src_stats, src_div, frames, coco, ids_by_phase, device, outdir, cat_map)
            r.update(run_method(mcfg, cfg, det0, src_stats, src_div, frames, coco, ids_by_phase, device, outdir, cat_map,
                                replay=r.pop("_trace")))
        except Exception as e:
            log("fail", stage="method", method=mcfg["name"], error=f"{type(e).__name__}: {e}")
            raise
        results.append(r)
        log("method", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    best = max(results, key=lambda r: r["map50_shift"])
    summary = {"results": results, "src_div": src_div, "secs_total": round(time.time() - T0),
               "headline": {"best_method": best["method"], "map50_shift": best["map50_shift"]}}
    json.dump(summary, open(f"{outdir}/summary.json", "w"), indent=1, default=float)
    log("table", rows=[f"{r['method']}: shift50={r['map50_shift']:.4f} clean0={r['map50_clean_first']:.4f} "
                        f"cleanR={r['map50_clean_last']:.4f} steps={r['adapt_steps']} J/f={r['j_per_frame']:.3f} p99={r['lat_p99']:.1f}ms"
                        for r in results])
    log("result", **summary["headline"], secs_total=summary["secs_total"], outdir=outdir)


if __name__ == "__main__":
    main()
