"""Budget-aware continual test-time adaptation (CTTA) for YOLO detectors on a COCO-C cyclic stream.

One run = one stream, many method configs (config.yaml). Every method starts from the same source model,
sees the same frames in the same order, and is scored with the same COCOeval procedure.
"""
import argparse, contextlib, copy, hashlib, io, json, os, subprocess, sys, time, zipfile
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
            jobs.append((f"{root}/val2017/{coco.imgs[i]['file_name']}", name, severity, seed * 100000 + len(jobs), size))
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
        self.momentum, self.anchor, self.update = 0.0, 0.0, False

    def forward(self, x):
        if self.update and self.momentum > 0:
            with torch.no_grad():
                m, a = self.momentum, self.anchor
                self.running_mean.mul_(1 - m).add_(x.mean((0, 2, 3)), alpha=m)
                self.running_var.mul_(1 - m).add_(x.var((0, 2, 3), unbiased=False), alpha=m)
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
    """Per-channel mean/var of chosen layer outputs, captured on every forward (with grad when enabled)."""

    def __init__(self, det, layers):
        self.out = {}
        for i in layers:
            det.model[i].register_forward_hook(lambda m, inp, o, i=i: self.out.__setitem__(i, o))
        self.layers = layers

    def stats(self):
        return {i: (o.mean((0, 2, 3)), o.var((0, 2, 3), unbiased=False)) for i, o in self.out.items()}


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
def run_method(mcfg, cfg, det0, src_stats, src_div, frames, coco, ids_by_phase, device, outdir, cat_map):
    from ultralytics.utils.nms import non_max_suppression
    name = mcfg["name"]
    det = copy.deepcopy(det0)
    bns = [m for m in det.modules() if isinstance(m, AdaBN)]
    taps = Taps(det, cfg["taps"])
    stem_end = mcfg.get("stem_end", 5)
    adapt_bns = bns if mcfg.get("params", "all") == "all" else [b for b in bns if b.layer < stem_end]
    for b in bns:
        b.momentum, b.anchor = mcfg.get("bn_momentum", 0.0), mcfg.get("bn_anchor", 0.0)
    grad = mcfg.get("adapt", "none") == "grad"
    params = [p for b in adapt_bns for p in (b.weight, b.bias)] if grad else []
    for p in params:
        p.requires_grad_(True)
    src_params = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=mcfg.get("lr", 1e-4)) if grad else None
    w_align, w_ent = mcfg.get("loss", {}).get("align", 1.0), mcfg.get("loss", {}).get("ent", 0.0)
    alpha = mcfg.get("ema_alpha", 0.1)
    sched = mcfg.get("schedule", "always")
    K = mcfg.get("period", 10)
    tau_hi = src_div["q99"] * mcfg.get("thr_mult", 1.0)
    tau_lo = src_div["q90"] * mcfg.get("thr_mult", 1.0)
    rho = mcfg.get("budget_frac", 0.2)          # budget = rho * (measured always-adapt step cost) per batch
    burst = mcfg.get("burst", 8)                # token-bucket depth in steps
    reset_after = mcfg.get("reset_after", 0)    # consecutive in-domain batches before restoring source params

    imgs, lbs, meta = frames
    B = cfg["batch"]
    n_batches = (len(imgs) + B - 1) // B
    ema_stats, d_ema = None, None
    dets_by_phase = [[] for _ in ids_by_phase]
    trace = {k: np.zeros(n_batches, np.float32) for k in ("div", "adapt", "want", "lat_ms", "energy_j")}
    bucket, step_cost, base_cost, calm = burst, None, None, 0
    energy = Energy(device)

    # warm-up on clean source-like zeros so cudnn autotune/allocator do not land on frame 0
    with torch.no_grad():
        for _ in range(5):
            det(torch.zeros(B, 3, cfg["imgsz"], cfg["imgsz"], device=device))
    for b in bns:
        b.update = False
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()
    e_start, t_start = energy.joules(), time.time()
    e_prev = e_start
    for t in range(n_batches):
        sl = slice(t * B, min((t + 1) * B, len(imgs)))
        t0 = time.perf_counter()
        x = torch.from_numpy(imgs[sl]).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)

        # ---- decide: adapt this batch?  (decision uses previous batches' signal only)
        want = d_ema is not None and d_ema > tau_hi
        if sched == "always":
            do = grad
        elif sched == "periodic":
            do = grad and t % K == 0
        elif sched == "trigger":
            do = grad and want
        elif sched == "budget":
            do = grad and want and bucket >= 1.0
        else:
            raise ValueError(sched)
        for b in bns:
            b.update = mcfg.get("bn_momentum", 0.0) > 0

        with torch.set_grad_enabled(do):
            y, raw = det(x)
            st = taps.stats()
            d = divergence({i: (m.detach(), v.detach()) for i, (m, v) in st.items()}, src_stats)
            if do:
                loss = 0.0
                if w_align:
                    if ema_stats is None:
                        ema_stats = {i: (m.detach(), v.detach()) for i, (m, v) in st.items()}
                    cur = {i: ((1 - alpha) * ema_stats[i][0] + alpha * m, (1 - alpha) * ema_stats[i][1] + alpha * v)
                           for i, (m, v) in st.items()}
                    loss = loss + w_align * sum(kl(m, v, *src_stats[i]) for i, (m, v) in cur.items()) / len(cur)
                    ema_stats = {i: (m.detach(), v.detach()) for i, (m, v) in cur.items()}
                if w_ent:
                    s = torch.cat([z.flatten(2) for z in raw["scores"]], 2) if isinstance(raw["scores"], list) else raw["scores"]
                    p = s.sigmoid().amax(1)  # [B, anchors] max class prob
                    conf = p > mcfg.get("ent_conf", 0.3)
                    pe = p[conf].clamp(1e-6, 1 - 1e-6)
                    if pe.numel():
                        loss = loss + w_ent * -(pe * pe.log() + (1 - pe) * (1 - pe).log()).mean()
                if torch.is_tensor(loss):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            elif ema_stats is not None:  # keep test-stat EMA current even when not adapting
                ema_stats = {i: ((1 - alpha) * ema_stats[i][0] + alpha * m.detach(), (1 - alpha) * ema_stats[i][1] + alpha * v.detach())
                             for i, (m, v) in st.items()}
        d_ema = d if d_ema is None else 0.7 * d_ema + 0.3 * d

        # ---- clean-return restoration
        if reset_after and grad:
            calm = calm + 1 if d_ema < tau_lo else 0
            if calm == reset_after:
                with torch.no_grad():
                    for p, s0 in zip(params, src_params):
                        p.copy_(s0)
                    for b in bns:
                        b.running_mean.copy_(b.src_mean); b.running_var.copy_(b.src_var)
                opt.state.clear()

        out = non_max_suppression(y.detach(), conf_thres=0.001, iou_thres=0.7, max_det=100)
        sync()
        lat = (time.perf_counter() - t0) * 1000
        e_now = energy.joules()
        e_batch, e_prev = e_now - e_prev, e_now

        # ---- budget accounting on *measured* energy: learn per-batch cost of skip vs adapt online
        if sched == "budget":
            if do:
                step_cost = e_batch if step_cost is None else 0.9 * step_cost + 0.1 * e_batch
            else:
                base_cost = e_batch if base_cost is None else 0.9 * base_cost + 0.1 * e_batch
            if do and base_cost is not None:
                extra = max(e_batch - base_cost, 0.0)
                unit = max((step_cost or e_batch) - base_cost, 1e-6)
                bucket -= extra / unit
            elif do:
                bucket -= 1.0
            bucket = min(bucket + rho, burst)

        trace["div"][t], trace["adapt"][t], trace["want"][t], trace["lat_ms"][t], trace["energy_j"][t] = d, float(do), float(want), lat, e_batch
        for k, o in enumerate(out):
            fi = sl.start + k
            p, img_id = meta[fi]
            r, dw, dh = lbs[fi]
            o = o.cpu().numpy()
            for x1, y1, x2, y2, sc, c in o:
                x1, x2 = (x1 - dw) / r, (x2 - dw) / r
                y1, y2 = (y1 - dh) / r, (y2 - dh) / r
                dets_by_phase[p].append({"image_id": int(img_id), "category_id": cat_map[int(c)],
                                         "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                         "score": round(float(sc), 5)})
        if not np.isfinite(d):
            log("warn", method=name, batch=t, issue="non-finite divergence")
    sync()
    e_total, wall = energy.joules() - e_start, time.time() - t_start
    for p in params:
        p.requires_grad_(False)

    ev = evaluate(coco, dets_by_phase, ids_by_phase)
    phases = cfg["stream"]["phases"]
    shift = [i for i, n in enumerate(phases) if n != "clean"]
    lat = trace["lat_ms"][5:]
    res = {"method": name, "map50_shift": np.mean([ev[i][1] for i in shift]), "map_shift": np.mean([ev[i][0] for i in shift]),
           "map50_clean_first": ev[0][1], "map50_clean_last": ev[-1][1],
           "map50_by_phase": {f"{i}:{n}": round(ev[i][1], 4) for i, n in enumerate(phases)},
           "adapt_steps": int(trace["adapt"].sum()), "adapt_frac": float(trace["adapt"].mean()),
           "energy_j": e_total, "j_per_frame": e_total / len(imgs), "energy_source": energy.source,
           "lat_p50": float(np.percentile(lat, 50)), "lat_p95": float(np.percentile(lat, 95)), "lat_p99": float(np.percentile(lat, 99)),
           "fps": len(imgs) / wall, "secs": round(wall, 1)}
    res.update(trigger_metrics(trace, meta, B, phases, cfg.get("trigger_window", 40)) if sched in ("trigger", "budget") else {})
    np.savez_compressed(f"{outdir}/trace_{name}.npz", **trace)
    with open(f"{outdir}/dets_{name}.json", "w") as f:
        json.dump(dets_by_phase, f)
    return res


def trigger_metrics(trace, meta, B, phases, W):
    """Shift-event reliability (EmbodiTTA-style counts, adapted to batches): onsets = rising edges of the adapt-want signal."""
    phase_of_batch = np.array([meta[min(t * B, len(meta) - 1)][0] for t in range(len(trace["adapt"]))])
    bounds = [int(np.argmax(phase_of_batch == p)) for p in range(1, len(phases))]
    a = trace["want"] > 0
    onsets = [t for t in range(len(a)) if a[t] and (t == 0 or not a[t - 1])]
    hit, delays = 0, []
    for b in bounds:
        o = [t for t in onsets if b <= t <= b + W]
        if o:
            hit += 1
            delays.append(o[0] - b)
    in_win = lambda t: any(b <= t <= b + W for b in bounds)
    # adapting during the first clean phase is a false trigger; so is re-triggering inside a stationary phase
    false = sum(1 for t in onsets if not in_win(t))
    clean0 = int(a[phase_of_batch == 0].sum())
    return {"shifts_detected": hit, "shifts_total": len(bounds), "false_onsets": false,
            "delay_batches_mean": float(np.mean(delays)) if delays else None, "adapt_in_clean0": clean0}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
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
    root = os.environ.get("DATA_DIR", "/tmp/coco")
    ann = fetch_coco(root)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(ann)
    sp = split_ids(coco, cfg["source_images"], seed=0)
    st = cfg["stream"]
    rng = np.random.RandomState(cfg["seed"])
    pool = sp[st["split"]]
    ids_by_phase = [rng.choice(pool, st["per_phase"], replace=False).tolist() for _ in st["phases"]]
    frames = build_frames(coco, root, ids_by_phase, st["phases"], st["severity"], cfg["imgsz"], cfg["seed"])

    det0, bns = load_model(cfg["model"], device)
    log("model", weights=cfg["model"], params=sum(p.numel() for p in det0.parameters()), bn_layers=len(bns),
        bn_affine=sum(b.weight.numel() * 2 for b in bns), stem_bn_affine=sum(b.weight.numel() * 2 for b in bns if b.layer < 5))

    # source feature statistics + per-batch divergence distribution on the clean source split
    src_imgs, _, _ = build_frames(coco, root, [sp["source"]], ["clean"], 1, cfg["imgsz"], cfg["seed"])
    taps = Taps(det0, cfg["taps"])
    acc, divs, B = {}, [], cfg["batch"]
    with torch.no_grad():
        for k in range(0, len(src_imgs), B):
            x = torch.from_numpy(src_imgs[k:k + B]).to(device).permute(0, 3, 1, 2).float().div_(255)
            det0(x)
            for i, o in taps.out.items():
                s = acc.setdefault(i, [0.0, 0.0, 0])
                s[0] = s[0] + o.sum((0, 2, 3)); s[1] = s[1] + (o ** 2).sum((0, 2, 3)); s[2] += o.numel() // o.shape[1]
        src_stats = {i: (s0 / n, s1 / n - (s0 / n) ** 2) for i, (s0, s1, n) in acc.items()}
        for k in range(0, len(src_imgs) - B + 1, B):
            x = torch.from_numpy(src_imgs[k:k + B]).to(device).permute(0, 3, 1, 2).float().div_(255)
            det0(x)
            divs.append(divergence(taps.stats(), src_stats))
    for i in cfg["taps"]:
        det0.model[i]._forward_hooks.clear()
    src_div = {f"q{q}": float(np.percentile(divs, q)) for q in (50, 90, 95, 99)}
    log("source", images=len(src_imgs), **{k: round(v, 4) for k, v in src_div.items()})
    del src_imgs
    torch.save({"src_stats": src_stats, "src_div": src_div}, f"{outdir}/source_profile.pt")

    cat_map = coco80_to_coco91_class()
    results = []
    only = os.environ.get("METHODS")
    for mcfg in cfg["methods"]:
        if only and mcfg["name"] not in only.split(","):
            continue
        try:
            r = run_method(mcfg, cfg, det0, src_stats, src_div, frames, coco, ids_by_phase, device, outdir, cat_map)
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
