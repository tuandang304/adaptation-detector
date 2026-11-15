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

    imgs, lbs, meta = frames
    B = cfg["batch"]
    n_batches = (len(imgs) + B - 1) // B
    ema_stats, d_fast, d_slow = None, None, None
    dets_by_phase = [[] for _ in ids_by_phase]
    trace = {k: np.zeros(n_batches, np.float32) for k in ("div", "adapt", "want", "event", "bn_upd", "restore", "lat_ms", "energy_j")}
    need_sig = sched in ("trigger", "budget") or bn_gate == "ood" or bool(reset_after) or bool(cp_delta)
    need_stats = need_sig or (grad and bool(w_align))
    d = 0.0
    calm, burst_left, cool = 0, 0, 0
    budget = {}
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
        if replay is not None:
            do, upd = bool(replay["adapt"][t]), bool(replay["bn_upd"][t])
        for b in bns:
            b.update = upd

        with torch.set_grad_enabled(do):
            y, raw = det(x)
            st = taps.stats() if need_stats else {}
            if need_sig:
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
        elif cp_delta and t >= 3 and cool == 0 and abs(np.log((d_fast + 1e-8) / (d_slow + 1e-8))) > cp_delta:
            event, burst_left, cool, d_slow = True, cp_burst, cp_cool, d_fast
            if reset_on_shift:
                restore_source(params, src_params, bns, opt)
                ema_stats, restored = None, True
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
           "map50_by_phase": {f"{i}:{n}": round(ev[i][1], 4) for i, n in enumerate(phases)},
           "adapt_steps": int(trace["adapt"].sum()), "adapt_frac": float(trace["adapt"].mean()), "bn_upd_frac": float(trace["bn_upd"].mean()),
           "eval_pass": {k: round(v, 3) if isinstance(v, float) else v for k, v in cost.items()}, "secs": round(wall, 1), **budget}
    if cp_delta:
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
    if st.get("same_clean_ids"):
        first = ids_by_phase[st["phases"].index("clean")]
        ids_by_phase = [first if n == "clean" else ids for n, ids in zip(st["phases"], ids_by_phase)]
    frames = build_frames(coco, root, ids_by_phase, st["phases"], st["severity"], cfg["imgsz"], cfg["seed"])

    det0, bns = load_model(cfg["model"], device)
    log("model", weights=cfg["model"], params=sum(p.numel() for p in det0.parameters()), bn_layers=len(bns),
        bn_affine=sum(b.weight.numel() * 2 for b in bns), stem_bn_affine=sum(b.weight.numel() * 2 for b in bns if b.layer < 5))

    # source feature statistics + per-batch divergence distribution on the clean source split
    src_imgs, _, _ = build_frames(coco, root, [sp["source"]], ["clean"], 1, cfg["imgsz"], cfg["seed"])
    align_keys = [str(k) for k in cfg["taps"]]
    sig_keys = [str(k) for k in cfg.get("signal_taps", cfg["taps"])]
    taps = Taps(det0, sorted(set(align_keys + sig_keys)))
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
    divs = []
    halves = np.array_split(g.permutation(N), 2)
    for f in range(2):  # out-of-fold: stats from one half, signal on random batches of the other, smoothed like d_fast
        ref = expected_stats(halves[1 - f], 1000)
        df = None
        for t in range(400):
            st_ = batch_stats(g.choice(halves[f], B, replace=False))
            d = divergence({k: st_[k] for k in sig_keys}, ref)
            df = d if df is None else 0.7 * df + 0.3 * d
            if t >= 3:
                divs.append(df)
    src_div = {f"q{q}": float(np.percentile(divs, q)) for q in (50, 90, 95, 99)}
    log("source", images=len(src_imgs), **{k: round(v, 4) for k, v in src_div.items()})
    del src_imgs
    torch.save({"src_stats": src_stats, "src_div": src_div}, f"{outdir}/source_profile.pt")

    cat_map = coco80_to_coco91_class()
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
