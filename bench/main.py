"""Entry point. Reads configs/node.yaml and runs it. Outputs go to $ORX_OUTPUT_DIR; headline lines are greppable tags."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

OUT = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs"))


def env_info() -> dict:
    import torch
    import ultralytics
    return {"commit": os.environ.get("ORX_COMMIT"), "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "ultralytics": ultralytics.__version__, "nproc": os.cpu_count(), "platform": platform.platform()}


def _pool_eval(split_items: list[tuple[str, dict, dict, list[dict]]]) -> tuple[dict, dict]:
    """Single ignore-aware COCOeval over all shifted images pooled together (removes small-split distortion)."""
    from . import evaluate

    pooled_gt: dict = {"images": [], "annotations": [], "categories": split_items[0][1]["categories"]}
    pooled_ign: dict = {}
    pooled_dets: list[dict] = []
    next_ann_id = 1
    for idx, (_split, gt, ign, dets) in enumerate(split_items):
        off = (idx + 1) * 1_000_000
        for im in gt["images"]:
            pooled_gt["images"].append({**im, "id": int(im["id"]) + off})
        for ann in gt["annotations"]:
            pooled_gt["annotations"].append({**ann, "id": next_ann_id, "image_id": int(ann["image_id"]) + off})
            next_ann_id += 1
        for k, v in ign.items():
            pooled_ign[str(int(k) + off)] = v
        for d in dets:
            pooled_dets.append({**d, "image_id": int(d["image_id"]) + off})
    return evaluate.coco_eval(pooled_gt, pooled_dets), evaluate.coco_eval(pooled_gt, pooled_dets, ignore=pooled_ign)


def evaluate_ckpt(
    tag: str,
    ckpt: str,
    root: Path,
    imgsz: int,
    splits: list[str],
    device: str,
    out: Path,
    export_gt: bool = False,
    causal_eval: bool = False,
) -> dict:
    """Score a checkpoint with the unified evaluator; detections are kept so a model can be rescored later."""
    from . import data as D, efficiency, evaluate, predict

    pred = predict.load_predictor(ckpt, D.names_of(root), device=device)
    res: dict = {"tag": tag, "ckpt": ckpt, "metrics": {}}
    pooled_items: list[tuple[str, dict, dict, list[dict]]] = []
    if export_gt:
        (out / "gt").mkdir(parents=True, exist_ok=True)
    for split in splits:
        gt_path, ign_path = root / f"coco_{split}.json", root / f"ignore_{split}.json"
        gt = json.loads(gt_path.read_text())
        ign = json.loads(ign_path.read_text())
        if export_gt:
            (out / "gt" / f"coco_{split}.json").write_text(gt_path.read_text())
            (out / "gt" / f"ignore_{split}.json").write_text(ign_path.read_text())
        t0 = time.time()
        dets = predict.predict_split(pred, root, split, imgsz, gt)
        with gzip.open(out / f"dets_{tag}_{split}_{imgsz}.json.gz", "wt") as f:
            json.dump(dets, f)
        m, mi = evaluate.coco_eval(gt, dets), evaluate.coco_eval(gt, dets, ignore=ign)
        res["metrics"][f"{split}@{imgsz}"] = {"std": m, "ignore_aware": mi, "infer_sec": round(time.time() - t0, 1)}
        print(f"[eval] {tag} {split}@{imgsz}: {evaluate.fmt(mi)} | std mAP50_95={m['mAP50_95']:.2f}", flush=True)
        if split not in ("valid", "test"):
            pooled_items.append((split, gt, ign, dets))
    if len(pooled_items) >= 2:
        mp, mip = _pool_eval(pooled_items)
        res["metrics"][f"pooled_shift@{imgsz}"] = {"std": mp, "ignore_aware": mip}
        res["pooled_shift_mAP50_95"] = mip["mAP50_95"]
        print(f"[eval] {tag} pooled_shift@{imgsz}: {evaluate.fmt(mip)} | std mAP50_95={mp['mAP50_95']:.2f}", flush=True)
    res["efficiency"] = efficiency.params_flops(pred.torch_module(), imgsz)
    if device != "cpu":
        res["efficiency"].update(efficiency.latency(pred, str(next((root / "valid" / "images").iterdir())), imgsz))
    if causal_eval and pooled_items:
        pred_z = predict.load_predictor(ckpt, D.names_of(root), device=device)
        tm = pred_z.torch_module()
        if hasattr(tm, "zero_new_weights"):
            tm.zero_new_weights()
            z_items, z_splits = [], {}
            for split, gt, ign, _ in pooled_items:
                zdets = predict.predict_split(pred_z, root, split, imgsz, gt)
                _, zmi = evaluate.coco_eval(gt, zdets), evaluate.coco_eval(gt, zdets, ignore=ign)
                z_splits[split] = zmi["mAP50_95"]
                z_items.append((split, gt, ign, zdets))
            _, zmip = _pool_eval(z_items)
            full_p = res.get("pooled_shift_mAP50_95", 0.0)
            drop = round(full_p - zmip["mAP50_95"], 3)
            res["causal_ablation"] = {
                "zero_pooled_shift_mAP50_95": zmip["mAP50_95"],
                "pooled_drop": drop,
                "zero_splits": z_splits,
            }
            print(f"[causal] {tag} full_pooled={full_p:.3f} zero_pooled={zmip['mAP50_95']:.3f} drop={drop:+.3f} splits={z_splits}", flush=True)
    return res


def run_dataset(cfg: dict, name: str, gpu: int) -> dict:
    """Prepare one dataset, fine-tune on it, score the best checkpoint on val. Headline = ignore-aware mAP50-95
    (identical to plain COCO AP for datasets without ignore regions).
    smoke: true = one full epoch (real data, real epoch time and memory); smoke: tiny = one epoch on 64 images."""
    import torch
    from . import data as D
    from .train import train
    root = D.prepare(name, cfg["data_dir"])
    if cfg.get("smoke") == "tiny":  # CPU check of every stage on a few images
        root = D.make_subset(root, root.parent / f"{name}_smoke", {"train": 64, "valid": 32})
    tag, out = f"{cfg['model']}_{name}_s{cfg['seed']}", OUT / name
    out.mkdir(parents=True, exist_ok=True)
    device = str(gpu) if torch.cuda.is_available() else "cpu"
    ckpt, info = train(cfg["model"], name, root, out, 1 if cfg.get("smoke") else cfg["epochs"], cfg["imgsz"],
                       cfg["seed"], device, cfg.get("extra"), cfg.get("mods"))
    shifted = []
    if cfg.get("shift_eval"):
        from . import shift
        t0 = time.time()
        shifted = shift.make(root)
        print(f"[data] ds={name} shifted splits {shifted} ready in {time.time() - t0:.0f}s", flush=True)
    res = evaluate_ckpt(
        tag, ckpt, root, cfg["imgsz"], cfg.get("eval_splits", ["valid"]) + shifted, device, out,
        export_gt=bool(cfg.get("export_gt", False)),
        causal_eval=bool(cfg.get("causal_eval", False)),
    )
    res.update(dataset=name, train=info, manifest=json.loads((root / "manifest.json").read_text())["splits"])
    m = res["metrics"][f"valid@{cfg['imgsz']}"]["ignore_aware"]
    # every scored split other than the source validation set counts as shifted (synthetic valid_* or real ones)
    other = [s for s in cfg.get("eval_splits", ["valid"]) + shifted if s not in ("valid", "test")]
    sh = {s.removeprefix("valid_"): res["metrics"][f"{s}@{cfg['imgsz']}"]["ignore_aware"]["mAP50_95"] for s in other}
    res["shift_mAP50_95"] = round(sum(sh.values()) / len(sh), 3) if sh else None
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print("[result] " + json.dumps({"dataset": name, "model": cfg["model"], "epochs": cfg["epochs"], "seed": cfg["seed"],
                                    "mods": cfg.get("mods", []), "mAP50_95": m["mAP50_95"],
                                    "shift_mAP50_95": res["shift_mAP50_95"],
                                    "pooled_shift_mAP50_95": res.get("pooled_shift_mAP50_95"),
                                    "causal_ablation": res.get("causal_ablation"),
                                    "shift": sh, "mAP50": m["mAP50"], "AP_small": m["AP_small"],
                                    "AP_medium": m["AP_medium"], "AP_large": m["AP_large"], **res["efficiency"],
                                    "train_hours": info["train_hours"], "ckpt": ckpt}), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/node.yaml")
    ap.add_argument("--only", help="comma-separated datasets to run in this process (set by the parent)")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.only:  # child: one GPU, its datasets in order
        cfg = yaml.safe_load(Path(a.config).read_text())
        for name in a.only.split(","):
            try:
                run_dataset(cfg, name, a.gpu)
            except Exception as e:
                print(f"[fail] stage=dataset:{name} error={type(e).__name__}: {e}", flush=True)
                raise
        return
    cfg = yaml.safe_load(Path(a.config).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    print("[config] " + json.dumps(cfg), flush=True)
    info = env_info()
    print("[env] " + json.dumps(info), flush=True)
    t0, summary, stage = time.time(), {"config": cfg, "env": info}, "setup"
    try:
        if cfg.get("probe"):
            stage = "probe"
            from . import probe
            summary["probe"] = probe.run(cfg["probe"])
        if cfg.get("gpu_plan"):
            stage = "train"
            import torch
            cmd = [sys.executable, "-m", "bench.main", "--config", a.config]
            if torch.cuda.device_count() >= len(cfg["gpu_plan"]):  # one process per GPU, side by side
                procs = [subprocess.Popen(cmd + ["--only", ",".join(ds), "--gpu", str(i)])
                         for i, ds in enumerate(cfg["gpu_plan"])]
                rcs = [p.wait() for p in procs]
            else:
                rcs = [subprocess.call(cmd + ["--only", ",".join(ds), "--gpu", "0"]) for ds in cfg["gpu_plan"]]
            done = {p.parent.name: json.loads(p.read_text()) for p in sorted(OUT.glob("*/result.json"))}
            summary["results"] = {k: {"metrics": v["metrics"], "shift_mAP50_95": v["shift_mAP50_95"],
                                      "pooled_shift_mAP50_95": v.get("pooled_shift_mAP50_95"),
                                      "causal_ablation": v.get("causal_ablation"),
                                      "efficiency": v["efficiency"], "train": v["train"],
                                      "manifest": v["manifest"]} for k, v in done.items()}
            if any(rcs):
                raise RuntimeError(f"dataset process exit codes {rcs}; finished: {sorted(done)}")
    except Exception as e:
        print(f"[fail] stage={stage} error={type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        summary["minutes"] = round((time.time() - t0) / 60, 1)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    head = {k: v["metrics"][f"valid@{cfg['imgsz']}"]["ignore_aware"]["mAP50_95"] for k, v in summary.get("results", {}).items()}
    sh = {k: v["shift_mAP50_95"] for k, v in summary.get("results", {}).items()}
    print("[result] " + json.dumps({"stage": stage, "minutes": summary["minutes"], "mAP50_95": head, "shift_mAP50_95": sh}), flush=True)


if __name__ == "__main__":
    main()
