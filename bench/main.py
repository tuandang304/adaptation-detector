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


def evaluate_ckpt(tag: str, ckpt: str, root: Path, imgsz: int, splits: list[str], device: str, out: Path) -> dict:
    """Score a checkpoint with the unified evaluator; detections are kept so a model can be rescored later."""
    from . import data as D, efficiency, evaluate, predict
    pred = predict.load_predictor(ckpt, D.names_of(root), device=device)
    res: dict = {"tag": tag, "ckpt": ckpt, "metrics": {}}
    for split in splits:
        gt = json.loads((root / f"coco_{split}.json").read_text())
        ign = json.loads((root / f"ignore_{split}.json").read_text())
        t0 = time.time()
        dets = predict.predict_split(pred, root, split, imgsz, gt)
        with gzip.open(out / f"dets_{tag}_{split}_{imgsz}.json.gz", "wt") as f:
            json.dump(dets, f)
        m, mi = evaluate.coco_eval(gt, dets), evaluate.coco_eval(gt, dets, ignore=ign)
        res["metrics"][f"{split}@{imgsz}"] = {"std": m, "ignore_aware": mi, "infer_sec": round(time.time() - t0, 1)}
        print(f"[eval] {tag} {split}@{imgsz}: {evaluate.fmt(mi)} | std mAP50_95={m['mAP50_95']:.2f}", flush=True)
    res["efficiency"] = efficiency.params_flops(pred.torch_module(), imgsz)
    if device != "cpu":
        res["efficiency"].update(efficiency.latency(pred, str(next((root / "valid" / "images").iterdir())), imgsz))
    return res


def run_dataset(cfg: dict, name: str, gpu: int) -> dict:
    """Prepare one dataset, fine-tune on it, score the best checkpoint on val. Headline = ignore-aware mAP50-95
    (identical to plain COCO AP for datasets without ignore regions)."""
    import torch
    from . import data as D
    from .train import train
    root = D.prepare(name, cfg["data_dir"])
    if cfg.get("smoke"):  # 1 epoch on a small subset: every stage runs, nothing is measured
        root = D.make_subset(root, root.parent / f"{name}_smoke", {"train": 64, "valid": 32})
    tag, out = f"{cfg['model']}_{name}_s{cfg['seed']}", OUT / name
    out.mkdir(parents=True, exist_ok=True)
    device = str(gpu) if torch.cuda.is_available() else "cpu"
    ckpt, info = train(cfg["model"], name, root, out, 1 if cfg.get("smoke") else cfg["epochs"], cfg["imgsz"],
                       cfg["seed"], device, cfg.get("extra"))
    res = evaluate_ckpt(tag, ckpt, root, cfg["imgsz"], cfg.get("eval_splits", ["valid"]), device, out)
    res.update(dataset=name, train=info, manifest=json.loads((root / "manifest.json").read_text())["splits"])
    (out / "result.json").write_text(json.dumps(res, indent=2))
    m = res["metrics"][f"valid@{cfg['imgsz']}"]["ignore_aware"]
    print("[result] " + json.dumps({"dataset": name, "model": cfg["model"], "epochs": cfg["epochs"], "seed": cfg["seed"],
                                    "mAP50_95": m["mAP50_95"], "mAP50": m["mAP50"], "AP_small": m["AP_small"],
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
            summary["results"] = {k: {"metrics": v["metrics"], "efficiency": v["efficiency"], "train": v["train"],
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
    print("[result] " + json.dumps({"stage": stage, "minutes": summary["minutes"], "mAP50_95": head}), flush=True)


if __name__ == "__main__":
    main()
