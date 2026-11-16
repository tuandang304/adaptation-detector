"""Entry point. Reads configs/node.yaml and runs it. Outputs go to $ORX_OUTPUT_DIR; headline lines are greppable tags."""
from __future__ import annotations

import argparse
import json
import os
import platform
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/node.yaml")
    a = ap.parse_args()
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
    except Exception as e:
        print(f"[fail] stage={stage} error={type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        summary["minutes"] = round((time.time() - t0) / 60, 1)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("[result] " + json.dumps({"stage": stage, "minutes": summary["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
