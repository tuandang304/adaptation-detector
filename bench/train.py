"""Fine-tune from the official COCO checkpoint. Only the shared knobs are set here; the recipe lives in configs/node.yaml."""
from __future__ import annotations

import time
from pathlib import Path


def train(model: str, tag: str, data_root: Path, out: Path, epochs: int, imgsz: int, seed: int, device: str,
          extra: dict | None = None, mods: list[str] | None = None) -> tuple[str, dict]:
    import torch
    from ultralytics import YOLO
    y = YOLO(f"{model}.pt")
    kw = dict(data=str(data_root / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=16, seed=seed, deterministic=True,
              device=device, project=str(out), name="train", exist_ok=True, plots=True, workers=4, cache="ram", val=True)
    kw.update(extra or {})
    t0, last = time.time(), [time.time()]

    def on_train_start(tr):
        g = tr.optimizer.param_groups[0]
        print(f"[model] ds={tag} model={model} params={sum(p.numel() for p in tr.model.parameters())} "
              f"tensors={len(list(tr.model.parameters()))} optimizer={type(tr.optimizer).__name__} "
              f"lr0={tr.args.lr0} momentum={g.get('momentum', g.get('betas'))} batch={tr.batch_size} "
              f"train_images={len(tr.train_loader.dataset)} iters_per_epoch={len(tr.train_loader)}", flush=True)

    def on_fit_epoch_end(tr):
        dt, last[0] = time.time() - last[0], time.time()
        loss = " ".join(f"{k.split('/')[-1].replace('_loss', '')}={float(v):.3f}"
                        for k, v in tr.label_loss_items(tr.tloss, prefix="train").items())
        m = tr.metrics
        mem = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        left = (tr.epochs - tr.epoch - 1) * dt / 60
        if tr.epoch >= tr.epochs:  # Ultralytics fires this once more after re-validating best.pt
            return
        print(f"[epoch {tr.epoch + 1}/{tr.epochs}] ds={tag} loss {loss} | val map={m.get('metrics/mAP50-95(B)', float('nan')):.4f} "
              f"map50={m.get('metrics/mAP50(B)', float('nan')):.4f} | lr={tr.optimizer.param_groups[0]['lr']:.5f} | "
              f"{dt:.0f}s eta={left:.0f}m mem={mem:.1f}G", flush=True)
        if (tr.epoch + 1) in (1, 3, 4, 10, 20, tr.epochs):
            target = tr.ema.ema if (getattr(tr, "ema", None) is not None and tr.ema.enabled) else tr.model
            if hasattr(target, "diag_report"):
                rep = target.diag_report()
                if rep:
                    print(f"[r12diag] ep={tr.epoch + 1} " + " ".join(f"{k}={v:.4f}" for k, v in rep.items()), flush=True)

    y.add_callback("on_train_start", on_train_start)
    y.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    print(f"[train] ds={tag} {kw}", flush=True)
    if mods:  # the trainer rebuilds the model from its yaml, so the changes are applied inside it
        from .mods import ModTrainer, on_r12_epoch_end, on_r12_epoch_start
        y.add_callback("on_train_epoch_start", on_r12_epoch_start)
        y.add_callback("on_train_epoch_end", on_r12_epoch_end)
        ModTrainer.mods = list(mods)
        y.train(trainer=ModTrainer, **kw)
    else:
        y.train(**kw)
    return str(out / "train" / "weights" / "best.pt"), {"train_hours": round((time.time() - t0) / 3600, 3),
                                                        "train_kwargs": {k: str(v) for k, v in kw.items()}}
