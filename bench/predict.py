"""Unified predictors: every model returns, per image, xyxy boxes (original pixels), scores, class idx in NAMES.

Class ids are mapped *by name* so checkpoints with a different label order (e.g. an extra 'others'
class) are evaluated on the dataset's own classes; unmapped classes are dropped and reported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _norm(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _name_map(names, target_names) -> dict[int, int]:
    target = {_norm(n): i for i, n in enumerate(target_names)}
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(k): target[_norm(v)] for k, v in items if _norm(v) in target}


class YoloPredictor:
    family = "ultralytics"

    def __init__(self, ckpt: str, names: list[str], device: str = "cpu", half: bool = False):
        from ultralytics import YOLO
        import ultralytics.utils.nms as nms
        # predict() has no multi_label argument and defaults to one class per box; Ultralytics val and RF-DETR's
        # top-k both score every box/class pair, so the shared evaluator uses multi-label here too.
        if not getattr(nms.non_max_suppression, "_multi_label", False):
            orig = nms.non_max_suppression
            nms.non_max_suppression = lambda *a, **k: orig(*a, **{**k, "multi_label": True})
            nms.non_max_suppression._multi_label = True
        self.model = YOLO(ckpt)
        self.device, self.half = device, half
        self.map = _name_map(self.model.names, names)
        print(f"[predict] {ckpt}: names={self.model.names} -> map={self.map}", flush=True)

    def __call__(self, paths: list[str], imgsz: int, conf: float = 0.001, max_det: int = 500, batch: int = 8):
        # default Ultralytics predict path (nms=None): one-to-many head + NMS, also for YOLO26/YOLOv10
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            res = self.model.predict(chunk, imgsz=imgsz, conf=conf, iou=0.7, max_det=max_det, device=self.device,
                                     verbose=False, **({"half": True} if self.half else {}))  # half=False still warns per call
            for p, r in zip(chunk, res):
                b = r.boxes
                cls = b.cls.cpu().numpy().astype(int)
                keep = np.array([c in self.map for c in cls], dtype=bool)
                yield p, b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep], \
                    np.array([self.map[c] for c in cls[keep]], dtype=int)

    def torch_module(self):
        return self.model.model


def load_predictor(ckpt: str, names: list[str], device: str = "cpu", half: bool = False):
    return YoloPredictor(ckpt, names, device, half)


def predict_split(pred, data_root: Path, split: str, imgsz: int, coco: dict, max_det: int = 1000) -> list[dict]:
    """COCO-results list for one split."""
    ids = {im["file_name"]: im["id"] for im in coco["images"]}
    paths = [str(data_root / split / "images" / im["file_name"]) for im in coco["images"]]
    out = []
    for n, (p, xyxy, sc, cl) in enumerate(pred(paths, imgsz=imgsz, max_det=max_det)):
        iid = ids[Path(p).name]
        for (x0, y0, x1, y1), s, c in zip(xyxy.tolist(), sc.tolist(), cl.tolist()):
            out.append({"image_id": iid, "category_id": int(c), "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
        if (n + 1) % 200 == 0:
            print(f"[predict] {split}@{imgsz}: {n + 1}/{len(paths)}", flush=True)
    return out
