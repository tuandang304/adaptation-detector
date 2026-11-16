"""Dataset preparation. Every dataset ends in the same layout under <data_dir>/<name>:

  data.yaml                            (Ultralytics dataset file; class names are read from here)
  {train,valid[,test]}/images/*        (symlinks to the source images)
  {train,valid[,test]}/labels/*.txt    (YOLO cls cx cy w h, normalised)
  coco_{valid,test}.json               (COCO ground truth for the unified evaluator)
  ignore_{valid,test}.json             (per-image ignore boxes; empty for datasets without ignore regions)
  manifest.json                        (source, per-split counts, file-list SHA256)

A converter yields, per image: (image path, [(class index, x0, y0, x1, y1)], [ignore (x0, y0, x1, y1)]) in pixels.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import yaml
from PIL import Image

Item = tuple[Path, list[tuple[int, float, float, float, float]], list[tuple[float, float, float, float]]]


def write_split(out: Path, split: str, names: list[str], items: Iterable[Item]) -> dict:
    img_dir, lab_dir = out / split / "images", out / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    images, anns, ignores, n_bad = [], [], {}, 0
    for img_id, (img_path, boxes, ign) in enumerate(items):
        link = img_dir / img_path.name
        assert not link.exists(), f"duplicate image name {img_path.name} in {split}"
        os.symlink(img_path.resolve(), link)
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        lines = []
        for cls, x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)  # clip to image
            if x1 <= x0 or y1 <= y0:
                n_bad += 1
                continue
            assert 0 <= cls < len(names), (img_path, cls)
            cw, ch = x1 - x0, y1 - y0
            lines.append(f"{cls} {(x0 + cw / 2) / w:.6f} {(y0 + ch / 2) / h:.6f} {cw / w:.6f} {ch / h:.6f}")
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cls, "bbox": [x0, y0, cw, ch],
                         "area": cw * ch, "iscrowd": 0})
        (lab_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        clipped = [(max(0, a), max(0, b), min(w, c), min(h, d)) for a, b, c, d in ign]
        n_bad += sum(c <= a or d <= b for a, b, c, d in clipped)
        ignores[img_id] = [[a, b, c - a, d - b] for a, b, c, d in clipped if c > a and d > b]
    if split != "train":
        (out / f"coco_{split}.json").write_text(json.dumps({
            "images": images, "annotations": anns,
            "categories": [{"id": i, "name": n, "supercategory": out.name} for i, n in enumerate(names)]}))
        (out / f"ignore_{split}.json").write_text(json.dumps(ignores))
    return {"images": len(images), "gt_boxes": len(anns), "ignore_boxes": sum(len(v) for v in ignores.values()),
            "degenerate_dropped": n_bad,
            "filelist_sha256": hashlib.sha256("\n".join(i["file_name"] for i in images).encode()).hexdigest()}


def finish(out: Path, names: list[str], manifest: dict) -> Path:
    test = "test: test/images\n" if (out / "test").exists() else ""
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\n{test}"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def names_of(root: Path) -> list[str]:
    d = yaml.safe_load((root / "data.yaml").read_text())["names"]
    return [d[i] for i in sorted(d)]


def _download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(".part")
    print(f"[data] downloading {url}", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dst)


# --- VisDrone2019-DET (official zips re-hosted by Ultralytics) -------------------------------------------------------
# Rows are x,y,w,h,score,category,truncation,occlusion. GT when score == 1 and category in 1..10;
# category 0 ('ignored regions') and 11 ('others') become ignore regions.
VISDRONE = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
VISDRONE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def _visdrone_items(src: Path) -> Iterable[Item]:
    for img_path in sorted((src / "images").glob("*.jpg")):
        boxes, ign = [], []
        for row in (src / "annotations" / f"{img_path.stem}.txt").read_text().strip().splitlines():
            x, y, bw, bh, score, cat = [int(v) for v in row.strip().rstrip(",").split(",")[:8]][:6]
            if cat in (0, 11) or score == 0:
                ign.append((x, y, x + bw, y + bh))
            else:
                boxes.append((cat - 1, x, y, x + bw, y + bh))
        yield img_path, boxes, ign


def prep_visdrone(out: Path) -> Path:
    raw, manifest = out.parent / "visdrone_raw", {"source": VISDRONE_URL, "splits": {}}
    raw.mkdir(parents=True, exist_ok=True)
    for split, name in {"train": "VisDrone2019-DET-train", "valid": "VisDrone2019-DET-val",
                        "test": "VisDrone2019-DET-test-dev"}.items():
        z = raw / f"{name}.zip"
        _download(f"{VISDRONE_URL}/{name}.zip", z)
        if not (raw / name).exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw)
        manifest["splits"][split] = write_split(out, split, VISDRONE, _visdrone_items(raw / name))
    return finish(out, VISDRONE, manifest)


PREP: dict[str, Callable[[Path], Path]] = {"visdrone": prep_visdrone}


def prepare(name: str, data_dir: str | Path) -> Path:
    """Build dataset `name` under data_dir (skipped when its manifest exists)."""
    out = Path(data_dir) / name
    if not (out / "manifest.json").exists():
        shutil.rmtree(out, ignore_errors=True)  # a half-built directory would trip the duplicate check
        PREP[name](out)
    m = json.loads((out / "manifest.json").read_text())
    print("[data] " + json.dumps({"dataset": name, "source": m["source"], "splits": m["splits"]}), flush=True)
    return out


def make_subset(root: Path, out: Path, n: dict[str, int], seed: int = 0) -> Path:
    """Smoke-test subset: n random images per split, with re-indexed COCO GT."""
    if (out / "data.yaml").exists():
        return out
    rng = random.Random(seed)
    for split, k in n.items():
        if not (root / split).exists():
            continue
        names = sorted(p.name for p in (root / split / "images").iterdir())
        keep = set(rng.sample(names, min(k, len(names))))
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        for name in sorted(keep):
            os.symlink((root / split / "images" / name).resolve(), out / split / "images" / name)
            lab = Path(name).with_suffix(".txt").name
            shutil.copy(root / split / "labels" / lab, out / split / "labels" / lab)
        if split != "train":
            coco = json.loads((root / f"coco_{split}.json").read_text())
            ign = json.loads((root / f"ignore_{split}.json").read_text())
            ids = {im["id"] for im in coco["images"] if im["file_name"] in keep}
            coco["images"] = [im for im in coco["images"] if im["id"] in ids]
            coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
            (out / f"coco_{split}.json").write_text(json.dumps(coco))
            (out / f"ignore_{split}.json").write_text(json.dumps({k2: v for k2, v in ign.items() if int(k2) in ids}))
    (out / "data.yaml").write_text((root / "data.yaml").read_text().replace(f"path: {root.resolve()}", f"path: {out.resolve()}"))
    shutil.copy(root / "manifest.json", out / "manifest.json")
    return out
