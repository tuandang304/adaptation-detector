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
import subprocess
import time
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


def fetch_kaggle(ref: str, raw: Path) -> Path:
    """Fetch a public Kaggle dataset; returns the directory holding its files.
    Inside a Kaggle kernel: kagglehub over HTTP with the kernel's own identity (a batch kernel cannot attach a
    dataset at run time, and kagglehub only tries to attach when KAGGLE_KERNEL_RUN_TYPE is set).
    Elsewhere: anonymous download, which Kaggle answers with 404 from cloud hosts such as Colab."""
    dst = raw / ref.split("/")[1]
    if (dst / ".done").exists():
        return dst
    raw.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading kaggle:{ref}", flush=True)
    t0 = time.time()
    if "KAGGLE_USER_SECRETS_TOKEN" in os.environ:
        import kagglehub
        os.environ["KAGGLEHUB_CACHE"] = str(raw / "kagglehub")
        run_type = os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
        try:
            dst = Path(kagglehub.dataset_download(ref))
        finally:
            if run_type is not None:
                os.environ["KAGGLE_KERNEL_RUN_TYPE"] = run_type
    else:
        z = raw / (ref.split("/")[1] + ".zip")
        subprocess.run(["curl", "-L", "-sS", "--fail", "--retry", "5", "--retry-delay", "10", "-o", str(z),
                        f"https://www.kaggle.com/api/v1/datasets/download/{ref}"], check=True)
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(dst)], check=True)
        z.unlink()
        (dst / ".done").touch()
    print(f"[data] kaggle:{ref} ready in {time.time() - t0:.0f}s at {dst}", flush=True)
    return dst


# --- Cityscapes (instance boxes from gtFine instanceIds; copy verified 2975 train / 500 val) -------------------------
# instanceIds pixels >= 1000 encode labelId * 1000 + instance index for the eight instance classes.
CITYSCAPES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
CITYSCAPES_IDS = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 31: 5, 32: 6, 33: 7}
CITYSCAPES_REF = "markmoawad/cityscapes-trainval"


def _cityscapes_boxes(args: tuple[str, str]) -> tuple[str, list]:
    import numpy as np
    img, inst = args
    a = np.asarray(Image.open(inst)).astype(np.int32)
    boxes = []
    for v in np.unique(a[a >= 1000]):
        if int(v) // 1000 in CITYSCAPES_IDS:
            ys, xs = np.nonzero(a == v)
            boxes.append((CITYSCAPES_IDS[int(v) // 1000], int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return img, boxes


def prep_cityscapes(out: Path) -> Path:
    from multiprocessing.pool import ThreadPool  # PIL decoding and numpy release the GIL
    raw = fetch_kaggle(CITYSCAPES_REF, out.parent / "raw")
    manifest = {"source": f"kaggle:{CITYSCAPES_REF} (Cityscapes leftImg8bit + gtFine instanceIds)", "splits": {}}
    for split, src in {"train": "train", "valid": "val"}.items():
        imgs = sorted((raw / "leftImg8bit" / src).rglob("*_leftImg8bit.png"))
        jobs = [(str(p), str(next((raw / "gtFine" / src).rglob(p.name.replace("_leftImg8bit", "_gtFine_instanceIds")))))
                for p in imgs]
        with ThreadPool(os.cpu_count() or 2) as pool:
            done = pool.map(_cityscapes_boxes, jobs, chunksize=16)
        manifest["splits"][split] = write_split(out, split, CITYSCAPES, ((Path(i), b, []) for i, b in done))
    return finish(out, CITYSCAPES, manifest)


# --- ExDark (official train/val/test list; annotation CSV checked equal to the authors' ground truth, 23 710 boxes) --
EXDARK = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
EXDARK_REF = "hongngnguyn/exdarkdataset"
EXDARK_LIST = "https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/master/Groundtruth/imageclasslist.txt"


def prep_exdark(out: Path) -> Path:
    import csv
    raw = fetch_kaggle(EXDARK_REF, out.parent / "raw")
    lst = out.parent / "raw" / "exdark_imageclasslist.txt"  # the fetched directory may be read-only cache
    _download(EXDARK_LIST, lst)
    boxes: dict[str, list] = {}
    with open(raw / "annotations.csv") as f:
        for r in csv.DictReader(f):
            boxes.setdefault(r["image_path"].rsplit(".", 1)[0].lower(), []).append(
                (int(r["class"]), int(r["x_tl"]), int(r["y_tl"]), int(r["x_br"]), int(r["y_br"])))
    files = {p.stem.lower(): p for p in (raw / "Dataset").rglob("*") if p.is_file()}
    rows = [ln.split() for ln in lst.read_text().splitlines()[1:] if ln.strip()]
    manifest = {"source": f"kaggle:{EXDARK_REF} + official imageclasslist.txt (columns: class, light, in/out, split)",
                "splits": {}, "light_types": {r[0].rsplit(".", 1)[0].lower(): int(r[2]) for r in rows}}
    for split, code in {"train": "1", "valid": "2", "test": "3"}.items():
        stems = sorted(r[0].rsplit(".", 1)[0].lower() for r in rows if r[4] == code)
        manifest["splits"][split] = write_split(out, split, EXDARK, ((files[s], boxes.get(s, []), []) for s in stems))
    return finish(out, EXDARK, manifest)


# --- RUOD (official layout: RUOD_pic/{train,test}, RUOD_ANN COCO json; 9800 train / 4200 test) -----------------------
# The official test set stays closed: validation is a fixed random tenth of the official training set.
RUOD_REF = "abedhossain/ruod-himu"


def _coco_items(js: dict, img_dir: Path, keep: set[str] | None = None) -> tuple[list[str], list[Item]]:
    cats = sorted(js["categories"], key=lambda c: c["id"])
    idx = {c["id"]: i for i, c in enumerate(cats)}
    by: dict[int, list] = {}
    for a in js["annotations"]:
        x, y, w, h = a["bbox"]
        by.setdefault(a["image_id"], []).append((idx[a["category_id"]], x, y, x + w, y + h))
    items = [(img_dir / im["file_name"], by.get(im["id"], []), []) for im in sorted(js["images"], key=lambda i: i["file_name"])
             if keep is None or im["file_name"] in keep]
    return [c["name"] for c in cats], items


def prep_ruod(out: Path) -> Path:
    raw = fetch_kaggle(RUOD_REF, out.parent / "raw")
    tr = json.loads(next((raw / "RUOD_ANN").glob("*train*.json")).read_text())
    te = json.loads(next((raw / "RUOD_ANN").glob("*test*.json")).read_text())
    files = sorted(im["file_name"] for im in tr["images"])
    val = set(random.Random(0).sample(files, len(files) // 10))
    names, train_items = _coco_items(tr, raw / "RUOD_pic" / "train", set(files) - val)
    manifest = {"source": f"kaggle:{RUOD_REF} (official RUOD layout); valid = seed-0 random tenth of official train",
                "splits": {"train": write_split(out, "train", names, train_items),
                           "valid": write_split(out, "valid", names, _coco_items(tr, raw / "RUOD_pic" / "train", val)[1]),
                           "test": write_split(out, "test", names, _coco_items(te, raw / "RUOD_pic" / "test")[1])}}
    return finish(out, names, manifest)


# --- BDD100K as a real-shift benchmark (Diverse-Weather style: train on daytime+clear, test on other conditions) ----
# Labels: one record per image with attributes.timeofday / attributes.weather and labels[].box2d.
# shortcut: train capped at 12000 and each test set at 2000 images (seed 0) to fit a T4 screening run;
# lift BDD_CAP before any number goes in a paper.
BDD = ["bus", "bike", "car", "motor", "person", "rider", "truck"]
BDD_ALIAS = {"bicycle": "bike", "motorcycle": "motor", "pedestrian": "person"}
BDD_REF = "solesensei/solesensei_bdd100k"
BDD_CAP = {"train": 12000, "valid": 2000, "night": 2000, "nightrain": 2000, "duskrain": 2000, "dayrain": 2000}
# split -> (label files it may draw from, timeofday, weather); train draws only daytime+clear, so the other
# conditions of the official train file are unseen and can be test images
BDD_SPLITS = {"train": (("train",), "daytime", "clear"), "valid": (("val",), "daytime", "clear"),
              "night": (("val",), "night", "clear"), "nightrain": (("train", "val"), "night", "rainy"),
              "duskrain": (("train", "val"), "dawn/dusk", "rainy"), "dayrain": (("train", "val"), "daytime", "rainy")}


def _bdd_items(recs: dict[str, list], img_root: Path, split: str) -> list[Item]:
    files, tod, weather = BDD_SPLITS[split]
    items = []
    for f in files:
        for r in recs[f]:
            a = r.get("attributes") or {}
            if a.get("timeofday") != tod or a.get("weather") != weather:
                continue
            boxes = []
            for lab in r.get("labels") or []:
                c = BDD_ALIAS.get(lab["category"], lab["category"])
                if c in BDD and "box2d" in lab:
                    b = lab["box2d"]
                    boxes.append((BDD.index(c), b["x1"], b["y1"], b["x2"], b["y2"]))
            items.append((img_root / f / r["name"], boxes, []))
    items.sort(key=lambda i: i[0].name)
    return sorted(random.Random(0).sample(items, min(BDD_CAP[split], len(items))), key=lambda i: i[0].name)


def prep_bdd(out: Path) -> Path:
    raw = fetch_kaggle(BDD_REF, out.parent / "raw")
    keep = ("name", "attributes", "labels")  # the label files also carry lane and drivable-area polygons
    recs = {}
    for f in ("train", "val"):
        path = next(raw.rglob(f"bdd100k_labels_images_{f}.json"))
        recs[f] = [{k: r.get(k) for k in keep} for r in json.loads(path.read_text())]
    img_root = next(p for p in raw.rglob("100k") if (p / "train").is_dir() and p.parent.name == "images")
    manifest = {"source": f"kaggle:{BDD_REF}; train = daytime+clear of official train, tests = other conditions; "
                          f"caps {BDD_CAP} (seed 0)", "splits": {}}
    for split in BDD_SPLITS:
        items = [i for i in _bdd_items(recs, img_root, split) if i[0].exists()]
        manifest["splits"][split] = write_split(out, split, BDD, items)
    n = {k: v["images"] for k, v in manifest["splits"].items()}
    assert n["train"] >= 8000 and all(v >= 300 for v in n.values()), f"unexpected BDD100K split sizes {n}"
    return finish(out, BDD, manifest)


PREP: dict[str, Callable[[Path], Path]] = {"visdrone": prep_visdrone, "cityscapes": prep_cityscapes,
                                           "exdark": prep_exdark, "ruod": prep_ruod, "bdd": prep_bdd}


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
