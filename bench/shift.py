"""Shifted copies of a validation split for screening: four fixed photometric degradations, same images and labels.

shortcut: hand-written degradations at one strength, enough to rank variants; use the real shifted sets
(Foggy Cityscapes, RUOD environment sets) and the standard corruption suite before any number goes in a paper.
"""
from __future__ import annotations

import shutil
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np


def _dark(x, rng):
    return 255 * 0.5 * (x / 255) ** 2.2


def _haze(x, rng):
    return 0.4 * x + 0.6 * 0.85 * 255


def _noise(x, rng):
    return x + rng.normal(0, 0.08 * 255, x.shape)


def _blur(x, rng):
    return cv2.GaussianBlur(x, (0, 0), 2.5)


KINDS = {"dark": _dark, "haze": _haze, "noise": _noise, "blur": _blur}


def _one(args) -> None:
    src, dst, kind, seed = args
    x = cv2.imread(str(src)).astype(np.float32)
    y = np.clip(KINDS[kind](x, np.random.default_rng(seed)), 0, 255).astype(np.uint8)
    assert cv2.imwrite(str(dst), y), dst


def make(root: Path, split: str = "valid") -> list[str]:
    """Write <split>_<kind>/images plus the matching ground-truth files; returns the new split names."""
    names = sorted(p.name for p in (root / split / "images").iterdir())
    out = []
    for kind in KINDS:
        s = f"{split}_{kind}"
        out.append(s)
        if (root / f"coco_{s}.json").exists():
            continue
        (root / s / "images").mkdir(parents=True, exist_ok=True)
        with ThreadPool(4) as pool:
            pool.map(_one, [(root / split / "images" / n, root / s / "images" / n, kind, i) for i, n in enumerate(names)])
        shutil.copy(root / f"ignore_{split}.json", root / f"ignore_{s}.json")
        shutil.copy(root / f"coco_{split}.json", root / f"coco_{s}.json")  # written last: marks the split complete
    return out
