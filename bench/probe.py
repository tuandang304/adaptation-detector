"""Print the directory structure of candidate dataset copies, so converters are written against what is really there."""
from __future__ import annotations

import collections
import json
import os
import time
from pathlib import Path


def tree(root: Path, depth: int = 4) -> dict:
    hist: collections.Counter = collections.Counter()
    odd: list[str] = []
    for dp, _, fns in os.walk(root):
        rel = Path(dp).relative_to(root).parts[:depth]
        for fn in fns:
            ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
            hist["/".join(rel) + " *." + ext] += 1
            if ext not in ("png", "jpg", "jpeg", "txt", "xml") and len(odd) < 40:
                odd.append(str(Path(dp).relative_to(root) / fn))
    return {"dirs": dict(sorted(hist.items())), "other_files": odd}


def sample(root: Path, exts: tuple[str, ...], n: int = 2) -> dict:
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().endswith(exts) and len(out) < n * len(exts):
                p = Path(dp) / fn
                key = str(p.relative_to(root))
                if sum(k.rsplit(".", 1)[-1] == fn.rsplit(".", 1)[-1] for k in out) < n:
                    out[key] = p.read_text(errors="replace")[:600]
    return out


def run(refs: list[str]) -> dict:
    import kagglehub
    rep = {}
    for ref in refs:
        t0 = time.time()
        try:
            root = Path(kagglehub.dataset_download(ref))
            t = tree(root)
            rep[ref] = {"root": str(root), "sec": round(time.time() - t0, 1), "files": sum(t["dirs"].values()), **t,
                        "samples": sample(root, (".txt", ".xml", ".json", ".yaml", ".csv"))}
        except Exception as e:  # noqa: BLE001 - a missing copy is a finding, not a failure
            rep[ref] = {"error": repr(e)[:400], "sec": round(time.time() - t0, 1)}
        print("[probe] " + json.dumps({ref: rep[ref]})[:6000], flush=True)
    return rep
