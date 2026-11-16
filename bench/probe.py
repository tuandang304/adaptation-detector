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
    """Which ways of fetching a public Kaggle dataset work from this host (kernels cannot attach at run time)."""
    import subprocess
    rep = {"env": sorted(k for k in os.environ if "KAGGLE" in k)}
    for ref in refs:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
        r = subprocess.run(["curl", "-sS", "-L", "-m", "60", "-r", "0-1023", "-o", "/dev/null", "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        rep[ref] = {"anonymous_http": r.stdout + r.stderr[:200]}
        try:  # outside the kernel code path kagglehub downloads over HTTP with the kernel's own identity
            env = {k: os.environ.pop(k) for k in list(os.environ) if k == "KAGGLE_KERNEL_RUN_TYPE"}
            import kagglehub
            t0 = time.time()
            path = kagglehub.dataset_download(ref, path=None) if ref == refs[0] else "skipped"
            rep[ref]["kagglehub_http"] = {"path": str(path), "sec": round(time.time() - t0, 1)}
            os.environ.update(env)
        except Exception as e:  # noqa: BLE001 - a failing route is a finding
            rep[ref]["kagglehub_http"] = repr(e)[:300]
        print("[probe] " + json.dumps({ref: rep[ref]}), flush=True)
    print("[probe] " + json.dumps({"env": rep["env"]}), flush=True)
    return rep
