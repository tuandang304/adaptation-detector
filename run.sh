#!/usr/bin/env bash
# Fixed run contract: install pinned deps (quietly), fetch weights, run the configured stream.
set -euo pipefail
export GITHUB_ACTIONS=1 PYTHONUNBUFFERED=1 YOLO_VERBOSE=False
pip install -q --no-input --disable-pip-version-check ultralytics==8.4.200 imagecorruptions==1.1.2 pycocotools nvidia-ml-py 2>&1 | grep -viE "warning|notice|dependency resolver|incompatible" || true
W=$(python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['model'])")
[ -f "$W" ] || curl -sSfL --retry 3 -o "$W" "https://github.com/ultralytics/assets/releases/download/v8.4.0/$W"
# per-node shard settings (stream, method subset, ...) live in a committed file so every run is reproducible from its commit
[ -f shard.env ] && { set -a; . ./shard.env; set +a; }
python ctta.py --config config.yaml
