"""Model registry: download released weights once, verify SHA-256, cache locally."""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/LittlePeanutAA/eKYC/releases/download/v2.0.0"

# name -> (file, sha256)
MODELS = {
    "card_corners": ("card_corners_yolov8s.pt", "646265ebd28cc9dd54a7f5b426a4d9e311daafadae28f31254602f6b020c2b6e"),
    "card_fields": ("card_fields_yolov8s.pt", "5ba485b6bdf7c9228574130972914f19b6284ec014836e64e067b7a8ceff8752"),
    "document_keypoints": ("document_keypoints_yolov8n_pose.pt", "ba98b2a793f361cc97d3834c9890c2602a0a2fea509514b9da6f7e0348e53ff5"),
    "vietocr": ("vietocr_vgg_transformer.pth", "380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59"),
}


def cache_dir() -> Path:
    return Path(os.environ.get("EKYC_WEIGHTS_DIR", Path.home() / ".cache" / "ekyc"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_weights(name: str) -> str:
    """Return a local path to the named weights, downloading them from the GitHub release if needed."""
    file, sha = MODELS[name]
    path = cache_dir() / file
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {file} ...", file=sys.stderr)  # keep stdout clean for JSON output
    urllib.request.urlretrieve(f"{RELEASE}/{file}", tmp)
    if _sha256(tmp) != sha:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {file}; delete the cache and retry")
    tmp.rename(path)
    return str(path)
