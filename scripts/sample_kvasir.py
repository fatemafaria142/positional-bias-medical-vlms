"""
Sample 30 image+mask pairs from Kvasir-SEG (Colon).

Usage:
    1. Download Kvasir-SEG (datasets.simula.no/kvasir-seg) and extract it,
       e.g. E:/downloads/Kvasir-SEG.
    2. Set SOURCE_ROOT below to that extracted folder (it should directly
       contain "images/" and "masks/" subfolders).
    3. Run: python scripts/sample_kvasir.py
    4. Output lands in samples/kvasir/ (images/, masks/, manifest_kvasir.csv) --
       a small, portable folder. Upload samples/kvasir/ to Drive, then delete
       the original full download to reclaim disk space.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common import sample_and_export

# EDIT THIS to wherever you extracted Kvasir-SEG
SOURCE_ROOT = Path("E:/downloads/Kvasir-SEG")

OUT_DIR = Path(__file__).parent.parent / "samples" / "kvasir"
TARGET_NOUN = "the polyp"
DATASET_NAME = "kvasir"


def build_pool(root: Path):
    pool = []
    img_dir, mask_dir = root / "images", root / "masks"
    for img_path in sorted(img_dir.glob("*.jpg")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask.sum() == 0:
            continue  # skip images with no visible polyp
        pool.append({"id": img_path.stem, "image_path": img_path, "mask": mask})
    return pool


if __name__ == "__main__":
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"SOURCE_ROOT not found: {SOURCE_ROOT}\nEdit SOURCE_ROOT in this script first.")
    pool = build_pool(SOURCE_ROOT)
    print(f"[info] found {len(pool)} candidate image+mask pairs with visible polyp")
    sample_and_export(pool, TARGET_NOUN, DATASET_NAME, OUT_DIR)
