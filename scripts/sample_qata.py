"""
Sample 30 image+mask pairs from QaTa-COV19 (Chest).

Usage:
    1. Download QaTa-COV19 (kaggle.com/datasets/aysendegerli/qatacov19-dataset,
       or the LViT release which also has the native .xlsx text) and extract
       it, e.g. E:/downloads/QaTa-COV19.
    2. Set SOURCE_ROOT below to the split folder you want to sample from (it
       should directly contain "img/" and "labelcol/" subfolders) -- e.g.
       .../QaTa-COV19/Train_Folder.
    3. Run: python scripts/sample_qata.py
    4. Output lands in samples/qata/ (images/, masks/, manifest_qata.csv) --
       a small, portable folder. Upload samples/qata/ to Drive, then delete
       the original full download to reclaim disk space.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common import sample_and_export

# EDIT THIS to wherever you extracted QaTa-COV19 (point at one split's folder)
SOURCE_ROOT = Path("E:/downloads/QaTa-COV19/Train_Folder")

OUT_DIR = Path(__file__).parent.parent / "samples" / "qata"
TARGET_NOUN = "the infection"
DATASET_NAME = "qata"


def build_pool(root: Path):
    pool = []
    img_dir, mask_dir = root / "img", root / "labelcol"
    for img_path in sorted(img_dir.glob("*")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask.sum() == 0:
            continue  # skip images with no visible infection
        pool.append({"id": img_path.stem, "image_path": img_path, "mask": mask})
    return pool


if __name__ == "__main__":
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"SOURCE_ROOT not found: {SOURCE_ROOT}\nEdit SOURCE_ROOT in this script first.")
    pool = build_pool(SOURCE_ROOT)
    print(f"[info] found {len(pool)} candidate image+mask pairs with visible infection")
    sample_and_export(pool, TARGET_NOUN, DATASET_NAME, OUT_DIR)
