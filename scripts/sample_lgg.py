"""
Sample 30 image+mask pairs from LGG Segmentation (Brain).

Usage:
    1. Download the dataset (Kaggle mirror: mateuszbuda/lgg-mri-segmentation)
       and extract it somewhere, e.g. E:/downloads/lgg-mri-segmentation.
    2. Set SOURCE_ROOT below to the "kaggle_3m" folder inside that extraction.
    3. Run: python scripts/sample_lgg.py
    4. Output lands in samples/lgg/ (images/, masks/, manifest_lgg.csv) --
       a small, portable folder. Upload samples/lgg/ to Drive, then delete
       the original full download to reclaim disk space.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common import sample_and_export

# EDIT THIS to wherever you extracted the LGG dataset
SOURCE_ROOT = Path("E:/downloads/lgg-mri-segmentation/kaggle_3m")

OUT_DIR = Path(__file__).parent.parent / "samples" / "lgg"
TARGET_NOUN = "the tumor"
DATASET_NAME = "lgg"


def build_pool(root: Path):
    pool = []
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        for mask_path in sorted(patient_dir.glob("*_mask.tif")):
            img_path = Path(str(mask_path).replace("_mask.tif", ".tif"))
            if not img_path.exists():
                continue
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            if mask.sum() == 0:
                continue  # skip slices with no visible tumor
            pool.append({"id": mask_path.stem.replace("_mask", ""), "image_path": img_path, "mask": mask})
    return pool


if __name__ == "__main__":
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"SOURCE_ROOT not found: {SOURCE_ROOT}\nEdit SOURCE_ROOT in this script first.")
    pool = build_pool(SOURCE_ROOT)
    print(f"[info] found {len(pool)} candidate image+mask pairs with visible tumor")
    sample_and_export(pool, TARGET_NOUN, DATASET_NAME, OUT_DIR)
