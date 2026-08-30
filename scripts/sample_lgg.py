"""
Sample 50 image+mask pairs from LGG Segmentation (Brain).

Usage (same steps whether you're on your own machine, a teammate's machine,
or Google Colab -- only the repo root's location changes, never these paths):
    1. Download the dataset (Kaggle mirror: mateuszbuda/lgg-mri-segmentation)
       and extract it so the "kaggle_3m" folder ends up at
       data/raw/lgg-mri-segmentation/kaggle_3m relative to the repo root.
    2. Run: python scripts/sample_lgg.py   (from anywhere -- path is
       resolved relative to this script's own location, not your cwd)
    3. Output lands in <repo_root>/samples/lgg/ (images/, masks/,
       manifest_lgg.csv) -- a small, portable folder. Upload samples/lgg/
       to Drive, then delete data/raw/lgg-mri-segmentation to reclaim disk
       space.

On Colab: clone/mount the repo, then extract the downloaded dataset to
<repo_root>/data/raw/lgg-mri-segmentation exactly as above -- no path edits
needed, since everything below is relative to the repo, not to any one
machine.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common import sample_and_export

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "data" / "raw" / "lgg-mri-segmentation" / "kaggle_3m"
OUT_DIR = REPO_ROOT / "samples" / "lgg"
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
