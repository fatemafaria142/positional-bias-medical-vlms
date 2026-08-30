"""
Sample 50 image+mask pairs from ISIC 2018 (Skin), Task 1 Validation split.

Usage (same steps whether you're on your own machine, a teammate's machine,
or Google Colab -- only the repo root's location changes, never these paths):
    1. Download the ISIC 2018 Task 1 Validation Input and GroundTruth zips
       from https://challenge.isic-archive.com/data/ (direct download, no
       login needed) and extract so you end up with:
         data/raw/ISIC2018/images/ISIC_XXXXXXX.jpg
         data/raw/ISIC2018/masks/ISIC_XXXXXXX_segmentation.png
       relative to the repo root. The Validation split (100 images, ~240MB)
       is used instead of the full Training split (2,594 images, ~11GB) --
       plenty for 50 samples, far smaller to download.
    2. Run: python scripts/sample_isic.py   (from anywhere -- path is
       resolved relative to this script's own location, not your cwd)
    3. Output lands in <repo_root>/samples/isic/ (images/, masks/,
       manifest_isic.csv) -- a small, portable folder. Upload samples/isic/
       to Drive, then delete data/raw/ISIC2018 to reclaim disk space.

On Colab: clone/mount the repo, then extract the downloaded dataset to
<repo_root>/data/raw/ISIC2018 exactly as above -- no path edits needed,
since everything below is relative to the repo, not to any one machine.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common import sample_and_export

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "data" / "raw" / "ISIC2018"
OUT_DIR = REPO_ROOT / "samples" / "isic"
TARGET_NOUN = "the lesion"
DATASET_NAME = "isic"


def build_pool(root: Path):
    pool = []
    img_dir, mask_dir = root / "images", root / "masks"
    for img_path in sorted(img_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{img_path.stem}_segmentation.png"
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask.sum() == 0:
            continue  # skip images with no visible lesion
        pool.append({"id": img_path.stem, "image_path": img_path, "mask": mask})
    return pool


if __name__ == "__main__":
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"SOURCE_ROOT not found: {SOURCE_ROOT}\nEdit SOURCE_ROOT in this script first.")
    pool = build_pool(SOURCE_ROOT)
    print(f"[info] found {len(pool)} candidate image+mask pairs with visible lesion")
    sample_and_export(pool, TARGET_NOUN, DATASET_NAME, OUT_DIR)
