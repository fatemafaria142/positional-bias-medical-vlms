"""
Sample 50 image+mask pairs from Kvasir-SEG (Colon).

Usage (same steps whether you're on your own machine, a teammate's machine,
or Google Colab -- only the repo root's location changes, never these paths):
    1. Download Kvasir-SEG (datasets.simula.no/kvasir-seg) and extract it so
       that it sits at data/raw/Kvasir-SEG relative to the repo root (i.e.
       the "images/" and "masks/" folders end up directly inside
       <repo_root>/data/raw/Kvasir-SEG/).
    2. Run: python scripts/sample_kvasir.py   (from anywhere -- path is
       resolved relative to this script's own location, not your cwd)
    3. Output lands in <repo_root>/samples/kvasir/ (images/, masks/,
       manifest_kvasir.csv) -- a small, portable folder. Upload
       samples/kvasir/ to Drive, then delete data/raw/Kvasir-SEG to reclaim
       disk space.

On Colab: clone/mount the repo, then extract the downloaded dataset to
<repo_root>/data/raw/Kvasir-SEG exactly as above -- no path edits needed,
since everything below is relative to the repo, not to any one machine.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from common import sample_and_export

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "data" / "raw" / "Kvasir-SEG"
OUT_DIR = REPO_ROOT / "samples" / "kvasir"
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
