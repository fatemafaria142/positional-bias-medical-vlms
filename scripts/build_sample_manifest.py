"""
Builds the fixed 30-samples-per-dataset manifest for the positional-bias study.

Covers the 3 easiest-to-work-with 2D datasets: Brain (LGG), Colon (Kvasir-SEG),
Chest (QaTa-COV19) -- chosen for no registration requirement, simple flat
folder structures, and extensive documentation. Ground truth (centroid, bbox,
left/right, upper/lower) always comes from each dataset's mask, never from
text, so LGG works identically to the other 2 despite having no native text
of its own.

For each of the 3 datasets, this script:
  1. Collects every (image, mask) pair that has a non-empty mask (target visible).
  2. Fixes a random seed and samples exactly 30 pairs.
  3. Derives ground truth from each mask: centroid, bbox, left/right of midline,
     upper/lower of midheight.
  4. Emits one row per (image, prompt) -- 30 * 5 = 150 rows per dataset -- into
     manifest.csv, using the exact P0-P4 strings from the plan.

Fill in the ROOT paths below to match where you actually downloaded each dataset.
Folder layouts are written to match each dataset's documented release structure,
but you MUST verify against what you actually have on disk before trusting output.
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
N_SAMPLES = 30
OUT_PATH = Path("manifest.csv")

PROMPTS = {
    "P0": "Please locate {x} in this image.",
    "P1": "Please locate {x} on the left side of this image.",
    "P2": "Please locate {x} on the right side of this image.",
    "P3": "Please locate {x} in the upper region of this image.",
    "P4": "Please locate {x} in the lower region of this image.",
}

# --------------------------------------------------------------------------
# EDIT THESE to your actual local download paths
# --------------------------------------------------------------------------
ROOTS = {
    "lgg": Path("data/lgg-mri-segmentation/kaggle_3m"),
    "kvasir": Path("data/Kvasir-SEG"),
    "qata": Path("data/QaTa-COV19/Train_Folder"),  # pool from one or all splits
}

TARGET_NOUN = {
    "lgg": "the tumor",
    "kvasir": "the polyp",
    "qata": "the infection",
}


def mask_ground_truth(mask: np.ndarray):
    """Return (centroid_x, centroid_y, bbox, side_lr, side_ud) for a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx, cy = xs.mean(), ys.mean()
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    h, w = mask.shape
    side_lr = "left" if cx < w / 2 else "right"
    side_ud = "upper" if cy < h / 2 else "lower"
    return cx, cy, bbox, side_lr, side_ud


# --------------------------------------------------------------------------
# Per-dataset pool builders -> list of dicts: {id, image_path, mask_array}
# --------------------------------------------------------------------------

def pool_lgg(root: Path):
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
                continue
            pool.append({"id": mask_path.stem, "image_path": img_path, "mask": mask})
    return pool


def pool_kvasir(root: Path):
    pool = []
    img_dir, mask_dir = root / "images", root / "masks"
    for img_path in sorted(img_dir.glob("*.jpg")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask.sum() == 0:
            continue
        pool.append({"id": img_path.stem, "image_path": img_path, "mask": mask})
    return pool


def pool_qata(root: Path):
    pool = []
    img_dir, mask_dir = root / "img", root / "labelcol"
    for img_path in sorted(img_dir.glob("*")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask.sum() == 0:
            continue
        pool.append({"id": img_path.stem, "image_path": img_path, "mask": mask})
    return pool


POOL_BUILDERS = {
    "lgg": pool_lgg,
    "kvasir": pool_kvasir,
    "qata": pool_qata,
}


def build_manifest():
    rng = random.Random(SEED)
    rows = []
    for dataset, root in ROOTS.items():
        if not root.exists():
            print(f"[skip] {dataset}: path not found -> {root}")
            continue
        pool = POOL_BUILDERS[dataset](root)
        if len(pool) < N_SAMPLES:
            print(f"[warn] {dataset}: only {len(pool)} valid pairs found, need {N_SAMPLES}")
        sample = rng.sample(pool, min(N_SAMPLES, len(pool)))
        noun = TARGET_NOUN[dataset]
        for item in sample:
            gt = mask_ground_truth(item["mask"])
            if gt is None:
                continue
            cx, cy, bbox, side_lr, side_ud = gt
            for prompt_id, template in PROMPTS.items():
                rows.append({
                    "dataset": dataset,
                    "sample_id": item["id"],
                    "image_path": str(item["image_path"]),
                    "gt_centroid_x": cx,
                    "gt_centroid_y": cy,
                    "gt_bbox_xmin": bbox[0],
                    "gt_bbox_ymin": bbox[1],
                    "gt_bbox_xmax": bbox[2],
                    "gt_bbox_ymax": bbox[3],
                    "gt_side_lr": side_lr,
                    "gt_side_ud": side_ud,
                    "prompt_id": prompt_id,
                    "prompt_text": template.format(x=noun),
                })
        print(f"[ok] {dataset}: sampled {len(sample)} images -> {len(sample) * 5} prompt rows")

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    build_manifest()
