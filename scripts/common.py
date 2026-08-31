"""
Shared constants and ground-truth logic used by all 3 per-dataset sampling
scripts (sample_lgg.py, sample_kvasir.py, sample_qata.py). No dataset-specific
code lives here -- this file alone needs no downloaded data to import.

Overview of what this module does
----------------------------------
Each per-dataset script (LGG brain-MRI, Kvasir polyps, QaTa-COV19, etc.) loads
its own images + segmentation masks into a common shape -- a "pool" of
{id, image_path, mask} dicts -- and hands that pool to sample_and_export()
below. This file has no idea which dataset it's dealing with; it only knows
how to turn (image, mask) pairs into a small, portable, benchmark-ready folder.

The pipeline, end to end:
  1. mask_ground_truth() -- for one binary mask, compute the ground-truth
     location info: a centroid point, a bounding box, and left/right +
     upper/lower side labels.
  2. sample_and_export() -- for a whole dataset:
       a. Randomly pick exactly N_SAMPLES=50 items using a fixed SEED=42,
          so re-running the script always picks the same 50 items.
       b. Copy each chosen image and mask into out_dir/images and
          out_dir/masks (a small subset, safe to upload/share, instead of
          keeping the full original dataset download around).
       c. For every sampled item, format all 5 PROMPTS templates (a plain
          "locate X" version plus left/right/upper/lower directional
          versions) and write one CSV row per (item, prompt) combination --
          i.e. 50 items x 5 prompts = 250 rows -- into
          out_dir/manifest_<dataset_name>.csv, alongside the ground-truth
          centroid/bbox/side values.

The end result per dataset is a self-contained folder (images/, masks/,
manifest_<name>.csv) that a vision-language model can be prompted against,
with ground truth already computed for scoring the model's answers.
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
N_SAMPLES = 50

PROMPTS = {
    "P0": "Please locate {x} in this image.",
    "P1": "Please locate {x} on the left side of this image.",
    "P2": "Please locate {x} on the right side of this image.",
    "P3": "Please locate {x} in the upper region of this image.",
    "P4": "Please locate {x} in the lower region of this image.",
}


def mask_ground_truth(mask: np.ndarray):
    """Return (centroid_x, centroid_y, bbox, side_lr, side_ud) for a binary mask.

    Three ways of describing "where the object is", from most to least precise:
      - mask:     the input itself -- a pixel-level map where 1 = object, 0 = background.
                   This is the ground truth; bbox and point below are both derived from it.
      - point:    a single (x, y) coordinate summarizing the object's location --
                   here, the centroid (average position of all object pixels).
      - bbox:     the smallest rectangle (xmin, ymin, xmax, ymax) that contains
                   every object pixel -- coarser than the mask, but standard for
                   object-detection-style tasks ("draw a box around X").
    """
    # np.nonzero on a 2D array returns (row_indices, col_indices) = (ys, xs),
    # i.e. the coordinates of every foreground ("object") pixel in the mask.
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None  # empty mask -- nothing to localize

    # Point: centroid = mean position of all object pixels (center of mass).
    cx, cy = xs.mean(), ys.mean()

    # Bbox: tightest rectangle enclosing every object pixel.
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())

    # Which half of the image the object's centroid falls in --
    # used later to build directional prompts ("...on the left side").
    h, w = mask.shape
    side_lr = "left" if cx < w / 2 else "right"
    side_ud = "upper" if cy < h / 2 else "lower"
    return cx, cy, bbox, side_lr, side_ud


def sample_and_export(pool, target_noun, dataset_name, out_dir: Path):
    """Given a pool of {id, image_path, mask} dicts:
    1. Sample exactly N_SAMPLES with a fixed seed.
    2. Copy each selected image + mask into out_dir/images and out_dir/masks
       (small, portable -- safe to upload to Drive and delete the full download).
    3. Write out_dir/manifest_<dataset_name>.csv with ground truth + all 5 prompts.
    """
    import shutil

    if len(pool) < N_SAMPLES:
        print(f"[warn] only {len(pool)} valid pairs found, need {N_SAMPLES}")

    # random.Random(SEED) (not the global `random` module) gives a private,
    # deterministic RNG -- rng.sample() always returns the same 50 items for
    # the same pool, so results are reproducible across runs and machines.
    rng = random.Random(SEED)
    sample = rng.sample(pool, min(N_SAMPLES, len(pool)))

    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in sample:
        # Ground truth for this one item: centroid point, bbox, side labels
        # (see mask_ground_truth() above for what each of these means).
        gt = mask_ground_truth(item["mask"])
        if gt is None:
            continue  # empty mask -- skip, nothing to localize/report
        cx, cy, bbox, side_lr, side_ud = gt

        # Copy the original image file as-is (preserves its extension).
        src_img = item["image_path"]
        dst_img = img_out / src_img.name
        shutil.copy2(src_img, dst_img)

        # Save the mask as a clean PNG (small, lossless, consistent format).
        # mask is 0/1-valued; *255 makes it viewable as a normal black/white image.
        dst_mask = mask_out / f"{item['id']}_mask.png"
        Image.fromarray((item["mask"].astype("uint8")) * 255).save(dst_mask)

        # One CSV row per (item, prompt) pair -- so each image contributes
        # 5 rows, one for each of the plain/left/right/upper/lower prompt
        # variants, all sharing the same ground-truth values computed above.
        for prompt_id, template in PROMPTS.items():
            rows.append({
                "dataset": dataset_name,
                "sample_id": item["id"],
                "original_filename": src_img.name,
                "image_path": f"images/{dst_img.name}",
                "mask_path": f"masks/{dst_mask.name}",
                "gt_centroid_x": cx,
                "gt_centroid_y": cy,
                "gt_bbox_xmin": bbox[0],
                "gt_bbox_ymin": bbox[1],
                "gt_bbox_xmax": bbox[2],
                "gt_bbox_ymax": bbox[3],
                "gt_side_lr": side_lr,
                "gt_side_ud": side_ud,
                "prompt_id": prompt_id,
                "prompt_text": template.format(x=target_noun),
            })

    # Write every row collected above into a single manifest CSV --
    # this file is the ground-truth answer key for the exported images.
    manifest_path = out_dir / f"manifest_{dataset_name}.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] {dataset_name}: copied {len(sample)} image+mask pairs -> {out_dir}")
    print(f"[ok] {dataset_name}: wrote {len(rows)} prompt rows -> {manifest_path}")
    print(f"[done] {dataset_name}: this folder is now safe to upload to Drive; "
          f"you can delete the original full dataset download.")
