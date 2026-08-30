"""
Shared constants and ground-truth logic used by all 3 per-dataset sampling
scripts (sample_lgg.py, sample_kvasir.py, sample_qata.py). No dataset-specific
code lives here -- this file alone needs no downloaded data to import.
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

    rng = random.Random(SEED)
    sample = rng.sample(pool, min(N_SAMPLES, len(pool)))

    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in sample:
        gt = mask_ground_truth(item["mask"])
        if gt is None:
            continue
        cx, cy, bbox, side_lr, side_ud = gt

        # Copy the original image file as-is (preserves its extension).
        src_img = item["image_path"]
        dst_img = img_out / src_img.name
        shutil.copy2(src_img, dst_img)

        # Save the mask as a clean PNG (small, lossless, consistent format).
        dst_mask = mask_out / f"{item['id']}_mask.png"
        Image.fromarray((item["mask"].astype("uint8")) * 255).save(dst_mask)

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

    manifest_path = out_dir / f"manifest_{dataset_name}.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] {dataset_name}: copied {len(sample)} image+mask pairs -> {out_dir}")
    print(f"[ok] {dataset_name}: wrote {len(rows)} prompt rows -> {manifest_path}")
    print(f"[done] {dataset_name}: this folder is now safe to upload to Drive; "
          f"you can delete the original full dataset download.")
