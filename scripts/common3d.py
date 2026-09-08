"""
Shared logic for turning 3D (volume, multi-organ label map) datasets into the
same kind of positional-bias manifest the 2D scripts (sample_lgg.py, etc.)
produce -- reused by sample_amos22_mri.py and (later) sample_neurite_oasis.py.

Why this differs from scripts/common.py (2D, single-target datasets):
  - A 3D volume has no single "the image" to prompt against -- a model can
    only be shown one 2D slice at a time. So for each organ/structure we
    pick that organ's *best* axial slice (the one with the most labeled
    voxels) and treat that slice as if it were a standalone 2D sample.
  - One subject contributes multiple labels (e.g. AMOS22 MRI has up to 15
    organs per volume), so the "pool" here is (subject, organ) pairs, not
    one item per subject.

Pipeline, end to end:
  1. best_slice_for_label() -- for one label id in one label volume, find
     the axial slice index with the most voxels of that label, and return
     the binary 2D mask at that slice.
  2. normalize_to_uint8() -- percentile-clip + rescale a raw MRI slice to a
     viewable 8-bit grayscale image (raw intensities are not 0-255).
  3. mask_ground_truth() -- identical to scripts/common.py: centroid, bbox,
     left/right + upper/lower side labels, but computed in image-space
     (array) coordinates -- this is what's being tested for positional
     bias, not true anatomical laterality.
  4. export_subject() -- for one subject's (image_vol, label_vol): for every
     organ id present, save the chosen slice's image+mask as PNGs and
     append one manifest row per (organ, prompt) combination.
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
N_SUBJECTS = 10

PROMPTS = {
    "P0": "Please locate {x} in this image.",
    "P1": "Please locate {x} on the left side of this image.",
    "P2": "Please locate {x} on the right side of this image.",
    "P3": "Please locate {x} in the upper region of this image.",
    "P4": "Please locate {x} in the lower region of this image.",
}


def best_slice_for_label(label_vol: np.ndarray, label_id: int, axis: int = 2):
    """Return (slice_index, mask2d) for the axial slice with the most voxels
    of `label_id`, or None if the label never appears in this volume.

    axis=2 assumes the volume is (X, Y, Z) with Z as the slice axis -- true
    for both AMOS22 MRI and Neurite OASIS as distributed (their Z spacing
    is coarser than X/Y, i.e. Z is the through-plane/slice axis).
    """
    is_label = label_vol == label_id
    other_axes = tuple(a for a in range(label_vol.ndim) if a != axis)
    counts_per_slice = is_label.sum(axis=other_axes)
    if counts_per_slice.max() == 0:
        return None  # label not present anywhere in this volume
    best_idx = int(counts_per_slice.argmax())
    mask2d = np.take(is_label, best_idx, axis=axis)
    return best_idx, mask2d


def normalize_to_uint8(slice2d: np.ndarray) -> np.ndarray:
    """Percentile-clip (1st/99th) + rescale a raw intensity slice to uint8
    so it's viewable as a normal grayscale PNG."""
    lo, hi = np.percentile(slice2d, [1, 99])
    if hi <= lo:
        return np.zeros_like(slice2d, dtype=np.uint8)
    clipped = np.clip(slice2d, lo, hi)
    return (((clipped - lo) / (hi - lo)) * 255).astype(np.uint8)


def mask_ground_truth(mask: np.ndarray):
    """Same contract as scripts/common.py's mask_ground_truth: given a 2D
    binary mask, return (centroid_x, centroid_y, bbox, side_lr, side_ud)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx, cy = xs.mean(), ys.mean()
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    h, w = mask.shape
    side_lr = "left" if cx < w / 2 else "right"
    side_ud = "upper" if cy < h / 2 else "lower"
    return cx, cy, bbox, side_lr, side_ud


def export_subject(
    subject_id: str,
    image_vol: np.ndarray,
    label_vol: np.ndarray,
    organ_map: dict,
    out_dir: Path,
    dataset_name: str,
    split: str,
    axis: int = 2,
):
    """For one subject: for every organ id in `organ_map` present in
    label_vol, save that organ's best slice (image + mask PNG) and return
    one manifest row per (organ, prompt) combination."""
    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_id, organ_name in organ_map.items():
        found = best_slice_for_label(label_vol, label_id, axis=axis)
        if found is None:
            continue  # this organ isn't annotated in this subject's volume
        slice_idx, mask2d = found

        gt = mask_ground_truth(mask2d)
        if gt is None:
            continue
        cx, cy, bbox, side_lr, side_ud = gt

        img2d = np.take(image_vol, slice_idx, axis=axis)
        img_uint8 = normalize_to_uint8(img2d)

        organ_slug = organ_name.replace(" ", "_").replace("/", "_")
        base_name = f"{subject_id}_{organ_slug}"
        dst_img = img_out / f"{base_name}.png"
        dst_mask = mask_out / f"{base_name}_mask.png"
        Image.fromarray(img_uint8).save(dst_img)
        Image.fromarray((mask2d.astype("uint8")) * 255).save(dst_mask)

        for prompt_id, template in PROMPTS.items():
            rows.append({
                "dataset": dataset_name,
                "subject_id": subject_id,
                "split": split,
                "organ": organ_name,
                "label_id": label_id,
                "slice_index": slice_idx,
                "volume_shape": "x".join(str(s) for s in label_vol.shape),
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
                "prompt_text": template.format(x=f"the {organ_name}"),
            })
    return rows


def write_manifest(rows: list, out_dir: Path, dataset_name: str):
    manifest_path = out_dir / f"manifest_{dataset_name}.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {dataset_name}: wrote {len(rows)} prompt rows -> {manifest_path}")
    return manifest_path
