"""Build dataset/acdc/ from the 20 downloaded ACDC training volumes.

Source: data/raw/acdc/<patient>/<patient>_frame01.nii.gz (+ _gt.nii.gz)
Downloaded from the public Human Heart Project Girder collection (ACDC, MICCAI 2017):
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb

Each ACDC ground-truth slice carries 3 distinct structures in one label map:
  1 = right ventricle (RV) cavity, 2 = myocardium, 3 = left ventricle (LV) cavity.
For each patient, one slice (chosen to have all 3 structures present, when possible) is
extracted as a 2D PNG, with one binary mask per structure. The manifest has one row per
(patient, structure, prompt) so a model's ability to find each of the 3 targets in the
same image can be checked separately -- not just whether it finds "a" cardiac target.

In addition to the per-region P0-P4 rows, one combined-prompt row (prompt_id "P_ALL") is
added per patient, asking the model to segment all present structures in a single call.
This tests a different thing than P0-P4: not "does it move correctly with positional
wording" (which only makes sense for one target at a time), but "can it attempt
multi-target output at all, and find all N targets in one shot." Positional variants
don't apply to this combined row (region/mask columns for a single target are left
blank; see target_regions/mask_paths instead).

Output layout:
  dataset/acdc/<patient>/<patient>.png
  dataset/acdc/<patient>/<patient>_mask_<structure>.png   (one per structure present)
  dataset/acdc/manifest_acdc.csv
"""
import csv
import os

import nibabel as nib
import numpy as np
from PIL import Image

RAW_DIR = os.path.join("data", "raw", "acdc")
OUT_DIR = os.path.join("dataset", "acdc")

STRUCTURES = {
    1: ("rv", "the right ventricle"),
    2: ("myocardium", "the myocardium"),
    3: ("lv", "the left ventricle"),
}

PROMPTS = {
    "P0": "Please segment {X} in this image.",
    "P1": "Please segment {X} on the left side of this image.",
    "P2": "Please segment {X} on the right side of this image.",
    "P3": "Please segment {X} in the upper region of this image.",
    "P4": "Please segment {X} in the lower region of this image.",
}


def join_natural(phrases):
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def combined_prompt_row(dataset, sample_id, original_filename, image_rel, width, height, targets):
    """targets: list of (slug, target_phrase, mask_rel)."""
    phrases = [t[1] for t in targets]
    prompt_text = f"Please segment {join_natural(phrases)} in this image."
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "region": "all",
        "original_filename": original_filename,
        "image_path": image_rel,
        "mask_path": "",
        "image_width": width,
        "image_height": height,
        "gt_centroid_x": "",
        "gt_centroid_y": "",
        "gt_bbox_xmin": "",
        "gt_bbox_ymin": "",
        "gt_bbox_xmax": "",
        "gt_bbox_ymax": "",
        "gt_side_lr": "",
        "gt_side_ud": "",
        "prompt_id": "P_ALL",
        "prompt_text": prompt_text,
        "target_regions": ";".join(t[0] for t in targets),
        "mask_paths": ";".join(t[2] for t in targets),
    }


def normalize_to_uint8(slice_2d):
    lo, hi = np.percentile(slice_2d, 1), np.percentile(slice_2d, 99)
    if hi <= lo:
        hi = slice_2d.max() if slice_2d.max() > 0 else 1
        lo = slice_2d.min()
    clipped = np.clip(slice_2d, lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6) * 255.0
    return scaled.astype(np.uint8)


def mask_ground_truth(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    cx, cy = xs.mean(), ys.mean()
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return cx, cy, bbox


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    patients = sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )

    rows = []
    skipped = []

    for pname in patients:
        pdir = os.path.join(RAW_DIR, pname)
        img_path = os.path.join(pdir, f"{pname}_frame01.nii.gz")
        gt_path = os.path.join(pdir, f"{pname}_frame01_gt.nii.gz")
        if not (os.path.exists(img_path) and os.path.exists(gt_path)):
            skipped.append((pname, "missing nii.gz"))
            continue

        img_vol = nib.load(img_path).get_fdata()
        gt_vol = nib.load(gt_path).get_fdata().astype(np.int16)

        z_mid = img_vol.shape[2] // 2
        z_order = sorted(range(img_vol.shape[2]), key=lambda z: abs(z - z_mid))
        # prefer a slice where all 3 structures are present; fall back to >=2, then >=1
        chosen_z = None
        for min_labels in (3, 2, 1):
            for z in z_order:
                labels_here = set(np.unique(gt_vol[:, :, z]).tolist()) & set(STRUCTURES)
                if len(labels_here) >= min_labels:
                    chosen_z = z
                    break
            if chosen_z is not None:
                break
        if chosen_z is None:
            skipped.append((pname, "no cardiac label in any slice"))
            continue

        img_slice = np.rot90(img_vol[:, :, chosen_z])
        gt_slice = np.rot90(gt_vol[:, :, chosen_z])

        height, width = img_slice.shape
        sample_id = pname
        patient_dir = os.path.join(OUT_DIR, sample_id)
        os.makedirs(patient_dir, exist_ok=True)
        image_rel = f"{sample_id}/{sample_id}.png"
        Image.fromarray(normalize_to_uint8(img_slice)).save(os.path.join(OUT_DIR, image_rel))

        present_labels = [lbl for lbl in STRUCTURES if np.any(gt_slice == lbl)]
        combined_targets = []

        for lbl in present_labels:
            slug, target_phrase = STRUCTURES[lbl]
            mask_bool = gt_slice == lbl
            cx, cy, bbox = mask_ground_truth(mask_bool)
            side_lr = "right" if cx > width / 2 else "left"
            side_ud = "upper" if cy < height / 2 else "lower"

            mask_rel = f"{sample_id}/{sample_id}_mask_{slug}.png"
            Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(os.path.join(OUT_DIR, mask_rel))
            combined_targets.append((slug, target_phrase, mask_rel))

            for pid, template in PROMPTS.items():
                rows.append(
                    {
                        "dataset": "acdc",
                        "sample_id": sample_id,
                        "region": slug,
                        "original_filename": f"{pname}_frame01.nii.gz (slice {chosen_z})",
                        "image_path": image_rel,
                        "mask_path": mask_rel,
                        "image_width": width,
                        "image_height": height,
                        "gt_centroid_x": round(float(cx), 2),
                        "gt_centroid_y": round(float(cy), 2),
                        "gt_bbox_xmin": bbox[0],
                        "gt_bbox_ymin": bbox[1],
                        "gt_bbox_xmax": bbox[2],
                        "gt_bbox_ymax": bbox[3],
                        "gt_side_lr": side_lr,
                        "gt_side_ud": side_ud,
                        "prompt_id": pid,
                        "prompt_text": template.format(X=target_phrase),
                        "target_regions": "",
                        "mask_paths": "",
                    }
                )

        if len(combined_targets) >= 2:
            rows.append(
                combined_prompt_row(
                    "acdc",
                    sample_id,
                    f"{pname}_frame01.nii.gz (slice {chosen_z})",
                    image_rel,
                    width,
                    height,
                    combined_targets,
                )
            )

    manifest_path = os.path.join(OUT_DIR, "manifest_acdc.csv")
    fieldnames = [
        "dataset",
        "sample_id",
        "region",
        "original_filename",
        "image_path",
        "mask_path",
        "image_width",
        "image_height",
        "gt_centroid_x",
        "gt_centroid_y",
        "gt_bbox_xmin",
        "gt_bbox_ymin",
        "gt_bbox_xmax",
        "gt_bbox_ymax",
        "gt_side_lr",
        "gt_side_ud",
        "prompt_id",
        "prompt_text",
        "target_regions",
        "mask_paths",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_samples = len({(r["sample_id"]) for r in rows})
    n_combined = sum(1 for r in rows if r["prompt_id"] == "P_ALL")
    n_isolated = len(rows) - n_combined
    n_region_groups = n_isolated // len(PROMPTS)
    print(
        f"Wrote {len(rows)} rows ({n_samples} images, {n_region_groups} region-groups x "
        f"{len(PROMPTS)} isolated prompts + {n_combined} combined P_ALL rows) to {manifest_path}"
    )
    if skipped:
        print("Skipped:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
