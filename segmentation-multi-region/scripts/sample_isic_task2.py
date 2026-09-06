"""Build dataset/isic_task2/ from the ISIC 2018 Task 2 (Lesion Attribute Detection) data.

Source (direct, anonymous download, no login):
  Images:       https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip
  Ground truth: https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task2_Validation_GroundTruth.zip

Task 2 provides 5 dermoscopic attribute masks per lesion image (pigment network,
negative network, streaks, milia-like cyst, globules), but coverage per image is sparse
and uneven (checked over the 100 validation images: pigment_network non-empty in 73/100,
globules 19/100, negative_network 9/100, streaks 6/100, milia_like_cyst 6/100 -- and only
26/100 images have 2+ non-empty attributes at all). To test whether a model can find more
than one distinct region in the same image, this script selects the 20 images with the
most non-empty attributes (all with >=2) and writes one region-group per attribute that
is actually present, each with its own P0-P4 prompt set.

In addition, one combined-prompt row (prompt_id "P_ALL") is added per image, asking the
model to segment all present attributes in a single call -- testing whether it can
attempt multi-target output at all, separately from the per-attribute P0-P4 rows.

Output layout: dataset/isic_task2/<sample_id>/<sample_id>.png,
dataset/isic_task2/<sample_id>/<sample_id>_mask_<attribute>.png (one per attribute present),
dataset/isic_task2/manifest_isic_task2.csv.
"""
import csv
import os
import zipfile

import numpy as np
from PIL import Image

RAW_DIR = os.path.join("data", "raw", "isic_task2")
INPUT_ZIP = os.path.join(RAW_DIR, "input.zip")
GT_DIR = os.path.join(RAW_DIR, "gt", "ISIC2018_Task2_Validation_GroundTruth")
OUT_DIR = os.path.join("dataset", "isic_task2")

ATTRIBUTES = {
    "pigment_network": "the pigment network pattern",
    "negative_network": "the negative network pattern",
    "streaks": "the streaks",
    "milia_like_cyst": "the milia-like cysts",
    "globules": "the globules",
}
N_SAMPLES = 20

PROMPTS = {
    "P0": "Please segment {X} in this image.",
    "P1": "Please segment {X} on the left side of this image.",
    "P2": "Please segment {X} on the right side of this image.",
    "P3": "Please segment {X} in the upper region of this image.",
    "P4": "Please segment {X} in the lower region of this image.",
}


def mask_ground_truth(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    cx, cy = xs.mean(), ys.mean()
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return cx, cy, bbox


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_files = [f for f in os.listdir(GT_DIR) if f.endswith(".png")]
    image_ids = sorted(set(f.split("_attribute_")[0] for f in all_files))

    non_empty_attrs_per_image = {}
    for img_id in image_ids:
        present = []
        for attr in ATTRIBUTES:
            p = os.path.join(GT_DIR, f"{img_id}_attribute_{attr}.png")
            if os.path.exists(p) and np.array(Image.open(p)).max() > 0:
                present.append(attr)
        if len(present) >= 2:
            non_empty_attrs_per_image[img_id] = present

    # most-regions-first, then by id for reproducibility
    chosen_ids = sorted(
        non_empty_attrs_per_image, key=lambda k: (-len(non_empty_attrs_per_image[k]), k)
    )[:N_SAMPLES]

    zf = zipfile.ZipFile(INPUT_ZIP)
    zip_prefix = "ISIC2018_Task1-2_Validation_Input/"

    rows = []
    for img_id in chosen_ids:
        sample_id = img_id
        patient_dir = os.path.join(OUT_DIR, sample_id)
        os.makedirs(patient_dir, exist_ok=True)

        with zf.open(f"{zip_prefix}{img_id}.jpg") as src:
            img = Image.open(src).convert("RGB")
            img.load()

        width, height = img.size
        image_rel = f"{sample_id}/{sample_id}.png"
        img.save(os.path.join(OUT_DIR, image_rel))

        combined_targets = []
        for attr in non_empty_attrs_per_image[img_id]:
            target_phrase = ATTRIBUTES[attr]
            mask_img = Image.open(os.path.join(GT_DIR, f"{img_id}_attribute_{attr}.png")).convert("L")
            mask_bool = np.array(mask_img) > 0
            cx, cy, bbox = mask_ground_truth(mask_bool)
            side_lr = "right" if cx > width / 2 else "left"
            side_ud = "upper" if cy < height / 2 else "lower"

            mask_rel = f"{sample_id}/{sample_id}_mask_{attr}.png"
            Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(os.path.join(OUT_DIR, mask_rel))
            combined_targets.append((attr, target_phrase, mask_rel))

            for pid, template in PROMPTS.items():
                rows.append(
                    {
                        "dataset": "isic_task2",
                        "sample_id": sample_id,
                        "region": attr,
                        "original_filename": f"{img_id}.jpg",
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
                    "isic_task2",
                    sample_id,
                    f"{img_id}.jpg",
                    image_rel,
                    width,
                    height,
                    combined_targets,
                )
            )

    manifest_path = os.path.join(OUT_DIR, "manifest_isic_task2.csv")
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

    n_combined = sum(1 for r in rows if r["prompt_id"] == "P_ALL")
    n_isolated = len(rows) - n_combined
    n_region_groups = n_isolated // len(PROMPTS)
    print(
        f"Wrote {len(rows)} rows ({len(chosen_ids)} images, {n_region_groups} region-groups x "
        f"{len(PROMPTS)} isolated prompts + {n_combined} combined P_ALL rows) to {manifest_path}"
    )


if __name__ == "__main__":
    main()
