"""Build dataset/chaos/ from the CHAOS Train_Sets MR (T1DUAL) data.

Source (public, no login, Zenodo mirror of the CHAOS grand-challenge dataset):
  https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip

The CT arm of CHAOS only has a liver mask (1 region) -- not enough to test whether a
model can find multiple distinct targets in one image. The MR arm's T1DUAL sequence
instead encodes 4 organs in one 8-bit grayscale label map per slice:
  63 = liver, 126 = right kidney, 189 = left kidney, 252 = spleen (0 = background).
There are exactly 20 MR patients in Train_Sets, so all 20 are used. For each patient,
the InPhase DICOM slice with the most distinct organ labels present is chosen (ties
broken by total organ pixel area), and one binary mask per organ actually present in
that slice is extracted -- so patients whose scan range doesn't cover all 4 organs still
contribute whichever organs are visible, rather than being dropped.

In addition, one combined-prompt row (prompt_id "P_ALL") is added per patient, asking
the model to segment all present organs in a single call -- testing whether it can
attempt multi-target output at all, separately from the per-organ P0-P4 rows.

Output layout:
  dataset/chaos/<patient_id>/<patient_id>.png
  dataset/chaos/<patient_id>/<patient_id>_mask_<organ>.png   (one per organ present)
  dataset/chaos/manifest_chaos.csv
"""
import csv
import os
import zipfile

import numpy as np
import pydicom
from PIL import Image

ZIP_PATH = os.path.join("data", "raw", "chaos", "Train_Sets.zip")
OUT_DIR = os.path.join("dataset", "chaos")

LABELS = {
    63: ("liver", "the liver"),
    126: ("right_kidney", "the right kidney"),
    189: ("left_kidney", "the left kidney"),
    252: ("spleen", "the spleen"),
}

PROMPTS = {
    "P0": "Please segment {X} in this image.",
    "P1": "Please segment {X} on the left side of this image.",
    "P2": "Please segment {X} on the right side of this image.",
    "P3": "Please segment {X} in the upper region of this image.",
    "P4": "Please segment {X} in the lower region of this image.",
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


def nearest_label(value):
    if value == 0:
        return None
    return min(LABELS, key=lambda lbl: abs(lbl - value))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    zf = zipfile.ZipFile(ZIP_PATH)
    names = zf.namelist()

    patients = sorted(
        set(
            n.split("/")[2]
            for n in names
            if n.startswith("Train_Sets/MR/") and len(n.split("/")) > 2 and n.split("/")[2]
        ),
        key=lambda x: int(x),
    )

    rows = []
    skipped = []

    for pid in patients:
        base = f"Train_Sets/MR/{pid}/T1DUAL"
        dicom_names = sorted(
            n for n in names if n.startswith(f"{base}/DICOM_anon/InPhase/") and n.endswith(".dcm")
        )
        ground_by_stem = {
            os.path.splitext(os.path.basename(n))[0]: n
            for n in names
            if n.startswith(f"{base}/Ground/") and n.endswith(".png")
        }
        if not dicom_names or not ground_by_stem:
            skipped.append((pid, "missing DICOM or Ground files"))
            continue

        best = None  # (n_labels, area, dicom_name, ground_name, label_masks)
        for dpath in dicom_names:
            stem = os.path.splitext(os.path.basename(dpath))[0]
            gpath = ground_by_stem.get(stem)
            if gpath is None:
                continue
            with zf.open(gpath) as f:
                raw = np.array(Image.open(f).convert("L"))
            present = {}
            for v in np.unique(raw):
                lbl = nearest_label(int(v))
                if lbl is not None:
                    present[lbl] = raw == v
            if not present:
                continue
            area = sum(m.sum() for m in present.values())
            score = (len(present), area)
            if best is None or score > best[0]:
                best = (score, dpath, gpath, present)

        if best is None:
            skipped.append((pid, "no organ label in any InPhase slice"))
            continue

        _, dpath, gpath, present = best
        with zf.open(dpath) as f:
            ds = pydicom.dcmread(f)
            img_arr = ds.pixel_array.astype(np.float32)
        height, width = img_arr.shape

        sample_id = f"patient{int(pid):03d}"
        patient_dir = os.path.join(OUT_DIR, sample_id)
        os.makedirs(patient_dir, exist_ok=True)
        image_rel = f"{sample_id}/{sample_id}.png"
        Image.fromarray(normalize_to_uint8(img_arr)).save(os.path.join(OUT_DIR, image_rel))

        combined_targets = []
        for lbl, mask_bool in present.items():
            slug, target_phrase = LABELS[lbl]
            cx, cy, bbox = mask_ground_truth(mask_bool)
            side_lr = "right" if cx > width / 2 else "left"
            side_ud = "upper" if cy < height / 2 else "lower"

            mask_rel = f"{sample_id}/{sample_id}_mask_{slug}.png"
            Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(os.path.join(OUT_DIR, mask_rel))
            combined_targets.append((slug, target_phrase, mask_rel))

            for pid_, template in PROMPTS.items():
                rows.append(
                    {
                        "dataset": "chaos",
                        "sample_id": sample_id,
                        "region": slug,
                        "original_filename": f"MR/{pid}/T1DUAL/{os.path.basename(dpath)}",
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
                        "prompt_id": pid_,
                        "prompt_text": template.format(X=target_phrase),
                        "target_regions": "",
                        "mask_paths": "",
                    }
                )

        if len(combined_targets) >= 2:
            rows.append(
                combined_prompt_row(
                    "chaos",
                    sample_id,
                    f"MR/{pid}/T1DUAL/{os.path.basename(dpath)}",
                    image_rel,
                    width,
                    height,
                    combined_targets,
                )
            )

    manifest_path = os.path.join(OUT_DIR, "manifest_chaos.csv")
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

    n_samples = len({r["sample_id"] for r in rows})
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
            print(f"  patient {name}: {reason}")


if __name__ == "__main__":
    main()
