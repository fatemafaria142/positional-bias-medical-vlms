"""
Stage the first 100 Neurite-OASIS subjects (by sorted subject ID) from the full
local release at neurite-oasis.v1.0/ into data/neurite_oasis_100/ (gitignored),
as the basis for a fresh Hugging Face dataset.

Image and label volumes are copied byte-for-byte from aligned_norm.nii.gz /
aligned_seg35.nii.gz -- no relabeling, no reprocessing. Label IDs are used
exactly as they already exist in the files (compact 1-35 scheme, 0 = background).

Usage:
    python scripts/stage_neurite_oasis_100.py
"""

import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "neurite-oasis.v1.0"
OUT_ROOT = REPO_ROOT / "data" / "neurite_oasis_100"
N_SUBJECTS = 100

# Label IDs exactly as they appear in aligned_seg35.nii.gz (compact 1-35 scheme).
OASIS_LABEL_MAP = {
    1: "left cerebral white matter", 2: "left cerebral cortex", 3: "left lateral ventricle",
    4: "left inferior lateral ventricle", 5: "left cerebellum white matter",
    6: "left cerebellum cortex", 7: "left thalamus", 8: "left caudate", 9: "left putamen",
    10: "left pallidum", 11: "3rd ventricle", 12: "4th ventricle", 13: "brain stem",
    14: "left hippocampus", 15: "left amygdala", 16: "left accumbens", 17: "left ventral DC",
    18: "left vessel", 19: "left choroid plexus", 20: "right cerebral white matter",
    21: "right cerebral cortex", 22: "right lateral ventricle",
    23: "right inferior lateral ventricle", 24: "right cerebellum white matter",
    25: "right cerebellum cortex", 26: "right thalamus", 27: "right caudate",
    28: "right putamen", 29: "right pallidum", 30: "right hippocampus", 31: "right amygdala",
    32: "right accumbens", 33: "right ventral DC", 34: "right vessel", 35: "right choroid plexus",
}

FIELDNAMES = [
    "dataset", "subject_id", "organ", "label_id",
    "image_path", "label_path", "volume_shape", "voxel_spacing_mm", "axcodes",
    "gt_centroid_vox_0", "gt_centroid_vox_1", "gt_centroid_vox_2",
    "gt_bbox_vox_0min", "gt_bbox_vox_1min", "gt_bbox_vox_2min",
    "gt_bbox_vox_0max", "gt_bbox_vox_1max", "gt_bbox_vox_2max",
    "prompt_text",
]


def build_prompt(organ: str) -> str:
    parts = organ.split()
    if parts[0].lower() in ("left", "right"):
        s = f"Segment the {parts[0].lower()} {' '.join(parts[1:])} in this image."
    else:
        s = f"Segment the {organ} in this image."
    return s[0].upper() + s[1:]


def centroid_and_bbox(mask: np.ndarray):
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        return None
    centroid = [float(c.mean()) for c in coords]
    bbox = [int(c.min()) for c in coords] + [int(c.max()) for c in coords]
    return centroid, bbox


def stage_subject(subject_id: str) -> list[dict]:
    src_dir = RAW_ROOT / subject_id
    src_image = src_dir / "aligned_norm.nii.gz"
    src_label = src_dir / "aligned_seg35.nii.gz"

    dest_dir = OUT_ROOT / "volumes"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_image = dest_dir / f"{subject_id}_image.nii.gz"
    dest_label = dest_dir / f"{subject_id}_label.nii.gz"

    # Byte-for-byte copy -- no modification of the label (or image) data.
    shutil.copy2(src_image, dest_image)
    shutil.copy2(src_label, dest_label)

    img_nii = nib.load(str(dest_image))
    label_nii = nib.load(str(dest_label))
    label_data = np.asarray(label_nii.dataobj)
    shape = label_data.shape
    zooms = img_nii.header.get_zooms()[:3]
    axcodes_str = "".join(nib.aff2axcodes(img_nii.affine))

    present_labels = set(np.unique(label_data).tolist()) - {0}
    rows = []
    for label_id, organ in OASIS_LABEL_MAP.items():
        if label_id not in present_labels:
            continue
        mask = label_data == label_id
        cb = centroid_and_bbox(mask)
        if cb is None:
            continue
        centroid, bbox = cb
        rows.append({
            "dataset": "neurite_oasis", "subject_id": subject_id, "organ": organ,
            "label_id": label_id,
            "image_path": f"volumes/{subject_id}_image.nii.gz",
            "label_path": f"volumes/{subject_id}_label.nii.gz",
            "volume_shape": "x".join(str(s) for s in shape),
            "voxel_spacing_mm": "x".join(f"{z:.3f}" for z in zooms),
            "axcodes": axcodes_str,
            "gt_centroid_vox_0": centroid[0], "gt_centroid_vox_1": centroid[1], "gt_centroid_vox_2": centroid[2],
            "gt_bbox_vox_0min": bbox[0], "gt_bbox_vox_1min": bbox[1], "gt_bbox_vox_2min": bbox[2],
            "gt_bbox_vox_0max": bbox[3], "gt_bbox_vox_1max": bbox[4], "gt_bbox_vox_2max": bbox[5],
            "prompt_text": build_prompt(organ),
        })
    return rows


def main():
    subject_ids = sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir() and p.name.startswith("OASIS_"))[:N_SUBJECTS]
    print(f"[plan] staging {len(subject_ids)} subjects: {subject_ids[0]} .. {subject_ids[-1]}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest_neurite_oasis_100.csv"
    all_rows = []
    for i, subject_id in enumerate(subject_ids, 1):
        rows = stage_subject(subject_id)
        all_rows.extend(rows)
        print(f"[{i}/{len(subject_ids)}] {subject_id}: {len(rows)} organs")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[done] staged {len(subject_ids)} subjects, {len(all_rows)} manifest rows -> {manifest_path}")


if __name__ == "__main__":
    main()
