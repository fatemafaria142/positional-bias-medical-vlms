"""
Stage the first 50 TotalSegmentatorV2 subjects (by sorted subject ID, from the
small v2.0.1 Zenodo/Dropbox 102-subject release) from the full local extraction
at totalsegmentator_v2.0.1/ into data/totalsegmentator_50/ (gitignored).

Each subject in the source release stores one CT volume (ct.nii.gz) plus 117
separate per-structure binary masks under segmentations/<structure>.nii.gz --
unlike Neurite-OASIS's single multi-label volume. Only the bilateral organs
below are copied; everything else (unilateral organs, ribs, vertebrae, skull,
etc.) is left out entirely, both to keep staged size small and because this
project's RQs only run on bilateral (left/right) structures.

Ribs are deliberately excluded even though they are bilateral: 12 pairs behave
as near-identical mirror structures and would dilute any aggregate RQ
statistic across organs (see team discussion -- treat as a single "rib"
category or sample a few, rather than staging all 24 individually, if ribs
are ever added later).

Usage:
    python scripts/stage_totalsegmentator_50.py
"""

import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "totalsegmentator_v2.0.1"
OUT_ROOT = REPO_ROOT / "data" / "totalsegmentator_50"
N_SUBJECTS = 50

# Bilateral organ types only. Label file stems are exactly as they appear
# under <subject>/segmentations/ in the source release (note: "iliac_vena",
# not "iliac_vein", and "clavicula", not "clavicle" -- both match the actual
# file names, not the more common English spelling).
BILATERAL_ORGAN_STEMS = {
    "adrenal_gland": "adrenal gland",
    "autochthon": "autochthon",
    "clavicula": "clavicle",
    "common_carotid_artery": "common carotid artery",
    "brachiocephalic_vein": "brachiocephalic vein",
    "femur": "femur",
    "gluteus_maximus": "gluteus maximus",
    "gluteus_medius": "gluteus medius",
    "gluteus_minimus": "gluteus minimus",
    "hip": "hip",
    "humerus": "humerus",
    "iliac_artery": "iliac artery",
    "iliac_vena": "iliac vein",
    "iliopsoas": "iliopsoas",
    "kidney": "kidney",
    "kidney_cyst": "kidney cyst",
    "lung_lower_lobe": "lung lower lobe",
    "lung_upper_lobe": "lung upper lobe",
    "scapula": "scapula",
    "subclavian_artery": "subclavian artery",
}
# lung_middle_lobe is excluded here -- it only exists as *_right, no left
# counterpart, so it is not a bilateral pair under this project's design.
# Ribs are excluded per the note above.

FIELDNAMES = [
    "dataset", "subject_id", "organ", "side",
    "image_path", "label_path", "volume_shape", "voxel_spacing_mm", "axcodes",
    "gt_centroid_vox_0", "gt_centroid_vox_1", "gt_centroid_vox_2",
    "gt_bbox_vox_0min", "gt_bbox_vox_1min", "gt_bbox_vox_2min",
    "gt_bbox_vox_0max", "gt_bbox_vox_1max", "gt_bbox_vox_2max",
    "prompt_text",
]


def build_prompt(side: str, organ: str) -> str:
    s = f"Segment the {side} {organ} in this image."
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
    src_image = src_dir / "ct.nii.gz"
    src_seg_dir = src_dir / "segmentations"

    dest_dir = OUT_ROOT / "volumes" / subject_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_image = dest_dir / "ct.nii.gz"
    shutil.copy2(src_image, dest_image)

    img_nii = nib.load(str(dest_image))
    shape = img_nii.shape
    zooms = img_nii.header.get_zooms()[:3]
    axcodes_str = "".join(nib.aff2axcodes(img_nii.affine))

    rows = []
    for stem, organ_name in BILATERAL_ORGAN_STEMS.items():
        for side in ("left", "right"):
            src_label = src_seg_dir / f"{stem}_{side}.nii.gz"
            if not src_label.exists():
                continue

            mask_nii = nib.load(str(src_label))
            mask = np.asarray(mask_nii.dataobj) > 0
            cb = centroid_and_bbox(mask)
            if cb is None:
                continue
            centroid, bbox = cb

            dest_label = dest_dir / f"{stem}_{side}.nii.gz"
            shutil.copy2(src_label, dest_label)

            rows.append({
                "dataset": "totalsegmentator_v2", "subject_id": subject_id,
                "organ": organ_name, "side": side,
                "image_path": f"volumes/{subject_id}/ct.nii.gz",
                "label_path": f"volumes/{subject_id}/{stem}_{side}.nii.gz",
                "volume_shape": "x".join(str(s) for s in shape),
                "voxel_spacing_mm": "x".join(f"{z:.3f}" for z in zooms),
                "axcodes": axcodes_str,
                "gt_centroid_vox_0": centroid[0], "gt_centroid_vox_1": centroid[1], "gt_centroid_vox_2": centroid[2],
                "gt_bbox_vox_0min": bbox[0], "gt_bbox_vox_1min": bbox[1], "gt_bbox_vox_2min": bbox[2],
                "gt_bbox_vox_0max": bbox[3], "gt_bbox_vox_1max": bbox[4], "gt_bbox_vox_2max": bbox[5],
                "prompt_text": build_prompt(side, organ_name),
            })
    return rows


def main():
    subject_ids = sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir() and p.name.startswith("s"))[:N_SUBJECTS]
    print(f"[plan] staging {len(subject_ids)} subjects: {subject_ids[0]} .. {subject_ids[-1]}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest_totalsegmentator_50.csv"
    all_rows = []
    for i, subject_id in enumerate(subject_ids, 1):
        rows = stage_subject(subject_id)
        all_rows.extend(rows)
        print(f"[{i}/{len(subject_ids)}] {subject_id}: {len(rows)} organ/side rows")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[done] staged {len(subject_ids)} subjects, {len(all_rows)} manifest rows -> {manifest_path}")


if __name__ == "__main__":
    main()
