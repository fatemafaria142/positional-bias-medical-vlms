"""
Add more left/right organ pairs to samples/text3dsam/manifest_text3dsam_rotated.csv,
mined from label maps that are ALREADY downloaded (no new data needed) -- each
subject's label.nii.gz is a full multi-organ map, and rotate_multiangle_text3dsam.py
already rotated the WHOLE label volume (every organ id) through all 7 angles, so
new organ rows can be computed straight from the existing volumes_rot/ files.

For each (dataset, subject, new organ pair) where both sides are actually present:
  - gt_side_axis0/1/2: computed ONCE from the angle-0 label (source of truth,
    unrotated), then copied unchanged to every angle -- same policy as
    rotate_multiangle_text3dsam.py (true anatomical side never changes just
    because the image was digitally rotated).
  - gt_centroid_vox_*/gt_bbox_vox_*: recomputed per angle from that angle's
    already-rotated label mask.

Appends to the existing manifest; does not touch the rows already there.

Usage:
    python scripts/add_extra_organs_text3dsam.py
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from common3d import mask_ground_truth_3d

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "samples" / "text3dsam"
MANIFEST_PATH = DATA_ROOT / "manifest_text3dsam_rotated.csv"

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]

# (left_label_id, right_label_id, organ_name) -- organ_name gets "left "/"right " prefixed.
EXTRA_ORGANS = {
    "CT_HaN-Seg": [
        (18, 19, "submandibular gland"),
        (27, 28, "parotid gland"),
        (1, 2, "carotid artery"),
    ],
    "MR_TotalSeg": [
        (9, 8, "adrenal gland"),
        (36, 37, "hip"),
        (26, 27, "iliac artery"),
    ],
    "CT_TotalSeg_cardiac": [
        (17, 18, "iliac vena"),
        (6, 5, "subclavian artery"),
        (8, 7, "common carotid artery"),
        (9, 10, "brachiocephalic vein"),
    ],
}


def subjects_for(dataset_name: str):
    vol_dir = DATA_ROOT / dataset_name / "volumes"
    return sorted(p.name[: -len("_label.nii.gz")] for p in vol_dir.glob("*_label.nii.gz"))


def main():
    new_rows = []
    for dataset_name, organ_specs in EXTRA_ORGANS.items():
        ds_dir = DATA_ROOT / dataset_name
        rot_dir = ds_dir / "volumes_rot"

        for subject_id in subjects_for(dataset_name):
            base_label_path = rot_dir / f"{subject_id}_rot000_label.nii.gz"
            base_img_nii = nib.load(str(rot_dir / f"{subject_id}_rot000_image.nii.gz"))
            base_label_arr = np.asarray(nib.load(str(base_label_path)).dataobj).astype(np.int32)
            axcodes = nib.aff2axcodes(base_img_nii.affine)
            zooms = base_img_nii.header.get_zooms()
            uniq = set(np.unique(base_label_arr).tolist())

            for left_id, right_id, organ_base in organ_specs:
                if left_id not in uniq or right_id not in uniq:
                    continue  # this organ isn't annotated for this subject

                fixed_sides = {}
                for side_label_id, side_name in [(left_id, "left"), (right_id, "right")]:
                    gt0 = mask_ground_truth_3d(base_label_arr == side_label_id, axcodes)
                    if gt0 is None:
                        continue
                    side0, side1, _side2_geometric = gt0[2]
                    # axis2 is this data's true L/R axis for every subject checked, but
                    # mask_ground_truth_3d's "centroid vs array-center" heuristic is
                    # unreliable on tightly-cropped ROI volumes (common here) whenever
                    # the crop isn't centered on the true body midline -- e.g. it
                    # mislabeled MR_totalseg_mr_s0171's right iliac artery as "L".
                    # The organ id itself already tells us the true side with
                    # certainty (that's how CVPR-BiomedSegFM's class scheme works),
                    # so axis2 is taken directly from side_name instead of computed.
                    side2 = "L" if side_name == "left" else "R"
                    fixed_sides[side_label_id] = (side0, side1, side2)  # fixed across angles

                for angle in ANGLES_DEG:
                    lbl_path = rot_dir / f"{subject_id}_rot{angle:03d}_label.nii.gz"
                    lbl_arr = np.asarray(nib.load(str(lbl_path)).dataobj).astype(np.int32)

                    for side_label_id, side_name in [(left_id, "left"), (right_id, "right")]:
                        if side_label_id not in fixed_sides:
                            continue
                        gt = mask_ground_truth_3d(lbl_arr == side_label_id, axcodes)
                        if gt is None:
                            continue  # rotated fully out of frame at this angle
                        (cx, cy, cz), bbox, _ignored_rotated_side = gt
                        organ_name = f"{side_name} {organ_base}"
                        sides = fixed_sides[side_label_id]

                        new_rows.append({
                            "dataset": dataset_name,
                            "subject_id": subject_id,
                            "split": "train",
                            "organ": organ_name,
                            "label_id": side_label_id,
                            "rotation_angle_deg": angle,
                            "image_path": f"volumes_rot/{subject_id}_rot{angle:03d}_image.nii.gz",
                            "label_path": f"volumes_rot/{subject_id}_rot{angle:03d}_label.nii.gz",
                            "volume_shape": "x".join(str(s) for s in lbl_arr.shape),
                            "voxel_spacing_mm": "x".join(f"{z:.3f}" for z in zooms),
                            "axcodes": "".join(axcodes),
                            "gt_centroid_vox_0": cx,
                            "gt_centroid_vox_1": cy,
                            "gt_centroid_vox_2": cz,
                            "gt_bbox_vox_0min": bbox[0],
                            "gt_bbox_vox_1min": bbox[1],
                            "gt_bbox_vox_2min": bbox[2],
                            "gt_bbox_vox_0max": bbox[3],
                            "gt_bbox_vox_1max": bbox[4],
                            "gt_bbox_vox_2max": bbox[5],
                            "gt_side_axis0": sides[0],
                            "gt_side_axis1": sides[1],
                            "gt_side_axis2": sides[2],
                            "prompt_text": f"Segment the {organ_name} in this image.",
                        })
            print(f"[{dataset_name}] {subject_id}: done")

    with MANIFEST_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing_rows = list(reader)

    all_rows = existing_rows + [{k: r[k] for k in fieldnames} for r in new_rows]
    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[done] added {len(new_rows)} rows -> {MANIFEST_PATH} (total {len(all_rows)})")


if __name__ == "__main__":
    main()
