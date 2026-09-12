"""
Build a multi-angle rotation sweep (0/30/60/90/120/150/180 degrees) of the
Text3DSAM bilateral-organ samples (samples/text3dsam/<dataset>/), for a
positional-bias-vs-rotation-angle study -- same idea as
scripts/rotate_multiangle_samples.py in the voxtell3d-rotation branch,
adapted to 3 Text3DSAM datasets -- MR_TotalSeg, CT_HaN-Seg,
CT_TotalSeg_cardiac -- each with 5 subjects x a left/right organ pair in
samples/text3dsam/manifest_text3dsam.csv. (Groups by whatever datasets are
actually present in that manifest, so trimming/adding datasets there is
all that's needed to change scope.)

Ground truth policy -- identical to the voxtell3d-rotation branch:
  - organ, label_id, prompt_text: IDENTICAL at every angle. Rotating the
    image doesn't change which organ the prompt is asking about.
  - gt_side_axis0/1/2 (TRUE anatomical L/R, A/P, S/I): also IDENTICAL at
    every angle, copied straight from the source (un-rotated) manifest
    row. A real right kidney is still the right kidney no matter how the
    volume is digitally rotated. This is only trustworthy at 0/90/180 if
    recomputed from the rotated array, so at oblique angles it is
    deliberately NOT recomputed.
  - gt_centroid_vox_*/gt_bbox_vox_*: recomputed fresh per angle from the
    rotated label mask, since the organ's voxels actually move on the grid.
  - a new rotation_angle_deg column records the applied angle.

Rotation implementation: scipy.ndimage.rotate within the plane of the two
non-S/I array axes (found per-volume from its own axcodes -- CT_* samples
here are SAR, MR_TotalSeg is ASR), reshape=False, order=1 (linear) for the
image and order=0 (nearest) for the label so interpolation never invents a
new label id. angle=0 is a plain copy. The NIfTI affine is saved unchanged.

Output layout (mirrors samples/text3dsam/<dataset>/):
    samples/text3dsam/<dataset>/volumes_rot/<subject>_rot{angle:03d}_{image,label}.nii.gz
    samples/text3dsam/manifest_text3dsam_rotated.csv  -- the ONE manifest,
        all 3 datasets x all subjects x all angles x both organs combined,
        same schema as the voxtell3d-rotation branch's per-dataset
        manifests plus rotation_angle_deg.

Usage:
    python scripts/rotate_multiangle_text3dsam.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import rotate as ndi_rotate

from common3d import mask_ground_truth_3d

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "samples" / "text3dsam"
BASE_MANIFEST = DATA_ROOT / "manifest_text3dsam.csv"

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]


def read_manifest_by_dataset_subject(manifest_path: Path):
    """Group manifest rows by (dataset, subject_id), preserving row order."""
    by_ds_subject = defaultdict(list)
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_ds_subject[(row["dataset"], row["subject_id"])].append(row)
    return by_ds_subject


def rotate_inplane(volume: np.ndarray, si_axis: int, angle_deg: float, order: int) -> np.ndarray:
    """Rotate `volume` by `angle_deg` within the plane of its two non-S/I
    axes, keeping the S/I axis fixed. reshape=False keeps the output the
    same shape as the input at every angle. angle=0 is returned unchanged
    (no interpolation loss for the baseline)."""
    if angle_deg == 0:
        return volume.copy()
    inplane_axes = tuple(a for a in range(3) if a != si_axis)
    return ndi_rotate(
        volume.astype(np.float64),
        angle_deg,
        axes=inplane_axes,
        reshape=False,
        order=order,
        mode="constant",
        cval=0.0,
    )


def process_subject(dataset_name: str, subject_id: str, rows: list, ds_dir: Path, vol_out: Path):
    img_path = ds_dir / rows[0]["image_path"].split("/", 1)[1]  # strip "<dataset>/" prefix
    lbl_path = ds_dir / rows[0]["label_path"].split("/", 1)[1]

    img_nii = nib.load(str(img_path))
    lbl_nii = nib.load(str(lbl_path))
    img_arr = np.asarray(img_nii.dataobj)
    lbl_arr = np.asarray(lbl_nii.dataobj).astype(np.int32)

    axcodes = nib.aff2axcodes(lbl_nii.affine)
    si_axis = next(i for i, code in enumerate(axcodes) if code in ("S", "I"))
    zooms = lbl_nii.header.get_zooms()

    out_rows = []
    for angle in ANGLES_DEG:
        rot_img = rotate_inplane(img_arr, si_axis, angle, order=1)
        if angle != 0:
            rot_img = rot_img.astype(img_arr.dtype)
        rot_lbl = np.rint(rotate_inplane(lbl_arr, si_axis, angle, order=0)).astype(np.int32)

        dst_image = vol_out / f"{subject_id}_rot{angle:03d}_image.nii.gz"
        dst_label = vol_out / f"{subject_id}_rot{angle:03d}_label.nii.gz"
        nib.save(nib.Nifti1Image(rot_img, img_nii.affine, img_nii.header), str(dst_image))
        nib.save(nib.Nifti1Image(rot_lbl, lbl_nii.affine, lbl_nii.header), str(dst_label))

        for r in rows:
            label_id = int(r["label_id"])
            gt = mask_ground_truth_3d(rot_lbl == label_id, axcodes)
            if gt is None:
                continue  # structure rotated fully out of frame (shouldn't happen near volume center)
            (cx, cy, cz), bbox, _rotated_side_ignored = gt

            out_rows.append({
                "dataset": dataset_name,
                "subject_id": subject_id,
                "split": r["split"],
                "organ": r["organ"],
                "label_id": label_id,
                "rotation_angle_deg": angle,
                "image_path": f"volumes_rot/{dst_image.name}",
                "label_path": f"volumes_rot/{dst_label.name}",
                "volume_shape": "x".join(str(s) for s in rot_lbl.shape),
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
                # TRUE anatomical side -- invariant to rotation, copied from
                # the source (un-rotated) manifest row, NOT recomputed here.
                "gt_side_axis0": r["gt_side_axis0"],
                "gt_side_axis1": r["gt_side_axis1"],
                "gt_side_axis2": r["gt_side_axis2"],
                "prompt_text": r["prompt_text"],
            })

    return out_rows


def main():
    by_ds_subject = read_manifest_by_dataset_subject(BASE_MANIFEST)
    datasets = sorted({ds for ds, _ in by_ds_subject.keys()})

    all_rows_combined = []
    for dataset_name in datasets:
        ds_dir = DATA_ROOT / dataset_name
        vol_out = ds_dir / "volumes_rot"
        vol_out.mkdir(parents=True, exist_ok=True)

        subject_ids = sorted({sid for ds, sid in by_ds_subject.keys() if ds == dataset_name})
        ds_rows = []
        for si, subject_id in enumerate(subject_ids, 1):
            rows = by_ds_subject[(dataset_name, subject_id)]
            out_rows = process_subject(dataset_name, subject_id, rows, ds_dir, vol_out)
            ds_rows.extend(out_rows)
            print(f"[{dataset_name}] [{si}/{len(subject_ids)}] {subject_id}: "
                  f"{len(ANGLES_DEG)} angles x {len(rows)} organ(s)")

        all_rows_combined.extend(ds_rows)

    combined_path = DATA_ROOT / "manifest_text3dsam_rotated.csv"
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows_combined[0].keys())
        writer.writeheader()
        writer.writerows(all_rows_combined)
    print(f"\n[done] wrote {len(all_rows_combined)} combined rows -> {combined_path}")
    print("Angles: " + ", ".join(str(a) for a in ANGLES_DEG) + " degrees.")
    print("gt_side_axis* is the TRUE anatomical side, carried over unchanged from each "
          "subject/organ's source row at every angle. gt_centroid_vox_*/gt_bbox_vox_* ARE "
          "recomputed per angle, since the organ's on-screen position does move.")


if __name__ == "__main__":
    main()
