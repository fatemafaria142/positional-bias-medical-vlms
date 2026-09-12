"""
Build a multi-angle rotation sweep (0/30/60/90/120/150/180 degrees) of 6
subjects per dataset (samples/amos22_mri, samples/neurite_oasis), split
across three annotator folders -- faria/, nipu/, hridoy/ -- 2 subjects each,
for a positional-bias-vs-rotation-angle study.

Difference from scripts/rotate_samples.py (a single, EXACT 180-degree flip)
and scripts/mirror_samples.py (a single L/R reflection): this script covers
the full angle sweep using continuous (interpolated) in-plane rotation about
each volume's own Superior/Inferior axis, read per-volume from its own NIfTI
affine (never assumed), and fans the subjects out across three annotators
instead of writing one shared samples/sample_rotation/ tree.

Ground truth policy -- this is the key difference from rotate_samples.py:
  - organ, label_id, prompt_text: IDENTICAL at every angle. Rotating the
    image doesn't change which organ the prompt is asking about.
  - gt_side_axis0/1/2 (TRUE anatomical L/R, A/P, S/I): also IDENTICAL at
    every angle, copied straight from the source (un-rotated) manifest row.
    A real right kidney is still the right kidney no matter how the volume
    is digitally rotated -- the patient hasn't changed. Recomputing this
    heuristic from the rotated array (as rotate_samples.py's 180-only case
    does) only stays valid at 0/90/180, since at oblique angles the array
    axes no longer line up with true L/R/A/P/S/I directions at all -- so
    here it is deliberately NOT recomputed.
  - gt_centroid_vox_*/gt_bbox_vox_*: recomputed fresh per angle from the
    rotated label mask. This DOES change with angle, because the organ's
    voxels have actually moved on the grid -- this is what lets a downstream
    analysis check whether the model's prediction tracks the (constant)
    prompt or drifts with the (changing) on-screen position.
  - a new rotation_angle_deg column records the applied angle.

Rotation implementation: scipy.ndimage.rotate within the plane of the two
non-S/I array axes, reshape=False (volume shape stays constant across every
angle for a subject), order=1 (linear) for the image and order=0 (nearest)
for the label so interpolation never invents a new label id. angle=0 is a
plain copy -- no interpolation loss for the baseline. The NIfTI affine is
saved unchanged (as in mirror_samples.py): oblique rotation has no simple
corrective affine, and -- per the policy above -- it no longer drives any
ground-truth computation anyway, so it stays purely informational.

Output layout (one tree per annotator, mirroring samples/<dataset>/):
    faria/samples/amos22_mri/{volumes/, manifest_amos22_mri.csv}
    faria/samples/neurite_oasis/{volumes/, manifest_neurite_oasis.csv}
    nipu/samples/...
    hridoy/samples/...

Usage:
    python scripts/rotate_multiangle_samples.py
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import rotate as ndi_rotate

from common3d import mask_ground_truth_3d, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = REPO_ROOT / "samples"

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]
ANNOTATORS = ["faria", "nipu", "hridoy"]  # first 2 subjects -> faria, next 2 -> nipu, last 2 -> hridoy

DATASETS = [
    {
        "name": "amos22_mri",
        "src_dir": SAMPLES_ROOT / "amos22_mri",
        "manifest": SAMPLES_ROOT / "amos22_mri" / "manifest_amos22_mri.csv",
    },
    {
        "name": "neurite_oasis",
        "src_dir": SAMPLES_ROOT / "neurite_oasis",
        "manifest": SAMPLES_ROOT / "neurite_oasis" / "manifest_neurite_oasis.csv",
    },
]


def read_manifest_by_subject(manifest_path: Path):
    """Group manifest rows by subject_id, preserving row order per subject."""
    by_subject = {}
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_subject.setdefault(row["subject_id"], []).append(row)
    return by_subject


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


def process_subject(subject_id: str, rows: list, src_dir: Path, vol_out: Path, dataset_name: str):
    """Rotate one subject's volume through every angle in ANGLES_DEG and
    return the manifest rows for all (angle, organ) combinations."""
    img_path = src_dir / rows[0]["image_path"]
    lbl_path = src_dir / rows[0]["label_path"]

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
                "image_path": f"volumes/{dst_image.name}",
                "label_path": f"volumes/{dst_label.name}",
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


def process_dataset(name: str, src_dir: Path, manifest_path: Path):
    by_subject = read_manifest_by_subject(manifest_path)
    subject_ids = list(by_subject.keys())
    if len(subject_ids) < 6:
        raise ValueError(f"[{name}] need at least 6 subjects, found {len(subject_ids)}")

    pairs = {
        "faria": subject_ids[0:2],
        "nipu": subject_ids[2:4],
        "hridoy": subject_ids[4:6],
    }

    for person, sids in pairs.items():
        out_dir = REPO_ROOT / person / "samples" / name
        vol_out = out_dir / "volumes"
        vol_out.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for si, subject_id in enumerate(sids, 1):
            rows = process_subject(subject_id, by_subject[subject_id], src_dir, vol_out, name)
            all_rows.extend(rows)
            print(f"[{name}/{person}] [{si}/{len(sids)}] {subject_id}: "
                  f"{len(ANGLES_DEG)} angles x {len(by_subject[subject_id])} organ(s)")

        write_manifest(all_rows, out_dir, name)


if __name__ == "__main__":
    for ds in DATASETS:
        if not ds["manifest"].exists():
            print(f"[skip] {ds['name']}: manifest not found at {ds['manifest']}")
            continue
        process_dataset(ds["name"], ds["src_dir"], ds["manifest"])

    print("\n[done] Multi-angle rotated samples written under "
          + ", ".join(f"{p}/samples/" for p in ANNOTATORS))
    print("Angles: " + ", ".join(str(a) for a in ANGLES_DEG) + " degrees.")
    print("gt_side_axis* is the TRUE anatomical side, carried over unchanged from each "
          "subject/organ's source row at every angle -- rotating the image doesn't change "
          "the patient. gt_centroid_vox_*/gt_bbox_vox_* ARE recomputed per angle, since the "
          "organ's on-screen position does move.")
