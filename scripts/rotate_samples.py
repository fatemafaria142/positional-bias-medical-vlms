"""
Build a rotated copy of the existing 3D positional-bias samples
(samples/amos22_mri, samples/neurite_oasis) under samples/sample_rotation/,
mirroring the exact same file layout and manifest schema -- so the existing
VoxTell notebooks (voxtell-amos-kaggle.ipynb, voxtell-oasis-kaggle.ipynb) can
run unchanged against the rotated data by only swapping one path:

    AMOS_DIR = RESEARCH_REPO / 'samples' / 'sample_rotation' / 'amos22_mri'

Why rotate (not just flip): each volume is rotated 180 degrees about its own
Superior/Inferior axis (read from that volume's own NIfTI affine, same as
common3d.axcodes_for -- never assumed). A 180-degree rotation about the
vertical body axis:
  - swaps Left <-> Right and Anterior <-> Posterior, while leaving
    Superior/Inferior unchanged -- exactly the axes a "left kidney" /
    "right hippocampus" style prompt cares about.
  - is a proper rotation (determinant +1), not a mirror reflection --
    physically the same as the patient being rotated 180 degrees on the
    table, so it stays an anatomically valid volume (unlike a left-right
    flip, which would mirror-image the anatomy).
  - is exact on the voxel grid (implemented as flip() along the two
    non-S/I array axes, no interpolation), so image and label masks stay
    pixel-perfect aligned and no resampling artifacts are introduced.

Ground truth (centroid, bbox, L/R/A/P/S/I side per axis) is recomputed from
scratch against the rotated array + its correctly-updated affine, reusing
common3d.mask_ground_truth_3d exactly as the original sampling scripts do --
so gt_side_axis* in the rotated manifest reflects the *rotated* anatomy, not
just a copy of the original labels.

Usage:
    python scripts/rotate_samples.py
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from common3d import mask_ground_truth_3d, write_manifest
import csv

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = REPO_ROOT / "samples"
OUT_ROOT = SAMPLES_ROOT / "sample_rotation"

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


def rotate_180_about_si(volume: np.ndarray, affine: np.ndarray, axcodes: tuple):
    """Rotate a volume + its affine 180 degrees about the array axis that
    corresponds to Superior/Inferior, per this volume's own axcodes.

    Implemented as an exact flip() along the two non-S/I axes (voxel-grid
    aligned, no interpolation) with the affine updated to match, so
    nib.aff2axcodes() on the returned affine is still correct for the
    rotated array.
    """
    si_axis = next(i for i, code in enumerate(axcodes) if code in ("S", "I"))
    flip_axes = [i for i in range(3) if i != si_axis]

    rotated = np.flip(volume, axis=flip_axes[0])
    rotated = np.flip(rotated, axis=flip_axes[1])
    rotated = np.ascontiguousarray(rotated)

    # new_index -> old_index: old_index[a] = (shape[a] - 1) - new_index[a]
    # for a in flip_axes, old_index[si_axis] = new_index[si_axis] unchanged.
    # M maps [new_index, 1] -> [old_index, 1]; new_affine = old_affine @ M.
    shape = volume.shape
    M = np.eye(4)
    for a in flip_axes:
        M[a, a] = -1
        M[a, 3] = shape[a] - 1
    new_affine = affine @ M
    return rotated, new_affine


def process_dataset(name: str, src_dir: Path, manifest_path: Path):
    out_dir = OUT_ROOT / name
    vol_out = out_dir / "volumes"
    vol_out.mkdir(parents=True, exist_ok=True)

    by_subject = read_manifest_by_subject(manifest_path)
    print(f"[{name}] {len(by_subject)} subject(s) to rotate")

    all_rows = []
    for si, (subject_id, rows) in enumerate(by_subject.items(), 1):
        img_path = src_dir / rows[0]["image_path"]
        lbl_path = src_dir / rows[0]["label_path"]

        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))

        img_arr = np.asarray(img_nii.dataobj)
        lbl_arr = np.asarray(lbl_nii.dataobj).astype(np.int32)

        axcodes = nib.aff2axcodes(lbl_nii.affine)

        rot_img, new_img_affine = rotate_180_about_si(img_arr, img_nii.affine, axcodes)
        rot_lbl, new_lbl_affine = rotate_180_about_si(lbl_arr, lbl_nii.affine, axcodes)
        new_axcodes = nib.aff2axcodes(new_lbl_affine)

        dst_image = vol_out / f"{subject_id}_image.nii.gz"
        dst_label = vol_out / f"{subject_id}_label.nii.gz"
        nib.save(nib.Nifti1Image(rot_img, new_img_affine, img_nii.header), str(dst_image))
        nib.save(nib.Nifti1Image(rot_lbl.astype(np.int32), new_lbl_affine, lbl_nii.header), str(dst_label))

        zooms = lbl_nii.header.get_zooms()
        for r in rows:
            label_id = int(r["label_id"])
            gt = mask_ground_truth_3d(rot_lbl == label_id, new_axcodes)
            if gt is None:
                continue  # structure vanished after rotation (shouldn't happen, but stay safe)
            (cx, cy, cz), bbox, (side0, side1, side2) = gt

            all_rows.append({
                "dataset": name,
                "subject_id": subject_id,
                "split": r["split"],
                "organ": r["organ"],
                "label_id": label_id,
                "image_path": f"volumes/{dst_image.name}",
                "label_path": f"volumes/{dst_label.name}",
                "volume_shape": "x".join(str(s) for s in rot_lbl.shape),
                "voxel_spacing_mm": "x".join(f"{z:.3f}" for z in zooms),
                "axcodes": "".join(new_axcodes),
                "gt_centroid_vox_0": cx,
                "gt_centroid_vox_1": cy,
                "gt_centroid_vox_2": cz,
                "gt_bbox_vox_0min": bbox[0],
                "gt_bbox_vox_1min": bbox[1],
                "gt_bbox_vox_2min": bbox[2],
                "gt_bbox_vox_0max": bbox[3],
                "gt_bbox_vox_1max": bbox[4],
                "gt_bbox_vox_2max": bbox[5],
                "gt_side_axis0": side0,
                "gt_side_axis1": side1,
                "gt_side_axis2": side2,
                "prompt_text": r["prompt_text"],
            })

        print(f"[{name}] [{si}/{len(by_subject)}] {subject_id}: "
              f"{axcodes} -> {new_axcodes}, {len(rows)} row(s)")

    write_manifest(all_rows, out_dir, name)
    return out_dir


if __name__ == "__main__":
    for ds in DATASETS:
        if not ds["manifest"].exists():
            print(f"[skip] {ds['name']}: manifest not found at {ds['manifest']}")
            continue
        process_dataset(ds["name"], ds["src_dir"], ds["manifest"])

    print(f"\n[done] Rotated samples written under {OUT_ROOT}")
    print("Same manifest schema and volumes/ layout as samples/amos22_mri and "
          "samples/neurite_oasis -- point AMOS_DIR/OASIS_DIR at "
          "samples/sample_rotation/<dataset> in the existing notebooks to run "
          "positional-bias comparisons against the rotated data.")
