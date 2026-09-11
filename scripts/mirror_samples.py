"""
Build a MIRRORED copy of the existing 3D positional-bias samples
(samples/amos22_mri, samples/neurite_oasis) under samples/sample_mirror/ --
same manifest schema/volumes layout as scripts/rotate_samples.py, but a
fundamentally different (and more adversarial) transform.

Difference from scripts/rotate_samples.py (180-degree PROPER rotation):
  rotate_samples.py flips two axes (e.g. L/R and A/P) and *also* negates the
  affine to match, so recomputed ground truth still reports the TRUE
  anatomical side (a real right kidney is still labeled R, just relocated in
  the array) -- that script tests whether a model follows real content vs. a
  fixed array position, while staying an anatomically valid volume.

  THIS script instead:
    1. Flips only ONE axis -- the Left/Right axis, read per-volume from its
       own NIfTI affine (never assumed) -- for both image and label mask,
       identically, so they stay pixel-aligned.
    2. Deliberately keeps the ORIGINAL affine unchanged on the saved file (a
       true mirror is a reflection, not a rotation -- there is no rigid
       transform that "corrects" it back to a valid affine the way a
       rotation does).
    3. Recomputes the manifest's gt_side_axis*/centroid/bbox against the
       FLIPPED mask using the OLD (unmodified) axcodes -- so the computed
       side label is NOT corrected for the flip and will actively disagree
       with the original manifest (e.g. a "right kidney" row's
       gt_side_axis0 can come out as L instead of R).

This is intentional: the point is a harder, adversarial lateral-bias probe.
The prompt text and organ name stay identical to the original sample ("right
kidney"), but the real structure now physically sits where the OLD
coordinate convention says "left" -- so a model that is truly reading image
content should still find it at its new (correctly-reported, if anatomically
backwards) location, while a model biased toward "prompt says right, so look
on the right side of the array" will be fooled into grabbing whatever
(wrong) structure now sits there instead.

Usage:
    python scripts/mirror_samples.py
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from common3d import mask_ground_truth_3d, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = REPO_ROOT / "samples"
OUT_ROOT = SAMPLES_ROOT / "sample_mirror"

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
    by_subject = {}
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_subject.setdefault(row["subject_id"], []).append(row)
    return by_subject


def mirror_lr(volume: np.ndarray, axcodes: tuple) -> np.ndarray:
    """Flip only the Left/Right array axis (per this volume's own axcodes).
    Exact on the voxel grid (numpy.flip, no interpolation)."""
    lr_axis = next(i for i, code in enumerate(axcodes) if code in ("L", "R"))
    return np.ascontiguousarray(np.flip(volume, axis=lr_axis))


def process_dataset(name: str, src_dir: Path, manifest_path: Path):
    out_dir = OUT_ROOT / name
    vol_out = out_dir / "volumes"
    vol_out.mkdir(parents=True, exist_ok=True)

    by_subject = read_manifest_by_subject(manifest_path)
    print(f"[{name}] {len(by_subject)} subject(s) to mirror")

    all_rows = []
    for si, (subject_id, rows) in enumerate(by_subject.items(), 1):
        img_path = src_dir / rows[0]["image_path"]
        lbl_path = src_dir / rows[0]["label_path"]

        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))

        img_arr = np.asarray(img_nii.dataobj)
        lbl_arr = np.asarray(lbl_nii.dataobj).astype(np.int32)

        # Original axcodes -- kept as-is and deliberately NOT recomputed
        # after the flip, since a mirror has no valid corrective affine.
        axcodes = nib.aff2axcodes(lbl_nii.affine)

        mirrored_img = mirror_lr(img_arr, axcodes)
        mirrored_lbl = mirror_lr(lbl_arr, axcodes)

        dst_image = vol_out / f"{subject_id}_image.nii.gz"
        dst_label = vol_out / f"{subject_id}_label.nii.gz"
        # Affine intentionally UNCHANGED from the original file.
        nib.save(nib.Nifti1Image(mirrored_img, img_nii.affine, img_nii.header), str(dst_image))
        nib.save(nib.Nifti1Image(mirrored_lbl.astype(np.int32), lbl_nii.affine, lbl_nii.header), str(dst_label))

        zooms = lbl_nii.header.get_zooms()
        for r in rows:
            label_id = int(r["label_id"])
            # Recomputed against the FLIPPED mask using the OLD axcodes ->
            # side label is not corrected for the flip, by design.
            gt = mask_ground_truth_3d(mirrored_lbl == label_id, axcodes)
            if gt is None:
                continue
            (cx, cy, cz), bbox, (side0, side1, side2) = gt

            all_rows.append({
                "dataset": name,
                "subject_id": subject_id,
                "split": r["split"],
                "organ": r["organ"],
                "label_id": label_id,
                "image_path": f"volumes/{dst_image.name}",
                "label_path": f"volumes/{dst_label.name}",
                "volume_shape": "x".join(str(s) for s in mirrored_lbl.shape),
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
                "gt_side_axis0": side0,
                "gt_side_axis1": side1,
                "gt_side_axis2": side2,
                "prompt_text": r["prompt_text"],
            })

        print(f"[{name}] [{si}/{len(by_subject)}] {subject_id}: mirrored, "
              f"{len(rows)} row(s) (axcodes kept as {axcodes})")

    write_manifest(all_rows, out_dir, name)
    return out_dir


if __name__ == "__main__":
    for ds in DATASETS:
        if not ds["manifest"].exists():
            print(f"[skip] {ds['name']}: manifest not found at {ds['manifest']}")
            continue
        process_dataset(ds["name"], ds["src_dir"], ds["manifest"])

    print(f"\n[done] Mirrored samples written under {OUT_ROOT}")
    print("Same manifest schema and volumes/ layout as samples/amos22_mri and "
          "samples/neurite_oasis, but gt_side_axis* is NOT corrected for the "
          "flip -- it will disagree with the original manifest for structures "
          "on the flipped axis, by design (see this file's module docstring).")
