"""
Shared logic for turning 3D (volume, multi-organ label map) datasets into a
text-guided *volumetric* segmentation manifest -- reused by
sample_amos22_mri.py and (later) sample_neurite_oasis.py.

These datasets are meant for 3D models (Text3DSAM, MedSAM3, ...) that take
a whole volume + a text prompt and return a whole predicted mask volume --
not 2D slice models. So, unlike scripts/common.py (2D single-target
datasets), nothing here slices a volume down to a single 2D image: each
subject's full image + label NIfTI volumes are kept as-is, and "ground
truth" is a full 3D description (voxel-space centroid + bounding box, plus
true anatomical side labels).

This is testing *cross-organ* positional bias, not prompt-phrasing bias:
one plain segmentation prompt per organ ("Segment the liver.") -- the
question is whether the model's predicted mask lands on the organ that was
actually asked for, or drifts toward wherever a *different* organ usually
sits. So there is exactly one prompt per (subject, organ).

Pipeline, end to end:
  1. axcodes_for() -- read a volume's NIfTI affine to find which array axis
     maps to which true anatomical direction (e.g. axis 0 -> Right,
     axis 1 -> Anterior, axis 2 -> Superior). This varies per file (even
     which axis is the thin/through-plane one differs between subjects in
     AMOS22 MRI), so it must be read per volume, never assumed fixed.
  2. mask_ground_truth_3d() -- for one label id's binary 3D mask: voxel
     centroid (x, y, z), voxel bounding box, and anatomical side labels
     (L/R, A/P, S/I) per axis, derived from axcodes_for().
  3. export_subject() -- copies one subject's image+label volumes into
     out_dir/volumes/ once (shared across all its organs -- a 3D model
     predicts against the same input volume per organ prompt), then
     appends one manifest row per organ with that organ's single prompt
     and 3D ground truth.
"""

import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

SEED = 42
N_SUBJECTS = 10

PROMPT_TEMPLATE = "Segment the {x} in this image."

# Anatomical opposites, for labeling the "low index" half of each axis
# given the axcode (which names the "high index" direction).
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}


def axcodes_for(nifti_path) -> tuple:
    """Return e.g. ('R', 'A', 'S'): the true anatomical direction that
    increasing each array axis (0, 1, 2) moves toward, read from this
    volume's own affine (NOT assumed -- it can differ between subjects)."""
    return nib.aff2axcodes(nib.load(str(nifti_path)).affine)


def mask_ground_truth_3d(mask: np.ndarray, axcodes: tuple):
    """Given a 3D binary mask and this volume's axcodes, return:
    (centroid_x, centroid_y, centroid_z, bbox, side_0, side_1, side_2)
    where bbox = (xmin, ymin, zmin, xmax, ymax, zmax) in voxel indices,
    and side_i is the true anatomical half ('R'/'L', 'A'/'P', or 'S'/'I')
    the mask's centroid falls in along array axis i.
    """
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        return None
    centroid = tuple(float(c.mean()) for c in coords)
    bbox = tuple(int(c.min()) for c in coords) + tuple(int(c.max()) for c in coords)

    sides = []
    for axis, (code, centroid_val) in enumerate(zip(axcodes, centroid)):
        size = mask.shape[axis]
        sides.append(code if centroid_val > size / 2 else _OPPOSITE[code])
    return centroid, bbox, tuple(sides)


def export_subject(
    subject_id: str,
    image_path: str,
    label_path: str,
    organ_map: dict,
    out_dir: Path,
    dataset_name: str,
    split: str,
):
    """For one subject: copy its image+label volumes once into
    out_dir/volumes/, then return one manifest row per organ id in
    `organ_map` that's actually present in the label volume."""
    vol_out = out_dir / "volumes"
    vol_out.mkdir(parents=True, exist_ok=True)

    dst_image = vol_out / f"{subject_id}_image.nii.gz"
    dst_label = vol_out / f"{subject_id}_label.nii.gz"
    shutil.copy2(image_path, dst_image)
    shutil.copy2(label_path, dst_label)

    label_vol = nib.load(str(dst_label)).get_fdata().astype(int)
    axcodes = axcodes_for(dst_label)
    zooms = nib.load(str(dst_label)).header.get_zooms()

    rows = []
    for label_id, organ_name in organ_map.items():
        gt = mask_ground_truth_3d(label_vol == label_id, axcodes)
        if gt is None:
            continue  # this organ isn't annotated in this subject's volume
        (cx, cy, cz), bbox, (side0, side1, side2) = gt

        rows.append({
            "dataset": dataset_name,
            "subject_id": subject_id,
            "split": split,
            "organ": organ_name,
            "label_id": label_id,
            "image_path": f"volumes/{dst_image.name}",
            "label_path": f"volumes/{dst_label.name}",
            "volume_shape": "x".join(str(s) for s in label_vol.shape),
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
            "prompt_text": PROMPT_TEMPLATE.format(x=organ_name),
        })
    return rows


def thin_axis_first_order(shape: tuple) -> tuple:
    """Return an axis permutation that moves the physically thinnest
    (smallest-extent) axis to position 0, keeping the other two in their
    original relative order.

    Text3DSAM resizes every volume to a fixed (128, 256, 256), i.e. it
    always treats array axis 0 as the coarse/through-plane axis. Real scans
    are acquired in different planes (axial/coronal/sagittal), so which
    array axis is actually thinnest varies per subject even when all
    volumes share the same RAS axcodes -- e.g. in samples/amos22_mri,
    amos_0508 (260x80x320) has its thin axis at position 1, while amos_0544
    (576x468x72) has it at position 2. Without this reordering, Resize
    silently stretches/squashes the wrong axis into the model's fixed 128
    slot for any subject whose thin axis isn't already axis 0.
    """
    thin_axis = int(np.argmin(shape))
    return (thin_axis,) + tuple(ax for ax in range(len(shape)) if ax != thin_axis)


def inverse_order(order: tuple) -> tuple:
    """Permutation that undoes np.transpose(volume, order)."""
    inv = [0] * len(order)
    for new_pos, old_pos in enumerate(order):
        inv[old_pos] = new_pos
    return tuple(inv)


def write_manifest(rows: list, out_dir: Path, dataset_name: str):
    manifest_path = out_dir / f"manifest_{dataset_name}.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {dataset_name}: wrote {len(rows)} prompt rows -> {manifest_path}")
    return manifest_path
