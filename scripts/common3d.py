"""
Shared logic for turning 3D (volume, multi-organ label map) datasets into a
text-guided *volumetric* segmentation manifest.

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
     axis 1 -> Anterior, axis 2 -> Superior). This varies per file, so it
     must be read per volume, never assumed fixed.
  2. mask_ground_truth_3d() -- for one label id's binary 3D mask: voxel
     centroid (x, y, z), voxel bounding box, and anatomical side labels
     (L/R, A/P, S/I) per axis, derived from axcodes_for().
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

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
    (centroid, bbox, sides) where centroid = (x, y, z) voxel indices,
    bbox = (xmin, ymin, zmin, xmax, ymax, zmax) in voxel indices,
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


def write_manifest(rows: list, out_dir: Path, dataset_name: str):
    manifest_path = out_dir / f"manifest_{dataset_name}.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {dataset_name}: wrote {len(rows)} prompt rows -> {manifest_path}")
    return manifest_path
