"""
One-off local script: stage the 4 AMOS22 MRI + 4 Neurite-OASIS subjects that exist in
data/raw/ but were never extracted into the per-person samples/ structure (only 6 of the
10 available subjects per dataset were ever staged, 2 per annotator, for the original
rotation-sweep study). Needed to go from 6 -> 10 subjects for the groundedness
investigation notebooks, since data/raw/ itself is gitignored and not visible to Kaggle.

For each new subject: copies the raw image+label as <subject>_rot000_{image,label}.nii.gz
into one existing annotator's samples/<dataset>/volumes/ folder (round-robin across
faria/nipu/hridoy), and appends one manifest row per organ present in that subject's
label volume (rotation_angle_deg=0 only -- this investigation doesn't use rotation).

gt_centroid/gt_bbox/gt_side computed here are for REFERENCE ONLY -- the groundedness
notebooks compute these fresh from the actual mask at runtime (see the AXCODES fix), so
they don't need to be in any particular convention to be usable; they're filled in here
just so the manifest stays informative, using plain nibabel (no nnU-Net dependency needed
for this local prep step).

Usage:
    python scripts/stage_extra_subjects.py
"""

import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

AMOS_LABEL_MAP = {
    "spleen": 1, "right kidney": 2, "left kidney": 3, "gallbladder": 4, "esophagus": 5,
    "liver": 6, "stomach": 7, "aorta": 8, "inferior vena cava": 9, "pancreas": 10,
    "right adrenal gland": 11, "left adrenal gland": 12, "duodenum": 13,
}
OASIS_LABEL_MAP = {
    "left cerebral white matter": 1, "left cerebral cortex": 2, "left lateral ventricle": 3,
    "left inferior lateral ventricle": 4, "left cerebellum white matter": 5,
    "left cerebellum cortex": 6, "left thalamus": 7, "left caudate": 8, "left putamen": 9,
    "left pallidum": 10, "3rd ventricle": 11, "4th ventricle": 12, "brain stem": 13,
    "left hippocampus": 14, "left amygdala": 15, "left accumbens": 16, "left ventral DC": 17,
    "left vessel": 18, "left choroid plexus": 19, "right cerebral white matter": 20,
    "right cerebral cortex": 21, "right lateral ventricle": 22,
    "right inferior lateral ventricle": 23, "right cerebellum white matter": 24,
    "right cerebellum cortex": 25, "right thalamus": 26, "right caudate": 27,
    "right putamen": 28, "right pallidum": 29, "right hippocampus": 30, "right amygdala": 31,
    "right accumbens": 32, "right ventral DC": 33, "right vessel": 34, "right choroid plexus": 35,
}

NEW_AMOS = [  # (subject_id, split, raw_image_path, raw_label_path, person)
    ("amos_0522", "train", "train/imagesTr/amos_0522.nii.gz", "train/labelsTr/amos_0522.nii.gz", "nipu"),
    ("amos_0532", "train", "train/imagesTr/amos_0532.nii.gz", "train/labelsTr/amos_0532.nii.gz", "faria"),
    ("amos_0554", "train", "train/imagesTr/amos_0554.nii.gz", "train/labelsTr/amos_0554.nii.gz", "hridoy"),
    ("amos_0575", "valid", "valid/imagesVa/amos_0575.nii.gz", "valid/labelsVa/amos_0575.nii.gz", "nipu"),
]
NEW_OASIS = [  # (subject_id, person)
    ("OASIS_OAS1_0151_MR1", "nipu"),
    ("OASIS_OAS1_0363_MR1", "faria"),
    ("OASIS_OAS1_0419_MR1", "hridoy"),
    ("OASIS_OAS1_0421_MR1", "nipu"),
]

_OPPOSITE = {"S": "I", "I": "S", "A": "P", "P": "A", "R": "L", "L": "R"}
AXCODES = ("S", "A", "R")  # matches the notebooks' empirically-verified nnU-Net axis order


def centroid_bbox_side(mask: np.ndarray):
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        return None
    centroid = [float(c.mean()) for c in coords]
    bbox = [int(c.min()) for c in coords] + [int(c.max()) for c in coords]
    sides = [code if cv > mask.shape[ax] / 2 else _OPPOSITE[code]
             for ax, (code, cv) in enumerate(zip(AXCODES, centroid))]
    return centroid, bbox, sides


def build_prompt(organ: str) -> str:
    parts = organ.split()
    if parts[0].lower() in ("left", "right"):
        s = f"Segment the {parts[0].lower()} {' '.join(parts[1:])} in this image."
    else:
        s = f"Segment the {organ} in this image."
    return s[0].upper() + s[1:]


def stage_subject(dataset_name, subject_id, split, image_src, label_src, person, label_map):
    dest_dir = REPO_ROOT / person / "samples" / dataset_name / "volumes"
    dest_dir.mkdir(parents=True, exist_ok=True)
    image_dest = dest_dir / f"{subject_id}_rot000_image.nii.gz"
    label_dest = dest_dir / f"{subject_id}_rot000_label.nii.gz"
    shutil.copy2(image_src, image_dest)
    shutil.copy2(label_src, label_dest)

    img_nii = nib.load(str(image_dest))
    label_nii = nib.load(str(label_dest))
    label_data = np.asarray(label_nii.dataobj)
    shape = label_data.shape
    zooms = img_nii.header.get_zooms()[:3]
    axcodes_str = "".join(nib.aff2axcodes(img_nii.affine))

    present_labels = set(np.unique(label_data).tolist()) - {0}
    rows = []
    for organ, label_id in label_map.items():
        if label_id not in present_labels:
            continue
        mask = label_data == label_id
        cbs = centroid_bbox_side(mask)
        if cbs is None:
            continue
        centroid, bbox, sides = cbs
        rows.append({
            "dataset": dataset_name, "subject_id": subject_id, "split": split, "organ": organ,
            "label_id": label_id, "rotation_angle_deg": 0,
            "image_path": f"volumes/{subject_id}_rot000_image.nii.gz",
            "label_path": f"volumes/{subject_id}_rot000_label.nii.gz",
            "volume_shape": "x".join(str(s) for s in shape),
            "voxel_spacing_mm": "x".join(f"{z:.3f}" for z in zooms),
            "axcodes": axcodes_str,
            "gt_centroid_vox_0": centroid[0], "gt_centroid_vox_1": centroid[1], "gt_centroid_vox_2": centroid[2],
            "gt_bbox_vox_0min": bbox[0], "gt_bbox_vox_1min": bbox[1], "gt_bbox_vox_2min": bbox[2],
            "gt_bbox_vox_0max": bbox[3], "gt_bbox_vox_1max": bbox[4], "gt_bbox_vox_2max": bbox[5],
            "gt_side_axis0": sides[0], "gt_side_axis1": sides[1], "gt_side_axis2": sides[2],
            "prompt_text": build_prompt(organ),
        })
    print(f"[staged] {person}/{dataset_name}/{subject_id}: {len(rows)} organs -> {image_dest.name}")
    return rows


def append_manifest(dataset_name, person, new_rows):
    manifest_path = REPO_ROOT / person / "samples" / dataset_name / f"manifest_{dataset_name}.csv"
    with open(manifest_path, newline="") as f:
        fieldnames = next(csv.reader(f))
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)
    print(f"[manifest] appended {len(new_rows)} rows -> {manifest_path}")


if __name__ == "__main__":
    RAW = REPO_ROOT / "data" / "raw"

    for subject_id, split, img_rel, lbl_rel, person in NEW_AMOS:
        rows = stage_subject("amos22_mri", subject_id, split,
                              RAW / "amos22-mri-dataset" / img_rel,
                              RAW / "amos22-mri-dataset" / lbl_rel,
                              person, AMOS_LABEL_MAP)
        append_manifest("amos22_mri", person, rows)

    for subject_id, person in NEW_OASIS:
        subj_dir = RAW / "neurite-oasis-extracted" / subject_id
        rows = stage_subject("neurite_oasis", subject_id, "all",
                              subj_dir / "aligned_norm.nii.gz",
                              subj_dir / "aligned_seg35.nii.gz",
                              person, OASIS_LABEL_MAP)
        append_manifest("neurite_oasis", person, rows)

    print("\n[done] staged 4 AMOS + 4 OASIS subjects across faria/nipu/hridoy")
