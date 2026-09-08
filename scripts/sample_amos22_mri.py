"""
Sample 10 subjects from AMOS22 MRI (abdominal, 15-organ segmentation) for
positional-bias probing, and export one representative axial slice per
organ per subject as a text-guided-segmentation manifest.

Usage:
    python scripts/sample_amos22_mri.py

Downloads the 10 sampled subjects' NIfTI volumes from Hugging Face
(MedOtter/amos22-mri-dataset) into data/raw/amos22-mri-dataset/ (only the
files actually needed, not the full 786 MB dataset), then writes:
    samples/amos22_mri/images/<subject>_<organ>.png
    samples/amos22_mri/masks/<subject>_<organ>_mask.png
    samples/amos22_mri/manifest_amos22_mri.csv

Each row in the manifest is one (subject, organ, prompt-variant) triple --
10 subjects x up to 15 organs x 5 prompts, minus any organ not annotated in
a given subject. See scripts/common3d.py for the slice-selection and
prompt logic shared with other 3D datasets.
"""

import random
from pathlib import Path

import nibabel as nib
from huggingface_hub import hf_hub_download

from common3d import N_SUBJECTS, SEED, export_subject, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "samples" / "amos22_mri"
RAW_DIR = REPO_ROOT / "data" / "raw" / "amos22-mri-dataset"
DATASET_NAME = "amos22_mri"
HF_REPO_ID = "MedOtter/amos22-mri-dataset"

# Standard AMOS22 15-class label map (label id -> organ name). Not every
# subject has every label annotated (e.g. prostate/uterus is sex-specific);
# export_subject() skips any label absent from a given volume.
ORGAN_MAP = {
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior vena cava",
    10: "pancreas",
    11: "right adrenal gland",
    12: "left adrenal gland",
    13: "duodenum",
    14: "urinary bladder",
    15: "prostate or uterus",
}


def list_all_subjects():
    """Return [(patient_id, split), ...] for all 60 subjects, derived from
    the repo's known train/valid file listing (40 + 20)."""
    train_ids = [
        "amos_0507", "amos_0508", "amos_0510", "amos_0514", "amos_0517", "amos_0518",
        "amos_0522", "amos_0530", "amos_0532", "amos_0538", "amos_0540", "amos_0541",
        "amos_0548", "amos_0551", "amos_0554", "amos_0555", "amos_0557", "amos_0558",
        "amos_0570", "amos_0571", "amos_0578", "amos_0580", "amos_0582", "amos_0583",
        "amos_0584", "amos_0585", "amos_0586", "amos_0587", "amos_0588", "amos_0589",
        "amos_0590", "amos_0591", "amos_0592", "amos_0593", "amos_0594", "amos_0595",
        "amos_0596", "amos_0597", "amos_0599", "amos_0600",
    ]
    valid_ids = [
        "amos_0544", "amos_0545", "amos_0546", "amos_0547", "amos_0549", "amos_0550",
        "amos_0552", "amos_0553", "amos_0556", "amos_0559", "amos_0561", "amos_0562",
        "amos_0563", "amos_0568", "amos_0572", "amos_0573", "amos_0575", "amos_0576",
        "amos_0577", "amos_0581",
    ]
    return [(pid, "train") for pid in train_ids] + [(pid, "valid") for pid in valid_ids]


def repo_paths(patient_id: str, split: str):
    if split == "train":
        return f"train/imagesTr/{patient_id}.nii.gz", f"train/labelsTr/{patient_id}.nii.gz"
    return f"valid/imagesVa/{patient_id}.nii.gz", f"valid/labelsVa/{patient_id}.nii.gz"


if __name__ == "__main__":
    all_subjects = list_all_subjects()
    print(f"[info] {len(all_subjects)} subjects available (40 train + 20 valid)")

    rng = random.Random(SEED)
    sampled = rng.sample(all_subjects, min(N_SUBJECTS, len(all_subjects)))
    print(f"[info] sampled {len(sampled)} subjects (seed={SEED}): "
          f"{[pid for pid, _ in sampled]}")

    all_rows = []
    for patient_id, split in sampled:
        img_rel, lbl_rel = repo_paths(patient_id, split)
        img_path = hf_hub_download(HF_REPO_ID, img_rel, repo_type="dataset", local_dir=RAW_DIR)
        lbl_path = hf_hub_download(HF_REPO_ID, lbl_rel, repo_type="dataset", local_dir=RAW_DIR)

        image_vol = nib.load(img_path).get_fdata()
        label_vol = nib.load(lbl_path).get_fdata().astype(int)

        rows = export_subject(
            subject_id=patient_id,
            image_vol=image_vol,
            label_vol=label_vol,
            organ_map=ORGAN_MAP,
            out_dir=OUT_DIR,
            dataset_name=DATASET_NAME,
            split=split,
        )
        print(f"[ok] {patient_id} ({split}): {len(rows) // len(set(r['prompt_id'] for r in rows) or {1})} "
              f"organ(s) exported" if rows else f"[warn] {patient_id}: no organs found")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("[error] no rows produced -- check downloads/label ids")

    write_manifest(all_rows, OUT_DIR, DATASET_NAME)
    print(f"[done] {DATASET_NAME}: samples/amos22_mri/ is self-contained and safe to "
          f"upload/share; you can delete {RAW_DIR} to reclaim disk space.")
