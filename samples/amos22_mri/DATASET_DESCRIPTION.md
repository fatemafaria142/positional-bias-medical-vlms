# AMOS22 MRI Dataset — Description

Source: [MedOtter/amos22-mri-dataset](https://huggingface.co/datasets/MedOtter/amos22-mri-dataset) on Hugging Face
Original challenge: [AMOS22](https://amos22.grand-challenge.org/) (arXiv:2206.08023)
License: CC-BY-4.0

## What it is

The MRI subset of AMOS22 — abdominal MRI scans with dense multi-organ segmentation
masks. Each **sample = one subject = one full 3D volume** (not a 2D slice). There is
no subject with multiple samples, and no sample containing multiple subjects: it's a
strict 1:1 mapping.

## Subjects per split

| Split | # Subjects (samples) | Images dir | Labels dir | JSONL manifest |
|---|---|---|---|---|
| train | 40 | `train/imagesTr/` | `train/labelsTr/` | `train.jsonl` |
| validation | 20 | `valid/imagesVa/` | `valid/labelsVa/` | `validation.jsonl` |
| **Total** | **60** | | | |

There is no separate test split published in this dataset repo.

## File format

- Each subject is one pair of NIfTI files: `amos_<id>.nii.gz` (image) +
  `amos_<id>.nii.gz` (label mask), same filename, different directory.
- Example (`amos_0507`): volume shape `(320, 290, 72)` voxels, voxel spacing
  `(1.19, 1.19, 2.5) mm` — in-plane resolution is fine, slice thickness is coarser,
  typical of abdominal MRI. Shape/spacing vary per subject.
- Label volumes are integer-encoded, one channel, values `0`–`13` (0 = background,
  1–13 = the annotated organs below).

## Annotated organs (15 structures, 13 non-background label ids)

Spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta,
inferior vena cava, pancreas, right adrenal gland, left adrenal gland, duodenum,
urinary bladder. (Kidney/adrenal-gland pairs share a generic "kidney"/"adrenal
gland" text label in the JSONL metadata in addition to the left/right specific ones.)

## JSONL manifest schema (`train.jsonl` / `validation.jsonl`)

```json
{
  "image": "data/nii/AMOS22/amos22/imagesTr/amos_0507.nii.gz",
  "mask": "data/nii/AMOS22/amos22/labelsTr/amos_0507.nii.gz",
  "label": ["spleen", "right kidney", "left kidney", "..."],
  "modality": "MRI",
  "dataset": "AMOS22_MRI",
  "official_split": "train",
  "patient_id": "amos_0507.nii.gz"
}
```
One JSON line per subject — so `train.jsonl` has 40 lines, `validation.jsonl` has 20 lines.

## How to pull it via Hugging Face

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MedOtter/amos22-mri-dataset",
    repo_type="dataset",
    local_dir="data/raw/amos22-mri-dataset",
)
```
