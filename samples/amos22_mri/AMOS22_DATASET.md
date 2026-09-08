# Datasets — 3D Multi-Region Positional Bias (AMOS22 MRI)

One public 3D medical imaging dataset is used here, for a **volumetric** (not
2D-slice) text-guided segmentation positional-bias study: given a full 3D volume
and a plain organ-name prompt, does a 3D model (Text3DSAM, MedSAM3) predict a mask
that actually lands on the requested organ, or drift toward where a *different*
organ typically sits? Unlike the project's 2D datasets (`DATASETS.md` at repo
root — ACDC/ISIC/CHAOS), the bias tested here is **cross-organ**, not
prompt-phrasing: there is exactly one plain prompt per organ, not left/right/
upper/lower variants, since the model receives the whole volume at once.

## Dataset overview

| Dataset | General description | Source link | Domain | Modality | Samples | Region segmentation |
|---|---|---|---|---|---|---|
| AMOS22 MRI | MRI subset of AMOS22 (MICCAI 2022), abdominal multi-organ segmentation benchmark. | [MedOtter/amos22-mri-dataset](https://huggingface.co/datasets/MedOtter/amos22-mri-dataset) (HF mirror of [amos22.grand-challenge.org](https://amos22.grand-challenge.org/)) | Abdominal organs | MRI | 10 subjects | Spleen, right/left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, pancreas, right/left adrenal gland, duodenum |

---

## AMOS22 MRI — Abdominal Multi-Organ Segmentation

**Domain:** Abdominal radiology. 3D MRI volumes, one scan per subject.

**Why this dataset:** AMOS22 densely annotates up to 15 abdominal organs per
volume in one combined label map — same subject, many real, adjacent,
similarly-sized targets — which is exactly what's needed to test whether a 3D
model's predicted mask for organ X drifts toward a neighboring organ Y's
location instead.

**Source:** [MedOtter/amos22-mri-dataset](https://huggingface.co/datasets/MedOtter/amos22-mri-dataset)
on Hugging Face — a repackaged mirror of the MRI portion of the original
[AMOS22 challenge](https://amos22.grand-challenge.org/) (arXiv:2206.08023).
License: CC-BY-4.0. Full dataset: 60 subjects (40 train / 20 validation),
786 MB, NIfTI (`.nii.gz`) format.

**Samples used:** 10 subjects, sampled deterministically (`random.Random(seed=42)`
over all 60, see `scripts/sample_amos22_mri.py`) — 7 from `train`
(`amos_0508, amos_0522, amos_0530, amos_0532, amos_0554, amos_0555, amos_0558`),
3 from `valid` (`amos_0544, amos_0553, amos_0575`).

**What was taken to segment:** Every organ actually annotated in this MRI
subset — verified directly against the raw label volumes, not assumed: label
ids only ever go up to **13** across all 10 downloaded subjects (ids 14/15,
urinary bladder and prostate/uterus, are part of AMOS22's full 15-class scheme
but are never annotated in this MRI subset). All 10 subjects have all 13
organs present — 130 region-groups total (10 subjects × 13 organs).

**Ground truth format:** Full 3D, per organ — voxel-space centroid `(x, y, z)`,
voxel bounding box, and true anatomical side per axis (L/R, A/P, S/I), the last
read from each volume's own NIfTI affine via `nib.aff2axcodes()` (never assumed
— it is *not* the same array axis for every subject in this dataset; e.g.
`amos_0508`'s through-plane axis is array axis 1, `amos_0530`'s is axis 2).
No 2D slice extraction — the full image + label volumes are kept as-is, since a
3D model consumes the whole volume, not a slice.

## File layout

```
samples/amos22_mri/
├── AMOS22_DATASET.md              # this file
├── manifest_amos22_mri.csv        # 130 rows: 10 subjects x 13 organs
└── volumes/
    ├── amos_0508_image.nii.gz     # one image+label pair per subject,
    ├── amos_0508_label.nii.gz     # shared across that subject's 13 organ rows
    ├── amos_0522_image.nii.gz
    ├── amos_0522_label.nii.gz
    └── ...
```

## Manifest columns (`manifest_amos22_mri.csv`)

| Column | Meaning |
|---|---|
| `subject_id`, `split` | e.g. `amos_0508`, `train` |
| `organ`, `label_id` | organ name and its integer id in the label volume (`label_vol == label_id` recovers that organ's binary mask) |
| `image_path`, `label_path` | relative paths under `volumes/` — shared by all 13 of that subject's rows |
| `volume_shape`, `voxel_spacing_mm`, `axcodes` | e.g. `576x468x72`, `0.694x0.694x3.000`, `RAS` |
| `gt_centroid_vox_0/1/2` | organ's voxel-space centroid |
| `gt_bbox_vox_{0,1,2}{min,max}` | organ's voxel-space bounding box |
| `gt_side_axis0/1/2` | true anatomical side per axis (e.g. `L`, `P`, `S`) |
| `prompt_text` | e.g. `"Segment the spleen in this image."` |

## Reproducing / regenerating

```bash
python scripts/sample_amos22_mri.py
```
Downloads only the 10 sampled subjects' volumes from Hugging Face into
`data/raw/amos22-mri-dataset/` (gitignored — safe to delete after running;
`samples/amos22_mri/` is self-contained and is what's tracked in git).

## Verification

Checked before pushing (all passed): every manifest `image_path`/`label_path`
exists; per-subject image/label shape and affine match; no NaNs or constant
(corrupt) images; label volumes integer-valued in range; every manifest
ground-truth row (centroid, bbox) independently recomputed from the raw
volumes and matched exactly; no duplicate `(subject, organ)` rows; all 10
subjects have all 13 organs; volumes byte-identical between disk and the
git-committed blob; a fresh clone of this branch loads all 20 volumes
correctly with nibabel. `.gitattributes` (repo root) marks `.nii.gz`/`.png`
`-text` so no clone can line-ending-corrupt them regardless of local Git
config.
