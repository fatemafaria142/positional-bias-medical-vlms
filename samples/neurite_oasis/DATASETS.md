# Datasets — 3D Multi-Region Positional Bias (Neurite OASIS)

One public 3D medical imaging dataset is used here, for the same **volumetric**
(not 2D-slice) text-guided segmentation positional-bias study as
`samples/amos22_mri/DATASETS.md`: given a full 3D volume and a plain
structure-name prompt, does a 3D model (Text3DSAM, MedSAM3) predict a mask that
actually lands on the requested structure, or drift toward where a *different*
structure typically sits? Same cross-organ (here, cross-structure) framing —
one plain prompt per structure, not directional variants.

## Dataset overview

| Dataset | General description | Source link | Domain | Modality | Samples | Region segmentation |
|---|---|---|---|---|---|---|
| Neurite OASIS | Brain MRI collection (414 subjects) from OASIS, processed with FreeSurfer + SAMSEG by the Neurite/HyperMorph team. | [surfer.nmr.mgh.harvard.edu/ftp/data/neurite](http://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar) (paper: [adalca/medical-datasets](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md)) | Neuroimaging (brain) | T1 MRI | 10 subjects (of 414) | 35 anatomical brain structures (17 pairs of left/right + brain stem) |

---

## Neurite OASIS — Brain MRI, 35-Structure Segmentation

**Domain:** Neuroimaging. 3D T1-weighted brain MRI, one scan per subject.

**Why this dataset:** Unlike AMOS22 (abdominal organs, mostly unpaired), Neurite
OASIS's 35-label segmentation is dominated by explicit **left/right structure
pairs** (17 pairs — e.g. `Left-Hippocampus`/`Right-Hippocampus`,
`Left-Thalamus`/`Right-Thalamus`, `Left-Amygdala`/`Right-Amygdala` — plus
`Brain-Stem`, `3rd-Ventricle`, `4th-Ventricle` unpaired). This is a sharper test
of lateral (left/right) positional bias specifically: if a model confuses
`Left-Hippocampus` for `Right-Hippocampus`, that's a clean, unambiguous failure
mode to detect, distinct from the AMOS22 organ-to-organ confusion case.

**Source:** [Neurite OASIS v1.0](http://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar)
(6.6 GB tar), documented at
[adalca/medical-datasets/neurite-oasis.md](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md).
Derived from the [OASIS cross-sectional dataset](https://oasis-brains.org)
(Marcus et al., *J. Cognitive Neuroscience*, 2007), processed for the HyperMorph
paper (Hoopes et al., arXiv:2101.01035, 2021). **Usage requires agreeing to the
[OASIS Data Use Agreement](https://oasis-brains.org/#access)** and citing both
papers (already downloaded locally by the user under that agreement, at
`data/raw/neurite-oasis.v1.0.tar`, which stays gitignored — never pushed to
GitHub).

**Files per subject** (12 total, from the dataset's own `README.md`, extracted
and verified against the actual tar contents):

| File | Shape | Space |
|---|---|---|
| `orig.nii.gz` | 256×256×256 | raw, scanner space |
| `norm.nii.gz` | 256×256×256 | skull-stripped + bias-corrected, scanner space |
| `aligned_orig.nii.gz` | 160×192×224 | raw, affine-aligned to common template (Talairach) |
| `aligned_norm.nii.gz` | 160×192×224 | corrected, affine-aligned to common template |
| `slice_orig.nii.gz`, `slice_norm.nii.gz` | 160×192 | single 2D coronal slice (not used here — 3D only) |
| `seg35.nii.gz`, `aligned_seg35.nii.gz` | matches `orig`/`aligned_orig` | 35-label anatomical segmentation |
| `seg4.nii.gz`, `aligned_seg4.nii.gz` | matches `orig`/`aligned_orig` | 4-label tissue-type segmentation (not used — too coarse for organ-wise prompts) |
| `slice_seg24.nii.gz`, `slice_seg4.nii.gz` | 160×192 | 2D-only segmentations (not used) |

**What will be taken to segment:** `aligned_norm.nii.gz` (image) +
`aligned_seg35.nii.gz` (label) per subject — the affine-aligned, common-template
versions, so every subject shares the same 160×192×224 shape and orientation
(`LIA` per `nib.aff2axcodes`), making cross-subject positional comparison
meaningful. All 35 non-background structures from `seg35_labels.txt` (embedded
in the tar, extracted and verified directly — not assumed):

```
1  Left-Cerebral-White-Matter    11 3rd-Ventricle               21 Right-Cerebral-Cortex
2  Left-Cerebral-Cortex          12 4th-Ventricle                22 Right-Lateral-Ventricle
3  Left-Lateral-Ventricle        13 Brain-Stem                   23 Right-Inf-Lat-Ventricle
4  Left-Inf-Lat-Ventricle        14 Left-Hippocampus              24 Right-Cerebellum-White-Matter
5  Left-Cerebellum-White-Matter  15 Left-Amygdala                 25 Right-Cerebellum-Cortex
6  Left-Cerebellum-Cortex        16 Left-Accumbens                26 Right-Thalamus
7  Left-Thalamus                 17 Left-Ventral-DC                27 Right-Caudate
8  Left-Caudate                  18 Left-Vessel                    28 Right-Putamen
9  Left-Putamen                  19 Left-Choroid-Plexus             29 Right-Pallidum
10 Left-Pallidum                 20 Right-Cerebral-White-Matter      30 Right-Hippocampus
                                                                     31 Right-Amygdala
                                                                     32 Right-Accumbens
                                                                     33 Right-Ventral-DC
                                                                     34 Right-Vessel
                                                                     35 Right-Choroid-Plexus
```

10 subjects will be sampled deterministically (`random.Random(seed=42)`, same
convention as `scripts/sample_amos22_mri.py`) from the 414 in `subjects.txt`
(also embedded in the tar). Not every subject is guaranteed to have every one
of the 35 labels present with nonzero voxels (small structures like
`Left-Vessel`/`Right-Vessel` can occasionally be empty in an individual
automated segmentation) — the sampling script skips any label absent in a
given subject, same as the AMOS22 pipeline.

**Ground truth format:** Full 3D, per structure — same scheme as AMOS22:
voxel-space centroid `(x, y, z)`, voxel bounding box, and true anatomical side
per axis (L/R, A/P/or similar, S/I), read from each volume's own NIfTI affine
via `nib.aff2axcodes()`, never assumed.

## Status

Description only, at this point — samples not yet generated. Next step:
`scripts/sample_neurite_oasis.py` (reusing `scripts/common3d.py`, the same
volumetric export logic as AMOS22) will extract the 10 sampled subjects'
`aligned_norm.nii.gz` + `aligned_seg35.nii.gz` from the local tar and write:

```
samples/neurite_oasis/
├── DATASETS.md                      # this file
├── manifest_neurite_oasis.csv       # 10 subjects x up to 35 structures
└── volumes/
    ├── OASIS_OAS1_XXXX_MR1_image.nii.gz
    ├── OASIS_OAS1_XXXX_MR1_label.nii.gz
    └── ...
```
