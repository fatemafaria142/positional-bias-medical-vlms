# Datasets — Segmentation Positional Bias Research

Three public 2D medical imaging datasets are used in this study. Each was chosen because
a single image contains **two or more distinct segmentable regions**, which is the core
requirement for this research: we need to know not just whether a model can find *a*
target when prompted, but whether it can find the *correct* one among several, and
whether its answer moves in the right direction when the prompt's positional wording
changes (left/right/upper/lower).

## General dataset description

What each dataset is, in general — independent of what we specifically took from it
for this study:

| Dataset | General description | Source link |
|---|---|---|
| ACDC | Public cardiac cine-MRI benchmark (MICCAI 2017), 150 patients across 5 diagnostic groups. | [humanheart-project.creatis.insa-lyon.fr](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| ISIC 2018 Task 2 | Part of the ISIC Archive; dermoscopic skin lesion images annotated with clinically relevant attributes used to assess melanoma risk. | [challenge.isic-archive.com](https://challenge.isic-archive.com/data/) |
| CHAOS | Public abdominal CT/MRI benchmark (ISBI 2019) of healthy subjects, with expert organ ground truth. | [Zenodo mirror](https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip) (official: [chaos.grand-challenge.org](https://chaos.grand-challenge.org/)) |

## What we actually used — already-built dataset summary

Below is what was specifically taken from each dataset for this study (not the full
original dataset — see "What was taken to segment" under each dataset for the exact
selection logic):

| Dataset | Domain | Modality | Samples | Region segmentation |
|---|---|---|---|---|
| ACDC | Cardiac (cardiomyopathy) | MRI | 20 | Right ventricle (RV), Myocardium, Left ventricle (LV) |
| ISIC 2018 Task 2 | Skin lesion (melanoma/nevus) | Dermoscopy | 20 | Pigment network, Negative network, Streaks, Milia-like cyst, Globules |
| CHAOS (MR) | Abdominal organs | MRI | 20 | Liver, Right kidney, Left kidney, Spleen |

Note: not every region appears in every image of a dataset — see "What was taken to
segment" under each dataset below for exactly how many of a dataset's images contain
each region.

---

## 1. ACDC — Automated Cardiac Diagnosis Challenge

**Domain:** Cardiology. Cine cardiac MRI, short-axis view, end-diastolic (ED) frame.

**Why this dataset:** ACDC is a standard, well-validated cardiac segmentation
benchmark covering 5 diagnostic groups (normal, myocardial infarction, dilated
cardiomyopathy, hypertrophic cardiomyopathy, abnormal right ventricle). Every slice
carries 3 anatomically distinct structures in one ground-truth label map, so it's a
clean test of whether a model can distinguish between 3 real, adjacent, similarly-shaped
targets in the same image.

**Source:** [Human Heart Project database](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) — a public mirror of the MICCAI 2017 ACDC challenge data. No account or registration required.

**Samples used:** 20 patients (`patient001`–`patient020`), one 2D slice per patient
(chosen so all 3 structures are visible in that slice).

**What was taken to segment:** ACDC's ground truth only ever labels 3 structures per
slice (RV, myocardium, LV) — there is no larger pool to choose from, so all 3 were
taken, for every patient. 20/20 sample images have all 3 regions present (60
region-groups total: 20 patients × 3 regions).

**Ground truth format:** One combined label-map mask per slice (multi-class), split
here into one binary PNG mask per region for scoring.

---

## 2. ISIC 2018 Task 2 — Lesion Attribute Detection

**Domain:** Dermatology. Dermoscopic (magnified skin surface) photography of
pigmented skin lesions.

**Why this dataset:** Unlike ISIC Task 1 (single lesion-boundary mask per image),
Task 2 annotates up to 5 distinct dermoscopic *attributes* — visual sub-patterns within
the same lesion that dermatologists use for diagnosis. This lets us test positional bias
on fine-grained, often small and irregularly placed targets, a harder case than the
larger, more regularly shaped organs in the other two datasets.

**Source:** [ISIC Archive](https://challenge.isic-archive.com/data/) — direct,
anonymous download, no login required.
- Images: `ISIC2018_Task1-2_Validation_Input.zip`
- Ground truth: `ISIC2018_Task2_Validation_GroundTruth.zip`

**What was taken to segment:** Task 2 defines 5 possible attributes, but coverage per
image is sparse and uneven across the full 100-image validation set — pigment_network
is non-empty in 73/100 images, globules in 19/100, negative_network in 9/100, streaks
in 6/100, milia_like_cyst in 6/100, and only 26/100 images have 2 or more non-empty
attributes at all. Rather than fixing one attribute (which would drop most images to a
single region) or sampling randomly (which would mostly pull in single-region images),
**the 20 images with the most non-empty attributes were selected**, and for each,
*every* attribute that is actually non-empty in that image was taken as a region — not
a fixed subset. Result: 16/20 images contributed 2 regions, 4/20 contributed 3 regions
(44 region-groups total, out of the 5 possible attributes per image).

**Ground truth format:** One separate binary PNG mask file per attribute per image.

---

## 3. CHAOS (MR) — Combined (CT-MR) Healthy Abdominal Organ Segmentation

**Domain:** Abdominal radiology. T1-weighted MRI (T1DUAL sequence).

**Why this dataset (and why MR, not CT):** CHAOS is a standard abdominal multi-organ
benchmark. Its CT arm ships liver-only ground truth (1 region), which does not meet
this study's ≥2-regions requirement, so the **MR arm** is used instead — same
collection, same patients, but with 4 organs labeled per slice.

**Source:** [CHAOS dataset, Zenodo mirror](https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip) — public, no login (the official grand-challenge.org host requires account registration; this mirror doesn't).

**What was taken to segment:** CHAOS-MR labels 4 organs per slice (liver, right
kidney, left kidney, spleen) — all 4 were taken as regions, for every one of the 20 MR
patients available in the Train_Sets release. For each patient, the one InPhase slice
with the most organ labels present was chosen (not a fixed slice index), so patients
whose scan range didn't fully cover all 4 organs still contribute whichever organs are
visible rather than being dropped. Result: 18/20 images have all 4 organs, 2/20 have 3
(78 region-groups total, out of the 4 possible organs per image).

**Ground truth format:** One combined grayscale label-map mask per slice, split here
into one binary PNG mask per organ.

---

## Common design across all 3 datasets

- **2D only** — every sample is a single 2D image (a native photo/scan, or one
  extracted slice from a 3D MRI volume), matching the localization study's scope.
- **≥2 regions per image, always** — every sample was screened/selected specifically
  to guarantee this; it was not incidental.
- **Same prompt template (P0–P4)** applied per region, with only the target noun
  (`{X}`) and the positional clause changing:

  | ID | Prompt |
  |----|--------|
  | P0 | "Please segment {X} in this image." |
  | P1 | "Please segment {X} on the left side of this image." |
  | P2 | "Please segment {X} on the right side of this image." |
  | P3 | "Please segment {X} in the upper region of this image." |
  | P4 | "Please segment {X} in the lower region of this image." |
  | P_ALL | "Please segment {region 1}, {region 2}, ... in this image." (all regions present in that image, named in one sentence) |

  P0–P4 test *positional steering* on one named target at a time. P_ALL is a separate,
  additional condition per image testing *multi-target completeness* — can the model
  find all N regions in a single call — since positional wording doesn't cleanly apply
  to a multi-target ask (see `target_regions`/`mask_paths` below).

- **Same manifest schema** (`manifest_<dataset>.csv`): one row per (image, region,
  prompt), carrying `image_path`, `mask_path`, ground-truth centroid/bbox/side, and the
  exact `prompt_text` to send to the model — so scoring code is identical across all 3
  datasets. `P_ALL` rows instead carry `target_regions` and `mask_paths` (semicolon-
  separated lists), since they name several regions at once rather than one.

## File layout

```
dataset/
  acdc/manifest_acdc.csv             (320 rows: 60 region-groups x 5 prompts + 20 P_ALL)
  isic_task2/manifest_isic_task2.csv (240 rows: 44 region-groups x 5 prompts + 20 P_ALL)
  chaos/manifest_chaos.csv           (410 rows: 78 region-groups x 5 prompts + 20 P_ALL)
```

Each dataset folder also contains one subfolder per sample, holding the 2D image PNG
and one binary mask PNG per region present in that sample, plus a `visual_check/`
folder with one presentation-ready ground-truth overlay PNG per sample (all regions
color-coded on the image, with a legend) for visual comparison against model output.
