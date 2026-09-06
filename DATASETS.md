# Datasets — Segmentation Positional Bias Research

Three public 2D medical imaging datasets are used in this study. Each was chosen because
a single image contains **two or more distinct segmentable regions**, which is the core
requirement for this research: we need to know not just whether a model can find *a*
target when prompted, but whether it can find the *correct* one among several, and
whether its answer moves in the right direction when the prompt's positional wording
changes (left/right/upper/lower).

| Dataset | Disease / Domain | Modality | Samples | Regions per image |
|---|---|---|---|---|
| ACDC | Cardiac (cardiomyopathy) | MRI | 20 | 3 |
| ISIC 2018 Task 2 | Skin lesion (melanoma/nevus) | Dermoscopy | 20 | 2–3 |
| CHAOS (MR) | Abdominal organs | MRI | 20 | 3–4 |

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

**Regions per image (3):**
| Region | Ground-truth label | Description |
|---|---|---|
| Right ventricle (RV) | 1 | Right heart chamber cavity |
| Myocardium | 2 | Heart muscle wall |
| Left ventricle (LV) | 3 | Left heart chamber cavity |

**Coverage achieved:** 20/20 sample images have all 3 regions present.

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

**Samples used:** 20 lesion images, selected as the 20 images (out of the 100-image
validation set) with the most non-empty attribute masks, so every sample has at least 2
usable regions. (Attribute coverage is naturally sparse and uneven across the full
100-image set — only 26 of the 100 images have 2+ non-empty attributes at all.)

**Regions per image (up to 5, whichever are present):**
| Region | Description |
|---|---|
| Pigment network | Network-like pigmented lines across the lesion |
| Negative network | Inverse (light) network pattern |
| Streaks | Radial linear projections at the lesion border |
| Milia-like cyst | Small white/yellow round structures |
| Globules | Round to oval pigmented structures |

**Coverage achieved:** 16/20 images have 2 regions, 4/20 images have 3 regions.

**Ground truth format:** One separate binary PNG mask file per attribute per image.

---

## 3. CHAOS (MR) — Combined (CT-MR) Healthy Abdominal Organ Segmentation

**Domain:** Abdominal radiology. T1-weighted MRI (T1DUAL sequence).

**Why this dataset (and why MR, not CT):** CHAOS is a standard abdominal multi-organ
benchmark. Its CT arm ships liver-only ground truth (1 region), which does not meet
this study's ≥2-regions requirement, so the **MR arm** is used instead — same
collection, same patients, but with 4 organs labeled per slice.

**Source:** [CHAOS dataset, Zenodo mirror](https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip) — public, no login (the official grand-challenge.org host requires account registration; this mirror doesn't).

**Samples used:** 20 patients (all MR patients available in the Train_Sets release),
one 2D slice per patient — the InPhase slice with the most organ labels present.

**Regions per image (up to 4):**
| Region | Ground-truth grayscale label |
|---|---|
| Liver | 63 |
| Right kidney | 126 |
| Left kidney | 189 |
| Spleen | 252 |

**Coverage achieved:** 18/20 images have all 4 organs, 2/20 have 3 organs.

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

- **Same manifest schema** (`manifest_<dataset>.csv`): one row per (image, region,
  prompt), carrying `image_path`, `mask_path`, ground-truth centroid/bbox/side, and the
  exact `prompt_text` to send to the model — so scoring code is identical across all 3
  datasets.

## File layout

```
dataset/
  acdc/manifest_acdc.csv            (300 rows: 60 region-groups x 5 prompts)
  isic_task2/manifest_isic_task2.csv (220 rows: 44 region-groups x 5 prompts)
  chaos/manifest_chaos.csv          (390 rows: 78 region-groups x 5 prompts)
```

Each dataset folder also contains one subfolder per sample, holding the 2D image PNG
and one binary mask PNG per region present in that sample.
