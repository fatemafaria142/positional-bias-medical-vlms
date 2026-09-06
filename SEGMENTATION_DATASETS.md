# Segmentation Positional-Bias Datasets

This document describes the 3 datasets built for the **segmentation** positional-bias
study (branch `segmentation-positional-bias`). Unlike the earlier localization study
(`position_bias_research_plan.md` — LGG/Kvasir/ISIC Task 1, one target per image),
these 3 datasets were chosen and built specifically so that **each image has 2 or more
distinct segmentable regions**, so we can test whether a model can find more than one
target in the same image, and whether its answer for each one shifts correctly when the
prompt's positional wording changes.

All 3 are public, direct-download (no paid API, no long-term account approval wait),
20 real samples each, same manifest schema, same P0–P4 prompt template.

## At a glance

| Dataset | Domain | Modality | Images | Regions/image | Region set |
|---|---|---|---|---|---|
| **ACDC** | Cardiac | MRI (2D slice from cine-MRI volume) | 20 | 3–3 (all 20 have all 3) | right ventricle, myocardium, left ventricle |
| **ISIC 2018 Task 2** | Skin lesion | Dermoscopy (2D photo) | 20 | 2–3 | pigment network, negative network, streaks, milia-like cyst, globules (whichever ≥2 are present per image) |
| **CHAOS (MR/T1DUAL)** | Abdominal organs | MRI (2D slice) | 20 | 3–4 (18/20 have all 4) | liver, right kidney, left kidney, spleen |

Every image was deliberately selected so it has **at least 2** non-empty regions —
this was a hard requirement, not incidental. (An earlier CHAOS build used the CT arm,
which is liver-only / 1 region, and was rejected for exactly this reason — see
"Why not CT" under CHAOS below.)

---

## 1. ACDC (Automated Cardiac Diagnosis Challenge)

- **Domain:** cardiac MRI, cine short-axis slices.
- **Source:** [Human Heart Project](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) (public Girder collection mirroring the MICCAI 2017 ACDC challenge). Public, no login, no registration.
- **What we downloaded:** for the first 20 patients (`patient001`–`patient020`), the end-diastolic (ED) frame volume + its ground-truth label volume: `<patient>_frame01.nii.gz` + `<patient>_frame01_gt.nii.gz`.
- **Regions (from the label map):** label 1 = right ventricle (RV) cavity, label 2 = myocardium, label 3 = left ventricle (LV) cavity. All 3 are anatomical structures, not disease lesions — ACDC's disease framing (5 pathology groups: normal, myocardial infarction, dilated/hypertrophic cardiomyopathy, abnormal RV) is about how these structures deform, not extra disease-specific regions.
- **2D extraction:** one axial slice per patient, chosen (searching outward from the mid-slice) to be the first slice where all 3 structures are present; all 20 patients succeeded on the first try.
- **Region coverage achieved:** 20/20 images have all 3 regions (60 region-groups total).
- **Script:** `scripts/sample_acdc.py` (reads from `data/raw/acdc/<patient>/`, needs `nibabel`).

## 2. ISIC 2018 Task 2 (Lesion Attribute Detection)

- **Domain:** dermoscopic skin lesion imaging.
- **Source:** ISIC Archive, direct anonymous download (same lesion images as the earlier ISIC Task 1 work, different ground truth):
  - Images: `https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip`
  - Ground truth: `https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task2_Validation_GroundTruth.zip`
- **Regions:** 5 dermoscopic attribute masks exist per lesion in principle (pigment network, negative network, streaks, milia-like cyst, globules), but coverage is sparse and uneven across the 100-image validation set: pigment_network non-empty in 73/100, globules 19/100, negative_network 9/100, streaks 6/100, milia_like_cyst 6/100 — and only **26/100 images have 2 or more non-empty attributes at all**.
- **Selection:** the 20 images with the most non-empty attributes were chosen (all with ≥2), rather than a random 20 — a random sample would likely have pulled in single-attribute images.
- **Region coverage achieved:** 16/20 images have 2 regions, 4/20 have 3 regions (44 region-groups total).
- **Script:** `scripts/sample_isic_task2.py` (reads from `data/raw/isic_task2/`).

## 3. CHAOS (Combined (CT-MR) Healthy Abdominal Organ Segmentation) — MR arm

- **Domain:** abdominal organs (liver, kidneys, spleen).
- **Source:** [Zenodo mirror of the CHAOS grand-challenge dataset](https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip) (public, no login — the official grand-challenge.org site requires an account, this mirror doesn't).
- **Why not CT:** CHAOS's CT arm ships **liver-only, binary** ground truth (1 region) — it was our first attempt and it does not meet the ≥2-regions requirement. We rebuilt using the **MR arm's T1DUAL sequence** instead, which has real multi-organ ground truth in the same archive (no extra download needed).
- **Regions (from the label map):** 8-bit grayscale label values `63 = liver`, `126 = right kidney`, `189 = left kidney`, `252 = spleen` (`0 = background`), confirmed by inspecting the actual pixel values in the downloaded masks.
- **2D extraction:** for each of the 20 MR patients, the InPhase DICOM slice with the most organ labels present (ties broken by total organ pixel area) was chosen — not a fixed mid-slice — so the sample favors the most information-rich slice per patient.
- **Region coverage achieved:** 18/20 images have all 4 organs, 2/20 have 3 organs, missing either a left or right kidney slice at that scan level (78 region-groups total).
- **Script:** `scripts/sample_chaos.py` (reads from `data/raw/chaos/Train_Sets.zip` directly — no extraction to disk needed, it reads members straight out of the zip; needs `pydicom`).

---

## Folder structure (same convention for all 3)

```
dataset/
  acdc/
    manifest_acdc.csv
    patient001/
      patient001.png
      patient001_mask_rv.png
      patient001_mask_myocardium.png
      patient001_mask_lv.png
    ... (20 patients)
  isic_task2/
    manifest_isic_task2.csv
    ISIC_00XXXXX/
      ISIC_00XXXXX.png
      ISIC_00XXXXX_mask_<attribute>.png   (one file per attribute present, 2-3 files)
    ... (20 images)
  chaos/
    manifest_chaos.csv
    patient001/
      patient001.png
      patient001_mask_liver.png
      patient001_mask_right_kidney.png
      patient001_mask_left_kidney.png
      patient001_mask_spleen.png
    ... (20 patients)
```

## Manifest schema (identical columns across all 3 CSVs)

One row = one **(image, region, prompt)** combination — so a single image contributes
multiple rows (one region-group × 5 prompts each), letting you check per-region whether
the model found the *correct* thing, not just *something*.

| Column | Meaning |
|---|---|
| `dataset` | `acdc` / `isic_task2` / `chaos` |
| `sample_id` | Patient or image id (folder name under `dataset/<name>/`) |
| `region` | Which target this row's mask/prompt is for (e.g. `lv`, `pigment_network`, `right_kidney`) |
| `original_filename` | Traceability back to the source file (patient nii.gz + slice index, ISIC jpg name, or DICOM path) |
| `image_path` | Path to the 2D image PNG, relative to `dataset/<name>/` |
| `mask_path` | Path to this region's binary mask PNG, relative to `dataset/<name>/` |
| `image_width`, `image_height` | Pixel dimensions of the image |
| `gt_centroid_x`, `gt_centroid_y` | Center of mass of this region's mask |
| `gt_bbox_xmin/ymin/xmax/ymax` | Tight bounding box of this region's mask |
| `gt_side_lr` | `left`/`right` of image midline, from the centroid |
| `gt_side_ud` | `upper`/`lower` of image midheight, from the centroid |
| `prompt_id` | `P0`–`P4` |
| `prompt_text` | The exact sentence to send to the model |

Row counts: ACDC 300 rows (60 region-groups × 5 prompts), ISIC Task 2 220 rows (44 × 5), CHAOS 390 rows (78 × 5).

## Prompts (P0–P4, same template across all 3 datasets and all regions)

`{X}` is substituted with the region's name (e.g. "the right ventricle", "the pigment
network pattern", "the spleen") — same controlled template, only the noun and the
positional clause change:

| ID | Template |
|----|----------|
| P0 | "Please segment {X} in this image." |
| P1 | "Please segment {X} on the left side of this image." |
| P2 | "Please segment {X} on the right side of this image." |
| P3 | "Please segment {X} in the upper region of this image." |
| P4 | "Please segment {X} in the lower region of this image." |

Note the verb is **"segment"**, not "locate" — this branch's task is pixel-mask
segmentation specifically (unlike the earlier tier-agnostic localization study, which
used "locate" so it could be scored at the point/box/mask level of whatever model ran).

## Regenerating

Each script is idempotent and re-derives everything from `data/raw/<name>/`:

```
python scripts/sample_acdc.py
python scripts/sample_isic_task2.py
python scripts/sample_chaos.py
```

`data/raw/` is not meant to be committed or shared — only `dataset/<name>/` (images +
masks + manifest) needs to go to teammates; it's small enough to zip and hand off
directly, no re-downloading needed on their end.
