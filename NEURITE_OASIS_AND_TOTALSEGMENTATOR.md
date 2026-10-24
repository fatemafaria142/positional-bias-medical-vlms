# Neurite-OASIS vs. TotalSegmentatorV2 — Dataset Reference

A side-by-side reference for the two richest bilateral (left/right) datasets available for this
project's positional-bias research: the one currently in use (Neurite-OASIS) and the strongest
candidate for extending beyond the brain (TotalSegmentatorV2).

> **Staging status:** Neurite-OASIS is fully staged and used by every RQ in this repo today
> (`results/neurite-oasis/`). TotalSegmentatorV2 is **not staged** — everything below about it
> is general knowledge of the public dataset, not verified against local files.

---

## 1. Neurite-OASIS

### Overview
A preprocessed, analysis-ready neuroimaging dataset built by the Neurite lab (Adrian Dalca et
al., MIT/MGH) on top of the original **OASIS-1** brain MRI collection. Repackages T1-weighted
brain scans with standardized preprocessing and dense FreeSurfer anatomical segmentations.

| | |
|---|---|
| **Modality** | T1-weighted MRI (brain only) |
| **Base cohort** | OASIS-1 — ~414 subjects, ages 18–96 (healthy aging + mild/moderate Alzheimer's) |
| **Preprocessing** | Skull-stripped, intensity-normalized, affine-registered to a common atlas, resampled to a consistent voxel grid (typically 256³ at 1mm isotropic) |
| **Label source** | FreeSurfer `aseg35` parcellation (dense, 35 labels) |
| **Staged in this repo** | Yes — 100 subjects, `scripts/stage_neurite_oasis_100.py` → `Fatema142/neurite-oasis-100-rq-analysis` on Hugging Face |
| **Label format** | Each left/right structure is **already a separate label ID** — no need to derive sides from position |

### Organs (16 bilateral pairs used by every RQ)

| # | Organ pair | Prompt pair |
|---|---|---|
| 1 | Cerebral white matter | "left cerebral white matter" / "right cerebral white matter" |
| 2 | Cerebral cortex | "left cerebral cortex" / "right cerebral cortex" |
| 3 | Lateral ventricle | "left lateral ventricle" / "right lateral ventricle" |
| 4 | Inferior lateral ventricle | "left inferior lateral ventricle" / "right inferior lateral ventricle" |
| 5 | Cerebellum white matter | "left cerebellum white matter" / "right cerebellum white matter" |
| 6 | Cerebellum cortex | "left cerebellum cortex" / "right cerebellum cortex" |
| 7 | Thalamus | "left thalamus" / "right thalamus" |
| 8 | Caudate | "left caudate" / "right caudate" |
| 9 | Putamen | "left putamen" / "right putamen" |
| 10 | Pallidum | "left pallidum" / "right pallidum" |
| 11 | Hippocampus | "left hippocampus" / "right hippocampus" |
| 12 | Amygdala | "left amygdala" / "right amygdala" |
| 13 | Accumbens | "left accumbens" / "right accumbens" |
| 14 | Ventral DC | "left ventral DC" / "right ventral DC" |
| 15 | Vessel | "left vessel" / "right vessel" |
| 16 | Choroid plexus | "left choroid plexus" / "right choroid plexus" |

**Excluded (midline, no left/right counterpart):** 3rd ventricle, 4th ventricle, brain stem.

### Why it's used here
- Clean, consistent FreeSurfer labels give reliable ground truth for left/right pairs.
- 16 bilateral structures is a large, anatomically diverse set for testing whether a laterality
  effect is organ-specific or consistent across anatomy.
- Pre-registered to a common atlas space — every subject's anatomy sits in the same orientation,
  simplifying the RAS-axis handling this project's pipeline relies on.

### Access
Distributed via the Neurite lab's public data releases, commonly alongside their registration
tooling (`neurite`/`voxelmorph` GitHub ecosystems). Requires the standard OASIS-1 data-use
acknowledgment.

---

## 2. TotalSegmentatorV2

### Overview
A large-scale, automatically-and-manually-curated whole-body CT segmentation dataset/toolkit
(Wasserthal et al.), built to provide dense multi-organ labels across essentially the entire
body — not just one anatomical region. Version 2 expanded the original 104-class release to
roughly 117 classes, adding kidney cysts, several named vessels, costal cartilages, and a more
precisely delineated spinal cord.

| | |
|---|---|
| **Modality** | CT (whole-body) |
| **Subjects** | ~1,200+ CT scans in the full public release (varies by exact version/subset) |
| **Label source** | Combination of a trained nnU-Net model + manual correction, covering ~117 structures per scan |
| **Staged in this repo** | **No** — would need a new staging script, modeled on `stage_neurite_oasis_100.py` |
| **Label format** | Bilateral structures are **already separate named classes** (e.g. `kidney_left`/`kidney_right`) — same convenience as OASIS, no connected-component side-splitting needed |

### Organs — bilateral (usable for this project's design)

| Category | Structures |
|---|---|
| **Organs / glands** | Kidney, kidney cyst (v2), adrenal gland |
| **Lung lobes** | Upper lobe (L/R), lower lobe (L/R), middle lobe (right only — no left middle lobe anatomically) |
| **Vessels** | Iliac artery, iliac vein, subclavian artery (v2), common carotid artery (v2), brachiocephalic vein (v2) |
| **Muscles** | Gluteus maximus, gluteus medius, gluteus minimus, autochthon (paraspinal), iliopsoas |
| **Skeleton** | Hip bone, humerus, scapula, clavicle, femur, ribs 1–12 (24 individual pairs if counted separately) |

That's roughly **10–12 genuinely bilateral structure *types*** (before even counting ribs
separately) — several times AMOS22's 2, and approaching OASIS's 16.

### Organs — unilateral / midline (not usable under the current bilateral-only RQ design)

Spleen, liver, gallbladder, pancreas, stomach, esophagus, trachea, thyroid gland, small bowel,
duodenum, colon, urinary bladder, prostate, aorta, inferior vena cava, portal + splenic vein,
superior vena cava, brachiocephalic trunk, pulmonary vein, heart, brain, spinal cord, skull,
sternum, costal cartilages, sacrum, all vertebrae (C1–C7, T1–T12, L1–L5).

### Why it would be a strong extension
- Pre-split left/right labels mean **no new axis-detection or side-inference work** — same
  advantage that makes OASIS easy, unlike AMOS22 (which has only 2 bilateral pairs) or KiTS23
  (whose standard release doesn't split left/right kidney into separate label IDs at all).
- Whole-body CT is a genuinely different domain from OASIS's brain MRI, so results would test
  whether a laterality effect generalizes across both modality and anatomy — not just within
  the brain.
- Large public subject count gives plenty of room to pick a labeled subset comparable in size
  to OASIS's 100 or AMOS's ~100 (CT validation).

### Caveats before staging
- **Rib pairs (24 total) would dilute results** if each one is treated as its own RQ-run organ
  pair — better rolled up into one aggregate "rib" comparison, or a representative few (e.g.
  rib 1, rib 6, rib 12) rather than all 24.
- VoxTell's actual prompt-vocabulary coverage of TotalSegmentator's more obscure structures
  (e.g. `autochthon`, `iliopsoas`) is unverified — worth a cheap one-volume sanity check (same
  as recommended for AMOS) before committing to a full staging run.
- No local evidence yet that VoxTell was specifically evaluated on TotalSegmentatorV2's full
  117-class scheme vs. a reduced subset — the project's own doc (`NEURITE_OASIS_DATASET.md`)
  only confirms it's one of the four benchmark datasets used, not which classes.

### Access
Publicly released via Zenodo, with an official `TotalSegmentator` pip package for inference
(not needed here — only the raw CT + label volumes and manifest matter for staging).

---

## 3. Side-by-Side Comparison

| | Neurite-OASIS | TotalSegmentatorV2 |
|---|---|---|
| Modality | T1 MRI (brain) | CT (whole-body) |
| Staged in this repo | ✅ Yes (100 subjects) | ❌ No |
| Bilateral structure types | 16 | ~10–12 (+ 24 rib pairs if counted individually) |
| Labels pre-split by side | ✅ Yes | ✅ Yes |
| Anatomical region | Brain only | Whole body (abdomen, thorax, pelvis, limbs, spine) |
| Public subject count | ~414 (full OASIS-1) | ~1,200+ (full public release) |
| Used in VoxTell's own eval (per this repo's notes) | N/A (this project's primary dataset) | ✅ Yes, alongside AMOS22-MRI, KiTS23, HaN-Seg |
| Staging effort to use here | None (already done) | New staging script needed (same shape as the deleted `stage_neurite_oasis_100.py`) |

**Bottom line:** OASIS remains the richest, already-working dataset. TotalSegmentatorV2 is the
best next step if extending beyond the brain — far more bilateral coverage than AMOS22, with
the same "labels already split by side" convenience that makes OASIS easy to work with, just
requiring new staging work and a quick VoxTell-vocabulary sanity check first.
