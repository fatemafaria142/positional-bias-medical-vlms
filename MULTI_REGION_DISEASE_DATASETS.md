# Candidate 2D Multi-Region Disease Datasets for Positional Bias Research

Datasets previously used in this repo (ISIC, Kvasir, LGG) are **single-region** per image
(one lesion / one polyp / one tumor mask). To study positional bias when a model must
localize **multiple distinct regions within the same disease image**, the datasets below
are better suited: each is 2D, each image carries **more than one region/class to segment**,
and each is tied to a specific disease.

| # | Dataset | Disease / Domain | Modality (2D) | Regions per image | Size | Access |
|---|---------|-------------------|----------------|--------------------|------|--------|
| 1 | **ChestX-Det** | Thoracic disease (chest X-ray) | X-ray | Up to 13 disease classes per image with pixel-level masks: Atelectasis, Calcification, Cardiomegaly, Consolidation, Diffuse Nodule, Effusion, Emphysema, Fibrosis, Fracture, Mass, Nodule, Pleural Thickening, Pneumothorax | ~3,578 images | [GitHub: Deepwise-AILab/ChestX-Det-Dataset](https://github.com/Deepwise-AILab/ChestX-Det-Dataset) |
| 2 | **IDRiD** (Indian Diabetic Retinopathy Image Dataset) | Diabetic retinopathy | Retinal fundus photo | 4 lesion types per image — microaneurysms (MA), hemorrhages (HE), hard exudates (EX), soft exudates (SE) — plus optic disc | 81 images w/ pixel-level masks (516 total in full dataset) | [IEEE DataPort / idrid.grand-challenge.org](https://idrid.grand-challenge.org/) |
| 3 | **ISIC 2018 Task 2** (Lesion Attribute Detection) | Melanoma / skin lesion | Dermoscopy | 5 dermoscopic attribute masks per lesion image — pigment network, negative network, streaks, milia-like cysts, globules/dots | 2,594 images | [challenge.isic-archive.com](https://challenge.isic-archive.com/data/) |
| 4 | **ACDC** (Automated Cardiac Diagnosis Challenge) | Cardiomyopathy (5 pathology groups) | Cardiac cine-MRI (2D slices) | 3 regions per slice — left ventricle cavity, right ventricle cavity, myocardium | 150 patients, ~1,900 annotated slices | [acdc.creatis.insa-lyon.fr](https://www.creatis.insa-lyon.fr/Challenge/acdc/) |
| 5 | **BraTS** (2D slices, e.g. BraTS2020) | Glioma (brain tumor) | Brain MRI (T1/T1ce/T2/FLAIR, sliced to 2D) | 3 nested tumor sub-regions per slice — necrotic/non-enhancing core, peritumoral edema, enhancing tumor | 369+ patients, thousands of 2D slices | [BraTS via Synapse / Kaggle mirrors](https://www.med.upenn.edu/cbica/brats/) |
| 6 | **CHAOS** (Combined Healthy Abdominal Organ Segmentation) | Abdominal organ disease workup | CT & T1/T2 MRI (2D slices) | 4 organs per slice — liver, right kidney, left kidney, spleen | 40 patients, ~1,500+ 2D slices across modalities | [chaos.grand-challenge.org](https://chaos.grand-challenge.org/) |
| 7 | **Synapse Multi-Organ (BTCV)** | Abdominal cancer/pre-op staging | Abdominal CT (2D slices) | Up to 13 organs per slice — liver, spleen, kidneys, pancreas, gallbladder, stomach, aorta, IVC, adrenal glands, etc. | 30 patients, ~3,700 2D slices | [Synapse: syn3193805](https://www.synapse.org/#!Synapse:syn3193805) |
| 8 | **AMOS** (Abdominal Multi-Organ Segmentation) | Multi-organ abdominal disease | CT & MRI (2D slices) | Up to 15 organs per slice — expands on BTCV with more organ classes and cases | 600 CT + 60 MRI scans | [amos22.grand-challenge.org](https://amos22.grand-challenge.org/) |

## Why these fit positional bias research

- **True multi-region layout**: each image has several distinct, independently labeled
  regions at different spatial positions (top/bottom/left/right/center), so a
  prompt-position or answer-order bias would actually have something to bias *between*.
- **Disease-anchored**: every dataset above segments pathology-relevant structures
  (tumor sub-compartments, disease lesions, diseased organs), not generic scene objects —
  consistent with the medical framing of the existing ISIC/Kvasir/LGG work.
- **2D-native or 2D-sliceable**: fundus/dermoscopy/X-ray sets are natively 2D; the
  CT/MRI volumes (ACDC, BraTS, CHAOS, Synapse, AMOS) are standardly consumed as 2D
  axial/short-axis slices in prior segmentation literature, so the existing VLM
  pipelines (Kosmos2, Florence2, OWLv2, GroundingDINO, CLIPSeg, PaliGemma, etc.) can be
  reused with a per-slice or per-image prompting loop.
- **Good difficulty spread**: ChestX-Det and IDRiD have irregular, small, scattered
  lesions (harder localization → more room for positional bias to show); ACDC/CHAOS/
  Synapse/AMOS have larger, more regularly positioned organs (useful as a lower-bias
  baseline for comparison).

## Segmentation mask confirmation

All 8 datasets provide **true pixel-level segmentation masks** (not just bounding
boxes), verified per-dataset:

| Dataset | Mask format | Note |
|---|---|---|
| ChestX-Det | Pixel-level polygon/mask annotations | Its predecessor, **ChestX-Det10, is box-only** — do not use that version for segmentation |
| IDRiD | Binary TIFF masks per lesion type | Only 81 of the 516 images have pixel-level masks (the segmentation sub-challenge subset) |
| ISIC 2018 Task 2 | Binary PNG mask per attribute | One mask file per attribute per lesion |
| ACDC | Per-pixel label map (0=bg, 1=RV, 2=myo, 3=LV) | NIfTI format, 2D slices extracted from volumes |
| BraTS | Per-voxel label map (label 1/2/4) | NIfTI format, 2D slices extracted from volumes |
| CHAOS | Per-pixel organ label (DICOM/PNG ground truth) | Separate mask sets for CT vs. MRI tasks |
| Synapse (BTCV) | Per-voxel organ label map | NIfTI format, 2D slices extracted from volumes |
| AMOS | Per-voxel organ label map | NIfTI format, 2D slices extracted from volumes |

## Suggested next step

Pick 2–3 of these (e.g. one lesion-style dataset like ChestX-Det or IDRiD, plus one
organ-style dataset like ACDC or CHAOS) to keep the experiment scope comparable to the
existing three single-region datasets, rather than onboarding all eight at once.
