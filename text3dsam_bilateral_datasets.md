# Text3DSAM — Top 8 Bilateral (Left/Right) Datasets

Source: [CVPR-BiomedSegFM](https://huggingface.co/datasets/junma/CVPR-BiomedSegFM) (`CVPR25_TextSegFMData_with_class.json`), the training/evaluation corpus for [Text3DSAM](https://github.com/mirthAI/Text3DSAM). Ranked by number of explicit left/right labeled structure pairs (richest bilateral coverage first) — narrowed from the full 68 sub-datasets to these 8 for ease of experimentation.

| # | Dataset | Modality | # L/R pairs | Bilateral structures |
|---|---|---|---|---|
| 1 | `MR_TotalSeg` | MR | 12 | Kidney, adrenal gland, iliac artery, iliac vena, humerus, femur, hip, gluteus maximus, gluteus medius, gluteus minimus, autochthon, iliopsoas |
| 2 | `CT_TotalSeg_muscles` | CT | 10 | Humerus, scapula, clavicula, femur, hip, gluteus maximus, gluteus medius, gluteus minimus, autochthon, iliopsoas |
| 3 | `CT_HaN-Seg` | CT | 8 | Carotid artery, cochlea, anterior eyeball segment, posterior eyeball segment, lacrimal gland, submandibular gland, optic nerve, parotid gland |
| 4 | `CT_TotalSeg_cardiac` | CT | 5 | Subclavian artery, common carotid artery, brachiocephalic vein, iliac artery, iliac vena |
| 5 | `CT_TotalSeg_organs` | CT | 3 | Kidney, adrenal gland, kidney cyst |
| 6 | `CT_AMOS` | CT | 2 | Kidney, adrenal gland |
| 7 | `CT_AbdomenAtlas` | CT | 2 | Kidney, adrenal gland |
| 8 | `MR_SegThy` | MR | 2 | Carotid artery, jugular vein |

**Note:** No Ultrasound sub-dataset in this corpus has true explicit left/right labels (`US_SegThy`, `US_Cardiac` have paired-by-anatomy but not left/right-tagged structures), so none made this top 8.

---
*Compiled from `CVPR25_TextSegFMData_with_class.json`, 2026-09-12.*
