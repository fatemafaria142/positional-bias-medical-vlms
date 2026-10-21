# Axis II — Experiment IV: MC Dropout Uncertainty Estimation

- **Experiment Folder:** `exp4_mc_dropout_uncertainty`
- **Generated (UTC):** `2026-10-21T18:53:09Z`
- **Total Detail Rows (`metrics.csv`):** 6900
- **Total Summary Rows (`summary.csv`):** 69
- **Saved Formats:** `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/`

## Experiment IV Overview

Runs N=5 stochastic forward passes per prompt with VoxTell's dropout layers unfrozen. Measures Epistemic Uncertainty (predictive variance & mutual information across passes) and Aleatoric Uncertainty (average per-pass binary entropy), plus pass-to-pass Dice stability and ECE.

## Wording Stability & Contradictory Prompt Uncertainty

| target | side | n_synonym_evals | canonical_epistemic_var | synonym_mean_epistemic_var | synonym_std_epistemic_var | synonym_max_min_ratio_epistemic_var | contradictory_epistemic_var | contradictory_gt_canonical_uncertainty | synonym_mean_aleatoric_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| caudate | left | 600 | 0.0224 | 0.0334 | 0.0345 | 36.7360 | 0.1323 | True | 0.1024 |
| caudate | right | 600 | 0.0234 | 0.0337 | 0.0482 | 50.2440 | 0.0179 | False | 0.0936 |
| hippocampus | left | 600 | 0.0240 | 0.0289 | 0.0307 | 32.7530 | 0.1551 | True | 0.0847 |
| hippocampus | right | 600 | 0.0173 | 0.0181 | 0.0129 | 21.5230 | 0.0155 | False | 0.0774 |
| putamen | left | 600 | 0.0168 | 0.0118 | 0.0093 | 25.7910 | 0.1009 | True | 0.0727 |
| putamen | right | 600 | 0.0125 | 0.0135 | 0.0124 | 28.1770 | 0.0504 | True | 0.0589 |
| thalamus | left | 600 | 0.0555 | 0.0265 | 0.0187 | 22.7740 | 0.0399 | False | 0.0536 |
| thalamus | right | 600 | 0.0431 | 0.0186 | 0.0167 | 38.6000 | 0.0961 | True | 0.0476 |

## Summary Table

| target | side | variant_type | variant_subtype | n_subjects | mean_dice_vs_gt | inter_pass_mean_dice | pass_volume_cv | roi_epistemic_variance | roi_epistemic_mi_bits | roi_aleatoric_entropy_bits | roi_predictive_entropy_bits | ece_roi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adrenal tumor | left | absent_organ_negative | lesion_left_absent | 100 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| caudate | left | contradictory | conflicting_left_on_right | 100 | 0.2442 | 0.5900 | 0.4019 | 0.1323 | 0.5573 | 0.1106 | 0.6679 | 0.6704 |
| caudate | left | lateralized | canonical_sentence | 100 | 0.7518 | 0.9448 | 0.0596 | 0.0224 | 0.0933 | 0.0666 | 0.1599 | 0.3411 |
| caudate | left | lateralized | short_term | 100 | 0.8503 | 0.9759 | 0.0199 | 0.0076 | 0.0304 | 0.0752 | 0.1056 | 0.2176 |
| caudate | left | location_based | anatomical_location_wording | 100 | 0.0016 | 0.6892 | 0.3660 | 0.0940 | 0.3881 | 0.2401 | 0.6282 | 0.9082 |
| caudate | left | paraphrase | paraphrase_1 | 100 | 0.8473 | 0.9750 | 0.0168 | 0.0076 | 0.0303 | 0.0814 | 0.1117 | 0.2190 |
| caudate | left | synonym | clinical_synonym_1 | 100 | 0.6981 | 0.9307 | 0.0919 | 0.0267 | 0.1127 | 0.0779 | 0.1906 | 0.4023 |
| caudate | left | synonym | clinical_synonym_2 | 100 | 0.7257 | 0.9112 | 0.1050 | 0.0420 | 0.1780 | 0.0736 | 0.2516 | 0.3303 |
| caudate | none | base | canonical_sentence | 100 | 0.7353 | 0.9490 | 0.0473 | 0.0229 | 0.0953 | 0.0849 | 0.1802 | 0.3528 |
| caudate | none | base | short_term | 100 | 0.8666 | 0.9714 | 0.0246 | 0.0092 | 0.0368 | 0.0883 | 0.1251 | 0.1842 |
| caudate | right | contradictory | conflicting_right_on_left | 100 | 0.7938 | 0.9598 | 0.0339 | 0.0179 | 0.0733 | 0.0708 | 0.1440 | 0.2844 |
| caudate | right | lateralized | canonical_sentence | 100 | 0.8188 | 0.9518 | 0.0508 | 0.0234 | 0.0967 | 0.0656 | 0.1622 | 0.2454 |
| caudate | right | lateralized | short_term | 100 | 0.8764 | 0.9791 | 0.0171 | 0.0061 | 0.0245 | 0.0762 | 0.1007 | 0.1785 |
| caudate | right | location_based | anatomical_location_wording | 100 | 0.0000 | 0.4251 | 0.9053 | 0.1376 | 0.5664 | 0.2200 | 0.7865 | 0.9328 |
| caudate | right | paraphrase | paraphrase_1 | 100 | 0.8779 | 0.9785 | 0.0176 | 0.0062 | 0.0245 | 0.0750 | 0.0995 | 0.1781 |
| caudate | right | synonym | clinical_synonym_1 | 100 | 0.8197 | 0.9601 | 0.0375 | 0.0175 | 0.0726 | 0.0667 | 0.1393 | 0.2513 |
| caudate | right | synonym | clinical_synonym_2 | 100 | 0.8522 | 0.9729 | 0.0236 | 0.0115 | 0.0472 | 0.0583 | 0.1055 | 0.2170 |
| hippocampus | left | contradictory | conflicting_left_on_right | 100 | 0.3458 | 0.5787 | 0.5028 | 0.1551 | 0.6465 | 0.0788 | 0.7253 | 0.6678 |
| hippocampus | left | lateralized | canonical_sentence | 100 | 0.7736 | 0.9518 | 0.0506 | 0.0240 | 0.1013 | 0.0643 | 0.1657 | 0.3097 |
| hippocampus | left | lateralized | short_term | 100 | 0.8058 | 0.9650 | 0.0328 | 0.0143 | 0.0591 | 0.0803 | 0.1394 | 0.2574 |
| hippocampus | left | location_based | anatomical_location_wording | 100 | 0.6158 | 0.7504 | 0.3116 | 0.0907 | 0.3864 | 0.1479 | 0.5343 | 0.4031 |
| hippocampus | left | paraphrase | paraphrase_1 | 100 | 0.8156 | 0.9583 | 0.0211 | 0.0177 | 0.0730 | 0.0796 | 0.1526 | 0.2404 |
| hippocampus | left | synonym | clinical_synonym_1 | 100 | 0.7918 | 0.9692 | 0.0138 | 0.0127 | 0.0521 | 0.0688 | 0.1209 | 0.2993 |
| hippocampus | left | synonym | clinical_synonym_2 | 100 | 0.8149 | 0.9669 | 0.0262 | 0.0140 | 0.0577 | 0.0675 | 0.1252 | 0.2664 |
| hippocampus | none | base | canonical_sentence | 100 | 0.7869 | 0.9588 | 0.0352 | 0.0195 | 0.0805 | 0.0657 | 0.1462 | 0.2966 |
| hippocampus | none | base | short_term | 100 | 0.8274 | 0.9739 | 0.0172 | 0.0070 | 0.0277 | 0.0982 | 0.1259 | 0.2387 |
| hippocampus | right | contradictory | conflicting_right_on_left | 100 | 0.7957 | 0.9648 | 0.0349 | 0.0155 | 0.0644 | 0.0673 | 0.1316 | 0.2885 |
| hippocampus | right | lateralized | canonical_sentence | 100 | 0.8077 | 0.9626 | 0.0170 | 0.0173 | 0.0718 | 0.0650 | 0.1367 | 0.2700 |
| hippocampus | right | lateralized | short_term | 100 | 0.8292 | 0.9771 | 0.0093 | 0.0068 | 0.0273 | 0.0804 | 0.1077 | 0.2425 |
| hippocampus | right | location_based | anatomical_location_wording | 100 | 0.7076 | 0.9063 | 0.0829 | 0.0422 | 0.1760 | 0.1190 | 0.2950 | 0.3372 |
| hippocampus | right | paraphrase | paraphrase_1 | 100 | 0.8237 | 0.9684 | 0.0302 | 0.0126 | 0.0516 | 0.0709 | 0.1224 | 0.2446 |
| hippocampus | right | synonym | clinical_synonym_1 | 100 | 0.8055 | 0.9617 | 0.0243 | 0.0177 | 0.0729 | 0.0664 | 0.1393 | 0.2704 |
| hippocampus | right | synonym | clinical_synonym_2 | 100 | 0.8284 | 0.9715 | 0.0182 | 0.0117 | 0.0480 | 0.0626 | 0.1106 | 0.2526 |
| kidney | left | absent_organ_negative | lateralized_absent | 100 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0029 | 0.0029 | 0.0005 |
| kidney | none | absent_organ_negative | base_absent | 100 | 1.0000 | 0.9938 | 0.0127 | 0.0021 | 0.0085 | 0.0092 | 0.0178 | 0.0049 |
| kidney | right | absent_organ_negative | lateralized_absent | 100 | 0.9600 | 0.9610 | 0.0927 | 0.0117 | 0.0504 | 0.0141 | 0.0645 | 0.0282 |
| lung nodule | right | absent_organ_negative | lung_lobe_right_upper_absent | 100 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0062 | 0.0062 | 0.0011 |
| putamen | left | contradictory | conflicting_left_on_right | 100 | 0.2516 | 0.6912 | 0.3286 | 0.1009 | 0.4139 | 0.1912 | 0.6051 | 0.7973 |
| putamen | left | lateralized | canonical_sentence | 100 | 0.8449 | 0.9601 | 0.0426 | 0.0168 | 0.0695 | 0.0828 | 0.1523 | 0.2204 |
| putamen | left | lateralized | short_term | 100 | 0.8985 | 0.9846 | 0.0146 | 0.0038 | 0.0154 | 0.0696 | 0.0850 | 0.1528 |
| putamen | left | location_based | anatomical_location_wording | 100 | 0.6298 | 0.9454 | 0.0584 | 0.0275 | 0.1158 | 0.0673 | 0.1830 | 0.4790 |
| putamen | left | paraphrase | paraphrase_1 | 100 | 0.8797 | 0.9786 | 0.0217 | 0.0059 | 0.0235 | 0.0854 | 0.1089 | 0.1765 |
| putamen | left | synonym | clinical_synonym_1 | 100 | 0.8872 | 0.9792 | 0.0094 | 0.0075 | 0.0301 | 0.0597 | 0.0898 | 0.1703 |
| putamen | left | synonym | clinical_synonym_2 | 100 | 0.8757 | 0.9743 | 0.0195 | 0.0094 | 0.0383 | 0.0713 | 0.1096 | 0.1822 |
| putamen | none | base | canonical_sentence | 100 | 0.8213 | 0.9726 | 0.0138 | 0.0092 | 0.0373 | 0.0849 | 0.1222 | 0.2646 |
| putamen | none | base | short_term | 100 | 0.9023 | 0.9834 | 0.0105 | 0.0036 | 0.0141 | 0.0793 | 0.0934 | 0.1422 |
| putamen | right | contradictory | conflicting_right_on_left | 100 | 0.7348 | 0.8991 | 0.1384 | 0.0504 | 0.2089 | 0.0875 | 0.2963 | 0.3372 |
| putamen | right | lateralized | canonical_sentence | 100 | 0.8744 | 0.9720 | 0.0271 | 0.0125 | 0.0521 | 0.0557 | 0.1078 | 0.1848 |
| putamen | right | lateralized | short_term | 100 | 0.9042 | 0.9831 | 0.0115 | 0.0043 | 0.0173 | 0.0701 | 0.0873 | 0.1415 |
| putamen | right | location_based | anatomical_location_wording | 100 | 0.6862 | 0.9340 | 0.0750 | 0.0382 | 0.1610 | 0.0454 | 0.2064 | 0.4148 |
| putamen | right | paraphrase | paraphrase_1 | 100 | 0.8988 | 0.9855 | 0.0059 | 0.0034 | 0.0133 | 0.0700 | 0.0833 | 0.1526 |
| putamen | right | synonym | clinical_synonym_1 | 100 | 0.8728 | 0.9696 | 0.0291 | 0.0139 | 0.0573 | 0.0552 | 0.1125 | 0.1834 |
| putamen | right | synonym | clinical_synonym_2 | 100 | 0.8793 | 0.9780 | 0.0188 | 0.0084 | 0.0341 | 0.0570 | 0.0912 | 0.1827 |
| thalamus | left | contradictory | conflicting_left_on_right | 100 | 0.0016 | 0.8497 | 0.4972 | 0.0399 | 0.1810 | 0.0391 | 0.2201 | 0.3392 |
| thalamus | left | lateralized | canonical_sentence | 100 | 0.7448 | 0.8958 | 0.1391 | 0.0555 | 0.2335 | 0.0638 | 0.2973 | 0.3510 |
| thalamus | left | lateralized | short_term | 100 | 0.9015 | 0.9783 | 0.0106 | 0.0090 | 0.0368 | 0.0495 | 0.0863 | 0.1457 |
| thalamus | left | location_based | anatomical_location_wording | 100 | 0.7789 | 0.9335 | 0.0788 | 0.0342 | 0.1431 | 0.0704 | 0.2135 | 0.2969 |
| thalamus | left | paraphrase | paraphrase_1 | 100 | 0.8908 | 0.9723 | 0.0283 | 0.0140 | 0.0577 | 0.0450 | 0.1027 | 0.1599 |
| thalamus | left | synonym | clinical_synonym_1 | 100 | 0.8769 | 0.9422 | 0.0804 | 0.0322 | 0.1394 | 0.0493 | 0.1886 | 0.1611 |
| thalamus | left | synonym | clinical_synonym_2 | 100 | 0.8434 | 0.9715 | 0.0260 | 0.0144 | 0.0599 | 0.0438 | 0.1038 | 0.2362 |

_Showing first 60 of 69 rows. Full table in `summary.csv` / `metrics.csv`._

