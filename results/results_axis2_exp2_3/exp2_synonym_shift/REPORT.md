# Axis II — Experiment II: Synonym & Paraphrase Shift Analysis

- **Experiment Folder:** `exp2_synonym_shift`
- **Generated (UTC):** `2026-10-21T12:57:28Z`
- **Total Detail Rows (`metrics.csv`):** 15400
- **Total Summary Rows (`summary.csv`):** 154
- **Saved Formats:** `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/`

## Experiment II Overview

Tracks (1) embedding change at the frozen Qwen3 output and after the Prompt Decoder, (2) prediction change (Dice/IoU/volume/centroid shift vs. canonical reference), (3) correlation between embedding distance and prediction shift, and (4) Dice performance against ground truth across synonyms, paraphrases, and location wordings.

## Embedding Distance vs. Prediction Shift Correlations

| target | n_observations | qwen3_vs_pred_shift_pearson_r | qwen3_vs_pred_shift_pearson_p | qwen3_vs_pred_shift_spearman_rho | qwen3_vs_pred_shift_spearman_p | prompt_dec_vs_pred_shift_pearson_r | prompt_dec_vs_pred_shift_pearson_p | prompt_dec_vs_pred_shift_spearman_rho | prompt_dec_vs_pred_shift_spearman_p | qwen3_vs_vol_shift_spearman_rho | prompt_dec_vs_vol_shift_spearman_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| caudate | 2000 | 0.6468 | 0.0000 | 0.6198 | 0.0000 | 0.7845 | 0.0000 | 0.6596 | 0.0000 | 0.5868 | 0.6097 |
| cerebellum cortex | 2000 | 0.3717 | 0.0000 | 0.7461 | 0.0000 | 0.4033 | 0.0000 | 0.7728 | 0.0000 | 0.5448 | 0.6151 |
| cerebellum white matter | 2000 | 0.4217 | 0.0000 | 0.5895 | 0.0000 | 0.5308 | 0.0000 | 0.6433 | 0.0000 | 0.4950 | 0.4903 |
| cerebral white matter | 2000 | 0.3845 | 0.0000 | 0.5435 | 0.0000 | 0.5682 | 0.0000 | 0.6894 | 0.0000 | 0.5068 | 0.6465 |
| hippocampus | 2000 | 0.3205 | 0.0000 | 0.5715 | 0.0000 | 0.4796 | 0.0000 | 0.6973 | 0.0000 | 0.3434 | 0.3822 |
| putamen | 2000 | 0.6907 | 0.0000 | 0.6620 | 0.0000 | 0.6596 | 0.0000 | 0.6658 | 0.0000 | 0.1240 | 0.1147 |
| thalamus | 2000 | 0.0162 | 0.4689 | 0.3144 | 0.0000 | -0.0017 | 0.9407 | 0.3300 | 0.0000 | 0.1634 | 0.1929 |

## Summary Table

| target | side | variant_type | variant_subtype | prompt_text | n_subjects | qwen3_cosine_dist_vs_ref | prompt_decoder_cosine_dist_vs_ref | similarity_map_pearson_r_vs_ref | mean_dice_vs_ref_prompt | mean_pred_change_1_minus_dice | mean_rel_volume_shift | mean_dice_vs_gt | mean_dice_drop_vs_ref_gt | wrong_side_rate | mean_rank_by_qwen3_dist | mean_rank_by_prompt_decoder_dist | mean_rank_by_pred_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| caudate | left | contradictory | conflicting_left_on_right | Segment the left caudate located on the right hemisphere in this image. | 100 | 0.0704 | 0.0566 | 0.9490 | 0.3398 | 0.6602 | 0.6739 | 0.2872 | 0.4645 | 0.0700 | 5.0000 | 5.0000 | 10.1000 |
| caudate | left | lateralized | short_term | left caudate | 100 | 0.1778 | 0.1715 | 0.9084 | 0.7915 | 0.2085 | 1.4200 | 0.8506 | -0.0989 | 0.0000 | 8.0000 | 8.8100 | 6.5200 |
| caudate | left | location_based | anatomical_location_wording | Segment the C-shaped basal ganglia nucleus bordering the left lateral ventricle in this image. | 100 | 0.2737 | 0.4084 | 0.8997 | 0.0021 | 0.9979 | 1.0643 | 0.0026 | 0.7491 | 0.0000 | 11.0000 | 11.0000 | 10.8800 |
| caudate | left | paraphrase | paraphrase_1 | Find the caudate that is on the left. | 100 | 0.1875 | 0.1900 | 0.9102 | 0.7960 | 0.2040 | 1.2129 | 0.8480 | -0.0963 | 0.0000 | 9.0000 | 10.0000 | 6.2500 |
| caudate | left | paraphrase | paraphrase_2 | Segment the caudate nucleus on the patient's left side. | 100 | 0.1018 | 0.0739 | 0.9738 | 0.8733 | 0.1267 | 0.2063 | 0.6918 | 0.0599 | 0.0000 | 7.0000 | 7.0000 | 4.0300 |
| caudate | left | paraphrase | paraphrase_3 | Locate the left-sided neostriatal caudate in this brain MRI. | 100 | 0.2048 | 0.1687 | 0.9494 | 0.8454 | 0.1546 | 0.7982 | 0.7938 | -0.0421 | 0.0000 | 10.0000 | 8.1900 | 4.3200 |
| caudate | left | synonym | clinical_synonym_1 | Segment the left caudate nucleus in this image. | 100 | 0.0239 | 0.0175 | 0.9963 | 0.9590 | 0.0410 | 0.0576 | 0.7333 | 0.0184 | 0.0000 | 1.0000 | 1.0000 | 1.0600 |
| caudate | left | synonym | clinical_synonym_2 | Segment the left striatal caudate in this image. | 100 | 0.0698 | 0.0487 | 0.9772 | 0.9257 | 0.0743 | 0.0544 | 0.7417 | 0.0100 | 0.0000 | 4.0000 | 4.0000 | 2.1600 |
| caudate | left | synonym | clinical_synonym_3 | Segment the left caudate head and body in this image. | 100 | 0.0724 | 0.0630 | 0.9857 | 0.6578 | 0.3422 | 0.4710 | 0.5456 | 0.2061 | 0.0000 | 6.0000 | 6.0000 | 8.2500 |
| caudate | left | synonym | clinical_synonym_4 | Segment the l. caudate in this image. | 100 | 0.0560 | 0.0418 | 0.9343 | 0.6521 | 0.3479 | 0.7673 | 0.5336 | 0.2181 | 0.1800 | 3.0000 | 3.0000 | 8.4600 |
| caudate | left | synonym | clinical_synonym_5 | Segment the caudate nucleus on the left side in this image. | 100 | 0.0438 | 0.0361 | 0.9875 | 0.8379 | 0.1621 | 0.2518 | 0.6585 | 0.0932 | 0.0000 | 2.0000 | 2.0000 | 3.9700 |
| caudate | right | contradictory | conflicting_right_on_left | Segment the right caudate located on the left hemisphere in this image. | 100 | 0.0523 | 0.0437 | 0.9688 | 0.9323 | 0.0677 | 0.0476 | 0.8010 | 0.0302 | 0.0000 | 3.0000 | 4.0000 | 5.0600 |
| caudate | right | lateralized | short_term | right caudate | 100 | 0.1817 | 0.2087 | 0.8483 | 0.8625 | 0.1375 | 0.2714 | 0.8773 | -0.0461 | 0.0000 | 9.0000 | 10.0000 | 9.9700 |
| caudate | right | location_based | anatomical_location_wording | Segment the C-shaped basal ganglia nucleus bordering the right lateral ventricle in this image. | 100 | 0.2917 | 0.4002 | 0.8734 | 0.0000 | 1.0000 | 0.9386 | 0.0000 | 0.8312 | 0.0000 | 11.0000 | 11.0000 | 11.0000 |
| caudate | right | paraphrase | paraphrase_1 | Find the caudate that is on the right. | 100 | 0.1690 | 0.1781 | 0.9195 | 0.8924 | 0.1076 | 0.1853 | 0.8769 | -0.0457 | 0.0000 | 8.0000 | 8.9900 | 7.9800 |
| caudate | right | paraphrase | paraphrase_2 | Segment the caudate nucleus on the patient's right side. | 100 | 0.1023 | 0.0827 | 0.9885 | 0.9324 | 0.0676 | 0.0721 | 0.8207 | 0.0106 | 0.0000 | 7.0000 | 7.0000 | 4.9600 |
| caudate | right | paraphrase | paraphrase_3 | Locate the right-sided neostriatal caudate in this brain MRI. | 100 | 0.1962 | 0.1634 | 0.9661 | 0.8896 | 0.1104 | 0.2051 | 0.8394 | -0.0082 | 0.0000 | 10.0000 | 8.0100 | 8.6000 |
| caudate | right | synonym | clinical_synonym_1 | Segment the right caudate nucleus in this image. | 100 | 0.0259 | 0.0189 | 0.9960 | 0.9670 | 0.0330 | 0.0538 | 0.8186 | 0.0127 | 0.0000 | 1.0000 | 1.0000 | 1.0200 |
| caudate | right | synonym | clinical_synonym_2 | Segment the right striatal caudate in this image. | 100 | 0.0706 | 0.0485 | 0.9916 | 0.9494 | 0.0506 | 0.0288 | 0.8515 | -0.0202 | 0.0000 | 5.0000 | 5.0000 | 2.8000 |
| caudate | right | synonym | clinical_synonym_3 | Segment the right caudate head and body in this image. | 100 | 0.0735 | 0.0600 | 0.9883 | 0.9075 | 0.0925 | 0.1642 | 0.7872 | 0.0441 | 0.0000 | 6.0000 | 6.0000 | 7.0900 |
| caudate | right | synonym | clinical_synonym_4 | Segment the r. caudate in this image. | 100 | 0.0540 | 0.0316 | 0.9729 | 0.9427 | 0.0573 | 0.1027 | 0.8426 | -0.0113 | 0.0000 | 4.0000 | 2.0500 | 4.1000 |
| caudate | right | synonym | clinical_synonym_5 | Segment the caudate nucleus on the right side in this image. | 100 | 0.0421 | 0.0327 | 0.9941 | 0.9441 | 0.0559 | 0.1000 | 0.8177 | 0.0136 | 0.0000 | 2.0000 | 2.9500 | 3.4200 |
| cerebellum cortex | left | contradictory | conflicting_left_on_right | Segment the left cerebellum cortex located on the right hemisphere in this image. | 100 | 0.0595 | 0.1044 | 0.9098 | 0.2170 | 0.7830 | 0.7963 | 0.2196 | 0.6385 | 0.0000 | 3.0000 | 5.0000 | 11.0000 |
| cerebellum cortex | left | lateralized | short_term | left cerebellum cortex | 100 | 0.1969 | 0.2365 | 0.8892 | 0.8925 | 0.1075 | 0.2177 | 0.8764 | -0.0183 | 0.0000 | 11.0000 | 11.0000 | 6.5400 |
| cerebellum cortex | left | location_based | anatomical_location_wording | Segment the outer gray matter cortical layer of the left cerebellar hemisphere in this image. | 100 | 0.1497 | 0.1909 | 0.9067 | 0.8953 | 0.1047 | 0.1023 | 0.8032 | 0.0550 | 0.0000 | 8.0000 | 8.0700 | 6.0100 |
| cerebellum cortex | left | paraphrase | paraphrase_1 | Find the cerebellum cortex that is on the left. | 100 | 0.1616 | 0.1918 | 0.9359 | 0.8965 | 0.1035 | 0.1638 | 0.8566 | 0.0016 | 0.0000 | 9.0000 | 8.9500 | 5.7500 |
| cerebellum cortex | left | paraphrase | paraphrase_2 | Segment the cerebellar cortex on the patient's left side. | 100 | 0.1079 | 0.1506 | 0.9283 | 0.8407 | 0.1593 | 0.0479 | 0.7854 | 0.0727 | 0.0000 | 7.0000 | 7.0000 | 8.8000 |
| cerebellum cortex | left | paraphrase | paraphrase_3 | Locate the left-sided cerebellar cortex in this brain MRI. | 100 | 0.1760 | 0.1955 | 0.9495 | 0.8710 | 0.1290 | 0.0512 | 0.8089 | 0.0493 | 0.0000 | 10.0000 | 9.9800 | 7.8800 |
| cerebellum cortex | left | synonym | clinical_synonym_1 | Segment the left cerebellar cortex in this image. | 100 | 0.0131 | 0.0211 | 0.9916 | 0.9486 | 0.0515 | 0.0146 | 0.8448 | 0.0134 | 0.0000 | 1.0000 | 1.0000 | 1.2100 |
| cerebellum cortex | left | synonym | clinical_synonym_2 | Segment the left hemispheric cerebellar cortex in this image. | 100 | 0.0313 | 0.0506 | 0.9795 | 0.9413 | 0.0587 | 0.0170 | 0.8484 | 0.0097 | 0.0000 | 2.0000 | 2.0000 | 2.5900 |
| cerebellum cortex | left | synonym | clinical_synonym_3 | Segment the left cerebellar gray matter cortex in this image. | 100 | 0.0657 | 0.0793 | 0.9798 | 0.9298 | 0.0702 | 0.0636 | 0.8188 | 0.0393 | 0.0000 | 4.0000 | 3.0000 | 3.7600 |
| cerebellum cortex | left | synonym | clinical_synonym_4 | Segment the l. cerebellar cortex in this image. | 100 | 0.0882 | 0.1119 | 0.8775 | 0.6657 | 0.3343 | 0.5063 | 0.6152 | 0.2429 | 0.0000 | 6.0000 | 6.0000 | 10.0000 |
| cerebellum cortex | left | synonym | clinical_synonym_5 | Segment the folia cortex of the left cerebellum in this image. | 100 | 0.0789 | 0.0856 | 0.9666 | 0.9406 | 0.0594 | 0.0321 | 0.8412 | 0.0169 | 0.0000 | 5.0000 | 4.0000 | 2.4600 |
| cerebellum cortex | right | contradictory | conflicting_right_on_left | Segment the right cerebellum cortex located on the left hemisphere in this image. | 100 | 0.0536 | 0.0901 | 0.9156 | 0.8991 | 0.1009 | 0.0868 | 0.7782 | 0.0099 | 0.0000 | 3.0000 | 5.0300 | 7.4000 |
| cerebellum cortex | right | lateralized | short_term | right cerebellum cortex | 100 | 0.2008 | 0.2799 | 0.8809 | 0.8493 | 0.1507 | 0.3246 | 0.8706 | -0.0825 | 0.0000 | 11.0000 | 11.0000 | 10.9200 |
| cerebellum cortex | right | location_based | anatomical_location_wording | Segment the outer gray matter cortical layer of the right cerebellar hemisphere in this image. | 100 | 0.1523 | 0.2059 | 0.8685 | 0.8951 | 0.1049 | 0.1071 | 0.8333 | -0.0452 | 0.0000 | 8.0000 | 9.1400 | 8.1000 |
| cerebellum cortex | right | paraphrase | paraphrase_1 | Find the cerebellum cortex that is on the right. | 100 | 0.1608 | 0.2071 | 0.9193 | 0.8568 | 0.1432 | 0.3133 | 0.8442 | -0.0561 | 0.0000 | 9.0000 | 9.8600 | 10.0600 |
| cerebellum cortex | right | paraphrase | paraphrase_2 | Segment the cerebellar cortex on the patient's right side. | 100 | 0.1090 | 0.1378 | 0.9470 | 0.9069 | 0.0931 | 0.1028 | 0.8468 | -0.0587 | 0.0000 | 7.0000 | 7.0000 | 6.4700 |
| cerebellum cortex | right | paraphrase | paraphrase_3 | Locate the right-sided cerebellar cortex in this brain MRI. | 100 | 0.1687 | 0.1863 | 0.9653 | 0.8963 | 0.1037 | 0.0529 | 0.8025 | -0.0144 | 0.0000 | 10.0000 | 8.0000 | 7.9300 |
| cerebellum cortex | right | synonym | clinical_synonym_1 | Segment the right cerebellar cortex in this image. | 100 | 0.0126 | 0.0201 | 0.9808 | 0.9552 | 0.0448 | 0.0436 | 0.8055 | -0.0174 | 0.0000 | 1.0000 | 1.0000 | 1.2900 |
| cerebellum cortex | right | synonym | clinical_synonym_2 | Segment the right hemispheric cerebellar cortex in this image. | 100 | 0.0333 | 0.0518 | 0.9727 | 0.9433 | 0.0567 | 0.0639 | 0.8127 | -0.0246 | 0.0000 | 2.0000 | 2.0000 | 3.1000 |
| cerebellum cortex | right | synonym | clinical_synonym_3 | Segment the right cerebellar gray matter cortex in this image. | 100 | 0.0666 | 0.0696 | 0.9638 | 0.9510 | 0.0490 | 0.0746 | 0.8212 | -0.0331 | 0.0000 | 4.0000 | 3.8300 | 2.0100 |
| cerebellum cortex | right | synonym | clinical_synonym_4 | Segment the r. cerebellar cortex in this image. | 100 | 0.0694 | 0.0683 | 0.9474 | 0.9406 | 0.0594 | 0.0297 | 0.8053 | -0.0172 | 0.0000 | 5.0000 | 3.1700 | 3.6500 |
| cerebellum cortex | right | synonym | clinical_synonym_5 | Segment the folia cortex of the right cerebellum in this image. | 100 | 0.0815 | 0.0994 | 0.9039 | 0.9238 | 0.0762 | 0.1308 | 0.8342 | -0.0461 | 0.0000 | 6.0000 | 5.9700 | 5.0700 |
| cerebellum white matter | left | contradictory | conflicting_left_on_right | Segment the left cerebellum white matter located on the right hemisphere in this image. | 100 | 0.0539 | 0.0744 | 0.8841 | 0.0542 | 0.9458 | 0.9462 | 0.0642 | 0.8078 | 0.0000 | 4.0000 | 4.0000 | 11.0000 |
| cerebellum white matter | left | lateralized | short_term | left cerebellum white matter | 100 | 0.1582 | 0.1738 | 0.9585 | 0.8956 | 0.1044 | 0.1845 | 0.8746 | -0.0026 | 0.0000 | 10.0000 | 10.0000 | 6.0700 |
| cerebellum white matter | left | location_based | anatomical_location_wording | Segment the deep white matter core inside the left cerebellar hemisphere in this image. | 100 | 0.1027 | 0.1256 | 0.9204 | 0.8584 | 0.1416 | 0.1602 | 0.8461 | 0.0259 | 0.0000 | 8.0000 | 8.0000 | 8.9200 |
| cerebellum white matter | left | paraphrase | paraphrase_1 | Find the cerebellum white matter that is on the left. | 100 | 0.1411 | 0.1554 | 0.9384 | 0.9010 | 0.0990 | 0.0378 | 0.8783 | -0.0064 | 0.0000 | 9.0000 | 9.0000 | 5.3600 |
| cerebellum white matter | left | paraphrase | paraphrase_2 | Segment the cerebellar white matter on the patient's left side. | 100 | 0.0746 | 0.1002 | 0.9378 | 0.8975 | 0.1025 | 0.0834 | 0.8595 | 0.0125 | 0.0000 | 7.0000 | 7.0000 | 5.8300 |
| cerebellum white matter | left | paraphrase | paraphrase_3 | Locate the left-sided cerebellar axonal tissue in this brain MRI. | 100 | 0.2354 | 0.2113 | 0.9586 | 0.8982 | 0.1018 | 0.1435 | 0.8349 | 0.0371 | 0.0000 | 11.0000 | 11.0000 | 5.3800 |
| cerebellum white matter | left | synonym | clinical_synonym_1 | Segment the left cerebellar white matter in this image. | 100 | 0.0096 | 0.0111 | 0.9960 | 0.9625 | 0.0375 | 0.0352 | 0.8740 | -0.0021 | 0.0000 | 1.0000 | 1.0000 | 2.5300 |
| cerebellum white matter | left | synonym | clinical_synonym_2 | Segment the left hemispheric cerebellar white matter in this image. | 100 | 0.0241 | 0.0295 | 0.9844 | 0.9680 | 0.0320 | 0.0174 | 0.8814 | -0.0095 | 0.0000 | 2.0000 | 2.0000 | 1.2900 |
| cerebellum white matter | left | synonym | clinical_synonym_3 | Segment the left cerebellar central white matter in this image. | 100 | 0.0446 | 0.0509 | 0.9917 | 0.9639 | 0.0361 | 0.0202 | 0.8712 | 0.0008 | 0.0000 | 3.0000 | 3.0000 | 2.1800 |
| cerebellum white matter | left | synonym | clinical_synonym_4 | Segment the l. cerebellar white matter in this image. | 100 | 0.0683 | 0.0860 | 0.9139 | 0.8761 | 0.1239 | 0.0239 | 0.8359 | 0.0361 | 0.0000 | 5.0000 | 5.5400 | 7.7400 |
| cerebellum white matter | left | synonym | clinical_synonym_5 | Segment the arbor vitae white matter of the left cerebellum in this image. | 100 | 0.0732 | 0.0842 | 0.9551 | 0.8259 | 0.1741 | 0.2843 | 0.7991 | 0.0728 | 0.0000 | 6.0000 | 5.4600 | 9.7000 |
| cerebellum white matter | right | contradictory | conflicting_right_on_left | Segment the right cerebellum white matter located on the left hemisphere in this image. | 100 | 0.0517 | 0.0574 | 0.9618 | 0.7652 | 0.2348 | 0.3476 | 0.7422 | 0.1456 | 0.0000 | 4.0000 | 5.0000 | 10.9900 |
| cerebellum white matter | right | lateralized | short_term | right cerebellum white matter | 100 | 0.1665 | 0.1617 | 0.9461 | 0.9213 | 0.0787 | 0.1322 | 0.9001 | -0.0123 | 0.0000 | 10.0000 | 10.0000 | 6.7500 |
| cerebellum white matter | right | location_based | anatomical_location_wording | Segment the deep white matter core inside the right cerebellar hemisphere in this image. | 100 | 0.0974 | 0.0916 | 0.9672 | 0.9064 | 0.0936 | 0.1252 | 0.8742 | 0.0136 | 0.0000 | 8.0000 | 7.9100 | 8.4400 |
| cerebellum white matter | right | paraphrase | paraphrase_1 | Find the cerebellum white matter that is on the right. | 100 | 0.1462 | 0.1349 | 0.9647 | 0.9236 | 0.0764 | 0.0334 | 0.8865 | 0.0013 | 0.0000 | 9.0000 | 9.0000 | 6.7000 |
| cerebellum white matter | right | paraphrase | paraphrase_2 | Segment the cerebellar white matter on the patient's right side. | 100 | 0.0752 | 0.0794 | 0.9888 | 0.9594 | 0.0406 | 0.0154 | 0.8925 | -0.0047 | 0.0000 | 6.0000 | 6.0000 | 3.9500 |

_Showing first 60 of 154 rows. Full table in `summary.csv` / `metrics.csv`._

