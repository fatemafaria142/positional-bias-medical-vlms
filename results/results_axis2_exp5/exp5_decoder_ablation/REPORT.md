# Axis II — Experiment V: 4-Way Image vs. Text Decoder Ablation

- **Experiment Folder:** `exp5_decoder_ablation`
- **Generated (UTC):** `2026-10-21T21:57:17Z`
- **Total Detail Rows (`metrics.csv`):** 39200
- **Total Summary Rows (`summary.csv`):** 28
- **Saved Formats:** `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/`

## Experiment V Overview

Compares the 4 combinations of Image Decoder (`VoxTellDecoder`) and Text Prompt Decoder (`TransformerDecoder`): `both_on`, `image_only`, `text_only`, and `both_off` across Experiment I (Base/Left/Right separation) and Experiment II (Synonym sensitivity).

## Summary Table

| ablation_setting | image_decoder | text_decoder | target | n_evals | canonical_dice_vs_matched_gt | canonical_dice_vs_opposite_gt | canonical_wrong_side_rate | dice_left_vs_right_overlap | dice_base_vs_left_overlap | centroid_shift_lr_vox | synonym_mean_dice_vs_matched_gt | synonym_mean_pred_change_vs_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both_off | Off | Off | caudate | 1400 | 0.0004 | 0.0004 | 0.0000 | 0.9376 | 0.6889 | 1.2467 | 0.0005 | 0.1779 |
| both_off | Off | Off | cerebellum cortex | 1400 | 0.0012 | 0.0012 | 0.0000 | 0.8159 | 0.7442 | 9.4079 | 0.0017 | 0.1850 |
| both_off | Off | Off | cerebellum white matter | 1400 | 0.0089 | 0.0089 | 0.0000 | 0.7766 | 0.7981 | 3.2647 | 0.0091 | 0.1090 |
| both_off | Off | Off | cerebral white matter | 1400 | 0.1588 | 0.1592 | 0.5300 | 0.8315 | 0.7604 | 12.9375 | 0.2130 | 0.1789 |
| both_off | Off | Off | hippocampus | 1400 | 0.0003 | 0.0003 | 0.0000 | 0.9441 | 0.6174 | 1.8796 | 0.0004 | 0.1182 |
| both_off | Off | Off | putamen | 1400 | 0.0032 | 0.0032 | 0.0000 | 0.9016 | 0.8545 | 1.4779 | 0.0032 | 0.0844 |
| both_off | Off | Off | thalamus | 1400 | 0.0051 | 0.0054 | 0.0000 | 0.8597 | 0.6499 | 1.9223 | 0.0061 | 0.1157 |
| both_on | On | On | caudate | 1400 | 0.7914 | 0.0000 | 0.0000 | 0.0000 | 0.4553 | 28.6223 | 0.6091 | 0.3134 |
| both_on | On | On | cerebellum cortex | 1400 | 0.8231 | 0.0022 | 0.0000 | 0.0000 | 0.4535 | 53.5627 | 0.8311 | 0.0834 |
| both_on | On | On | cerebellum white matter | 1400 | 0.8800 | 0.0000 | 0.0000 | 0.0000 | 0.5491 | 38.4496 | 0.8766 | 0.0662 |
| both_on | On | On | cerebral white matter | 1400 | 0.8592 | 0.0000 | 0.0000 | 0.0000 | 0.5696 | 55.9178 | 0.8959 | 0.1146 |
| both_on | On | On | hippocampus | 1400 | 0.7922 | 0.0000 | 0.0000 | 0.0000 | 0.5865 | 53.6974 | 0.7764 | 0.1361 |
| both_on | On | On | putamen | 1400 | 0.8649 | 0.0000 | 0.0000 | 0.0000 | 0.6894 | 53.8978 | 0.8306 | 0.1253 |
| both_on | On | On | thalamus | 1400 | 0.7891 | 0.0000 | 0.0000 | 0.0000 | 0.2912 | 23.6215 | 0.8678 | 0.1714 |
| image_only | On | Off | caudate | 1400 | 0.2192 | 0.0000 | 0.0000 | 0.0300 | 0.0570 | 29.8625 | 0.2405 | 0.6590 |
| image_only | On | Off | cerebellum cortex | 1400 | 0.0250 | 0.0000 | 0.0000 | 0.0000 | 0.0164 | 45.4987 | 0.0489 | 0.6281 |
| image_only | On | Off | cerebellum white matter | 1400 | 0.0004 | 0.0000 | 0.0000 | 0.5100 | 0.0458 | 49.6723 | 0.0014 | 0.4065 |
| image_only | On | Off | cerebral white matter | 1400 | 0.0000 | 0.0000 | 0.0000 | 0.3400 | 0.3823 | NaN | 0.0001 | 0.5247 |
| image_only | On | Off | hippocampus | 1400 | 0.0029 | 0.0000 | 0.0000 | 0.6200 | 0.1109 | 50.4431 | 0.1821 | 0.6947 |
| image_only | On | Off | putamen | 1400 | 0.0144 | 0.0000 | 0.0000 | 0.2100 | 0.9000 | NaN | 0.1424 | 0.5308 |
| image_only | On | Off | thalamus | 1400 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | NaN | 0.0462 | 0.3675 |
| text_only | Off | On | caudate | 1400 | 0.0002 | 0.0002 | 0.0000 | 0.9380 | 0.7585 | 0.7371 | 0.0003 | 0.1050 |
| text_only | Off | On | cerebellum cortex | 1400 | 0.0009 | 0.0008 | 0.0000 | 0.9109 | 0.8801 | 1.2406 | 0.0013 | 0.2768 |
| text_only | Off | On | cerebellum white matter | 1400 | 0.0041 | 0.0040 | 0.0000 | 0.9642 | 0.8897 | 0.8634 | 0.0046 | 0.1183 |
| text_only | Off | On | cerebral white matter | 1400 | 0.0780 | 0.0780 | 0.0700 | 0.9309 | 0.6483 | 2.3324 | 0.1291 | 0.3170 |
| text_only | Off | On | hippocampus | 1400 | 0.0003 | 0.0003 | 0.0000 | 0.9535 | 0.8448 | 0.3142 | 0.0003 | 0.1573 |
| text_only | Off | On | putamen | 1400 | 0.0022 | 0.0023 | 0.0000 | 0.8046 | 0.9349 | 1.8382 | 0.0022 | 0.1514 |
| text_only | Off | On | thalamus | 1400 | 0.0030 | 0.0031 | 0.0000 | 0.9748 | 0.7706 | 0.3206 | 0.0039 | 0.1512 |

