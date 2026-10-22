# Axis II — Experiment I: Does the Side Word Change the Output?

- **Experiment Folder:** `exp1_side_word_comparison`
- **Generated (UTC):** `2026-10-22T02:39:06Z`
- **Total Detail Rows (`metrics.csv`):** 6400
- **Total Summary Rows (`summary.csv`):** 63
- **Saved Formats:** `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/`

## Experiment I Overview

Tests whether location words (`left`, `right`, vs. `base` no-side) and contradictory/absent-organ prompts change VoxTell's 3D segmentation masks on Neurite-OASIS.

## Summary Table

| target | template_style | condition | n_subjects | mean_pred_voxels | mean_dice_vs_gt_left | mean_dice_vs_gt_right | mean_dice_vs_gt_bilateral | mean_dice_vs_matched_gt | mean_dice_vs_contradictory_opposite_gt | mean_dice_base_vs_left | mean_dice_base_vs_right | mean_dice_left_vs_right | mean_centroid_shift_lr_vox | wrong_side_rate | failure_nearly_identical_rate | left_right_disjoint_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adrenal tumor | lesion_base_absent | absent_organ_negative | 100 | 0.0000 | NaN | NaN | NaN | 1.0000 | NaN | NaN | NaN | NaN | NaN | 0.0000 | 0.0000 | 1.0000 |
| adrenal tumor | lesion_left_absent | absent_organ_negative | 100 | 0.0000 | NaN | NaN | NaN | 1.0000 | NaN | NaN | NaN | NaN | NaN | 0.0000 | 0.0000 | 1.0000 |
| adrenal tumor | lesion_right_absent | absent_organ_negative | 100 | 0.0000 | NaN | NaN | NaN | 1.0000 | NaN | NaN | NaN | NaN | NaN | 0.0000 | 0.0000 | 1.0000 |
| caudate | canonical_sentence | base | 100 | 7303.9100 | 0.3902 | 0.6477 | 0.7438 | 0.7438 | NaN | 0.4553 | 0.7265 | 0.0000 | 28.6223 | 0.0000 | 0.0000 | 1.0000 |
| caudate | canonical_sentence | contradictory_left_on_right | 100 | 1287.0400 | 0.2912 | 0.0905 | 0.2168 | 0.2912 | 0.0905 | 0.4553 | 0.7265 | 0.0000 | 28.6223 | 0.0700 | 0.0000 | 1.0000 |
| caudate | canonical_sentence | contradictory_right_on_left | 100 | 4485.9400 | 0.0000 | 0.8012 | 0.5336 | 0.8012 | 0.0000 | 0.4553 | 0.7265 | 0.0000 | 28.6223 | 0.0000 | 0.0000 | 1.0000 |
| caudate | canonical_sentence | left | 100 | 3568.9800 | 0.7516 | 0.0000 | 0.4730 | 0.7516 | 0.0000 | 0.4553 | 0.7265 | 0.0000 | 28.6223 | 0.0000 | 0.0000 | 1.0000 |
| caudate | canonical_sentence | right | 100 | 4370.9800 | 0.0000 | 0.8311 | 0.5507 | 0.8311 | 0.0000 | 0.4553 | 0.7265 | 0.0000 | 28.6223 | 0.0000 | 0.0000 | 1.0000 |
| caudate | short_term | base | 100 | 10283.1800 | 0.5549 | 0.5923 | 0.8688 | 0.8688 | NaN | 0.6329 | 0.6635 | 0.0000 | 29.2872 | 0.0000 | 0.0000 | 1.0000 |
| caudate | short_term | left | 100 | 5177.7000 | 0.8506 | 0.0000 | 0.5674 | 0.8506 | 0.0000 | 0.6329 | 0.6635 | 0.0000 | 29.2872 | 0.0000 | 0.0000 | 1.0000 |
| caudate | short_term | right | 100 | 5397.0700 | 0.0000 | 0.8773 | 0.6014 | 0.8773 | 0.0000 | 0.6329 | 0.6635 | 0.0000 | 29.2872 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | canonical_sentence | base | 100 | 98898.9600 | 0.4241 | 0.6397 | 0.7604 | 0.7604 | NaN | 0.4535 | 0.6695 | 0.0000 | 53.5626 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | canonical_sentence | contradictory_left_on_right | 100 | 12619.0800 | 0.2190 | 0.0773 | 0.1619 | 0.2190 | 0.0773 | 0.4535 | 0.6695 | 0.0000 | 53.5626 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | canonical_sentence | contradictory_right_on_left | 100 | 51583.7300 | 0.0015 | 0.7792 | 0.5008 | 0.7792 | 0.0015 | 0.4535 | 0.6695 | 0.0000 | 53.5626 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | canonical_sentence | left | 100 | 62667.6300 | 0.8577 | 0.0045 | 0.5693 | 0.8577 | 0.0045 | 0.4535 | 0.6695 | 0.0000 | 53.5626 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | canonical_sentence | right | 100 | 56343.9100 | 0.0000 | 0.7884 | 0.5128 | 0.7884 | 0.0000 | 0.4535 | 0.6695 | 0.0000 | 53.5626 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | short_term | base | 100 | 145167.2500 | 0.5853 | 0.5989 | 0.9025 | 0.9025 | NaN | 0.6457 | 0.6463 | 0.0179 | 49.1093 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | short_term | left | 100 | 76288.2700 | 0.8764 | 0.0246 | 0.6150 | 0.8764 | 0.0246 | 0.6457 | 0.6463 | 0.0179 | 49.1093 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum cortex | short_term | right | 100 | 74205.0300 | 0.0053 | 0.8706 | 0.5963 | 0.8706 | 0.0053 | 0.6457 | 0.6463 | 0.0179 | 49.1093 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | canonical_sentence | base | 100 | 25963.4000 | 0.5446 | 0.6138 | 0.8413 | 0.8413 | NaN | 0.5491 | 0.6368 | 0.0000 | 38.4497 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | canonical_sentence | contradictory_left_on_right | 100 | 842.9200 | 0.0636 | 0.0189 | 0.0426 | 0.0636 | 0.0189 | 0.5491 | 0.6368 | 0.0000 | 38.4497 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | canonical_sentence | contradictory_right_on_left | 100 | 9901.3400 | 0.0019 | 0.7446 | 0.4626 | 0.7446 | 0.0019 | 0.5491 | 0.6368 | 0.0000 | 38.4497 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | canonical_sentence | left | 100 | 15618.7800 | 0.8720 | 0.0000 | 0.5827 | 0.8720 | 0.0000 | 0.5491 | 0.6368 | 0.0000 | 38.4497 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | canonical_sentence | right | 100 | 15021.3600 | 0.0000 | 0.8879 | 0.5856 | 0.8879 | 0.0000 | 0.5491 | 0.6368 | 0.0000 | 38.4497 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | short_term | base | 100 | 33611.5400 | 0.5899 | 0.5942 | 0.8980 | 0.8980 | NaN | 0.6374 | 0.6488 | 0.0119 | 37.3970 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | short_term | left | 100 | 18459.3900 | 0.8746 | 0.0081 | 0.6065 | 0.8746 | 0.0081 | 0.6374 | 0.6488 | 0.0119 | 37.3970 | 0.0000 | 0.0000 | 1.0000 |
| cerebellum white matter | short_term | right | 100 | 16939.7000 | 0.0035 | 0.9001 | 0.6082 | 0.9001 | 0.0035 | 0.6374 | 0.6488 | 0.0119 | 37.3970 | 0.0000 | 0.0000 | 1.0000 |
| cerebral white matter | canonical_sentence | base | 100 | 439107.1000 | 0.5755 | 0.6382 | 0.8804 | 0.8804 | NaN | 0.5696 | 0.6582 | 0.0000 | 55.9177 | 0.0000 | 0.0000 | 1.0000 |
| cerebral white matter | canonical_sentence | contradictory_left_on_right | 100 | 12122.3900 | 0.0845 | 0.0019 | 0.0442 | 0.0845 | 0.0019 | 0.5696 | 0.6582 | 0.0000 | 55.9177 | 0.0000 | 0.0000 | 1.0000 |
| cerebral white matter | canonical_sentence | contradictory_right_on_left | 100 | 146526.9100 | 0.0009 | 0.6935 | 0.4240 | 0.6935 | 0.0009 | 0.5696 | 0.6582 | 0.0000 | 55.9177 | 0.0000 | 0.0000 | 1.0000 |
| cerebral white matter | canonical_sentence | left | 100 | 194988.9500 | 0.8189 | 0.0000 | 0.5187 | 0.8189 | 0.0000 | 0.5696 | 0.6582 | 0.0000 | 55.9177 | 0.0000 | 0.0000 | 1.0000 |
| cerebral white matter | canonical_sentence | right | 100 | 234255.6600 | 0.0000 | 0.8995 | 0.5881 | 0.8995 | 0.0000 | 0.5696 | 0.6582 | 0.0000 | 55.9177 | 0.0000 | 0.0000 | 1.0000 |
| cerebral white matter | short_term | base | 100 | 601914.0400 | 0.5957 | 0.6002 | 0.9142 | 0.9142 | NaN | 0.6531 | 0.6557 | 0.0121 | 56.3017 | 0.0000 | 0.0000 | 0.9900 |
| cerebral white matter | short_term | left | 100 | 300334.7500 | 0.9099 | 0.0063 | 0.6215 | 0.9099 | 0.0063 | 0.6531 | 0.6557 | 0.0121 | 56.3017 | 0.0000 | 0.0000 | 0.9900 |
| cerebral white matter | short_term | right | 100 | 297565.2100 | 0.0068 | 0.9178 | 0.6287 | 0.9178 | 0.0068 | 0.6531 | 0.6557 | 0.0121 | 56.3017 | 0.0000 | 0.0000 | 0.9900 |
| hippocampus | canonical_sentence | base | 100 | 8881.7200 | 0.5280 | 0.5463 | 0.7820 | 0.7820 | NaN | 0.5865 | 0.6459 | 0.0000 | 53.6974 | 0.0000 | 0.0000 | 1.0000 |
| hippocampus | canonical_sentence | contradictory_left_on_right | 100 | 1892.8200 | 0.4149 | 0.0821 | 0.2890 | 0.4149 | 0.0821 | 0.5865 | 0.6459 | 0.0000 | 53.6974 | 0.0600 | 0.0000 | 1.0000 |
| hippocampus | canonical_sentence | contradictory_right_on_left | 100 | 4560.3100 | 0.0000 | 0.7925 | 0.5184 | 0.7925 | 0.0000 | 0.5865 | 0.6459 | 0.0000 | 53.6974 | 0.0000 | 0.0000 | 1.0000 |
| hippocampus | canonical_sentence | left | 100 | 4282.7900 | 0.7757 | 0.0000 | 0.4969 | 0.7757 | 0.0000 | 0.5865 | 0.6459 | 0.0000 | 53.6974 | 0.0000 | 0.0000 | 1.0000 |
| hippocampus | canonical_sentence | right | 100 | 4785.2600 | 0.0000 | 0.8086 | 0.5330 | 0.8086 | 0.0000 | 0.5865 | 0.6459 | 0.0000 | 53.6974 | 0.0000 | 0.0000 | 1.0000 |
| hippocampus | short_term | base | 100 | 11759.6300 | 0.5296 | 0.5533 | 0.8265 | 0.8265 | NaN | 0.6394 | 0.6583 | 0.0128 | 50.6254 | 0.0000 | 0.0000 | 0.9300 |
| hippocampus | short_term | left | 100 | 5911.5600 | 0.8051 | 0.0135 | 0.5532 | 0.8051 | 0.0135 | 0.6394 | 0.6583 | 0.0128 | 50.6254 | 0.0000 | 0.0000 | 0.9300 |
| hippocampus | short_term | right | 100 | 6067.4500 | 0.0000 | 0.8293 | 0.5684 | 0.8293 | 0.0000 | 0.6394 | 0.6583 | 0.0128 | 50.6254 | 0.0000 | 0.0000 | 0.9300 |
| kidney | base_absent | absent_organ_negative | 100 | 1.8900 | NaN | NaN | NaN | 0.9900 | NaN | NaN | NaN | NaN | NaN | 0.0100 | 0.0000 | 0.9900 |
| kidney | lateralized_absent | absent_organ_negative | 200 | 50.6150 | NaN | NaN | NaN | 0.9750 | NaN | NaN | NaN | NaN | NaN | 0.0250 | 0.0000 | 0.9750 |
| lung nodule | lung_lobe_left_lower_absent | absent_organ_negative | 100 | 0.0000 | NaN | NaN | NaN | 1.0000 | NaN | NaN | NaN | NaN | NaN | 0.0000 | 0.0000 | 1.0000 |
| lung nodule | lung_lobe_right_upper_absent | absent_organ_negative | 100 | 0.0000 | NaN | NaN | NaN | 1.0000 | NaN | NaN | NaN | NaN | NaN | 0.0000 | 0.0000 | 1.0000 |
| putamen | canonical_sentence | base | 100 | 10359.8800 | 0.6301 | 0.5079 | 0.8117 | 0.8117 | NaN | 0.6894 | 0.5380 | 0.0000 | 53.8978 | 0.0000 | 0.0000 | 1.0000 |
| putamen | canonical_sentence | contradictory_left_on_right | 100 | 1938.2200 | 0.4182 | 0.0004 | 0.2388 | 0.4182 | 0.0004 | 0.6894 | 0.5380 | 0.0000 | 53.8978 | 0.0000 | 0.0000 | 1.0000 |
| putamen | canonical_sentence | contradictory_right_on_left | 100 | 4757.5400 | 0.0000 | 0.7708 | 0.4825 | 0.7708 | 0.0000 | 0.6894 | 0.5380 | 0.0000 | 53.8978 | 0.0000 | 0.0000 | 1.0000 |
| putamen | canonical_sentence | left | 100 | 5814.5300 | 0.8575 | 0.0000 | 0.5566 | 0.8575 | 0.0000 | 0.6894 | 0.5380 | 0.0000 | 53.8978 | 0.0000 | 0.0000 | 1.0000 |
| putamen | canonical_sentence | right | 100 | 6501.2400 | 0.0000 | 0.8724 | 0.5738 | 0.8724 | 0.0000 | 0.6894 | 0.5380 | 0.0000 | 53.8978 | 0.0000 | 0.0000 | 1.0000 |
| putamen | short_term | base | 100 | 13553.6900 | 0.6019 | 0.6045 | 0.9012 | 0.9012 | NaN | 0.6522 | 0.6543 | 0.0000 | 52.5071 | 0.0000 | 0.0000 | 1.0000 |
| putamen | short_term | left | 100 | 6659.4100 | 0.8968 | 0.0000 | 0.5952 | 0.8968 | 0.0000 | 0.6522 | 0.6543 | 0.0000 | 52.5071 | 0.0000 | 0.0000 | 1.0000 |
| putamen | short_term | right | 100 | 6847.2300 | 0.0000 | 0.9037 | 0.5997 | 0.9037 | 0.0000 | 0.6522 | 0.6543 | 0.0000 | 52.5071 | 0.0000 | 0.0000 | 1.0000 |
| thalamus | canonical_sentence | base | 100 | 4365.9000 | 0.2323 | 0.3960 | 0.3835 | 0.3835 | NaN | 0.2912 | 0.4636 | 0.0000 | 23.6215 | 0.0000 | 0.0000 | 1.0000 |
| thalamus | canonical_sentence | contradictory_left_on_right | 100 | 20.3000 | 0.0046 | 0.0000 | 0.0024 | 0.0046 | 0.0000 | 0.2912 | 0.4636 | 0.0000 | 23.6215 | 0.0000 | 0.0000 | 1.0000 |
| thalamus | canonical_sentence | contradictory_right_on_left | 100 | 5497.9900 | 0.0001 | 0.7553 | 0.4700 | 0.7553 | 0.0001 | 0.2912 | 0.4636 | 0.0000 | 23.6215 | 0.0000 | 0.0000 | 1.0000 |
| thalamus | canonical_sentence | left | 100 | 5598.5500 | 0.7786 | 0.0000 | 0.4897 | 0.7786 | 0.0000 | 0.2912 | 0.4636 | 0.0000 | 23.6215 | 0.0000 | 0.0000 | 1.0000 |
| thalamus | canonical_sentence | right | 100 | 5872.1700 | 0.0000 | 0.7995 | 0.5010 | 0.7995 | 0.0000 | 0.2912 | 0.4636 | 0.0000 | 23.6215 | 0.0000 | 0.0000 | 1.0000 |

_Showing first 60 of 63 rows. Full table in `summary.csv` / `metrics.csv`._

