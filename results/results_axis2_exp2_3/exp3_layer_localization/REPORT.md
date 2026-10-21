# Axis II — Experiment III: Layer-wise Localization of Side & Synonym Information

- **Experiment Folder:** `exp3_layer_localization`
- **Generated (UTC):** `2026-10-21T12:57:33Z`
- **Total Detail Rows (`metrics.csv`):** 245000
- **Total Summary Rows (`summary.csv`):** 112
- **Saved Formats:** `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/`

## Experiment III Overview

Measures feature distance (cosine distance, relative L2, Pearson r, and cross-attention divergence) between Left vs. Right prompts and Synonym vs. Reference prompts across all 14 stages of VoxTell: Frozen Qwen3 -> `project_text_embed` -> Transformer Decoder Layers 1..6 -> `LayerNorm` -> U-Net Multi-Scale Fusion Stages 0..4.

- **Saved Layer-wise Change Plot:** `layer_localization_plot.png`

## Summary Table

| comparison_type | stage_index | stage_name | n_pairs | mean_cosine_distance | mean_relative_l2_distance | mean_pearson_correlation | mean_cross_attn_cosine_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_vs_left | 0 | 00_qwen3_frozen | 700 | 0.0788 | 0.3952 | 0.9212 | NaN |
| base_vs_left | 1 | 01_project_text_embed | 700 | 0.1103 | 0.4672 | 0.8897 | NaN |
| base_vs_left | 2 | 02_transformer_layer_1 | 700 | 0.1133 | 0.4730 | 0.8867 | 0.0117 |
| base_vs_left | 3 | 03_transformer_layer_2 | 700 | 0.1181 | 0.4823 | 0.8819 | 0.0029 |
| base_vs_left | 4 | 04_transformer_layer_3 | 700 | 0.1224 | 0.4909 | 0.8777 | 0.0051 |
| base_vs_left | 5 | 05_transformer_layer_4 | 700 | 0.1191 | 0.4858 | 0.8809 | 0.0112 |
| base_vs_left | 6 | 06_transformer_layer_5 | 700 | 0.1185 | 0.4843 | 0.8816 | 0.0058 |
| base_vs_left | 7 | 07_transformer_layer_6 | 700 | 0.1147 | 0.4759 | 0.8854 | 0.0030 |
| base_vs_left | 8 | 08_transformer_norm_mask_emb | 700 | 0.1140 | 0.4728 | 0.8860 | NaN |
| base_vs_left | 9 | 09_unet_fusion_stage_0_12cubed | 700 | 0.2319 | 0.6536 | 0.7454 | NaN |
| base_vs_left | 10 | 10_unet_fusion_stage_1_24cubed | 700 | 0.0511 | 0.3316 | 0.7944 | NaN |
| base_vs_left | 11 | 11_unet_fusion_stage_2_48cubed | 700 | 0.0126 | 0.2136 | 0.7723 | NaN |
| base_vs_left | 12 | 12_unet_fusion_stage_3_96cubed | 700 | 0.1459 | 0.5599 | 0.8554 | NaN |
| base_vs_left | 13 | 13_unet_fusion_stage_4_192cubed | 700 | 0.0051 | 0.1640 | 0.8049 | NaN |
| base_vs_right | 0 | 00_qwen3_frozen | 700 | 0.0847 | 0.4106 | 0.9154 | NaN |
| base_vs_right | 1 | 01_project_text_embed | 700 | 0.1364 | 0.5207 | 0.8636 | NaN |
| base_vs_right | 2 | 02_transformer_layer_1 | 700 | 0.1367 | 0.5214 | 0.8633 | 0.0162 |
| base_vs_right | 3 | 03_transformer_layer_2 | 700 | 0.1388 | 0.5252 | 0.8612 | 0.0055 |
| base_vs_right | 4 | 04_transformer_layer_3 | 700 | 0.1387 | 0.5253 | 0.8613 | 0.0060 |
| base_vs_right | 5 | 05_transformer_layer_4 | 700 | 0.1315 | 0.5117 | 0.8685 | 0.0070 |
| base_vs_right | 6 | 06_transformer_layer_5 | 700 | 0.1291 | 0.5070 | 0.8709 | 0.0101 |
| base_vs_right | 7 | 07_transformer_layer_6 | 700 | 0.1237 | 0.4961 | 0.8763 | 0.0068 |
| base_vs_right | 8 | 08_transformer_norm_mask_emb | 700 | 0.1231 | 0.4941 | 0.8769 | NaN |
| base_vs_right | 9 | 09_unet_fusion_stage_0_12cubed | 700 | 0.0552 | 0.3591 | 0.8993 | NaN |
| base_vs_right | 10 | 10_unet_fusion_stage_1_24cubed | 700 | 0.0510 | 0.3821 | 0.7928 | NaN |
| base_vs_right | 11 | 11_unet_fusion_stage_2_48cubed | 700 | 0.0141 | 0.2613 | 0.8076 | NaN |
| base_vs_right | 12 | 12_unet_fusion_stage_3_96cubed | 700 | 0.3219 | 0.7962 | 0.8893 | NaN |
| base_vs_right | 13 | 13_unet_fusion_stage_4_192cubed | 700 | 0.0050 | 0.2148 | 0.7913 | NaN |
| left_vs_right | 0 | 00_qwen3_frozen | 700 | 0.1338 | 0.5166 | 0.8662 | NaN |
| left_vs_right | 1 | 01_project_text_embed | 700 | 0.1846 | 0.6063 | 0.8153 | NaN |
| left_vs_right | 2 | 02_transformer_layer_1 | 700 | 0.1919 | 0.6179 | 0.8081 | 0.0334 |
| left_vs_right | 3 | 03_transformer_layer_2 | 700 | 0.1979 | 0.6269 | 0.8021 | 0.0119 |
| left_vs_right | 4 | 04_transformer_layer_3 | 700 | 0.2009 | 0.6316 | 0.7992 | 0.0100 |
| left_vs_right | 5 | 05_transformer_layer_4 | 700 | 0.1937 | 0.6212 | 0.8064 | 0.0235 |
| left_vs_right | 6 | 06_transformer_layer_5 | 700 | 0.1907 | 0.6159 | 0.8094 | 0.0200 |
| left_vs_right | 7 | 07_transformer_layer_6 | 700 | 0.1821 | 0.6004 | 0.8181 | 0.0090 |
| left_vs_right | 8 | 08_transformer_norm_mask_emb | 700 | 0.1810 | 0.5980 | 0.8190 | NaN |
| left_vs_right | 9 | 09_unet_fusion_stage_0_12cubed | 700 | 0.2774 | 0.7168 | 0.6352 | NaN |
| left_vs_right | 10 | 10_unet_fusion_stage_1_24cubed | 700 | 0.1373 | 0.5534 | 0.4560 | NaN |
| left_vs_right | 11 | 11_unet_fusion_stage_2_48cubed | 700 | 0.0428 | 0.3058 | 0.3821 | NaN |
| left_vs_right | 12 | 12_unet_fusion_stage_3_96cubed | 700 | 0.1898 | 0.6129 | 0.8620 | NaN |
| left_vs_right | 13 | 13_unet_fusion_stage_4_192cubed | 700 | 0.0132 | 0.2068 | 0.3962 | NaN |
| synonym_vs_ref (contradictory) | 0 | 00_qwen3_frozen | 1400 | 0.0582 | 0.3406 | 0.9419 | NaN |
| synonym_vs_ref (contradictory) | 1 | 01_project_text_embed | 1400 | 0.0771 | 0.3932 | 0.9228 | NaN |
| synonym_vs_ref (contradictory) | 2 | 02_transformer_layer_1 | 1400 | 0.0772 | 0.3931 | 0.9227 | 0.0059 |
| synonym_vs_ref (contradictory) | 3 | 03_transformer_layer_2 | 1400 | 0.0796 | 0.3984 | 0.9203 | 0.0024 |
| synonym_vs_ref (contradictory) | 4 | 04_transformer_layer_3 | 1400 | 0.0815 | 0.4027 | 0.9184 | 0.0038 |
| synonym_vs_ref (contradictory) | 5 | 05_transformer_layer_4 | 1400 | 0.0793 | 0.3970 | 0.9207 | 0.0081 |
| synonym_vs_ref (contradictory) | 6 | 06_transformer_layer_5 | 1400 | 0.0788 | 0.3953 | 0.9212 | 0.0040 |
| synonym_vs_ref (contradictory) | 7 | 07_transformer_layer_6 | 1400 | 0.0756 | 0.3868 | 0.9245 | 0.0033 |
| synonym_vs_ref (contradictory) | 8 | 08_transformer_norm_mask_emb | 1400 | 0.0752 | 0.3845 | 0.9248 | NaN |
| synonym_vs_ref (contradictory) | 9 | 09_unet_fusion_stage_0_12cubed | 1400 | 0.0712 | 0.3578 | 0.9162 | NaN |
| synonym_vs_ref (contradictory) | 10 | 10_unet_fusion_stage_1_24cubed | 1400 | 0.0342 | 0.2966 | 0.9056 | NaN |
| synonym_vs_ref (contradictory) | 11 | 11_unet_fusion_stage_2_48cubed | 1400 | 0.0039 | 0.1431 | 0.9417 | NaN |
| synonym_vs_ref (contradictory) | 12 | 12_unet_fusion_stage_3_96cubed | 1400 | 0.1364 | 0.4984 | 0.9676 | NaN |
| synonym_vs_ref (contradictory) | 13 | 13_unet_fusion_stage_4_192cubed | 1400 | 0.0014 | 0.1207 | 0.9343 | NaN |
| synonym_vs_ref (lateralized) | 0 | 00_qwen3_frozen | 1400 | 0.1914 | 0.6193 | 0.8086 | NaN |
| synonym_vs_ref (lateralized) | 1 | 01_project_text_embed | 1400 | 0.2187 | 0.6610 | 0.7813 | NaN |
| synonym_vs_ref (lateralized) | 2 | 02_transformer_layer_1 | 1400 | 0.2127 | 0.6519 | 0.7873 | 0.0061 |
| synonym_vs_ref (lateralized) | 3 | 03_transformer_layer_2 | 1400 | 0.2148 | 0.6553 | 0.7852 | 0.0035 |

_Showing first 60 of 112 rows. Full table in `summary.csv` / `metrics.csv`._

