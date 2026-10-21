# Axis II: Sensitivity to Prompt Variation & Uncertainty Estimation — Comprehensive Guide

This document provides a complete, step-by-step reference for the **Axis II** experimental pipeline (`nipu/uncertainty` branch) using **VoxTell (`voxtell_v1.1` + `Qwen/Qwen3-Embedding-4B`)** on the **100-subject Neurite-OASIS** brain MRI dataset (`Fatema142/neurite-oasis-100-rq-analysis`).

---

## 1. Scientific Overview & Research Questions

From `Axis_II_Prompt_Sensitivity_Nipu.docx`, the goal of **Axis II** is to test whether positional words (`"left"`, `"right"`) and clinical wording variations (synonyms, paraphrases, location descriptions, contradictory prompts, and absent-organ prompts) genuinely drive VoxTell's 3D segmentation masks, where inside the network architecture laterality and wording signals emerge or vanish, and how stable/calibrated the model's predictive uncertainty is across phrasing.

| Research Question | Experiment & Deliverable Folder | How It Is Answered |
|---|---|---|
| **0. What prompts are tested?** | `exp0_frozen_prompts/` | Freezes the prompt catalog across all 16 bilateral OASIS brain structures (26 variants per structure + 8 absent-organ negatives = 424 prompts), plus a **19,000-row** 100-subject synonym manifest tracked directly in Git (`manifests/manifest_neurite_oasis_100_synonyms.csv`), and supplemental AMOS/Adrenal-Lesion/Lung-Lobe prompts. |
| **1. Does `"left"` vs. `"right"` change the output?** | `exp1_side_word_comparison/` | **Experiment I:** Runs `base` (no side), `left`, `right`, `contradictory` (`"left X located on the right hemisphere"`), and `absent_organ_negative` prompts on each scan. Measures pairwise Dice/IoU (`base_vs_left`, `base_vs_right`, `left_vs_right`), volume differences, centroid shifts, midline side classification, Dice against Left/Right/Bilateral GT, and failure flags (`nearly_identical` vs. `left_right_disjoint`). |
| **2. Are synonyms treated as the same concept?** | `exp2_synonym_shift/` | **Experiment II:** Evaluates 5 clinical synonyms, 3 syntactic paraphrases, 1 anatomical-location description, and 1 short term per side per structure. Measures embedding distance (Cosine, L2, Relative L2, Pearson $r$) at **both** the frozen `Qwen3-Embedding-4B` output and after the 6-layer `TransformerDecoder` (Prompt Decoder), measures mask prediction change (`1 - Dice(synonym, ref)`, volume shift, centroid shift, similarity map correlation), and computes **Spearman $\rho$** and **Pearson $r$** linking embedding shift to prediction shift. |
| **3. Where does side/wording information appear or vanish?** | `exp3_layer_localization/` | **Experiment III:** Tracks feature distance (`cosine_distance`, `relative_l2_distance`, `pearson_correlation`, and `cross_attn_cosine_distance`) across all **14 internal stages** of VoxTell (`00_qwen3_frozen` $\rightarrow$ `01_project_text_embed` $\rightarrow$ `02..07_transformer_layer_1..6` $\rightarrow$ `08_transformer_norm_mask_emb` $\rightarrow$ `09..13_unet_fusion_stage_0..4`) for `left_vs_right`, `base_vs_left`, `base_vs_right`, and `synonym_vs_ref`. |
| **4. How stable is the model to wording & contradictions?** | `exp4_mc_dropout_uncertainty/` | **Experiment IV:** Unfreezes only the dropout layers inside VoxTell and runs $N$ stochastic Monte Carlo (MC) forward passes per prompt. Computes voxel-wise and ROI-level **Epistemic Uncertainty** (predictive variance & mutual information across passes) and **Aleatoric Uncertainty** (average per-pass binary entropy), **Total Predictive Entropy**, inter-pass Dice stability, pass volume CV, and Expected Calibration Error (ECE). Checks whether synonyms destabilize uncertainty and whether contradictory/absent prompts trigger elevated uncertainty. |
| **5. Which pathway carries side & synonym information?** | `exp5_decoder_ablation/` | **Experiment V:** Runs the 4 combinations of **Image Decoder** (`VoxTellDecoder`) and **Text Prompt Decoder** (`TransformerDecoder`): `both_on`, `image_only`, `text_only`, and `both_off`. Repeats Experiments I and II under all 4 settings to isolate which decoder pathway carries laterality and synonym sensitivity. |

---

## 2. Repository File Structure & What Each File Does

```text
positional-bias-medical-vlms/
├── AXIS_II_EXPERIMENTS_GUIDE.md                   # This comprehensive guide
├── NEURITE_OASIS_DATASET.md                       # Documentation of the Neurite-OASIS dataset
├── voxtell-oasis-mechanistic-experiments-kaggle.ipynb  # Kaggle GPU notebook executing all Axis II experiments
│
├── manifests/                                     # Git-tracked manifests (cloned directly into Kaggle via GITHUB_PAT)
│   ├── manifest_neurite_oasis_100_synonyms.csv    # 19,000 rows (100 subjects × 190 Core-7 + negative prompt variants)
│   ├── frozen_prompts_neurite_oasis.csv           # 424 frozen prompts across all 16 bilateral OASIS structures + negatives
│   └── frozen_prompts_supplemental_amos_lesion_lung.csv  # 47 supplemental prompts (AMOS, adrenal tumor, lung lobe nodules)
│
├── scripts/                                       # Modular Python codebase called by the Kaggle notebook or CLI
│   ├── stage_neurite_oasis_100.py                 # Original staging script that prepared the 100 OASIS subjects for HF
│   ├── axis2_prompts.py                           # Deliverable 1: Frozen prompt catalog & 100-subject synonym manifest builder
│   ├── axis2_voxtell_engine.py                    # Core VoxTell inference, 14-stage hooks, MC dropout, ablation & I/O manager
│   ├── axis2_experiments.py                       # Implementations of Experiments I, II, III, IV, and V
│   └── run_axis2_oasis.py                         # Unified CLI & notebook orchestrator + zip packager
│
└── results_axis2/                                 # Dedicated per-experiment output folders (generated locally / in Kaggle)
    ├── exp0_frozen_prompts/
    ├── exp1_side_word_comparison/
    ├── exp2_synonym_shift/
    ├── exp3_layer_localization/
    ├── exp4_mc_dropout_uncertainty/
    └── exp5_decoder_ablation/
```

---

## 3. Detailed Explanation of Every Python File in `scripts/`

### 3.1 `scripts/axis2_prompts.py` — Frozen Prompt Catalog & Git Synonym Manifest Generator
This file implements **Section 1 ("Prompt selection")** and **Deliverable 1** of `Axis_II_Prompt_Sensitivity_Nipu.docx`.

- **What it contains:**
  - `OASIS_LABEL_MAP`: Exact 1–35 FreeSurfer label mapping matching `aligned_seg35.nii.gz` in Neurite-OASIS.
  - `OASIS_BILATERAL_PAIRS`: Maps all 16 bilateral brain structures (`hippocampus`, `thalamus`, `caudate`, `putamen`, `cerebral white matter`, `cerebellum white matter`, `cerebellum cortex`, `pallidum`, `amygdala`, `accumbens`, `ventral DC`, `cerebral cortex`, `lateral ventricle`, `inferior lateral ventricle`, `choroid plexus`, `vessel`) to their `(left_label_id, right_label_id)` pair.
  - `CORE_7_STRUCTURES`: The 7 bilateral structures where VoxTell v1.1 has strong baseline segmentation (`hippocampus`, `thalamus`, `caudate`, `putamen`, `cerebral white matter`, `cerebellum white matter`, `cerebellum cortex`).
  - `EXTENDED_11_STRUCTURES`: Adds `pallidum`, `amygdala`, `accumbens`, and `ventral DC`.
  - `OASIS_SYNONYM_SPEC`: Curated clinical synonyms (aligned with VoxTell's `Qwen3-Embedding-4B` vocabulary), syntactic paraphrases, and anatomical-location descriptions for every structure.
  - `OASIS_ABSENT_ORGAN_PROMPTS`: Cross-domain negative prompts absent from brain MRI (`"kidney"`, `"left kidney"`, `"right kidney"`, `"adrenal tumor"`, `"left adrenal tumor"`, `"right adrenal tumor"`, `"nodule in right upper lobe"`, `"nodule in left lower lobe"`).
- **Variant types generated per structure (26 prompts per structure):**
  1. `base` (2 prompts): `"Segment the {organ} in this image."` and short term `"{organ}"`.
  2. `lateralized` (4 prompts): `"Segment the left/right {organ} in this image."` and short term `"left/right {organ}"`.
  3. `synonym` (10 prompts — 5 per side): e.g., `"Segment the left hippocampal formation in this image."`, `"Segment the left archicortical hippocampus in this image."`, `"Segment the hippocampus left side in this image."`, `"Segment the l. hippocampus in this image."`, `"Segment the cornu ammonis of the left temporal lobe in this image."`.
  4. `paraphrase` (6 prompts — 3 per side):
     - `"Find the {organ} that is on the {side}."`
     - `"Segment the {organ} on the patient's {side} side."`
     - `"Locate the {side}-sided {organ_synonym} in this brain MRI."`
  5. `location_based` (2 prompts — 1 per side): Describes the structure by its anatomical neighborhood rather than its canonical name (e.g., `"Segment the structure in the left medial temporal lobe in this image."`).
  6. `contradictory` (2 prompts — 1 per side): Conflicting spatial wording (`"Segment the left {organ} located on the right hemisphere in this image."`).
- **Key functions:**
  - `build_oasis_prompt_catalog(structures, include_absent_negatives)`: Returns the frozen list of `PromptRecord` objects.
  - `build_supplemental_amos_lesion_lung_prompts()`: Generates the supplemental frozen prompt table for AMOS (`kidney`, `adrenal gland`), adrenal tumor lesions, and lung lobe nodules (`right upper lobe` vs. `left lower lobe`).
  - `export_frozen_prompt_table(output_dir, structures, oasis_manifest_csv)`: Writes `frozen_prompts_neurite_oasis.csv`, `manifest_neurite_oasis_100_synonyms.csv` (100 subjects $\times$ 190 prompts = 19,000 rows), `frozen_prompts_supplemental_amos_lesion_lung.csv`, `frozen_prompts_all.json`, and `README_frozen_prompts.md`, and syncs the CSV manifests to `manifests/` in the Git repository.

---

### 3.2 `scripts/axis2_voxtell_engine.py` — Core VoxTell Architecture Engine & Reproducibility Manager
This file interfaces directly with VoxTell's PyTorch modules (`VoxTellModel`, `VoxTellDecoder`, `TransformerDecoder`, `TransformerDecoderLayer`) without monkeypatching or breaking the installed package.

- **How VoxTell works internally (verified from `VoxTell-main/` and `2511.11450v1.pdf`):**
  1. **Orientation (`load_volume_ras`, `load_seg_ras`):** VoxTell's reader (`NibabelIOWithReorient`) loads NIfTI files into RAS-reoriented voxel space where array axes are ordered `(S, A, R)`:
     - Axis 0 = Superior–Inferior (`S`)
     - Axis 1 = Anterior–Posterior (`A`)
     - Axis 2 = Right–Left (`R`, `RL_AXIS = 2`) — increasing voxel index along axis 2 moves toward the patient's anatomical **Right**.
  2. **Image Encoder (`encode_patch`):** A 6-stage 3D `ResidualEncoder` processes a $192 \times 192 \times 192$ image patch (`skips = model.encoder(patch)`). It never sees the text prompt. Stage-4 bottleneck features ($320 \times 12 \times 12 \times 12$) are projected via `model.project_bottleneck_embed` into $S = 12^3 = 1728$ spatial memory tokens of dimension `query_dim = 2048`.
  3. **Text Backbone & Prompt Decoder (`run_prompt_decoder_trajectory`):**
     - Prompts are wrapped with VoxTell's instruction (`"Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents\nQuery: {prompt}"`), passed through frozen `Qwen/Qwen3-Embedding-4B`, and last-token pooled to a $2560$-d vector (`qwen3_embed`).
     - `model.project_text_embed` projects the $2560$-d vector to `tgt0` ($2048$-d).
     - `model.transformer_decoder` runs 6 pre-norm `TransformerDecoderLayer` layers (`layers[0..5]`). Because `normalize_before=True`, each layer executes `forward_pre` (cross-attention over the 1728 3D image tokens + FFN, with **no** self-attention between different queries in a batch).
     - `model.transformer_decoder.norm` produces the final `mask_embedding` ($2048$-d), which `model.project_to_decoder_channels[0..4]` projects into 5 stage-specific channel vectors.
  4. **Multi-Scale Mask Fusion Decoder (`decode_with_ablation`):**
     - `VoxTellDecoder` upsamples encoder skip connections across 5 resolution stages ($12^3 \rightarrow 24^3 \rightarrow 48^3 \rightarrow 96^3 \rightarrow 192^3$).
     - At intermediate stages, 32-head `torch.einsum('b c h w d, b n c -> b n h w d', x, mask_embedding_reshaped)` fusion maps are concatenated with spatial features.
     - At the final stage ($192^3$, 32 channels), `torch.einsum('b c h w d, b n c -> b n h w d', x, mask_embeddings[-1])` produces the segmentation logits.
- **Key Engine Optimizations & Features:**
  - **Single-Encoder-Pass Sliding Window (`run_sliding_window_custom`):** Instead of calling `predictor.predict_single_image` separately for every prompt variant or ablation setting, `encode_patch` runs **once** per sliding-window tile and evaluates all prompts in chunks through the lightweight decoder.
  - **MC Dropout Context (`mc_dropout_context`):** In VoxTell's `plans.json`, the convolutional U-Net has `dropout_op: null`, whereas every layer of `transformer_decoder` has `nn.Dropout(p=0.1)` (`dropout`, `dropout1`, `dropout2`, `dropout3`) and `nn.MultiheadAttention(..., dropout=0.1)`. `mc_dropout_context` keeps all normalization layers in `.eval()` mode while switching all `nn.Dropout` and `nn.MultiheadAttention` modules to `.train(True)` (plus optional `spatial_dropout_p` on bottleneck features) so $N$ stochastic passes sample from the trained dropout distribution.
  - **Lossless RLE Mask Compression (`encode_mask_rle`, `decode_mask_rle`):** Compresses any $160 \times 224 \times 192$ 3D binary mask into a few hundred bytes of zlib+base64 Run-Length Encoding so **every single row** in `metrics.csv` and `results.json` carries its exact 3D mask for instant reconstruction without blowing up disk space.
  - **Multi-Format Output Manager (`ExperimentOutputManager`):** Writes `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md`, `masks/*.npz`, `masks/*.nii.gz`, and `checkpoints/<subject_id>.json`.

---

### 3.3 `scripts/axis2_experiments.py` — Implementations of Experiments I–V

#### Experiment I: `run_experiment_1_side_words` $\rightarrow$ `exp1_side_word_comparison/`
- **What it runs:** For each subject and structure, runs `base`, `left`, `right` (in both `canonical_sentence` and `short_term` styles), `contradictory_left_on_right`, `contradictory_right_on_left`, plus the 8 `absent_organ_negative` prompts.
- **Metrics recorded in `metrics.csv` and `summary.csv`:**
  - Pairwise mask overlap: `dice_base_vs_left`, `dice_base_vs_right`, `dice_left_vs_right`, `iou_base_vs_left`, `iou_base_vs_right`, `iou_left_vs_right`.
  - Volume differences: `n_pred_voxels`, `vol_diff_left_minus_right`, `vol_diff_base_minus_left`, `vol_diff_base_minus_right`.
  - Spatial location & shift: `pred_centroid_0..2`, `centroid_shift_left_vs_right_vox`, `centroid_shift_left_vs_right_norm` (normalized by 3D volume diagonal), `pred_side_midline` (`L` or `R` relative to the anatomical midline), `frac_voxels_right_of_midline`, `frac_voxels_left_of_midline`.
  - Ground-truth accuracy: `dice_vs_gt_left`, `dice_vs_gt_right`, `dice_vs_gt_bilateral`, `dice_vs_matched_gt`, `dice_vs_contradictory_opposite_gt` (testing whether a `"left"` prompt segments the `"right"` GT or returns 0 overlap), `wrong_side_error`.
  - Failure & success flags: `failure_nearly_identical_flag` (`True` if `dice_bl > 0.90`, `dice_br > 0.90`, and `dice_lr > 0.90`), `left_right_disjoint_flag` (`True` if `dice_lr < 0.05`).

#### Experiment II: `run_experiment_2_and_3_synonyms_and_layers` $\rightarrow$ `exp2_synonym_shift/`
- **What it runs:** Evaluates all synonym, paraphrase, anatomical-location, short-term, and contradictory variants against the canonical lateralized reference prompt (`"Segment the left/right {organ} in this image."`) and base prompt.
- **Metrics recorded in `metrics.csv`, `summary.csv`, and `embedding_vs_prediction_correlations.csv`:**
  1. **Embedding shift at Frozen Qwen3 (`2560-d`):** `qwen3_cosine_dist_vs_ref`, `qwen3_l2_dist_vs_ref`, `qwen3_rel_l2_dist_vs_ref`, `qwen3_pearson_r_vs_ref`, `qwen3_cosine_dist_vs_base`.
  2. **Embedding shift after Prompt Decoder (`2048-d` `mask_embedding`):** `prompt_decoder_cosine_dist_vs_ref`, `prompt_decoder_l2_dist_vs_ref`, `prompt_decoder_rel_l2_dist_vs_ref`, `prompt_decoder_pearson_r_vs_ref`, `prompt_decoder_cosine_dist_vs_base`.
  3. **Similarity map comparison:** `similarity_map_pearson_r_vs_ref`, `similarity_map_cosine_dist_vs_ref` (comparing the final U-Net `einsum` spatial similarity maps).
  4. **Prediction shift vs. reference prompt:** `dice_vs_ref_prompt`, `pred_change_1_minus_dice_ref`, `iou_vs_ref_prompt`, `volume_shift_voxels`, `rel_volume_shift`, `centroid_shift_vs_ref_vox`, `centroid_shift_vs_ref_norm`.
  5. **Effect on ground-truth performance:** `dice_vs_gt`, `ref_dice_vs_gt`, `dice_drop_vs_ref_gt`, `dice_vs_opposite_gt`, `wrong_side_error`.
  6. **Correlation table (`embedding_vs_prediction_correlations.csv`):** Per-structure **Pearson $r$** and **Spearman rank $\rho$** (with $p$-values) between embedding distance (at Qwen3 and at Prompt Decoder) and prediction shift (`1 - Dice(synonym, ref)` and `rel_volume_shift`).

#### Experiment III: `run_experiment_2_and_3_synonyms_and_layers` $\rightarrow$ `exp3_layer_localization/`
- **What it runs:** Measures the internal representation gap across all **14 stages** of VoxTell during the same pass as Experiment II:
  - Stage `0`: `00_qwen3_frozen` (Frozen Qwen3-Embedding-4B output, 2560-d)
  - Stage `1`: `01_project_text_embed` (Initial query state `tgt0`, 2048-d)
  - Stages `2..7`: `02_transformer_layer_1` .. `07_transformer_layer_6` (Query state after each cross-attention + FFN block, plus `cross_attn_cosine_distance` between the $12 \times 12 \times 12$ cross-attention weight maps)
  - Stage `8`: `08_transformer_norm_mask_emb` (Post-LayerNorm `mask_embedding`, 2048-d)
  - Stages `9..13`: `09_unet_fusion_stage_0_12cubed` .. `13_unet_fusion_stage_4_192cubed` (Pooled 3D spatial `einsum` fusion maps across the 5 U-Net decoder scales)
- **Comparisons tracked:** `left_vs_right`, `base_vs_left`, `base_vs_right`, and `synonym_vs_ref (synonym / paraphrase / location_based / contradictory)`.

#### Experiment IV: `run_experiment_4_mc_dropout_uncertainty` $\rightarrow$ `exp4_mc_dropout_uncertainty/`
- **What it runs:** Unfreezes VoxTell's dropout layers and performs $N$ (`MC_DROPOUT_PASSES = 10`) stochastic forward passes per prompt across `base`, `lateralized`, `synonym`, `paraphrase`, `location_based`, `contradictory`, and `absent_organ_negative` prompts.
- **Uncertainty formulas implemented (`compute_mc_uncertainty_metrics`):**
  Let $p_k(v) = \sigma(\text{logit}_k(v)) \in (0, 1)$ be the predicted foreground probability at voxel $v$ on pass $k \in \{1, \dots, N\}$, and $\bar{p}(v) = \frac{1}{N}\sum_{k=1}^N p_k(v)$ be the mean predictive probability:
  - **Epistemic Uncertainty (Predictive Variance):**
    $$\text{Var}(v) = \frac{1}{N} \sum_{k=1}^N \big(p_k(v) - \bar{p}(v)\big)^2$$
  - **Total Predictive Entropy (in bits):**
    $$H(\bar{p}(v)) = -\bar{p}(v)\log_2 \bar{p}(v) - (1 - \bar{p}(v))\log_2(1 - \bar{p}(v))$$
  - **Aleatoric Uncertainty (Expected Entropy across passes, in bits):**
    $$H_{\text{aleatoric}}(v) = \frac{1}{N} \sum_{k=1}^N H(p_k(v))$$
  - **Epistemic Mutual Information (in bits):**
    $$I(y; \theta \mid x, v) = H(\bar{p}(v)) - H_{\text{aleatoric}}(v)$$
- **Metrics recorded in `metrics.csv`, `summary.csv`, and `synonym_uncertainty_stability.csv`:**
  - Whole-volume and ROI-level (`roi_*`, computed over the union of foreground predictions across passes) averages of `mean_prob`, `epistemic_variance`, `max_epistemic_variance`, `epistemic_mi_bits`, `aleatoric_entropy_bits`, `predictive_entropy_bits`.
  - Pass-to-pass mask consistency: `inter_pass_mean_dice` (mean pairwise Dice across all $\binom{N}{2}$ pass pairs), `pass_volume_mean`, `pass_volume_std`, `pass_volume_cv` (coefficient of variation).
  - Calibration: `ece_roi` (10-bin Expected Calibration Error) and `dice_mean_mask_vs_gt`.
  - Wording stability table (`synonym_uncertainty_stability.csv`): Compares `canonical_epistemic_var`, `synonym_mean_epistemic_var`, `synonym_std_epistemic_var`, `synonym_max_min_ratio_epistemic_var`, and `contradictory_epistemic_var` (`contradictory_gt_canonical_uncertainty`).

#### Experiment V: `run_experiment_5_decoder_ablation` $\rightarrow$ `exp5_decoder_ablation/`
- **What it runs:** Evaluates the 4 combinations of **Image Decoder** (`VoxTellDecoder`) and **Text Decoder** (`TransformerDecoder`):
  | Setting | Image Decoder (`VoxTellDecoder`) | Text Decoder (`TransformerDecoder`) | What Happens Inside VoxTell |
  |---|---|---|---|
  | `both_on` | **On** | **On** | Standard VoxTell model: 6-layer cross-attention refines `tgt0` using bottleneck image tokens, and 5-stage U-Net fuses `stage_mask_embeddings` with skip connections. |
  | `image_only` | **On** | **Off** | Bypasses the 6 cross-attention transformer layers (`tgt0` goes directly to `LayerNorm` and `project_to_decoder_channels`), while the full 5-stage U-Net image decoder remains active. Tests whether the text query projection + U-Net decoder alone can separate left/right without transformer cross-attention. |
  | `text_only` | **Off** | **On** | Runs the full 6-layer cross-attention `TransformerDecoder`, but bypasses the learned U-Net convolutional decoder stages (`decoder.stages` / `decoder.transpconvs`), taking the direct dot-product (`einsum`) between the encoder stem features and `stage_mask_embeddings[0]`. |
  | `both_off` | **Off** | **Off** | Bypasses both the 6-layer `TransformerDecoder` and the U-Net decoder stages (direct dot-product of unrefined `tgt0` projection with raw encoder stem features). |
- **Metrics recorded:** Repeats Experiment I (`canonical_dice_vs_matched_gt`, `canonical_wrong_side_rate`, `dice_left_vs_right_overlap`, `centroid_shift_lr_vox`) and Experiment II (`synonym_mean_dice_vs_matched_gt`, `synonym_mean_pred_change_vs_ref`) under all 4 settings.

---

### 3.4 `scripts/run_axis2_oasis.py` — Unified CLI & Notebook Orchestrator
Provides both a Python function (`run_axis2_pipeline`) and a command-line interface so you can run all experiments or individual experiments from Kaggle or a terminal:

```bash
python /kaggle/working/research_repo/scripts/run_axis2_oasis.py \
    --oasis-dir /kaggle/working/neurite_oasis_100 \
    --results-dir /kaggle/working/results \
    --model-dir /kaggle/working/checkpoints/voxtell_v1.1 \
    --experiments all \
    --structure-scope core_7 \
    --max-subjects 100 \
    --mc-passes 10
```

---

## 4. Step-by-Step Workflow in the Kaggle Notebook (`voxtell-oasis-mechanistic-experiments-kaggle.ipynb`)

When you open `voxtell-oasis-mechanistic-experiments-kaggle.ipynb` on Kaggle (with a **GPU T4 x2** or **P100** accelerator enabled and your `GITHUB_PAT` stored in **Add-ons $\rightarrow$ Secrets**):

1. **Cell 0 (Environment Install — no restart needed):**
   - Installs `voxtell>=0.1.2` (with `--no-deps`) plus only its missing runtime deps (`nnunetv2`, `acvl-utils`, `dynamic-network-architectures`, `positional-encodings`, `einops`, ...).
   - Writes a pip constraints file pinning every compiled package Kaggle already has (`numpy`, `scipy`, `torch`, `torchvision`, `transformers`, ...) to its current version, so nothing already loaded in the kernel is replaced on disk. This is what makes **Save Version (Save & Run All)** work without a kernel restart.
   - If `voxtell` is already installed (e.g. via Kaggle's Dependency Manager), it skips immediately.
2. **Cell 1–3 (GPU Check, Working Directory `/kaggle/working/results`, Import Check):**
   - Verifies CUDA GPU availability and package versions.
3. **Cell 4 (Clone `nipu/uncertainty` Branch from GitHub):**
   - Pulls your code and `manifests/manifest_neurite_oasis_100_synonyms.csv` into `/kaggle/working/research_repo` using `UserSecretsClient().get_secret("GITHUB_PAT")`.
4. **Cell 4b (Download 100-Subject Neurite-OASIS Dataset from Hugging Face):**
   - Downloads `Fatema142/neurite-oasis-100-rq-analysis` (`200` NIfTI volumes + `manifest_neurite_oasis_100.csv`) into `/kaggle/working/neurite_oasis_100`.
5. **Cell 5 & 5b (Download `mrokuss/VoxTell` `voxtell_v1.1` + `fp16` Qwen3 Safeguard):**
   - Loads `VoxTellPredictor` and converts `Qwen/Qwen3-Embedding-4B` to `float16` (reducing text-backbone VRAM from ~16 GB to ~8 GB so it never OOMs on Kaggle's 15 GB T4/P100 GPUs) while casting output embeddings back to `float32`.
   - Adds `/kaggle/working/research_repo/scripts` to `sys.path`.
6. **Cell 6 (`exp0_frozen_prompts`):**
   - Loads `manifests/manifest_neurite_oasis_100_synonyms.csv` from the cloned Git repo and exports the frozen prompt tables (`CSV`, `JSON`, `MD`) into `/kaggle/working/results/exp0_frozen_prompts/`.
7. **Cell 7 (Experiment Configuration):**
   - Configure `STRUCTURE_SCOPE = 'core_7'` (or `'extended_11'` / `'all_16'`), `MAX_SUBJECTS = 100` (or slice across runs with `START_INDEX` and `END_INDEX`), `MC_DROPOUT_PASSES = 10`, and `RESUME = True`.
8. **Cell 8 (`exp1_side_word_comparison`):** Runs **Experiment I** and displays the summary table.
9. **Cell 9 (`exp2_synonym_shift` & `exp3_layer_localization`):** Runs **Experiments II & III** jointly in a single pass per volume and displays the correlation and layer-localization tables.
10. **Cell 10 (`exp4_mc_dropout_uncertainty`):** Runs **Experiment IV (MC Dropout Uncertainty)** and displays the wording stability and contradictory uncertainty table.
11. **Cell 11 (`exp5_decoder_ablation`):** Runs **Experiment V (4-Way Decoder Ablation)** and displays the 4-setting comparison table.
12. **Cell 12 (Zip Archive):** Packages `/kaggle/working/results` into `/kaggle/working/results_axis2_archive.zip` for one-click download from Kaggle's Output panel.

---

## 5. Multi-Format Output Structure & Reproducibility Guarantee

Every experiment writes to its own dedicated folder under `RESULTS_ROOT` (`/kaggle/working/results/`):

```text
/kaggle/working/results/
├── exp0_frozen_prompts/
│   ├── frozen_prompts_neurite_oasis.csv
│   ├── manifest_neurite_oasis_100_synonyms.csv
│   ├── frozen_prompts_supplemental_amos_lesion_lung.csv
│   ├── frozen_prompts_all.json
│   └── README_frozen_prompts.md
│
├── exp1_side_word_comparison/
│   ├── metrics.csv                                # Subject × structure × prompt rows + RLE 3D masks
│   ├── summary.csv                                # Aggregated table by target & condition
│   ├── results.json                               # Full structured JSON (metadata + summary + metrics)
│   ├── summary.json                               # Lightweight summary JSON
│   ├── REPORT.md                                  # Formatted Markdown report & summary table
│   ├── checkpoints/                               # Per-subject <subject_id>.json for instant resume
│   └── masks/                                     # Packed .npz mask bundles + sample .nii.gz NIfTI masks
│
├── exp2_synonym_shift/
│   ├── metrics.csv                                # Synonym embedding shifts & mask shifts + RLE 3D masks
│   ├── summary.csv                                # Aggregated per-synonym metrics
│   ├── embedding_vs_prediction_correlations.csv   # Pearson r & Spearman rho per structure
│   ├── results.json
│   ├── summary.json
│   ├── REPORT.md
│   ├── checkpoints/
│   └── masks/
│
├── exp3_layer_localization/
│   ├── metrics.csv                                # 14-stage distances per subject × structure × pair
│   ├── summary.csv                                # Stage-by-stage distance curves (stages 0..13)
│   ├── results.json
│   ├── summary.json
│   ├── REPORT.md
│   └── checkpoints/
│
├── exp4_mc_dropout_uncertainty/
│   ├── metrics.csv                                # Per-subject × prompt epistemic & aleatoric UQ + RLE mean masks
│   ├── summary.csv                                # Aggregated uncertainty table by prompt variant
│   ├── synonym_uncertainty_stability.csv          # Wording stability & contradictory vs canonical UQ
│   ├── results.json
│   ├── summary.json
│   ├── REPORT.md
│   ├── checkpoints/
│   └── masks/                                     # .npz 3D probability/variance/entropy maps + .nii.gz maps
│
└── exp5_decoder_ablation/
    ├── metrics.csv                                # All 4 decoder settings × prompts + RLE 3D masks
    ├── summary.csv                                # 4-way ablation table (both_on, image_only, text_only, both_off)
    ├── results.json
    ├── summary.json
    ├── REPORT.md
    ├── checkpoints/
    └── masks/
```

### How to Reconstruct Any 3D Mask or Uncertainty Map Anytime Without Re-running VoxTell

1. **From `metrics.csv` or `results.json` (using lossless RLE strings):**
   ```python
   import pandas as pd
   from axis2_voxtell_engine import decode_mask_rle, str_to_shape, save_mask_nifti

   df = pd.read_csv("results/exp1_side_word_comparison/metrics.csv")
   row = df[(df["subject_id"] == "OASIS_OAS1_0001_MR1") & (df["target"] == "hippocampus") & (df["condition"] == "left")].iloc[0]

   shape = str_to_shape(row["volume_shape_ras"])
   mask_3d = decode_mask_rle(row["pred_mask_rle_b64"], shape)
   save_mask_nifti(mask_3d, "reconstructed_left_hippocampus.nii.gz")
   ```

2. **From `masks/*_uncertainty_maps.npz` (for Experiment IV 3D probability, epistemic variance, and aleatoric entropy maps):**
   ```python
   import numpy as np

   bundle = np.load("results/exp4_mc_dropout_uncertainty/masks/OASIS_OAS1_0001_MR1_exp4_uncertainty_maps.npz")
   orig_shape = tuple(bundle["orig_shape"])
   bbox = bundle["bbox"]
   # Reconstruct full 3D epistemic variance map in RAS space:
   full_var = np.zeros(orig_shape, dtype=np.float32)
   slicer = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
   full_var[slicer] = bundle["OASIS_P0003_epistemic_var_crop"].astype(np.float32)
   ```

---

## 6. 1-to-1 Verification Table Against `Axis_II_Prompt_Sensitivity_Nipu.docx`

| Section in `Axis_II_Prompt_Sensitivity_Nipu.docx` | Requirements in `.docx` | Where It Is Implemented & Saved |
|---|---|---|
| **1. Prompt selection**<br>*(Deliverable 1: Final frozen prompt table for OASIS, AMOS, and the lesion/lung-lobe dataset)* | - Pull base prompts from OASIS<br>- 5–10 variants per target: `Base (no side)`, `Lateralized` (`left`/`right`), `Synonyms and paraphrases`, `Nonsense or contradictory` (wrong side / absent organ)<br>- Repeat for AMOS, Adrenal Tumor, and Lung Lobes (`"nodule in right upper lobe"` vs. `"nodule in left lower lobe"`)<br>- Record ID, dataset, target, side, variant type, and whether it matches the image side | **Notebook Cell 6 (`exp0_frozen_prompts/`)** & Git `./manifests/`:<br>- `frozen_prompts_neurite_oasis.csv` (26 variants/structure + absent negatives = 424 prompts)<br>- `manifest_neurite_oasis_100_synonyms.csv` (19,000 rows for all 100 subjects)<br>- `frozen_prompts_supplemental_amos_lesion_lung.csv` (47 prompts across AMOS, adrenal tumor, lung lobes)<br>- `frozen_prompts_all.json` & `README_frozen_prompts.md` |
| **2. Experiment I: Does the side word change the output?**<br>*(Deliverable 2: Output-difference results for base/left/right)* | - Run `base`, `left`, `right`, and contradictory prompts on the same scans<br>- Pairwise Dice & IoU (`base_vs_left`, `base_vs_right`, `left_vs_right`)<br>- Difference in predicted volume (`vol_diff_*`)<br>- Difference in location (`centroid_shift_*`, midline side `L`/`R`)<br>- Dice against ground truth (`dice_vs_gt_left`, `dice_vs_gt_right`, `dice_vs_matched_gt`, `dice_vs_contradictory_opposite_gt`)<br>- Failure criterion (`failure_nearly_identical_flag`) & disjoint check (`left_right_disjoint_flag`) | **Notebook Cell 8 (`exp1_side_word_comparison/`)**:<br>- `metrics.csv`, `summary.csv`<br>- `results.json`, `summary.json`, `REPORT.md`<br>- `masks/*.npz` + RLE masks + `.nii.gz` |
| **3. Experiment II: Synonym shift analysis**<br>*(Deliverable 3: Synonym analysis: embedding distance vs. prediction change)* | - **Embedding change:** Cosine distance, L2 distance, and correlation at **frozen Qwen3 output** (`qwen3_*`) AND after **Prompt Decoder** (`prompt_decoder_*`) vs. reference and base prompt<br>- **Prediction change:** Dice (`1 - Dice`) and volume/centroid shift vs. reference prompt and base prompt<br>- **Link between the two:** Correlate embedding distance with prediction change (`embedding_vs_prediction_correlations.csv`, Pearson $r$ & Spearman $\rho$)<br>- **Effect on performance:** Dice against GT (`dice_vs_gt`, `dice_drop_vs_ref_gt`)<br>- **Other angles:** Rank synonyms by distance vs. output ranking (`rank_by_qwen3_dist`, `rank_by_prompt_decoder_dist`, `rank_by_pred_change`) + compare similarity maps per synonym (`similarity_map_*` + saved $6^3$ similarity maps in `.npz`) | **Notebook Cell 9 (`exp2_synonym_shift/`)**:<br>- `metrics.csv`, `summary.csv`<br>- `embedding_vs_prediction_correlations.csv`<br>- `results.json`, `summary.json`, `REPORT.md`<br>- `masks/*_exp2_synonym_masks_packed.npz` (includes per-synonym masks & $6^3$ similarity maps) + `.nii.gz` |
| **4. Layer localization**<br>*(Deliverable 4: Layer-wise change plot)* | - Measure difference between prompt variants (`left_vs_right`, `base_vs_left/right`, and `synonym_vs_ref`) at:<br>  1. Text encoder (`00_qwen3_frozen`)<br>  2. Prompt decoder (`01_project_text_embed` + `02..07_transformer_layer_1..6` + `08_transformer_norm_mask_emb` + cross-attention distance)<br>  3. Fusion & each decoder stage (`09..13_unet_fusion_stage_0..4`)<br>- Plot the gap against stage depth | **Notebook Cell 9 (`exp3_layer_localization/`)**:<br>- `layer_localization_plot.png` *(auto-generated plot of Cosine & Relative L2 distance across all 14 stages)*<br>- `metrics.csv`, `summary.csv`, `results.json`, `summary.json`, `REPORT.md` |
| **5. Uncertainty estimation (MC dropout)**<br>*(Deliverable 5: Uncertainty maps with mean and variance per prompt)* | - Unfreeze only dropout layers, keep everything else fixed, run $N$ stochastic passes per prompt<br>- Compute mean and variance per voxel and per mask<br>- **Epistemic uncertainty:** variance across passes (`epistemic_variance`) & mutual information (`epistemic_mi_bits`)<br>- **Aleatoric uncertainty:** average per-pass predictive entropy (`aleatoric_entropy_bits`)<br>- Compare across prompt variants (synonyms stability vs. contradictory wrong-side prompts) | **Notebook Cell 10 (`exp4_mc_dropout_uncertainty/`)**:<br>- `metrics.csv`, `summary.csv`<br>- `synonym_uncertainty_stability.csv`<br>- `results.json`, `summary.json`, `REPORT.md`<br>- `masks/*_exp4_uncertainty_maps.npz` (saves 3D `mean_prob`, `epistemic_var`, and `aleatoric_ent` maps for every prompt) + `.nii.gz` |
| **6. Decoder ablation (image vs. text decoder)**<br>*(Deliverable 6: Decoder on/off ablation table)* | - Run 4 settings: `Both on` (On/On), `Image only` (On/Off), `Text only` (Off/On), `Both off` (Off/Off)<br>- For each setting, repeat Experiments I and II and compare outputs and prompt sensitivity | **Notebook Cell 11 (`exp5_decoder_ablation/`)**:<br>- `metrics.csv`, `summary.csv`<br>- `results.json`, `summary.json`, `REPORT.md`<br>- `masks/*_exp5_ablation_masks_packed.npz` + `.nii.gz` |

---

## 7. VoxTell Model Size & Parameter Breakdown

### 7.1 Disk / Download Size (from Hugging Face)
| Component | Hugging Face Repo & File | Download Size |
|---|---|---:|
| **VoxTell 3D Segmentation Checkpoint** | `mrokuss/VoxTell` $\rightarrow$ `voxtell_v1.1/fold_0/checkpoint_final.pth` | **1.65 GB** (`1,692.38 MB`) |
| **VoxTell Precomputed Embedding Bank** | `mrokuss/VoxTell` $\rightarrow$ `embeddings/voxtell_v1.1/text_embeddings.npz` | **64.19 MB** |
| **Frozen Text Encoder (`Qwen3-Embedding-4B`)** | `Qwen/Qwen3-Embedding-4B` (`model-00001` + `model-00002.safetensors`) | **7.51 GB** (`7,686.16 MB`) |
| **Total Combined Download Size** | Checkpoint + Embedding Bank + Qwen3-4B | **~9.22 GB** |

### 7.2 Parameter Count & Memory Footprint
| Submodule | Parameter Count | Share of `VoxTellModel` | Memory (`fp32` / `fp16`) |
|---|---:|---:|---:|
| **1. 3D `ResidualEncoder` (Image Backbone)** | `90,296,576` (**90.30M**) | 20.5% | ~344.5 MB (`fp32`) |
| **2. 6-Layer `TransformerDecoder` (Prompt Decoder)** | `251,858,944` (**251.86M**) | 57.2% | ~960.8 MB (`fp32`) |
| **3. Projections (`project_bottleneck_embed` + `project_text_embed` + `project_to_decoder_channels`)** | `85,698,592` (**85.70M**) | 19.5% | ~326.9 MB (`fp32`) |
| **4. 3D `VoxTellDecoder` (U-Net Upsampling & Mask Fusion)** | `12,175,429` (**12.18M**) | 2.8% | ~46.4 MB (`fp32`) |
| **Total `VoxTellModel` (3D Segmentation Network)** | **`440,029,541` (440.03M)** | **100.0%** | **~1.68 GB (`fp32`)** |
| **Frozen `Qwen/Qwen3-Embedding-4B` (Text Backbone)** | **~4,020,000,000 (~4.02B)** | — | **~7.51 GB (`fp16`, offloaded to CPU after prompt embedding)** |

---

## 8. Estimated Runtime Comparison (Kaggle GPU vs. Local M5 MacBook Pro)

| Experiment | Decoder Passes / Subject | **Kaggle GPU (T4 / P100)**<br>*(per subject)* | **Kaggle GPU**<br>*(10 Subjects)* | **Kaggle GPU**<br>*(All 100 Subjects)* | **Local M5 (16 GB)**<br>*(All 100 Subjects)* |
|---|---:|---:|---:|---:|---:|
| **One-Time Setup** (Download HF + Load Model + Embed Prompts) | — | — | **~3–4 min** | **~3–4 min** | **~8–12 min** |
| **Exp 0: Frozen Prompt Table** | 0 | Instant | **< 2 sec** | **< 2 sec** | **< 2 sec** |
| **Exp I: Side Word (`base`/`left`/`right`/contradictory/absent)** | 64 | ~12–15 sec | **~2.5 min** | **~20–25 min** | **~1.5–2 hrs** |
| **Exp II + III: Synonym Shift + 14-Stage Layer Localization** *(run together)* | 182 | ~30–40 sec | **~5–7 min** | **~50–65 min** | **~3.5–5 hrs** |
| **Exp IV: MC Dropout Uncertainty** (`N = 10` passes $\times$ 88 prompts) | 880 | ~2.2–2.8 min | **~22–28 min** | **~3.5–4.5 hrs** | **~15–20 hrs** |
| **Exp V: 4-Way Decoder Ablation** (`both_on`, `image_only`, `text_only`, `both_off`) | $4 \times 84 = 336$ | ~40–50 sec | **~7–8 min** | **~1.1–1.4 hrs** | **~4.5–6 hrs** |
| **TOTAL (All 5 Experiments)** | **1,462 passes/subj** | **~3.8–4.8 min/subj** | **~38–48 min** | **~6.0–7.5 hours** | **~25–33 hours** |

