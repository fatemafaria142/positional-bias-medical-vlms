# How to Run This Study, End to End

A practical runbook for building the dataset, running all 13 models, and comparing their answers to ground truth. For the *why* behind any step here, see `position_bias_research_plan.md` — this file is just the *do this, in this order* version.

## Phase 1 — Build your 50-sample dataset per domain (one-time setup)

1. **Download each dataset** to `data/raw/<name>/` relative to the repo root. See each script's docstring for the exact download link and expected folder layout:
   - `scripts/sample_kvasir.py` → Kvasir-SEG, direct anonymous download, no login.
   - `scripts/sample_isic.py` → ISIC 2018 (Task 1, Validation split), direct anonymous download, no login.
   - `scripts/sample_lgg.py` → LGG Segmentation, needs a Kaggle account + API token (`kaggle datasets download -d mateuszbuda/lgg-mri-segmentation`).
2. **Run that dataset's sampling script**:
   ```
   python scripts/sample_lgg.py
   python scripts/sample_kvasir.py
   python scripts/sample_isic.py
   ```
   Each one filters to images with a visible finding, samples exactly 50 with a fixed random seed (so everyone gets the same 50), copies the 50 images + masks into `samples/<dataset>/`, and writes `manifest_<dataset>.csv` with ground truth + all 5 prompts (250 rows per dataset).
3. **Sanity-check a few samples visually**:
   ```
   python scripts/visualize_ground_truth.py lgg
   python scripts/visualize_ground_truth.py kvasir
   python scripts/visualize_ground_truth.py isic
   ```
   Open 2-3 PNGs from `samples/<dataset>/visual_check/` and confirm the red dot/box actually sits on the side the label claims (left/right of the yellow vertical line, upper/lower of the yellow horizontal line). Delete `visual_check/` afterward — it's a debug aid, not part of the dataset.
4. **Delete `data/raw/<name>/`** once its `samples/<dataset>/` folder is confirmed correct — you no longer need the full original download.
5. **Upload the three `samples/<dataset>/` folders to Google Drive.** This is what your teammate clones/mounts — small, self-contained, no re-downloading needed. Total size is roughly 130MB across all 3 (10MB LGG + 2.2MB Kvasir-SEG + 118MB ISIC).

## Phase 2 — Run each model on each dataset

6. **For each of the 13 models**, **for each of the 3 datasets**, **for each of the 50 images**, **for each of the 5 prompts (P0-P4)**: load `image_path` from that dataset's manifest, send it to the model along with `prompt_text` (adapted into that model's required syntax — see the Input column in `position_bias_research_plan.md`'s Models section), and record the model's raw output (mask, bbox, or point, depending on its tier).

   Per-model input syntax quick reference:
   | Model | What to send |
   |---|---|
   | CLIPSeg, Moondream2 (`/segment`), Grounding DINO, OWLv2, CogVLM-grounding, Qwen2.5-VL, InternVL2.5, Molmo-7B-D, Molmo2-8B | `prompt_text` as free-text, no reformatting |
   | PaliGemma 2 | `segment {X}` syntax (not the full sentence) |
   | Florence-2 | Task tokens: `<CAPTION_TO_PHRASE_GROUNDING>` then `<REGION_TO_SEGMENTATION>` |
   | Kosmos-2 | `prompt_text` wrapped in its `<grounding>` special token |
   | DeepSeek-VL2 | `prompt_text` prefixed with its grounding special token |

7. **Parse the raw output into one standard representation per tier**:
   - Mask-tier (CLIPSeg, Moondream2, PaliGemma 2, Florence-2): keep the predicted mask; also compute its centroid (average of white pixels) for the position check.
   - Bbox-tier (Grounding DINO, OWLv2, Kosmos-2, CogVLM-grounding, Qwen2.5-VL, InternVL2.5, DeepSeek-VL2): compute box-center = `((xmin+xmax)/2, (ymin+ymax)/2)`.
   - Point-tier (Molmo-7B-D, Molmo2-8B): use the (x, y) output directly.
8. **Save one row per (model, dataset, image, prompt)** into a results file, e.g. `results.csv`, with columns:
   `model, dataset, sample_id, prompt_id, pred_x, pred_y, pred_mask_path (mask-tier only), pred_bbox_xmin/ymin/xmax/ymax (bbox-tier only)`.

## Phase 3 — Compare predictions to ground truth

9. **Join each results row back to its manifest row** (match on `dataset` + `sample_id` + `prompt_id`) to pull in `gt_centroid_x/y`, `gt_bbox_xmin/ymin/xmax/ymax`, `gt_side_lr/ud`, and `mask_path`.
10. **Compute, per row**:
    - **Side check**: is `pred_x` vs (image width / 2) and `pred_y` vs (image height / 2) the same side as `gt_side_lr` / `gt_side_ud`? → label as `correct_side` or `wrong_side`.
    - **Shift check**: distance between this row's prediction and that same model+image's P0 prediction → label as `shifted` or `no_shift` (tolerance: ~5% of image width/height).
    - **Shape/overlap** (secondary, not a position metric): Dice against `mask_path` for mask-tier models, IoU against `gt_bbox_*` for bbox-tier models.
11. **Aggregate**:
    - Group by `model` → rank all 13 by wrong-side rate and prompt-following accuracy (answers RQ1).
    - Group by `dataset` × `prompt` → check whether wording alone shifts answers when the prompt already matches ground truth (answers RQ2).
    - Filter to rows where the prompted side is *wrong* relative to ground truth → check whether the model follows the text anyway (answers RQ3).
12. **Report per-domain and per-model breakdowns** — this is your final result table, comparing all 13 models across all 3 domains (Brain/LGG, Colon/Kvasir-SEG, Skin/ISIC 2018) and all 5 prompts.

## Reference

- Full rationale, research questions, and dataset details: `position_bias_research_plan.md`
- Ground-truth computation, worked with a real example: see "Beginner Walkthrough" section in the plan
- Manifest column reference: see "Manifest column reference" table in the plan
