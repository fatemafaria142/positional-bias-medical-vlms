# Prompts Used

This file lists every prompt used in the positional-bias experiments, and how each one gets adapted per model before it's actually sent.

> **Note:** the wording below (P0–P4) is the *controlled semantic content*, not always the literal string sent to a model. Several models expect their own task syntax instead of free-text chat, so **the base prompt has to be rewritten per model** before it's passed in — see "Per-model adaptation" below. When adding a new model, always check whether it needs its own rewrite rule rather than assuming the base P0–P4 text can be sent as-is.

## Base prompt set (P0–P4)

`{X}` is the target noun, fixed per dataset (same noun across all 5 prompts for that dataset) so swapping it never becomes a second variable alongside position.

| ID | Template |
|----|----------|
| P0 | "Please locate {X} in this image." |
| P1 | "Please locate {X} on the left side of this image." |
| P2 | "Please locate {X} on the right side of this image." |
| P3 | "Please locate {X} in the upper region of this image." |
| P4 | "Please locate {X} in the lower region of this image." |

`{X}` per dataset:
- LGG (brain MRI) → "the tumor"
- Kvasir-SEG (colon) → "the polyp"
- ISIC 2018 (skin) → "the lesion"

**Design constraint:** no variant may be more detailed, confident, or clinically informative than another except for the intended positional wording.

## Ground truth: mask → bbox, mask → point

Every dataset's ground truth starts as a binary segmentation **mask** (the answer key — never sent to a model). Both a bounding box and a point are derivable from it, one-way only (a bbox/point cannot reconstruct the mask's shape). Implemented once in `scripts/common.py`'s `mask_ground_truth()` and reused by every dataset's sampling script:

- **Mask → bbox**: `xs, ys = np.nonzero(mask)` (coordinates of every foreground pixel), then the tightest enclosing rectangle is `(xs.min(), ys.min(), xs.max(), ys.max())`.
- **Mask → point**: the centroid (center of mass) of those same foreground pixels — `cx, cy = xs.mean(), ys.mean()`.
- **Side labels** (`left`/`right`, `upper`/`lower`) are then derived from that centroid vs. half the image width/height.

These three (`gt_bbox_*`, `gt_centroid_x/y`, `gt_side_lr/ud`) all live in each dataset's `manifest_<dataset>.csv` alongside `prompt_text`, so any model — mask-tier, bbox-tier, or point-tier — can be scored against the same ground truth regardless of what format its own predictions come back in.

## Datasets

50 samples per dataset (150 images total), each paired with all 5 prompts (P0–P4) → 250 (image, prompt) rows per dataset, 750 total.

| Dataset | Domain | What's located (`{X}`) | Modality | Source |
|---|---|---|---|---|
| LGG (brain MRI) | Brain | the tumor | 2D (MRI slices) | [kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) |
| Kvasir-SEG | Colon | the polyp | 2D (endoscopy frames) | [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/) — direct download: [datasets.simula.no/downloads/kvasir-seg.zip](https://datasets.simula.no/downloads/kvasir-seg.zip) |
| ISIC 2018 (Task 1, Validation split) | Skin | the lesion | 2D (dermoscopy images) | [challenge.isic-archive.com/data](https://challenge.isic-archive.com/data/) — [Validation Input](https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip) + [Validation GroundTruth](https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip) |

Sampling and ground-truth derivation (centroid, bbox, left/right, upper/lower) is done per dataset by `scripts/sample_lgg.py`, `scripts/sample_kvasir.py`, and `scripts/sample_isic.py`, each writing its own `samples/<dataset>/manifest_<dataset>.csv`.

## Per-model adaptation — change the prompt based on the model

Every model takes text input, but several need their own task syntax instead of the raw "Please locate..." sentence. **Always rewrite the base prompt for the specific model being called** — don't send P0–P4 verbatim except to models that explicitly take free-text chat with no special formatting.

| Model | Provider | Params | Input format | Output format | Rewrite rule |
|---|---|---|---|---|---|
| CLIPSeg | IDS Lab (Uni Göttingen) / CIDAS | ~150M | Image + free-text phrase, no special formatting | Pixel mask (single call) | None — send P0–P4 sentence as-is |
| Moondream2 | moondream (vikhyatk) | 2B | Image + free-text object name, sent to its `/segment` endpoint | Pixel mask (also `/point`, `/detect`) | Strip "Please locate " down to just the noun phrase + positional clause — the endpoint takes an object description, not a full instruction sentence |
| PaliGemma 2 (3B) | Google | 3B | Image + text using fixed task syntax `segment {X}` | Pixel mask (single call) | Strip "Please locate " → prefix with `segment ` → "Please locate the polyp on the right side of this image." → "segment the polyp on the right side of this image" |
| Florence-2 | Microsoft | 0.23B / 0.77B | Image + task tokens, two calls | Pixel mask (two calls) | Not the P0–P4 sentence directly — call 1 uses `<CAPTION_TO_PHRASE_GROUNDING>` with the noun+position phrase to get a box, call 2 feeds that box into `<REGION_TO_SEGMENTATION>` |
| Grounding DINO | IDEA Research | ~172M (base) | Image + free-text query, dedicated detector (not a chat model) | Bounding box + confidence score | None — send P0–P4 sentence (or the noun+position phrase) as-is |
| OWLv2 | Google | ~154M–350M | Image + free-text query, dedicated detector (not a chat model) | Bounding box + confidence score | None — send P0–P4 sentence as-is |
| Kosmos-2 | Microsoft | ~1.6B | Image + text wrapped in its `<grounding>` special token | Bounding box | Strip "Please locate " → wrap remaining phrase in `<grounding><phrase> {phrase} </phrase>` — e.g. `<grounding><phrase> the tumor on the left side of this image </phrase>` |
| Qwen2.5-VL-3B | Alibaba (Qwen) | 3B | Image + free-text chat | Bounding box, point, or JSON coordinates | None strictly required — free-text chat model; can send P0–P4 as-is, optionally ask explicitly for JSON/box format in the instruction |
| InternVL2.5 (1B–8B) | OpenGVLab / Shanghai AI Lab | 1B–8B | Image + free-text chat | Bounding box | None — send P0–P4 sentence as-is |
| LocateAnything-3B | NVIDIA | 3B | Image (native resolution, up to 2.5K) + free-text task template (categories, referring expressions, pointing queries) | Text with structured coordinate tokens: boxes `<box> x1, y1, x2, y2 </box>` or points `<box> x, y </box>` | None required — free-text referring-expression grounding; P0–P4 sentence can be sent as the task template directly |
| Molmo2-8B | Allen Institute for AI | 8B | Image + free-text chat, e.g. "point to {X}" | Single (x, y) point, improved pointing architecture | Strip "Please locate " → prefix with `Point to ` (matches PixMo's pointing-task phrasing) — "Please locate the tumor on the left side of this image." → "Point to the tumor on the left side of this image." |
| North-Micro-Vision-Instruct | Cohere Labs | 2.4B (2B LM + 400M vision encoder) | Interleaved text + image(s), chat template (`apply_chat_template`) | Text; grounding boxes returned as `[x1, y1, x2, y2]` normalized to a 0–1000 scale | None required for the instruction itself — free-text chat; box coordinates must be rescaled back to pixels: `x_px = x/1000 * image_width` (same for y/height) when parsing the output |

For any new model, check its model card first to see whether it needs one of these rewrites, and add a `to_<model>_prompt()` function (following the pattern in `models/paligemma/run_paligemma.py`, and the Kosmos-2/MolmoE notebooks) rather than sending the raw template.

## Where this is implemented

- `scripts/common.py` — defines the base `PROMPTS` dict (P0–P4) and writes `prompt_id`/`prompt_text` into each dataset's manifest.
- `models/paligemma/run_paligemma.py` — `to_paligemma_prompt()`.
- `Kosmos2_Positional_Bias.ipynb` — `to_kosmos_phrase()`.
- `MolmoE_Positional_Bias.ipynb` — `to_molmo_prompt()`.

Every results CSV records both columns so the rewrite is auditable: `prompt_text` (the controlled P0–P4 sentence) and `model_prompt` (what was actually sent to that model).
