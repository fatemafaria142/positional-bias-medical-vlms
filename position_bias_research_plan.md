# Trust the Prompt or Trust the Pixels? Positional Bias in Zero-Shot Medical Image Localization

## Overview

- **Task framing:** "Locate {X}". Scored at whatever output level each model naturally produces (mask, bbox, or point).
- **Prompts:** P0 to P4, positional wording only, `{X}` swapped per dataset to match the actual target (tumor, infection, polyp, or lesion).
- **Zero-shot only.**
- **13 primary models**, all with published or benchmarked grounding capability. Pixel-mask tier (4): CLIPSeg, Moondream2, PaliGemma 2, Florence-2. Bbox tier (7): Grounding DINO, OWLv2, Kosmos-2, CogVLM-grounding, Qwen2.5-VL-7B, InternVL2.5, DeepSeek-VL2. Point tier (2): Molmo-7B-D, Molmo2-8B.
- **3 datasets, all 2D:** LGG Segmentation (brain), Kvasir-SEG/Kvasir-VQA-x1 (colon), QaTa-COV19 (chest), 30 fixed samples each — chosen as the 3 easiest to work with (no registration, simple flat folder structures, well-documented, small-to-moderate size). Ground truth (centroid, bbox, left/right, upper/lower) always comes from each dataset's mask, never from text, so LGG works identically to the other 2 despite having no native text of its own.
- **5,850 primary predictions** total (13 models times 3 datasets times 30 images times 5 prompts).

## Objective
Measure positional bias in text-prompted medical VLMs. When only the positional wording of the prompt changes (image and target fixed), does the model's reported location actually shift accordingly, or does it ignore or misinterpret laterality and region cues?

## Research Questions

- **RQ1:** When a prompt specifies a position (left, right, upper, or lower), does the model's reported location shift in that direction, relative to its own unprompted (P0) answer on the same image?
- **RQ2:** When the prompted position matches the image's true target location, does the model's answer stay consistent with ground truth across all P0 to P4 phrasings, or does wording alone introduce disagreement?
- **RQ3:** When the prompted position does not match the image's true target location, does the model follow the prompt anyway and report the wrong location, indicating it privileges the text instruction over the visual evidence?

## Frequently Asked Questions

**Q: Will each model take my text input?**

A: Yes, all 13. Every model in this plan accepts your P0 to P4 sentence, or that sentence adapted into its required syntax, as text, paired with the image, and every one of them returns a location in some form. None require anything other than a text string plus the image.

**Q: Why are we doing localization instead of segmentation?**

A: Three reasons.

First, the research question itself is about *where*, not about exact boundary shape. A pixel mask answers a harder question than the one this study is actually asking. A point or a box answers "where" just as validly, with less noise from boundary precision, shape, and thresholding, all of which have nothing to do with position.

Second, coverage. When the task was framed as "segment {X}", only 3 of the models could produce a real pixel mask in one step, and most of the rest would have needed a second model (SAM) bolted on just to get a mask at all, which would then measure bias in a combined system rather than in the model itself. Framed as "locate {X}", the task is scoreable at whatever output level each model already produces, mask, box, or point, so all 13 models are usable directly, without a workaround.

Third, cleaner attribution. Segmentation shape variation and positional bias are two different things, and mixing them into one Dice score makes it hard to tell which one caused a low score. Localization isolates the position question on its own.

## Primary Experiment: Positional Prompt to Localization ("Locate" formatting)

The core task is **localization**: the question is *where*, not *exact boundary shape*. This makes the task scoreable at whatever output level a model naturally produces (point, bbox, or mask), and makes 13 models directly usable.

**Prompt set (P0 to P4), parameterized by target `{X}`:**
- P0: "Please locate {X} in this image."
- P1: "Please locate {X} on the left side of this image."
- P2: "Please locate {X} on the right side of this image."
- P3: "Please locate {X} in the upper region of this image."
- P4: "Please locate {X} in the lower region of this image."

**Prompt-design constraint:** all variants must be semantically controlled. No variant may be more detailed, confident, or clinically informative than another except for the intended positional wording. `{X}` is fixed per dataset (same noun across all 5 prompts) so swapping it never becomes a second confounding variable alongside position.

**`{X}` per dataset:**

| Dataset | Domain | `{X}` | Why |
|---|---|---|---|
| LGG Segmentation | Brain | "the tumor" | Correct, LGG masks are glioma tumor regions |
| Kvasir-SEG / Kvasir-VQA-x1 | Colon | "the polyp" | Endoscopic polyp localization |
| QaTa-COV19 | Chest | "the infection" | X-ray COVID-19 infection regions, not tumors |

**Note on per-model calling convention:** "Please locate {X}..." is the *semantic content* kept controlled across all 13 models, not always the literal string sent. Every model listed below does accept a text input, but a few expect their own task syntax rather than free-text chat. See the Input column in the model table.

**Worked P0–P4 prompt strings, one per dataset:**

| Dataset | P0 | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|---|
| LGG Segmentation | "Please locate the tumor in this image." | "Please locate the tumor on the left side of this image." | "Please locate the tumor on the right side of this image." | "Please locate the tumor in the upper region of this image." | "Please locate the tumor in the lower region of this image." |
| Kvasir-SEG / -VQA-x1 | "Please locate the polyp in this image." | "Please locate the polyp on the left side of this image." | "Please locate the polyp on the right side of this image." | "Please locate the polyp in the upper region of this image." | "Please locate the polyp in the lower region of this image." |
| QaTa-COV19 | "Please locate the infection in this image." | "Please locate the infection on the left side of this image." | "Please locate the infection on the right side of this image." | "Please locate the infection in the upper region of this image." | "Please locate the infection in the lower region of this image." |

**Important — do not use each dataset's native text as the prompt.** QaTa-COV19's `.xlsx` clinical descriptions vary in length, detail, and confidence from case to case. Feeding any of that in as `{X}`'s surrounding context would violate the plan's core constraint (line 48: no prompt variant may be more detailed or informative than another except for positional wording), and it would confound "did the model move because of position wording" with "did the model move because this case's report said something more specific." For the primary P0–P4 experiment, every model receives only the fixed, generated sentence above (or its model-specific syntax equivalent) — nothing from the dataset's own annotation file.

Native text's role is limited to two things: (1) it confirms a dataset is genuinely paired multimodal data, useful if you want to cite it as image+text+mask in your write-up; (2) it is available for an *optional* secondary experiment — e.g., replacing P0 with the dataset's real report text once, per image, to see whether realistic clinical phrasing shifts the model's answer differently than your synthetic P0 does. That would be a separate analysis from RQ1–RQ3, not a substitute for the controlled P0–P4 set. Of these 3 datasets, only QaTa-COV19 has this native text; LGG and Kvasir-SEG do not, though Kvasir-VQA-x1 (if you use it instead of plain Kvasir-SEG) adds real location Q&A on top of Kvasir-SEG's images.

## Zero-Shot Only

No in-context demonstrations, no few-shot conditioning. Each model receives the image and a single P0 to P4 prompt cold. This is the cleanest test of the model's own trained spatial understanding.

## Models: 13 confirmed grounding-capable, zero-shot, Hugging Face, no paid API

All 13 have **published or benchmarked localization capability** (RefCOCO or equivalent), not general VQA models guessing at position. All are free Hugging Face weight downloads, no API billing. Every one of them takes your text prompt directly, the "Input" column shows exactly what form that text needs to take. Sorted by output type.

### Pixel mask output (4 models)

| # | Model | Input | Output | Params | License | Link |
|---|-------|-------|--------|--------|---------|------|
| 1 | CLIPSeg | Image + free-text phrase, no special formatting needed | Pixel mask, single call | ~150M | Apache-2.0 | [huggingface.co/CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) |
| 2 | Moondream2 | Image + free-text object name, sent to its `/segment` endpoint | Pixel mask (also `/point`, `/detect` available) | 2B | Apache-2.0 | [huggingface.co/vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) |
| 3 | PaliGemma 2 (3B) | Image + text using the fixed syntax `segment {X}` | Pixel mask, single call | 3B | MIT (weights) | [huggingface.co/google/paligemma2-3b-pt-224](https://huggingface.co/google/paligemma2-3b-pt-224) |
| 4 | Florence-2 | Image + text using task tokens `<CAPTION_TO_PHRASE_GROUNDING>` then `<REGION_TO_SEGMENTATION>` | Pixel mask, two calls | 0.23B/0.77B | MIT | [huggingface.co/microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) |

### Bounding box output (7 models)

| # | Model | Input | Output | Params | License | Link |
|---|-------|-------|--------|--------|---------|------|
| 5 | Grounding DINO | Image + free-text query, dedicated detector, not a chat model | Bounding box + confidence score | ~172M (base) | Apache-2.0 | [huggingface.co/IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) |
| 6 | OWLv2 | Image + free-text query, dedicated detector, not a chat model | Bounding box + confidence score | ~154M to 350M | Apache-2.0 (weights), MIT (code) | [huggingface.co/google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) |
| 7 | Kosmos-2 | Image + text wrapped in its `<grounding>` special token | Bounding box | ~1.6B | MIT | [huggingface.co/microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) |
| 8 | CogVLM-grounding | Image + free-text chat, grounding-specialist checkpoint | Bounding box | 17B (7B vision + language) | Code Apache-2.0, weights under CogVLM Model License (research and commercial use permitted) | [huggingface.co/THUDM/cogvlm-grounding-generalist-hf](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf) |
| 9 | Qwen2.5-VL-7B | Image + free-text chat | Bounding box, point, or JSON coordinates | 7B | Apache-2.0 | [huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| 10 | InternVL2.5 (1B to 8B) | Image + free-text chat | Bounding box | 1B to 8B | Open-source, verify the exact license on the size you use | [huggingface.co/OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |
| 11 | DeepSeek-VL2-Tiny/Small | Image + text prefixed with its grounding special token | Bounding box (format can be inconsistent per user reports) | 1.0B/2.8B activated (MoE) | Code MIT, weights under DeepSeek Model License | [huggingface.co/deepseek-ai/deepseek-vl2-small](https://huggingface.co/deepseek-ai/deepseek-vl2-small) |

### Point output (2 models)

| # | Model | Input | Output | Params | License | Link |
|---|-------|-------|--------|--------|---------|------|
| 12 | Molmo-7B-D | Image + free-text chat, e.g. "point to {X}" | Single (x, y) point | 7B | Apache-2.0 | [huggingface.co/allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) |
| 13 | Molmo2-8B | Image + free-text chat | Single (x, y) point, improved pointing architecture | 8B | Apache-2.0 | [huggingface.co/allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) |

**Hardware note:** all 13 run on Kaggle's free T4, Colab Pro, or CPU without a large GPU, except CogVLM-grounding, which is best run on Colab Pro (its 17B size is tight on Kaggle's 16GB T4 without quantization).

**Metrics by output type:**

| Output | Metric |
|---|---|
| Pixel mask (4 models) | Mask-centroid shift, Dice(P0, Pn) consistency, wrong-side rate |
| Bbox (7 models) | Box-center shift in the prompted direction, wrong-side rate |
| Point (2 models) | Point shift in the prompted direction, wrong-side rate. This is the most direct signal, since there is no shape noise at all |

## Datasets (3)

All 3 are **2D** and chosen specifically as the easiest to work with: no registration required, simple/flat folder structures, small-to-moderate size, and extensively documented (all 3 are among the most-tutorialized medical imaging datasets that exist). A ground-truth mask lets you derive a precise location (centroid, bounding box, left/right, upper/lower) for comparison against each model's answer.

**LGG has no native text; QaTa-COV19 does (and Kvasir-VQA-x1 adds real location Q&A on top of plain Kvasir-SEG).** This only affects the optional bonus uses of native text (provenance, secondary real-report-text experiment, stratification by clinical severity) — it doesn't affect the core P0–P4 experiment, which never uses native text as input for any of the 3 datasets anyway.

| # | Domain | Dataset | Dimension | Official / most reliable source | Practical download mirror | Native text? |
|---|---|---|---|---|---|---|
| 1 | Brain | **LGG Segmentation Dataset** | **2D** (FLAIR slice PNGs) | [The Cancer Imaging Archive, TCGA-LGG collection](https://www.cancerimagingarchive.net/collection/tcga-lgg/) | [kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | No |
| 2 | Colon | **Kvasir-VQA-x1** (VLQA subtask, built on Kvasir-SEG/HyperKvasir) | **2D** (single endoscopy frames) | [github.com/simula/Kvasir-VQA-x1](https://github.com/Simula/Kvasir-VQA-x1) | [github.com/simula/ImageCLEFmed-MEDVQA-GI-2023](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2023) | Yes — real location questions ("where is the abnormality?") paired with a segmentation-mask answer |
| 3 | Chest | QaTa-COV19 | **2D** (single X-ray images) | [qatacov.live](http://qatacov.live/) | [kaggle.com/datasets/aysendegerli/qatacov19-dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset), text via [github.com/HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT) | Yes — `.xlsx` free-text clinical descriptions, paired per image |

### Dataset Details

**LGG Segmentation (Brain, MRI):** TCGA-LGG collection (TCIA), 110 patients' pre-operative FLAIR MRI slices with manually-drawn glioma tumor masks, popularized via the Kaggle mirror. Already 2D — individual slice PNGs, not volumes. No native text; use your generated P0–P4 only. No registration required; a very widely-tutorialized dataset (common U-Net teaching example).

**Kvasir-SEG / Kvasir-VQA-x1 (Colon, Endoscopy):** Kvasir-SEG is 1,000 GI endoscopy images with manually-verified polyp masks and bounding boxes (Simula Research Lab, widely used polyp-segmentation benchmark). Kvasir-VQA-x1 builds on the same underlying image pool (HyperKvasir/Kvasir-Instrument) and adds 159,549 question-answer pairs, including its VLQA subtask which pairs an explicit location question ("where is the abnormality?") with a segmentation-mask answer — a genuine text+mask pairing, not a caption bolted on afterward.

**QaTa-COV19 (Chest, X-ray):** ~2,100 chest X-rays of confirmed COVID-19 cases with pixel-level infection (ground-glass opacity) masks, originally from Qatar University and Tampere University. The LViT project's release adds a `.xlsx` file per split (train/val/test) with a free-text clinical description for every image, used in their text-guided segmentation work — this is real radiologist-style text, not model-generated.

## Data Requirement

For P1 to P4 to be meaningful, each evaluation image needs a known true target position, derived from its ground-truth mask:
- Centroid, for point comparison
- Bounding box, for box comparison
- Left/right of midline, upper/lower of midheight, for the categorical wrong-side check

This separates two failure modes:
- **Prompt-following failure**: model ignores the positional word, its answer barely moves between P0 and P1 to P4.
- **Prompt-following but wrong**: the answer does move with the prompt, but toward the wrong side. This reveals the model trusts the text over the image content.

## Beginner Walkthrough: How Image, Ground Truth, and Model Output Actually Map Together

This section traces one single (image, prompt) task through the whole pipeline with real numbers, so the moving parts in the Comparison Specification below make sense before you read it in the abstract.

**The three files involved, per dataset:**
- An **image file** (e.g. `polyp_014.jpg`) — this is the picture you show the model.
- A **mask file** (e.g. `polyp_014_mask.jpg`) — a black-and-white image, same size as the picture, where white pixels mark exactly where the tumor/lesion/polyp/mass is. This is your ground truth. You never send the mask to the model — it's answer key, not input.
- Your **prompt text** (one of the 5 P0–P4 sentences) — generated from the plan, not from the dataset.

**Step 1 — Turn the mask into 5 numbers (this is "deriving ground truth"):**

Say `polyp_014_mask.jpg` is 256×256 pixels, and the white (polyp) pixels are clustered around x=180, y=60. The code (`mask_ground_truth()` in `scripts/build_sample_manifest.py`) does:
1. Find every white pixel's (x, y) coordinate.
2. Average them → **centroid** = (180, 60). This one point stands in for "where the polyp is."
3. Find the min/max x and y among white pixels → **bounding box**, e.g. (150, 40, 210, 80).
4. Compare centroid x (180) to half the image width (256/2 = 128): 180 > 128, so **side_lr = "right"**.
5. Compare centroid y (60) to half the image height (128): 60 < 128, so **side_ud = "upper"**.

So this one mask file becomes 5 plain numbers/labels: `centroid=(180,60)`, `bbox=(150,40,210,80)`, `side_lr=right`, `side_ud=upper`. That's your ground truth for this image — computed once, reused for every model and every prompt on this image.

**Step 2 — `manifest.csv` pairs that ground truth with a prompt.** One row of the manifest the script builds looks like:

| dataset | sample_id | image_path | gt_centroid_x | gt_centroid_y | gt_side_lr | gt_side_ud | prompt_id | prompt_text |
|---|---|---|---|---|---|---|---|---|
| kvasir | polyp_014 | data/Kvasir-SEG/images/polyp_014.jpg | 180 | 60 | right | upper | P2 | "Please locate the polyp on the right side of this image." |

This row is one complete task: "send this image, with this text, to a model." There are 5 such rows per image (one per P0–P4), all sharing the same ground truth columns since the ground truth doesn't change — only the prompt does.

**Step 3 — Send image + prompt_text to a model, get a raw answer back.** You never send anything from the `gt_*` columns to the model — only `image_path` (loaded as pixels) and `prompt_text`. The model's raw answer depends on its tier:
- A mask-tier model (e.g. CLIPSeg) returns its own black-and-white mask of where *it* thinks the polyp is.
- A bbox-tier model (e.g. Grounding DINO) returns 4 numbers, e.g. `(30, 45, 90, 85)`.
- A point-tier model (e.g. Molmo) returns one (x, y) pair directly, e.g. `(60, 65)`.

**Step 4 — Reduce the model's raw answer to one point, the same way you reduced the ground-truth mask.** If it returned a mask, average its white pixels → predicted centroid. If it returned a bbox, take the box center: `((30+90)/2, (45+85)/2) = (60, 65)`. If it returned a point already, use it as-is. Either way, you now have one predicted (x, y), directly comparable to the ground-truth centroid (180, 60) from Step 1.

**Step 5 — Compare the predicted point to the ground truth, three ways:**
1. **Side check:** is predicted x (60) > 128? No → predicted side_lr = "left". The prompt asked for "right" (P2). Predicted "left" ≠ prompted "right" → this is a **wrong-side** result — the model pointed the opposite way from what the text asked.
2. **Shift check:** compare this P2 prediction (60, 65) to the *same model's* P0 prediction on the *same image* (its answer with no positional wording at all). If P0's answer was also near (60, 65), the model didn't move at all when told to go right — a **prompt-following failure**, a different failure mode from wrong-side.
3. **Distance/overlap:** Euclidean distance between predicted (60,65) and ground truth (180,60), normalized by the image diagonal — a magnitude, not just a right/wrong label. (For mask/bbox models you also compute Dice/IoU here, a separate shape-quality number, not a position number.)

**Step 6 — Record and repeat.** Each comparison becomes one row in your results file: `model, dataset, image, prompt, predicted_point, gt_point, correct_or_wrong_side, shifted_or_not, distance`. Do this for every (model × dataset × image × prompt) combination — 13 × 3 × 30 × 5 = 5,850 rows total — then group and average those rows by model, by dataset, and by prompt to answer RQ1–RQ3 (e.g. "what % of Grounding DINO's answers on Kvasir-SEG were wrong-side when the prompt said 'right'?").

The Comparison Specification below formalizes exactly this same walkthrough for all three model tiers at once.

## Comparison Specification

This section specifies exactly what happens between "you give the model an image and a prompt" and "you have a bias score": the pipeline, and how each dataset's ground truth maps to each model's output type.

### Pipeline

1. **Input:** one image from one of the 3 datasets, paired with one prompt from P0 to P4 (text sent in whatever format that model's Input column requires).
2. **Model call:** the model runs once, zero-shot, and returns its raw output: a mask, a bounding box, or a point, depending on its tier.
3. **Parse:** the raw output is converted into one standard representation per tier: a binary mask array for mask models, four coordinates (xmin, ymin, xmax, ymax) for bbox models, one coordinate (x, y) for point models.
4. **Compare to ground truth:** the parsed output is compared against that same image's ground truth (derived from its dataset mask, see Data Requirement above), using the metric for that tier (see table below).
5. **Record:** one result row per (model, dataset, image, prompt): the metric value, the wrong-side/correct-side label, and whether the answer moved from P0 at all.
6. **Aggregate:** results are grouped by model, by dataset, and by prompt to answer RQ1 to RQ3.

### Dataset to Output-Type Mapping

Every dataset supplies the same kind of ground truth (a mask), so the same 30 images per dataset feed all three model tiers. Only the *comparison representation* changes per tier, not the dataset or the ground truth source.

| Dataset | Domain | Ground truth source | Compared against mask models (4) | Compared against bbox models (7) | Compared against point models (2) |
|---|---|---|---|---|---|
| LGG Segmentation | Brain | Tumor mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| Kvasir-SEG / Kvasir-VQA-x1 | Colon | Polyp mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| QaTa-COV19 | Chest | Infection mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |

### Metric by Tier

| Tier | Predicted representation | Ground truth representation | Metric |
|---|---|---|---|
| Pixel mask | Binary mask | Binary mask | Dice(predicted, ground truth) for shape agreement; centroid of predicted mask compared to centroid of ground truth mask for the shift/wrong-side check |
| Bounding box | (xmin, ymin, xmax, ymax) | (xmin, ymin, xmax, ymax), derived from mask | IoU(predicted box, ground truth box) for overlap; box-center of each compared for the shift/wrong-side check |
| Point | (x, y) | Centroid (x, y), derived from mask | Euclidean distance between predicted point and ground truth centroid, normalized by image diagonal, for the shift/wrong-side check |

### Wrong-Side Classification (applies identically to all three tiers)

Once each prediction is reduced to a single (x, y) reference point (its own centroid, box-center, or raw point), the same rule applies everywhere:

- **Left/right:** predicted x compared to image width divided by 2.
- **Upper/lower:** predicted y compared to image height divided by 2.
- **Correct-side:** predicted side matches the prompted side (P1 to P4).
- **Wrong-side:** predicted side is the opposite of the prompted side.
- **No-shift (prompt-following failure):** predicted point on P1 to P4 is within a small tolerance (for example, 5 percent of image width or height) of the predicted point on P0, meaning the model did not move its answer at all.

### Worked Example (one row per tier)

| Model | Dataset | Prompt | Raw output | Parsed | Ground truth | Metric result |
|---|---|---|---|---|---|---|
| CLIPSeg (mask) | Kvasir-SEG | P1: locate the polyp, left | Binary mask | Mask + its centroid | Polyp mask + its centroid | Dice = 0.71; predicted centroid on the left; correct-side |
| Grounding DINO (bbox) | QaTa-COV19 | P2: locate the infection, right | (120, 80, 210, 160) | Box + its center | Infection mask converted to box | IoU = 0.44; predicted center on the left; wrong-side |
| Molmo-7B-D (point) | LGG Segmentation | P0: locate the tumor | (134, 97) | Point | Mask centroid at (140, 101) | Distance = 7.4 px normalized; same side as P0; no ground-truth mismatch to check at P0 |

## Sampling Protocol

- Fixed set of **30 samples per dataset** (same 30 images across all models and all 5 prompts).
- Each model runs 3 datasets times 30 images times 5 prompts (P0 to P4) = **450 predictions per model**.
- 13 models times 450 = **5,850 predictions total**.

## Method

1. Derive positional ground truth (centroid, bbox, left/right, upper/lower) for all 90 images from mask annotations.
2. Generate P0 to P4 for every image, keeping wording strictly controlled per the constraint above.
3. Run all 13 primary models zero-shot on all 5 prompt variants for the same 30-image, 3-dataset set, each scored at its native output level (mask, box, or point).
4. Compute per-model: location shift in the prompted direction, consistency (Dice for mask models, IoU for box models), and wrong-side/failure rate.
5. Rank all 13 models by lowest positional bias (highest prompt-following accuracy on correct-side cases, lowest wrong-side rate).
6. Report per-domain and per-model breakdowns.

## Other Files

- `archived_segmentation_models_table.md`, archived reference table of published segmentation architectures. Kept for reference/comparison only, not part of this active plan.
