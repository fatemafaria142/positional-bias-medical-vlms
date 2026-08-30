# Trust the Prompt or Trust the Pixels? Positional Bias in Zero-Shot Medical Image Localization

## Overview

- **Task framing:** "Locate {X}". Scored at whatever output level each model naturally produces (mask, bbox, or point).
- **Prompts:** P0 to P4, positional wording only, `{X}` swapped per dataset to match the actual target (tumor, infection, polyp, or lesion).
- **Zero-shot only.**
- **15 primary models**, all with published or benchmarked grounding capability. Pixel-mask tier (6): CLIPSeg, Moondream2, PaliGemma 2, Florence-2, EVF-SAM, SEEM. Bbox tier (7): Grounding DINO, OWLv2, Kosmos-2, CogVLM-grounding, Qwen2.5-VL-7B, InternVL2.5, DeepSeek-VL2. Point tier (2): Molmo-7B-D, Molmo2-8B.
- **5 datasets:** LGG (brain), MosMedData+ (lung), Kvasir-SEG (colon), QaTa-COV19 (chest), ISIC 2018 (skin), 40 fixed samples each.
- **15,000 primary predictions** total (15 models times 5 datasets times 40 images times 5 prompts).

## Objective
Measure positional bias in text-prompted medical VLMs. When only the positional wording of the prompt changes (image and target fixed), does the model's reported location actually shift accordingly, or does it ignore or misinterpret laterality and region cues?

## Research Questions

- **RQ1:** When a prompt specifies a position (left, right, upper, or lower), does the model's reported location shift in that direction, relative to its own unprompted (P0) answer on the same image?
- **RQ2:** When the prompted position matches the image's true target location, does the model's answer stay consistent with ground truth across all P0 to P4 phrasings, or does wording alone introduce disagreement?
- **RQ3:** When the prompted position does not match the image's true target location, does the model follow the prompt anyway and report the wrong location, indicating it privileges the text instruction over the visual evidence?

## Frequently Asked Questions

**Q: Will each model take my text input?**

A: Yes, all 15. Every model in this plan accepts your P0 to P4 sentence, or that sentence adapted into its required syntax, as text, paired with the image, and every one of them returns a location in some form. None require anything other than a text string plus the image.

**Q: Why are we doing localization instead of segmentation?**

A: Three reasons.

First, the research question itself is about *where*, not about exact boundary shape. A pixel mask answers a harder question than the one this study is actually asking. A point or a box answers "where" just as validly, with less noise from boundary precision, shape, and thresholding, all of which have nothing to do with position.

Second, coverage. When the task was framed as "segment {X}", only 3 of the models could produce a real pixel mask in one step, and most of the rest would have needed a second model (SAM) bolted on just to get a mask at all, which would then measure bias in a combined system rather than in the model itself. Framed as "locate {X}", the task is scoreable at whatever output level each model already produces, mask, box, or point, so all 15 models are usable directly, without a workaround.

Third, cleaner attribution. Segmentation shape variation and positional bias are two different things, and mixing them into one Dice score makes it hard to tell which one caused a low score. Localization isolates the position question on its own.

## Primary Experiment: Positional Prompt to Localization ("Locate" formatting)

The core task is **localization**: the question is *where*, not *exact boundary shape*. This makes the task scoreable at whatever output level a model naturally produces (point, bbox, or mask), and makes 15 models directly usable.

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
| MosMedData+ | Lung | "the infection" | CT ground-glass opacities are infection findings, not tumors |
| Kvasir-SEG | Colon | "the polyp" | Endoscopic polyp localization |
| QaTa-COV19 | Chest | "the infection" | X-ray COVID-19 infection regions, not tumors |
| ISIC 2018 | Skin | "the lesion" | Standard ISIC terminology |

**Note on per-model calling convention:** "Please locate {X}..." is the *semantic content* kept controlled across all 15 models, not always the literal string sent. Every model listed below does accept a text input, but a few expect their own task syntax rather than free-text chat. See the Input column in the model table.

## Zero-Shot Only

No in-context demonstrations, no few-shot conditioning. Each model receives the image and a single P0 to P4 prompt cold. This is the cleanest test of the model's own trained spatial understanding.

## Models: 15 confirmed grounding-capable, zero-shot, Hugging Face, no paid API

All 15 have **published or benchmarked localization capability** (RefCOCO or equivalent), not general VQA models guessing at position. All are free Hugging Face weight downloads, no API billing. Every one of them takes your text prompt directly, the "Input" column shows exactly what form that text needs to take. Sorted by output type.

### Pixel mask output (6 models)

| # | Model | Input | Output | Params | License | Link |
|---|-------|-------|--------|--------|---------|------|
| 1 | CLIPSeg | Image + free-text phrase, no special formatting needed | Pixel mask, single call | ~150M | Apache-2.0 | [huggingface.co/CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) |
| 2 | Moondream2 | Image + free-text object name, sent to its `/segment` endpoint | Pixel mask (also `/point`, `/detect` available) | 2B | Apache-2.0 | [huggingface.co/vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) |
| 3 | PaliGemma 2 (3B) | Image + text using the fixed syntax `segment {X}` | Pixel mask, single call | 3B | MIT (weights) | [huggingface.co/google/paligemma2-3b-pt-224](https://huggingface.co/google/paligemma2-3b-pt-224) |
| 4 | Florence-2 | Image + text using task tokens `<CAPTION_TO_PHRASE_GROUNDING>` then `<REGION_TO_SEGMENTATION>` | Pixel mask, two calls | 0.23B/0.77B | MIT | [huggingface.co/microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) |
| 5 | EVF-SAM | Image + free-text phrase, fused into SAM's prompt embedding | Pixel mask, single call | 1.32B | Not confirmed on HF, verify before use | [huggingface.co/spaces/YxZhang/evf-sam](https://huggingface.co/spaces/YxZhang/evf-sam) |
| 6 | SEEM | Image + free-text phrase (also accepts point, box, or scribble) | Pixel mask | ~341M (Focal-L) | AFL-3.0 (OSI-approved, permits commercial use) | [huggingface.co/xdecoder/SEEM](https://huggingface.co/xdecoder/SEEM) |

### Bounding box output (7 models)

| # | Model | Input | Output | Params | License | Link |
|---|-------|-------|--------|--------|---------|------|
| 7 | Grounding DINO | Image + free-text query, dedicated detector, not a chat model | Bounding box + confidence score | ~172M (base) | Apache-2.0 | [huggingface.co/IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) |
| 8 | OWLv2 | Image + free-text query, dedicated detector, not a chat model | Bounding box + confidence score | ~154M to 350M | Apache-2.0 (weights), MIT (code) | [huggingface.co/google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) |
| 9 | Kosmos-2 | Image + text wrapped in its `<grounding>` special token | Bounding box | ~1.6B | MIT | [huggingface.co/microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) |
| 10 | CogVLM-grounding | Image + free-text chat, grounding-specialist checkpoint | Bounding box | 17B (7B vision + language) | Code Apache-2.0, weights under CogVLM Model License (research and commercial use permitted) | [huggingface.co/THUDM/cogvlm-grounding-generalist-hf](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf) |
| 11 | Qwen2.5-VL-7B | Image + free-text chat | Bounding box, point, or JSON coordinates | 7B | Apache-2.0 | [huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| 12 | InternVL2.5 (1B to 8B) | Image + free-text chat | Bounding box | 1B to 8B | Open-source, verify the exact license on the size you use | [huggingface.co/OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |
| 13 | DeepSeek-VL2-Tiny/Small | Image + text prefixed with its grounding special token | Bounding box (format can be inconsistent per user reports) | 1.0B/2.8B activated (MoE) | Code MIT, weights under DeepSeek Model License | [huggingface.co/deepseek-ai/deepseek-vl2-small](https://huggingface.co/deepseek-ai/deepseek-vl2-small) |

### Point output (2 models)

| # | Model | Input | Output | Params | License | Link |
|---|-------|-------|--------|--------|---------|------|
| 14 | Molmo-7B-D | Image + free-text chat, e.g. "point to {X}" | Single (x, y) point | 7B | Apache-2.0 | [huggingface.co/allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) |
| 15 | Molmo2-8B | Image + free-text chat | Single (x, y) point, improved pointing architecture | 8B | Apache-2.0 | [huggingface.co/allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) |

**Hardware note:** all 15 run on Kaggle's free T4, Colab Pro, or CPU without a large GPU, except CogVLM-grounding, which is best run on Colab Pro (its 17B size is tight on Kaggle's 16GB T4 without quantization).

**Metrics by output type:**

| Output | Metric |
|---|---|
| Pixel mask (6 models) | Mask-centroid shift, Dice(P0, Pn) consistency, wrong-side rate |
| Bbox (7 models) | Box-center shift in the prompted direction, wrong-side rate |
| Point (2 models) | Point shift in the prompted direction, wrong-side rate. This is the most direct signal, since there is no shape noise at all |

## Datasets (5)

All 5 work directly for localization: a ground-truth mask lets you derive a precise location (centroid, bounding box) for comparison against each model's answer. No dataset changes needed.

| # | Domain | Dataset | Official / most reliable source | Practical download mirror | Text prompts included |
|---|---|---|---|---|---|
| 1 | Brain | LGG Segmentation Dataset | [The Cancer Imaging Archive, TCGA-LGG collection](https://www.cancerimagingarchive.net/collection/tcga-lgg/) | [kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | No, must generate |
| 2 | Lung | MosMedData+ | [mosmed.ai/datasets/covid19_1110](https://mosmed.ai/datasets/covid19_1110) (CC BY-NC-ND 3.0) | [github.com/HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT) | Yes, via LViT's release |
| 3 | Colon | Kvasir-SEG | [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/) | Same as official | No, must generate |
| 4 | Chest | QaTa-COV19 | [qatacov.live](http://qatacov.live/) | [kaggle.com/datasets/aysendegerli/qatacov19-dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset) | Yes |
| 5 | Skin | ISIC 2018 | [isic-archive.com](https://www.isic-archive.com/), "Challenge 2018, Task 1: Segmentation" split | Same as official | No, must generate |

## Data Requirement

For P1 to P4 to be meaningful, each evaluation image needs a known true target position, derived from its ground-truth mask:
- Centroid, for point comparison
- Bounding box, for box comparison
- Left/right of midline, upper/lower of midheight, for the categorical wrong-side check

This separates two failure modes:
- **Prompt-following failure**: model ignores the positional word, its answer barely moves between P0 and P1 to P4.
- **Prompt-following but wrong**: the answer does move with the prompt, but toward the wrong side. This reveals the model trusts the text over the image content.

## Comparison Specification

This section specifies exactly what happens between "you give the model an image and a prompt" and "you have a bias score": the pipeline, and how each dataset's ground truth maps to each model's output type.

### Pipeline

1. **Input:** one image from one of the 5 datasets, paired with one prompt from P0 to P4 (text sent in whatever format that model's Input column requires).
2. **Model call:** the model runs once, zero-shot, and returns its raw output: a mask, a bounding box, or a point, depending on its tier.
3. **Parse:** the raw output is converted into one standard representation per tier: a binary mask array for mask models, four coordinates (xmin, ymin, xmax, ymax) for bbox models, one coordinate (x, y) for point models.
4. **Compare to ground truth:** the parsed output is compared against that same image's ground truth (derived from its dataset mask, see Data Requirement above), using the metric for that tier (see table below).
5. **Record:** one result row per (model, dataset, image, prompt): the metric value, the wrong-side/correct-side label, and whether the answer moved from P0 at all.
6. **Aggregate:** results are grouped by model, by dataset, and by prompt to answer RQ1 to RQ3.

### Dataset to Output-Type Mapping

Every dataset supplies the same kind of ground truth (a mask), so the same 40 images per dataset feed all three model tiers. Only the *comparison representation* changes per tier, not the dataset or the ground truth source.

| Dataset | Domain | Ground truth source | Compared against mask models (6) | Compared against bbox models (7) | Compared against point models (2) |
|---|---|---|---|---|---|
| LGG Segmentation | Brain | Tumor mask | Full mask, Dice/IoU | Mask converted to bbox (min/max x, y) | Mask centroid |
| MosMedData+ | Lung | Infection mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| Kvasir-SEG | Colon | Polyp mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| QaTa-COV19 | Chest | Infection mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| ISIC 2018 | Skin | Lesion mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |

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

- Fixed set of **40 samples per dataset** (same 40 images across all models and all 5 prompts).
- Each model runs 5 datasets times 40 images times 5 prompts (P0 to P4) = **1,000 predictions per model**.
- 15 models times 1,000 = **15,000 predictions total**.

## Method

1. Derive positional ground truth (centroid, bbox, left/right, upper/lower) for all 200 images from mask annotations.
2. Generate P0 to P4 for every image, keeping wording strictly controlled per the constraint above.
3. Run all 15 primary models zero-shot on all 5 prompt variants for the same 40-image, 5-dataset set, each scored at its native output level (mask, box, or point).
4. Compute per-model: location shift in the prompted direction, consistency (Dice for mask models, IoU for box models), and wrong-side/failure rate.
5. Rank all 15 models by lowest positional bias (highest prompt-following accuracy on correct-side cases, lowest wrong-side rate).
6. Report per-domain and per-model breakdowns.

## Other Files

- `archived_segmentation_models_table.md`, archived reference table of published segmentation architectures. Kept for reference/comparison only, not part of this active plan.
