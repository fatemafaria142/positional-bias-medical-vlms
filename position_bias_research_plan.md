# Trust the Prompt or Trust the Pixels? Positional Bias in Zero-Shot Medical Image Localization

## Overview

- **Task framing:** "Locate {X}". Scored at whatever output level each model naturally produces (mask, bbox, or point).
- **Prompts:** P0 to P4, positional wording only, `{X}` swapped per dataset to match the actual target (tumor, infection, polyp, or lesion).
- **Zero-shot only.**
- **13 primary models**, all with published or benchmarked grounding capability. Pixel-mask tier (4): CLIPSeg, Moondream2, PaliGemma 2, Florence-2. Bbox tier (7): Grounding DINO, OWLv2, Kosmos-2, CogVLM-grounding, Qwen2.5-VL-7B, InternVL2.5, DeepSeek-VL2. Point tier (2): Molmo-7B-D, Molmo2-8B.
- **5 datasets:** LGG/TextBraTS (brain), Stanford Thyroid Cine-clip (thyroid), Kvasir-SEG/Kvasir-VQA-x1 (colon), QaTa-COV19 (chest), PI-CAI (prostate), 30 fixed samples each.
- **9,750 primary predictions** total (13 models times 5 datasets times 30 images times 5 prompts).

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
| LGG Segmentation / TextBraTS | Brain | "the tumor" | Correct, glioma tumor regions |
| Stanford Thyroid Cine-clip | Thyroid | "the nodule" | Ultrasound-visible thyroid nodule regions |
| Kvasir-SEG / Kvasir-VQA-x1 | Colon | "the polyp" | Endoscopic polyp localization |
| QaTa-COV19 | Chest | "the infection" | X-ray COVID-19 infection regions, not tumors |
| PI-CAI | Prostate | "the lesion" | MRI-visible PI-RADS-scored prostate lesion regions |

**Note on per-model calling convention:** "Please locate {X}..." is the *semantic content* kept controlled across all 13 models, not always the literal string sent. Every model listed below does accept a text input, but a few expect their own task syntax rather than free-text chat. See the Input column in the model table.

**Worked P0–P4 prompt strings, one per dataset:**

| Dataset | P0 | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|---|
| LGG / TextBraTS | "Please locate the tumor in this image." | "Please locate the tumor on the left side of this image." | "Please locate the tumor on the right side of this image." | "Please locate the tumor in the upper region of this image." | "Please locate the tumor in the lower region of this image." |
| Stanford Thyroid | "Please locate the nodule in this image." | "Please locate the nodule on the left side of this image." | "Please locate the nodule on the right side of this image." | "Please locate the nodule in the upper region of this image." | "Please locate the nodule in the lower region of this image." |
| Kvasir-SEG / -VQA-x1 | "Please locate the polyp in this image." | "Please locate the polyp on the left side of this image." | "Please locate the polyp on the right side of this image." | "Please locate the polyp in the upper region of this image." | "Please locate the polyp in the lower region of this image." |
| QaTa-COV19 | "Please locate the infection in this image." | "Please locate the infection on the left side of this image." | "Please locate the infection on the right side of this image." | "Please locate the infection in the upper region of this image." | "Please locate the infection in the lower region of this image." |
| PI-CAI | "Please locate the lesion in this image." | "Please locate the lesion on the left side of this image." | "Please locate the lesion on the right side of this image." | "Please locate the lesion in the upper region of this image." | "Please locate the lesion in the lower region of this image." |

**Important — do not use each dataset's native text as the prompt.** TextBraTS's radiology-style annotations, the Stanford Thyroid TI-RADS/diagnosis fields, QaTa-COV19's `.xlsx` clinical descriptions, and PI-CAI's PI-RADS/PSAd report fields all vary in length, detail, and confidence from case to case. Feeding any of that in as `{X}`'s surrounding context would violate the plan's core constraint (line 48: no prompt variant may be more detailed or informative than another except for positional wording), and it would confound "did the model move because of position wording" with "did the model move because this case's report said something more specific." For the primary P0–P4 experiment, every model receives only the fixed, generated sentence above (or its model-specific syntax equivalent) — nothing from the dataset's own annotation file.

Native text's role is limited to two things: (1) it confirms each dataset is genuinely paired multimodal data, useful if you want to cite the dataset as image+text+mask in your write-up; (2) it is available for an *optional* secondary experiment — e.g., replacing P0 with the dataset's real report text once, per image, to see whether realistic clinical phrasing shifts the model's answer differently than your synthetic P0 does. That would be a separate analysis from RQ1–RQ3, not a substitute for the controlled P0–P4 set.

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

## Datasets (5)

All 5 work directly for localization: a ground-truth mask lets you derive a precise location (centroid, bounding box) for comparison against each model's answer. No dataset changes needed.

| # | Domain | Dataset | Official / most reliable source | Practical download mirror | Text prompts included | Native text type |
|---|---|---|---|---|---|---|
| 1 | Brain | **TextBraTS** (built on BraTS2020) | [arxiv.org/abs/2506.16784](https://arxiv.org/html/2506.16784v2) | Released alongside the paper, derived from BraTS2020 training set (369 volumes) | Yes | Expert-written free-text radiology-style annotations, paired per MRI volume |
| 2 | Thyroid | **Stanford AIMI Thyroid Ultrasound Cine-clip** | [stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5) | Same as official (registration required) | Yes (structured, not free-text) | TI-RADS descriptors, lesion size/location, and histopathological diagnosis, per nodule, alongside pixel-level masks (192 cine-clips, 17,412 images) |
| 3 | Colon | **Kvasir-VQA-x1** (VLQA subtask, built on Kvasir-SEG/HyperKvasir) | [github.com/simula/Kvasir-VQA-x1](https://github.com/Simula/Kvasir-VQA-x1) | [github.com/simula/ImageCLEFmed-MEDVQA-GI-2023](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2023) | Yes | Real location questions ("where is the abnormality?") paired with a segmentation-mask answer |
| 4 | Chest | QaTa-COV19 | [qatacov.live](http://qatacov.live/) | [kaggle.com/datasets/aysendegerli/qatacov19-dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset), text via [github.com/HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT) | Yes | `.xlsx` free-text clinical descriptions, paired per image |
| 5 | Prostate | **PI-CAI** + `picai_labels` | [pi-cai.grand-challenge.org](https://pi-cai.grand-challenge.org/) | [github.com/DIAGNijmegen/picai_labels](https://github.com/DIAGNijmegen/picai_labels) | Yes (structured, not free-text) | PI-RADS score, PSA density, and radiology-report-derived lesion location, per case, alongside lesion masks (~1,500 public cases, T2W/DWI/ADC) |

### Dataset Details

**LGG Segmentation / TextBraTS (Brain, MRI):** LGG Segmentation (TCGA-LGG collection, TCIA) is 110 patients' pre-operative FLAIR MRI slices with manually-drawn glioma tumor masks, popularized via the Kaggle mirror. TextBraTS pairs the same style of scan (BraTS2020's 369 multimodal MRI volumes: T1, T1ce, T2, FLAIR) with per-volume tumor segmentation masks *and* expert-written free-text radiology-style descriptions of tumor location, size, and appearance — released alongside its 2025 paper specifically to support text-guided segmentation research. Use TextBraTS's images if you want native text; LGG's Kaggle mirror is the easier download if you only need image+mask and plan to generate your own prompts.

**Stanford AIMI Thyroid Ultrasound Cine-clip (Thyroid, Ultrasound):** 192 cine-clip (video) ultrasound exams from 167 patients, 17,412 individual frames total, each with a pixel-level nodule segmentation mask. Alongside the mask, every nodule carries structured clinical text: TI-RADS descriptors (composition, echogenicity, shape, margin, echogenic foci), lesion size and location, and the histopathological diagnosis (benign/malignant) from biopsy. Released by Stanford's AIMI Center; download requires free registration (data use agreement, no cost) at the Stanford AIMI portal.

**Kvasir-SEG / Kvasir-VQA-x1 (Colon, Endoscopy):** Kvasir-SEG is 1,000 GI endoscopy images with manually-verified polyp masks and bounding boxes (Simula Research Lab, widely used polyp-segmentation benchmark). Kvasir-VQA-x1 builds on the same underlying image pool (HyperKvasir/Kvasir-Instrument) and adds 159,549 question-answer pairs, including its VLQA subtask which pairs an explicit location question ("where is the abnormality?") with a segmentation-mask answer — a genuine text+mask pairing, not a caption bolted on afterward.

**QaTa-COV19 (Chest, X-ray):** ~2,100 chest X-rays of confirmed COVID-19 cases with pixel-level infection (ground-glass opacity) masks, originally from Qatar University and Tampere University. The LViT project's release adds a `.xlsx` file per split (train/val/test) with a free-text clinical description for every image, used in their text-guided segmentation work — this is real radiologist-style text, not model-generated.

**PI-CAI + picai_labels (Prostate, MRI):** PI-CAI (Prostate Imaging: Cancer AI) is a multi-center public dataset of ~1,500 biparametric prostate MRI cases (T2W, DWI, ADC sequences) from Dutch and Norwegian hospitals, built for the PI-CAI Grand Challenge. The companion `picai_labels` repo provides expert-verified lesion segmentation masks, and each case's radiology report carries structured text: PI-RADS score (1–5 cancer-suspicion rating), PSA density, and lesion location as read by the radiologist.

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

Every dataset supplies the same kind of ground truth (a mask), so the same 30 images per dataset feed all three model tiers. Only the *comparison representation* changes per tier, not the dataset or the ground truth source.

| Dataset | Domain | Ground truth source | Compared against mask models (4) | Compared against bbox models (7) | Compared against point models (2) |
|---|---|---|---|---|---|
| LGG Segmentation / TextBraTS | Brain | Tumor mask | Full mask, Dice/IoU | Mask converted to bbox (min/max x, y) | Mask centroid |
| Stanford Thyroid Cine-clip | Thyroid | Nodule mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| Kvasir-SEG / Kvasir-VQA-x1 | Colon | Polyp mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| QaTa-COV19 | Chest | Infection mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| PI-CAI | Prostate | Lesion mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |

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
- Each model runs 5 datasets times 30 images times 5 prompts (P0 to P4) = **750 predictions per model**.
- 13 models times 750 = **9,750 predictions total**.

## Method

1. Derive positional ground truth (centroid, bbox, left/right, upper/lower) for all 150 images from mask annotations.
2. Generate P0 to P4 for every image, keeping wording strictly controlled per the constraint above.
3. Run all 13 primary models zero-shot on all 5 prompt variants for the same 30-image, 5-dataset set, each scored at its native output level (mask, box, or point).
4. Compute per-model: location shift in the prompted direction, consistency (Dice for mask models, IoU for box models), and wrong-side/failure rate.
5. Rank all 13 models by lowest positional bias (highest prompt-following accuracy on correct-side cases, lowest wrong-side rate).
6. Report per-domain and per-model breakdowns.

## Other Files

- `archived_segmentation_models_table.md`, archived reference table of published segmentation architectures. Kept for reference/comparison only, not part of this active plan.
