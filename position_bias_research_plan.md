# Trust the Prompt or Trust the Pixels? Positional Bias in Zero-Shot Medical Image Localization

## Overview

- **Task framing:** "Locate {X}". Scored at whatever output level each model naturally produces (mask, bbox, or point).
- **Prompts:** P0 to P4, positional wording only, `{X}` swapped per dataset to match the actual target (tumor, infection, polyp, or lesion).
- **Zero-shot only.**
- **13 primary models**, all with published or benchmarked grounding capability. Pixel-mask tier (4): CLIPSeg, Moondream2, PaliGemma 2, Florence-2. Bbox tier (7): Grounding DINO, OWLv2, Kosmos-2, CogVLM-grounding, Qwen2.5-VL-7B, InternVL2.5, DeepSeek-VL2. Point tier (2): Molmo-7B-D, Molmo2-8B.
- **3 datasets, all 2D:** LGG Segmentation (brain), Kvasir-SEG/Kvasir-VQA-x1 (colon), ISIC 2018 (skin), 50 fixed samples each — chosen for reliable, direct (no-registration or no-Kaggle-required) download and small size. QaTa-COV19 (originally used for Chest) was dropped: its only Kaggle source is the full 9,258-image, ~4-5GB "Extended" release with no smaller mirror available, too large for this project's disk constraints. Ground truth (centroid, bbox, left/right, upper/lower) always comes from each dataset's mask, never from text, so LGG and ISIC work identically to Kvasir despite having no native text of their own.
- **9,750 primary predictions** total (13 models times 3 datasets times 50 images times 5 prompts).

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
| ISIC 2018 | Skin | "the lesion" | Standard ISIC terminology |

**Note on per-model calling convention:** "Please locate {X}..." is the *semantic content* kept controlled across all 13 models, not always the literal string sent. Every model listed below does accept a text input, but a few expect their own task syntax rather than free-text chat. See the Input column in the model table.

**Worked P0–P4 prompt strings, one per dataset:**

| Dataset | P0 | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|---|
| LGG Segmentation | "Please locate the tumor in this image." | "Please locate the tumor on the left side of this image." | "Please locate the tumor on the right side of this image." | "Please locate the tumor in the upper region of this image." | "Please locate the tumor in the lower region of this image." |
| Kvasir-SEG / -VQA-x1 | "Please locate the polyp in this image." | "Please locate the polyp on the left side of this image." | "Please locate the polyp on the right side of this image." | "Please locate the polyp in the upper region of this image." | "Please locate the polyp in the lower region of this image." |
| ISIC 2018 | "Please locate the lesion in this image." | "Please locate the lesion on the left side of this image." | "Please locate the lesion on the right side of this image." | "Please locate the lesion in the upper region of this image." | "Please locate the lesion in the lower region of this image." |

**Important — do not use each dataset's native text as the prompt.** None of the 3 active datasets have usable free-text native annotations bundled with the images we downloaded (see Dataset Details below), so this caveat is largely moot for the current 3-dataset set, but the rule still holds if native text is ever reintroduced: no prompt variant may be more detailed or informative than another except for positional wording (line 48), so real report text can't just be substituted in as `{X}`'s context without confounding "did the model move because of position wording" with "did the model move because this case's report said something more specific."

Of these 3 datasets, only Kvasir-VQA-x1 (an optional upgrade over plain Kvasir-SEG) has real native text — its location Q&A pairs. LGG and ISIC 2018 have none.

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

All 3 are **2D** and chosen for reliable, small, direct-download access. A ground-truth mask lets you derive a precise location (centroid, bounding box, left/right, upper/lower) for comparison against each model's answer.

**None of the 3 active datasets have usable native text.** Kvasir-VQA-x1 is available as an optional upgrade over plain Kvasir-SEG for real location Q&A, but the base Kvasir-SEG/LGG/ISIC files used here do not carry native text. This only affects the optional bonus uses of native text (provenance, secondary real-report-text experiment, stratification by clinical severity) — it doesn't affect the core P0–P4 experiment, which never uses native text as input for any of the 3 datasets anyway.

| # | Domain | Dataset | Dimension | Official / most reliable source | Practical download mirror | Native text? |
|---|---|---|---|---|---|---|
| 1 | Brain | **LGG Segmentation Dataset** | **2D** (FLAIR slice PNGs) | [The Cancer Imaging Archive, TCGA-LGG collection](https://www.cancerimagingarchive.net/collection/tcga-lgg/) | [kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | No |
| 2 | Colon | **Kvasir-SEG** (Kvasir-VQA-x1 optional upgrade) | **2D** (single endoscopy frames) | [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/) | Direct download, no login: [datasets.simula.no/downloads/kvasir-seg.zip](https://datasets.simula.no/downloads/kvasir-seg.zip) | No (base Kvasir-SEG); Yes via Kvasir-VQA-x1 upgrade — real location questions ("where is the abnormality?") paired with a segmentation-mask answer |
| 3 | Skin | **ISIC 2018** (Task 1, Validation split) | **2D** (dermoscopy images) | [challenge.isic-archive.com/data](https://challenge.isic-archive.com/data/) | Direct download, no login: [Validation Input](https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip) (~240MB) + [Validation GroundTruth](https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip) (~0.75MB) | No |

**Important — ISIC 2018 has two tasks; we use Task 1 only.** The ISIC 2018 challenge page describes "2,594 images and 12,970 corresponding ground truth response masks (5 for each image)" — that 5-masks-per-image statistic is **Task 2** (dermoscopic attribute detection: pigment network, negative network, streaks, milia-like cysts, globules — 5 separate binary maps per image). This study uses **Task 1** (lesion boundary segmentation), which has exactly **1 mask per image** (`ISIC_XXXXXXX.jpg` ↔ `ISIC_XXXXXXX_segmentation.png`), matching every other dataset in this plan. Also note we use the **Validation** split (100 images, ~240MB), not the Training split (2,594 images, ~11GB) — the Training Input zip is far larger than this project needs; Validation alone comfortably supplies the 50 samples required, at a fraction of the download size.

### Dataset Details

**LGG Segmentation (Brain, MRI):** TCGA-LGG collection (TCIA), 110 patients' pre-operative FLAIR MRI slices with manually-drawn glioma tumor masks, popularized via the Kaggle mirror. Already 2D — individual slice PNGs, not volumes. No native text; use your generated P0–P4 only. Requires a Kaggle account/API token to download (no anonymous direct link found).

**Kvasir-SEG (Colon, Endoscopy):** 1,000 GI endoscopy images with manually-verified polyp masks and bounding boxes (Simula Research Lab, widely used polyp-segmentation benchmark). Has a direct, anonymous download link (no Kaggle/registration needed) — the only one of the 3 datasets with this advantage. Kvasir-VQA-x1 (github.com/simula/Kvasir-VQA-x1) is available as an optional upgrade if you want native location Q&A text on top of the same image pool.

**ISIC 2018 (Skin, Dermoscopy):** Direct, anonymous download (no Kaggle/registration needed) from the ISIC Challenge archive. We use the Task 1 Validation split (100 dermoscopic images with single-mask lesion boundaries) — see the Task 1 vs Task 2 note above. **Caveat:** ISIC dermoscopy images are often tightly cropped/zoomed on the lesion, so some lesions fill most of the frame with a centroid very close to true center — verified visually (`scripts/visualize_ground_truth.py isic`), where one sampled image's lesion spanned nearly the entire frame, making its left/right and upper/lower label a thin margin rather than a clear-cut side. This doesn't break the pipeline, but wrong-side/correct-side results on near-center ISIC samples should be read with that in mind — they carry a weaker positional signal than a clearly off-center lesion would.

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
- An **image file** (e.g. `cju6yywx1whbb0871ksgfgf9f.jpg`) — this is the picture you show the model.
- A **mask file** (e.g. `cju6yywx1whbb0871ksgfgf9f_mask.png`) — a black-and-white image, same size as the picture, where white pixels mark exactly where the tumor/lesion/polyp/mass is. This is your ground truth. You never send the mask to the model — it's answer key, not input.
- Your **prompt text** (one of the 5 P0–P4 sentences) — generated from the plan, not from the dataset.

**Step 1 — Turn the mask into 5 numbers (this is "deriving ground truth"):**

This is a real, verified example — not a hypothetical — taken directly from `samples/kvasir/manifest_kvasir.csv`, sample `cju6yywx1whbb0871ksgfgf9f`. Its image is 622×530 pixels (width × height), and its mask (the polyp region) has 27,645 white pixels. The code (`mask_ground_truth()` in `scripts/common.py`) does:

1. **Find every white pixel's (x, y) coordinate** in the mask — all 27,645 of them.
2. **Average them** → **centroid** = (402.37, 164.05). This one point stands in for "where the polyp is."
3. **Find the min/max x and y among white pixels** → **bounding box** = (300, 74, 503, 254).
4. **Compare centroid x (402.37) to half the image width** (622 / 2 = 311): 402.37 > 311, so **side_lr = "right"**.
5. **Compare centroid y (164.05) to half the image height** (530 / 2 = 265): 164.05 < 265, so **side_ud = "upper"**.

So this one mask file becomes 5 plain numbers/labels: `centroid=(402.37, 164.05)`, `bbox=(300, 74, 503, 254)`, `side_lr=right`, `side_ud=upper`. That's your ground truth for this image — computed once, reused for every model and every prompt on this image. (You can verify this yourself: `python scripts/visualize_ground_truth.py kvasir`, then open `samples/kvasir/visual_check/cju6yywx1whbb0871ksgfgf9f.png` — the red dot and box will sit exactly here, clearly right of and above the image's yellow midline/midheight crosshair.)

**Ground-truth computation table (shareable summary of the above):**

| Step | What we compute | Formula | Real value (this example) |
|---|---|---|---|
| 0 | Image size | width × height, from the actual image file | 622 × 530 px |
| 1 | Mask pixel coordinates | every white (foreground) pixel's (x, y) in the mask | 27,645 pixels |
| 2 | Centroid | mean of all pixel x's, mean of all pixel y's | (402.37, 164.05) |
| 3 | Bounding box | (min x, min y, max x, max y) among mask pixels | (300, 74, 503, 254) |
| 4 | Left/right threshold | width / 2 | 622 / 2 = 311 |
| 5 | `gt_side_lr` | "right" if centroid_x > width/2, else "left" | 402.37 > 311 → **right** |
| 6 | Upper/lower threshold | height / 2 | 530 / 2 = 265 |
| 7 | `gt_side_ud` | "upper" if centroid_y < height/2, else "lower" | 164.05 < 265 → **upper** |

**Step 2 — each dataset's own `manifest_<dataset>.csv` pairs that ground truth with a prompt.** Each of the 3 sampling scripts (`scripts/sample_lgg.py`, `scripts/sample_kvasir.py`, `scripts/sample_isic.py`) copies its own 50 sampled images and masks into a small `samples/<dataset>/` folder — small enough to upload to Google Drive on its own, so you can delete the full original dataset download afterward and only keep this folder. One row of `samples/kvasir/manifest_kvasir.csv` looks like:

| dataset | sample_id | image_path | mask_path | gt_centroid_x | gt_centroid_y | gt_side_lr | gt_side_ud | prompt_id | prompt_text |
|---|---|---|---|---|---|---|---|---|---|
| kvasir | cju6yywx1whbb0871ksgfgf9f | images/cju6yywx1whbb0871ksgfgf9f.jpg | masks/cju6yywx1whbb0871ksgfgf9f_mask.png | 402.37 | 164.05 | right | upper | P2 | "Please locate the polyp on the right side of this image." |

Note `image_path` is relative to `samples/kvasir/` itself, not the original full-dataset download — the image file was physically copied there, so the manifest stays valid even after you delete the original download. This row is one complete task: "send this image, with this text, to a model." There are 5 such rows per image (one per P0–P4), all sharing the same ground truth columns since the ground truth doesn't change — only the prompt does.

**Manifest column reference.** Every row already carries the ground truth needed for all three model tiers (mask, bbox, point) at once — you don't need a separate manifest per tier, you just read whichever column matches the model you're scoring:

| Column | What it is | Which tier uses it |
|---|---|---|
| `image_path` | The actual image to send to the model | All tiers — this is the model input |
| `mask_path` | The full ground-truth mask file | Mask-tier — Dice/IoU against the whole shape |
| `gt_bbox_xmin`, `gt_bbox_ymin`, `gt_bbox_xmax`, `gt_bbox_ymax` | Ground-truth bounding box, derived from the mask | Bbox-tier — IoU + box-center comparison |
| `gt_centroid_x`, `gt_centroid_y` | Ground-truth center point, derived from the mask | Point-tier — distance comparison |
| `gt_side_lr`, `gt_side_ud` | left/right and upper/lower labels, derived from the centroid | All tiers — wrong-side classification (RQ1–RQ3) |
| `sample_id`, `original_filename` | The dataset's own real identifier and literal source filename | Traceability back to the source dataset, not used in scoring |
| `dataset` | Which of the 3 datasets this row belongs to | Grouping/aggregation |
| `prompt_id`, `prompt_text` | Which of P0–P4 this row is, and the exact sentence sent | All tiers — this is the other model input |

**Step 3 — Send image + prompt_text to a model, get a raw answer back.** You never send anything from the `gt_*` columns to the model — only `image_path` (loaded as pixels) and `prompt_text`. The model's raw answer depends on its tier:
- A mask-tier model (e.g. CLIPSeg) returns its own black-and-white mask of where *it* thinks the polyp is.
- A bbox-tier model (e.g. Grounding DINO) returns 4 numbers, e.g. `(30, 45, 90, 85)`.
- A point-tier model (e.g. Molmo) returns one (x, y) pair directly, e.g. `(60, 65)`.

**Step 4 — Reduce the model's raw answer to one point, the same way you reduced the ground-truth mask.** If it returned a mask, average its white pixels → predicted centroid. If it returned a bbox, take the box center: `((30+90)/2, (45+85)/2) = (60, 65)`. If it returned a point already, use it as-is. Either way, you now have one predicted (x, y), directly comparable to the ground-truth centroid (402.37, 164.05) from Step 1.

**Step 5 — Compare the predicted point to the ground truth, three ways:**
1. **Side check:** is predicted x (60) > 311 (half the image's 622px width)? No → predicted side_lr = "left". The prompt asked for "right" (P2, per the real ground truth side_lr computed in Step 1). Predicted "left" ≠ prompted "right" → this is a **wrong-side** result — the model pointed the opposite way from what the text asked.
2. **Shift check:** compare this P2 prediction (60, 65) to the *same model's* P0 prediction on the *same image* (its answer with no positional wording at all). If P0's answer was also near (60, 65), the model didn't move at all when told to go right — a **prompt-following failure**, a different failure mode from wrong-side.
3. **Distance/overlap:** Euclidean distance between predicted (60, 65) and ground truth (402.37, 164.05), normalized by the image diagonal — a magnitude, not just a right/wrong label. (For mask/bbox models you also compute Dice/IoU here, a separate shape-quality number, not a position number.)

**Step 6 — Record and repeat.** Each comparison becomes one row in your results file: `model, dataset, image, prompt, predicted_point, gt_point, correct_or_wrong_side, shifted_or_not, distance`. Do this for every (model × dataset × image × prompt) combination — 13 × 3 × 50 × 5 = 9,750 rows total — then group and average those rows by model, by dataset, and by prompt to answer RQ1–RQ3 (e.g. "what % of Grounding DINO's answers on Kvasir-SEG were wrong-side when the prompt said 'right'?").

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

Every dataset supplies the same kind of ground truth (a mask), so the same 50 images per dataset feed all three model tiers. Only the *comparison representation* changes per tier, not the dataset or the ground truth source.

| Dataset | Domain | Ground truth source | Compared against mask models (4) | Compared against bbox models (7) | Compared against point models (2) |
|---|---|---|---|---|---|
| LGG Segmentation | Brain | Tumor mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
| Kvasir-SEG / Kvasir-VQA-x1 | Colon | Polyp mask | Full mask, Dice/IoU | Mask converted to bbox | Mask centroid |
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
| Grounding DINO (bbox) | ISIC 2018 | P2: locate the lesion, right | (120, 80, 210, 160) | Box + its center | Lesion mask converted to box | IoU = 0.44; predicted center on the left; wrong-side |
| Molmo-7B-D (point) | LGG Segmentation | P0: locate the tumor | (134, 97) | Point | Mask centroid at (140, 101) | Distance = 7.4 px normalized; same side as P0; no ground-truth mismatch to check at P0 |

## Sampling Protocol

- Fixed set of **50 samples per dataset** (same 50 images across all models and all 5 prompts).
- Each model runs 3 datasets times 50 images times 5 prompts (P0 to P4) = **750 predictions per model**.
- 13 models times 750 = **9,750 predictions total**.

## Method

1. Derive positional ground truth (centroid, bbox, left/right, upper/lower) for all 150 images from mask annotations.
2. Generate P0 to P4 for every image, keeping wording strictly controlled per the constraint above.
3. Run all 13 primary models zero-shot on all 5 prompt variants for the same 50-image, 3-dataset set, each scored at its native output level (mask, box, or point).
4. Compute per-model: location shift in the prompted direction, consistency (Dice for mask models, IoU for box models), and wrong-side/failure rate.
5. Rank all 13 models by lowest positional bias (highest prompt-following accuracy on correct-side cases, lowest wrong-side rate).
6. Report per-domain and per-model breakdowns.

## Other Files

- `HOW_TO_RUN.md` — the practical, do-this-in-order runbook: build the dataset, run all 13 models, compare to ground truth, phase by phase. Share this with teammates who need to execute the study, not just understand its design.
- `archived_segmentation_models_table.md`, archived reference table of published segmentation architectures. Kept for reference/comparison only, not part of this active plan.
