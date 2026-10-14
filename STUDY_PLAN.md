# Study Plan: Rotation Robustness & Interpretability of Medical Vision-Language Models

Branch: `faria/voxtell3d-rotation-medical-segmentation`

Items marked **TBD** are open decisions to resolve before running experiments.

---

## 1. Detection with Rotation

Goal: rotate the input image and measure how well 2D localization models still find the prompted left or right structure, and whether they land on the opposite side.

### Overview

- **Models (9):** Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B, Molmo2-8B, North-Micro-Vision-Instruct, Florence-2.
- **Datasets (2):** AMOS22-MRI and Neurite-OASIS, one 2D slice per subject.
- **Prompts (2):** left and right.
- **Rotation angles (7):** 0°, 30°, 60°, 90°, 120°, 150°, 180°.
- **Metrics:** mean IoU, wrong-side rate, mean normalized distance, detect rate, and the drop vs. 0°.
- **Exceptions:** Molmo2-8B returns points only, so it has no IoU. North-Micro-Vision-Instruct gets IoU only if it returns a box (to check).

### 1.1 Experiment design

| Item | Detail |
|---|---|
| Task | Prompted localization: given an image and a left/right prompt, return a box (or a point) |
| Models | **9** 2D localization models (listed in the overview) |
| Datasets | **2**: AMOS22-MRI and Neurite-OASIS (same as the VoxTell segmentation runs) |
| Input | One 2D slice per subject, extracted from the volume *(proposed: slice with the largest mask of the target pair)* |
| Structures | Bilateral pairs only. AMOS22-MRI: **TBD** *(proposed: kidney, adrenal gland)*. Neurite-OASIS: **TBD** *(proposed: hippocampus, lateral ventricle)* |
| Subjects | **TBD** per dataset |
| Prompts | **2**: left, right, per dataset and structure (see 1.2) |
| Rotation angles | **7**: 0°, 30°, 60°, 90°, 120°, 150°, 180° |
| Rotation method | In-plane rotation of the 2D slice around its center. Image size kept fixed, with fill = background value. TBD whether 90° multiples are also run without interpolation |
| Ground truth | Box (and centroid) computed from the ground-truth mask in the same slice, rotated by the same angle |

### 1.2 Prompts (by dataset)

Each structure is prompted twice, once with **left** and once with **right**. Use the rows for the dataset being run. Structures are *(proposed)*, final list **TBD**.

| Dataset | Structure | Left prompt | Right prompt |
|---|---|---|---|
| AMOS22-MRI | Kidney | `"Locate the left kidney in this image."` | `"Locate the right kidney in this image."` |
| AMOS22-MRI | Adrenal gland | `"Locate the left adrenal gland in this image."` | `"Locate the right adrenal gland in this image."` |
| Neurite-OASIS | Hippocampus | `"Locate the left hippocampus in this image."` | `"Locate the right hippocampus in this image."` |
| Neurite-OASIS | Lateral ventricle | `"Locate the left lateral ventricle in this image."` | `"Locate the right lateral ventricle in this image."` |

Prompt form by model *(proposed, confirm each model's API)*:
- **Grounding DINO, OWLv2** (open-vocabulary detectors): pass only the phrase, e.g. `"left kidney"`.
- **Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B, North-Micro-Vision-Instruct, Florence-2:** pass the instruction form above (`"Locate the left kidney in this image."`), adapted to each model's grounding format.
- **Molmo2-8B:** same instruction, but the answer is a point, not a box.

The same prompt is used at every rotation angle. Only the image rotates.

### 1.3 Procedure
1. Extract one 2D slice per subject, and compute the ground-truth box and centroid from the mask.
2. For each angle, rotate the slice and the ground-truth box/centroid together.
3. For each model and each prompt (left, right), run inference and store the predicted box or point.
4. Compute the metrics (see overview) in the rotated frame, and the wrong-side rate against the true side of the structure.
5. Aggregate over subjects and report the drop vs. 0°.

### 1.4 Task and metrics

**Task:** for each 2D slice rotated by angle θ and each prompt (left or right), the model returns a box (or a point for Molmo2-8B) for the prompted structure. The output is compared with the ground-truth box and centroid of that structure in the same rotated slice.

| Metric | Definition | Applies to |
|---|---|---|
| IoU | Area of overlap between predicted box and ground-truth box, divided by the area of their union. Averaged over samples | 8 box models (not Molmo2-8B) |
| Wrong-side rate | Share of predictions whose center is closer to the centroid of the **opposite-side** structure than to the centroid of the prompted structure | All 9 |
| Mean normalized distance | Distance between predicted center (or point) and ground-truth centroid, divided by the image diagonal | All 9 |
| Detect rate | Share of calls that return a usable box or point | All 9 |
| Drop vs. 0° | Metric at 0° minus the same metric at angle θ | All metrics |

Reported per model, per dataset, per prompt (left, right) and per angle.

---

## 2. Segmentation with Rotation

Only **in-domain datasets** are used for each model.

### 2.1 Datasets per model

**VoxTell** (5 datasets, CT and MRI)

| # | Dataset | Domain | Modality | Dim. | Paired structures | Example prompt |
|---|---|---|---|---|---|---|
| 1 | AMOS22-MRI | Abdomen | MRI (abdominal; sequence: **TBD**) | 3D | Kidneys, adrenal glands | `"Segment the left kidney in this image."` |
| 2 | Neurite-OASIS | Brain | MRI (T1-weighted structural) | 3D | Hippocampus, thalamus, ventricles, cerebral hemispheres | `"Segment the left hippocampus in this image."` |
| 3 | KiTS23 | Abdomen (kidney) | CT (contrast-enhanced) | 3D | Left/right kidney | `"Segment the right kidney in this image."` |
| 4 | TotalSegmentatorV2 | Whole body (thorax, abdomen, bones) | CT | 3D | Lungs, kidneys, adrenal glands, ribs (structures: **TBD**) | `"Segment the left lung in this image."` |
| 5 | HanSeg-CT | Head and neck | CT | 3D | Parotid and submandibular glands (confirm labels) | `"Segment the left parotid gland in this image."` |

**Text3DSAM** (2 in-domain datasets, CT and MRI)

| # | Dataset | Domain | Modality | Dim. | Paired structures | Example prompt |
|---|---|---|---|---|---|---|
| 1 | CT_HaN-Seg | Head and neck | CT | 3D | Parotid gland | `"Segment the left parotid gland in this image."` |
| 2 | MR_TotalSeg | Whole body (musculoskeletal, abdomen) | MRI (sequence: **TBD**) | 3D | Gluteus maximus (also kidneys) | `"Segment the left gluteus maximus in this image."` |

**Medical-SAM3** (2 in-domain datasets, radiography and ultrasound)

| # | Dataset | Domain | Modality | Dim. | Paired structures | Example prompt |
|---|---|---|---|---|---|---|
| 1 | Chest ImaGenome (built on MIMIC-CXR) | Chest (thorax) | Radiography (chest X-ray) | 2D | Left lung, right lung | `"Segment the left lung in this image."` |
| 2 | TRUSTED (renal ultrasound, paired with CT) | Abdomen (kidney) | Ultrasound (3D transabdominal) | 3D | Left kidney, right kidney (48 patients, 96 kidneys) | `"Segment the left kidney in this image."` |

### 2.2 Experiment settings

| Question | Answer |
|---|---|
| How many models? | 3 |
| How many datasets? | 9 in total (5 + 2 + 2), model-specific |
| Which metrics? | Dice, IoU, HD95, wrong-side rate, centroid shift |
| How many prompts tested? | 2 (left, right) |
| Rotation angles | 0°, 30°, 60°, 90°, 120°, 150°, 180° |

Notes:
- Chest X-ray is 2D, so its rotation is in-plane. The other datasets are 3D volumes.
- Report each metric per rotation angle and as the drop relative to 0°.

### 2.3 Task and metrics

**Task:** for each volume (or 2D image for Chest X-ray) rotated by angle θ and each prompt (left or right), the model segments the prompted structure. The predicted mask is compared with the ground-truth mask of that structure.

| Metric | Definition |
|---|---|
| Dice | 2 x overlap of predicted and ground-truth mask, divided by the sum of their sizes |
| IoU | Overlap of predicted and ground-truth mask, divided by their union |
| HD95 | 95th-percentile Hausdorff distance between predicted and ground-truth mask boundaries (smaller is better) |
| Wrong-side rate | Share of predictions that overlap the **opposite-side** structure's ground truth more than the prompted structure's ground truth |
| Centroid shift | Distance between predicted centroid and ground-truth centroid, normalized by the volume diagonal |
| Drop vs. 0° | Metric at 0° minus the same metric at angle θ |

Empty predictions score Dice 0 and are excluded from HD95 and centroid shift; report how many were empty. Reported per model, per dataset, per prompt and per angle.

---

## 3. Image-Text Contribution (Word Attribution)

- **Scope:** Segmentation model only, **0° rotation only**, bilateral images.
- **Models:** **VoxTell only** (1 model).
- **Datasets:** VoxTell's 5 datasets from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT).
- **Prompts:** **2** (left, right), the same template as Section 2, e.g. `"Segment the left kidney in this image."`.
- **Goal:** Quantify how much the image and each prompt word contribute to the prediction, and whether anatomical terms dominate positional words such as "left" and "right".
- **Methods:** Shapley values and Integrated Gradients, run on the same images and prompts so results are comparable.

### 3.1 Shapley values
- Players are the image and each prompt word. Absent words are masked or replaced with a neutral token. An absent image is replaced with a baseline image.
- **Image:** contribution of the image (or image regions/patches) to the output.
- **Text:** contribution of each word.

### 3.2 Integrated Gradients (IG)
- Attributes the output to input features by integrating gradients from a baseline to the actual input.
- **Image:** attribution across pixels or patches (baseline: black or blank image).
- **Text:** attribution across tokens, taken on the token embeddings (baseline: empty or padding prompt).
- **Use:** compare the **total attribution magnitude** of the image inputs and the word inputs (modality share).
- **Best for:** gradient-based modality attribution.

### Output score being explained
**Mean mask probability inside the ground-truth region.** For each image and prompt, take the model's predicted probability map and average it over the voxels (or pixels) where the ground-truth mask of the prompted structure is 1. This single number is the score that both Shapley and Integrated Gradients explain.

- It is differentiable, so Integrated Gradients can use it.
- It can be recomputed after removing a word or the image, so Shapley can use it.
- It drops when the model moves the mask to the wrong side, which is what the left/right question needs.
- Dice is not used as the explained score. It is used only to measure the deletion test.

### Task and metrics

**Task:** at 0° rotation, for each image and prompt (left or right), assign an attribution to the image and to each prompt word, using both Shapley and Integrated Gradients, for the explained score above.

| Metric | Definition |
|---|---|
| Modality share | Total absolute attribution of the image divided by the total of the image and all words, for each method |
| Per-word attribution | Attribution of each word (organ word, side word), averaged across images |
| Organ-vs-side ratio | Attribution of the organ word divided by attribution of the side word ("left" or "right") |
| Method agreement | Spearman rank correlation of the word attributions between Shapley and IG |
| Deletion test | Dice drop after removing the highest-attributed word (and, separately, the highest-attributed image regions), compared with a random removal |
| Left/right check | Whether the side word carries attribution, and whether the attributed image region falls on the matching side (share of image attribution inside the ground-truth region) |

Reported per dataset and per prompt (left, right).

---

## 4. Internal Embedding Shift (Left/Right Embedding Swap)

- **Scope:** Segmentation model only, **0° rotation only**, bilateral images.
- **Models:** **VoxTell only** (1 model).
- **Datasets:** VoxTell's 5 datasets from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT).
- **Prompt(s) used:** **left** and **right** prompts (paired, same image). Bilateral prompt: **TBD** whether to include.
- **Goal:** Test whether the model's output follows the prompt embedding. Inject the embedding of one side into the run of the other and see what changes.

### Method (activation patching)
1. Run the image with the **left** prompt and cache the internal embeddings (prompt/text embedding, or fused features at chosen layers).
2. Run the same image with the **right** prompt and cache its embeddings.
3. **Swap:** re-run the **left** prompt but inject the **right** embedding. Then re-run the **right** prompt but inject the **left** embedding.
4. Compare the output with the clean left and right runs.

### Conditions
| Run | Prompt text | Injected embedding | If output follows the embedding | If output follows the text/image |
|---|---|---|---|---|
| Clean left | left | left | (reference) | (reference) |
| Clean right | right | right | (reference) | (reference) |
| Swap A | left | **right** | segments the right side | segments the left side |
| Swap B | right | **left** | segments the left side | segments the right side |

- **Where to inject:** at the **text-image fusion**, which in VoxTell is the transformer decoder, where the text query cross-attends to the image features.
  - **Input stays the same:** the prompt text is not changed. For Swap A the prompt is still "left"; for Swap B it is still "right".
  - **Modify the embedding:** inside the model, replace the embedding of the side word ("left") with the embedding of the other side word ("right"), and the reverse for Swap B.
  - **Layer by layer, separate runs:**
    1. Run 1: make the replacement at **fusion layer 1**, then let the model continue normally.
    2. Run 2: make the replacement at **fusion layer 2** only.
    3. Run 3: make the replacement at **fusion layer 3** only.
  - Each run has one injection layer. The layers are not combined.
  - Compare the three runs to see which layer's replacement moves the mask to the other side.
  - The swap is not applied at the image encoder or after fusion.

### Task and metrics

**Task:** at 0° rotation, run the swap test (Swap A and Swap B) with the replacement made at fusion layer 1, 2 or 3 in separate runs, and check whether the predicted mask follows the injected side or the prompt side.

| Metric | Definition |
|---|---|
| Dice vs. embedding-side ground truth | Dice between the swapped output and the ground-truth mask of the side whose embedding was injected |
| Dice vs. prompt-side ground truth | Dice between the swapped output and the ground-truth mask of the side named in the prompt |
| Swap success rate | Share of cases where the Dice vs. the embedding-side ground truth is higher than the Dice vs. the prompt-side ground truth |
| Centroid shift | Distance the predicted centroid moves from the clean run toward the other side, normalized by the volume diagonal |
| Per-layer swap success | Swap success rate for injection at layer 1, layer 2 and layer 3, shown as a curve |

Reported per dataset, for Swap A and Swap B separately.

### Controls
- Inject an embedding from the **same** side (no-op) as a noise floor.
- Inject a random or unrelated-organ embedding to separate "follows the embedding" from "breaks the output".
- Cross-check on the same images with the real left/right prompts.

- **Open points:** Which embedding is swapped (text only, or fused image-text), and whether to patch the whole tensor or only selected tokens.

---

## 5. Layer-wise Embedding Divergence Under Rotation

| Question | Answer |
|---|---|
| Models | **VoxTell only** (1 model) |
| How many datasets? | VoxTell's 5 datasets from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) |
| Rotation angles | 0° as the reference vs. each of 30°, 60°, 90°, 120°, 150°, 180° |
| Prompt(s) used | 2 (left, right), fixed across angles |
| Layers probed | Encoder stages, text projection, transformer decoder (fusion) layers, decoder stages (see `model_layer_summary.md`) |

- **Goal:** Find which layer or component first breaks when the image is rotated (encoder, fusion, or decoder).
- **Setup:** Run the same image at 0° and at each rotation with the same prompt. Extract activations at each layer using forward hooks.
- **Alignment:** Rotate the feature maps back by the inverse angle (or compare pooled embeddings) so spatial positions correspond. Verify the round trip before trusting results.
- **Metrics:**
  - Cosine distance between 0° and rotated embeddings, per layer.
  - Linear CKA between 0° and rotated embeddings, per layer (computed across a batch of images).
- **Outputs:**
  - Plot of divergence vs. layer depth, one curve per angle.
  - Identify the first layer where divergence jumps, and compare it with the drop in Dice/IoU from Section 2.
- **Controls:**
  - Compare 90° multiples (no interpolation) against off-axis angles such as 45°.
  - Include a no-rotation repeat (same image twice) as the noise floor.
- **Open points:** Decide whether to compare token-level or pooled embeddings, and how many images are needed for stable CKA.

---

## 6. Uncertainty Quantification

| Question | Answer |
|---|---|
| Models | **TBD** |
| How many datasets? | **TBD** |
| Rotation angles | **TBD** (same as Sections 1 and 2) |
| Prompt(s) used | **TBD** |

### 6.1 Methods
- **Primary: test-time augmentation (TTA) variance.** Run the model on several rotated views, rotate the outputs back, and measure disagreement.
  - Segmentation: per-pixel variance or entropy of the averaged probability map.
  - Detection: spread of box positions and variance of scores.
  - Use a fixed TTA set that does not depend on the true rotation, to avoid leaking the answer.
- **Secondary: MC dropout** (10-20 passes per angle). Dropout is kept active at inference, and uncertainty is the variance or entropy across passes. Compare its results with TTA at every angle.
  - Check first that each model has dropout layers. If one does not, note it as a limitation for that model.
- **Only these two methods are used.** Both are reported side by side in 5.2.
- **Caveat:** TTA variance measures rotation sensitivity. A model that is wrong at every angle can look confident, which is why MC dropout is used alongside it.

### 6.2 What to report
1. **Uncertainty vs. rotation angle:** mean uncertainty at each angle (does it rise with rotation?).
2. **Failure detection:** does high uncertainty predict low Dice or a missed detection? Report AUROC and risk-coverage curves, for TTA and MC dropout separately.
3. **Calibration:** expected calibration error and reliability diagrams, 0° vs. rotated angles.
4. **Link to Section 5:** compare uncertainty with layer-wise embedding divergence.

---

## 7. Prompt Sensitivity (VoxTell Only)

| Question | Answer |
|---|---|
| How many models? | **VoxTell only** (1 model) |
| How many datasets? | VoxTell's 5 datasets from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) |
| Rotation | 0° only *(proposed)* |
| Metrics | **TBD** (Dice, IoU, HD95 for positive prompts; false-positive rate for negative prompts) |
| How many targets (organs/lesions)? | **TBD** |
| How many prompt types? | 2 (location wording, negative prompts) + canonical reference |

### 7.1 Anatomical-location wording

**What it is:** the same target, asked for by **where it is** instead of by its name.

**Question:** can VoxTell find the structure from location words alone?

| Dataset | Normal prompt (reference) | Location-wording prompt *(proposed)* |
|---|---|---|
| AMOS22-MRI, KiTS23 (kidney) | `"Segment the left kidney in this image."` | `"Segment the organ on the left side of the back of the abdomen."` |
| AMOS22-MRI (adrenal gland) | `"Segment the left adrenal gland in this image."` | `"Segment the small gland above the left kidney."` |
| Neurite-OASIS (hippocampus) | `"Segment the left hippocampus in this image."` | `"Segment the structure in the left medial temporal lobe."` |
| TotalSegmentatorV2 (lung) | `"Segment the left lung in this image."` | `"Segment the air-filled organ in the left side of the chest."` |
| HanSeg-CT (parotid gland) | `"Segment the left parotid gland in this image."` | `"Segment the salivary gland in front of the left ear."` |

- **New prompts:** 1 per structure and side (left and right).
- **Metrics:** Dice and IoU for the location prompt, and the drop relative to the normal prompt. Also the wrong-side rate.
- **How to read it:** a small drop means the model understands the location words. A large drop means it relies on the structure's name.

### 7.2 Negative prompts (target absent)

**What it is:** a prompt for a structure that is **not in the image**.

**Question:** does VoxTell return an empty mask (correct), or does it invent a mask (hallucination)?

| Dataset | Negative prompt *(proposed)* | Why the target is absent |
|---|---|---|
| AMOS22-MRI, KiTS23 (abdomen) | `"Segment the left hippocampus in this image."` | Hippocampus is in the brain, not in an abdominal scan |
| Neurite-OASIS (brain) | `"Segment the left kidney in this image."` | Kidney is not in a brain scan |
| TotalSegmentatorV2 | `"Segment the left hippocampus in this image."` | Only valid for scans that exclude the head (**TBD**: check) |
| HanSeg-CT (head and neck) | `"Segment the left kidney in this image."` | Kidney is not in a head-and-neck scan |

- **New prompts:** 1 per image.
- **Metrics:**
  - False-positive rate: share of negative prompts that return a non-empty mask.
  - Predicted mask size on negatives.
  - Model confidence on negatives.
- **Control:** run the matched positive prompt on the same dataset, so the false-positive rate can be read against the true-positive rate.
- **How to read it:** a low false-positive rate means the model correctly says "not here". A high rate means it hallucinates masks.
- **Open:** whether to also use a wrong-side variant, which is a negative where only the other side is missing (**TBD**).

### 7.3 Notes
- Keep the images and ground-truth masks fixed; change only the prompt text.
- Use the same prompt set on all 5 datasets so results are comparable.
- Report which prompt types cause the largest drops.

---

## Summary of Experiments

Scope: all experiments use **bilateral structures** (paired left/right).

**Prompt types (2 base):** **left** and **right**, using the repo's template, e.g. `"Segment the left kidney in this image."` / `"Segment the right kidney in this image."`. Each structure is prompted with both. (Earlier drafts also proposed a "bilateral/both" prompt; it is dropped to match the repo's left/right template.)

**Rotation angles (Experiments 1, 2, 5, 6):** 0°, 30°, 60°, 90°, 120°, 150°, 180° (7 angles, as in the existing VoxTell rotation sweep).

**Source of values:** models, datasets, angles and prompt wording come from the repo (`positional-bias-benchmark-plan.md`, `experiment_support_table.md`, the rotation prediction CSVs). Items marked *(proposed)* are my suggestions for you to confirm.

| # | Experiment | What it does | Models | Datasets | Metrics | Prompts |
|---|---|---|---|---|---|---|
| 1 | Detection with rotation | Rotates the input image and measures how box/point localization falls with angle, and whether it lands on the opposite side. | 9 2D localization models: Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B, Molmo2-8B, North-Micro-Vision-Instruct, Florence-2 (not yet run) | Same as the VoxTell segmentation runs: AMOS22-MRI and Neurite-OASIS (2). The 2D models get one 2D slice per subject, and the ground-truth box is taken from the mask in that slice | Mean IoU (not Molmo2, point-only), wrong-side rate, mean normalized distance, detect rate; drop vs. 0° | 2 (left, right) x 7 angles |
| 2 | Segmentation with rotation | Rotates the input volume and measures how segmentation quality falls with angle, and how often the mask lands on the opposite-side organ. | 3: VoxTell, Text3DSAM, Medical-SAM3 | In-domain only. VoxTell: AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT (5). Text3DSAM: CT_HaN-Seg, MR_TotalSeg (2). Medical-SAM3: Chest ImaGenome, TRUSTED (2) | Dice, IoU, HD95, wrong-side (opposite-organ) rate, centroid shift; drop vs. 0° | 2 (left, right) x 7 angles |
| 3 | Image-text contribution / word attribution | At 0° only, measures how much the image and each prompt word contribute to the prediction, and whether anatomical words dominate positional words (left/right). Two methods on the same inputs: **Shapley** and **Integrated Gradients**. | 1: VoxTell only | VoxTell's 5 from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) | Image-vs-text share, per-word attribution (anatomical vs. positional), Spearman rank correlation between the 2 methods, deletion-test Dice drop | 2 (left, right), e.g. `"Segment the left kidney in this image."` |
| 4 | Internal embedding shift (left/right swap) | At 0° only, injects the right embedding into the left-prompt run and the left embedding into the right-prompt run, then checks whether the output follows the embedding or the prompt/image. | 1: VoxTell only | VoxTell's 5 from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) | Swap success rate, Dice/IoU vs. embedding-side and prompt-side ground truth, centroid shift, per-layer swap success; no-op and random-embedding controls | 2 (left, right) |
| 5 | Layer-wise embedding divergence | Compares embeddings at 0° and rotated angles layer by layer to find where the model breaks (encoder, fusion, decoder). | 1: VoxTell only (layers listed in `model_layer_summary.md`: encoder stages, text projection, transformer decoder, decoder stages) | VoxTell's 5 from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) | Cosine distance and CKA per layer; noise floor from a same-image repeat; compared with the Dice drop from Experiment 2 | **2 (left, right)**, fixed across all angles; 6 non-zero angles vs. the 0° reference |
| 6 | Uncertainty quantification | Estimates uncertainty with TTA variance (primary) and MC dropout (secondary) and checks whether it rises with rotation and flags failures. | 1: VoxTell *(proposed; MC dropout needs dropout layers, check first)* | AMOS22-MRI, Neurite-OASIS | Uncertainty vs. angle, AUROC (failure detection), risk-coverage, ECE | 2 (left, right) x 7 angles |
| 7 | Prompt sensitivity (VoxTell only) | Tests two prompt types: anatomical-location wording of the same target, and negative prompts (target absent) to test for hallucinated masks. | 1: VoxTell only | VoxTell's 5 from Section 2.1 (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) | Dice/IoU drop vs. canonical (location wording); false-positive rate, predicted mask size and confidence on negatives | 2 types: **location wording** (1 new prompt per structure, e.g. `"organ in the right upper quadrant"`) + **negative prompts** (1 per image, an organ from the other anatomy, e.g. `"Segment the spleen in this image."` on brain MRI). Canonical left/right is the reference |

Still to confirm: number of subjects per dataset, which structures (e.g., kidney, adrenal gland, hippocampus), and the *(proposed)* items above.
