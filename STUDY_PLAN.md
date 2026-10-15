# Study Plan: Rotation Robustness and Positional Bias in Medical Vision-Language Models

---

## 1. 2D Localization Under Rotation

Goal: rotate the input image and measure how well 2D localization models still find the prompted left or right structure, and whether they land on the opposite side.

**Research questions:**
- Does rotation make the model's box/point land on the opposite-side structure more than chance, rather than just becoming less accurate on the correct side (wrong-side rate vs. IoU drop)?
- Is the direction of the bias (which side is favored) consistent across models and angles, which would show a systematic positional bias rather than random noise?

### Overview

| Setting | Value |
|---|---|
| Models (9) | Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B, Molmo2-8B, North-Micro-Vision-Instruct, Florence-2 |
| Datasets (2) | AMOS22-MRI and Neurite-OASIS, one 2D slice per subject |
| Prompts (2) | left and right |
| Rotation angles (7) | 0°, 30°, 60°, 90°, 120°, 150°, 180° |
| Metrics | mean IoU, wrong-side rate, mean normalized distance, detect rate, and the drop vs. 0° |
| Exceptions | Molmo2-8B returns points only, so it has no IoU. North-Micro-Vision-Instruct gets IoU only if it returns a box |

### 1.1 Experiment design

| Item | Detail |
|---|---|
| Task | Prompted localization: given an image and a left/right prompt, return a box (or a point) |
| Models | **9** 2D localization models (listed in the overview) |
| Datasets | **2**: AMOS22-MRI and Neurite-OASIS (same as the VoxTell segmentation runs) |
| Input | One 2D slice per subject, extracted from the volume *(proposed: slice with the largest mask of the target pair)* |
| Structures | All bilateral pairs |
| Subjects | As many as possible, per dataset |
| Prompts | **2**: left, right, per dataset and structure (see 1.2) |
| Rotation angles | **7**: 0°, 30°, 60°, 90°, 120°, 150°, 180° |
| Rotation method | In-plane rotation of the 2D slice around its center. Image size kept fixed, with fill = background value. 90° multiples are also run without interpolation, as a control against interpolation artifacts at off-axis angles |
| Ground truth | Box (and centroid) computed from the ground-truth mask in the same slice, rotated by the same angle |

### 1.2 Prompts (by dataset)

Each structure is prompted twice, once with **left** and once with **right**. Use the rows for the dataset being run. All bilateral pairs are included.

Two templates are used: the **original template** and one **paraphrase template** (same meaning, different wording), to run a paraphrase-robustness check alongside the main left/right comparison.

**Original template**

| Dataset | Structure | Left prompt | Right prompt |
|---|---|---|---|
| AMOS22-MRI | Kidney | `"Locate the left kidney in this image."` | `"Locate the right kidney in this image."` |
| AMOS22-MRI | Adrenal gland | `"Locate the left adrenal gland in this image."` | `"Locate the right adrenal gland in this image."` |
| Neurite-OASIS | Hippocampus | `"Locate the left hippocampus in this image."` | `"Locate the right hippocampus in this image."` |
| Neurite-OASIS | Thalamus | `"Locate the left thalamus in this image."` | `"Locate the right thalamus in this image."` |
| Neurite-OASIS | Ventricle | `"Locate the left ventricle in this image."` | `"Locate the right ventricle in this image."` |
| Neurite-OASIS | Cerebral hemisphere | `"Locate the left cerebral hemisphere in this image."` | `"Locate the right cerebral hemisphere in this image."` |

**Paraphrase template** (same meaning, reworded, for the paraphrase-robustness check)

| Dataset | Structure | Left prompt (paraphrase) | Right prompt (paraphrase) |
|---|---|---|---|
| AMOS22-MRI | Kidney | `"Find the kidney that is on the left."` | `"Find the kidney that is on the right."` |
| AMOS22-MRI | Adrenal gland | `"Find the adrenal gland that is on the left."` | `"Find the adrenal gland that is on the right."` |
| Neurite-OASIS | Hippocampus | `"Find the hippocampus that is on the left."` | `"Find the hippocampus that is on the right."` |
| Neurite-OASIS | Thalamus | `"Find the thalamus that is on the left."` | `"Find the thalamus that is on the right."` |
| Neurite-OASIS | Ventricle | `"Find the ventricle that is on the left."` | `"Find the ventricle that is on the right."` |
| Neurite-OASIS | Cerebral hemisphere | `"Find the cerebral hemisphere that is on the left."` | `"Find the cerebral hemisphere that is on the right."` |

Both templates are run at every rotation angle. Only the image rotates.

### 1.4 Task and metrics

**Task:** for each 2D slice rotated by angle θ and each prompt (left or right), the model returns a box (or a point for Molmo2-8B) for the prompted structure. The output is compared with the ground-truth box and centroid of that structure in the same rotated slice.

| Metric | Definition | Applies to | Why it is used |
|---|---|---|---|
| IoU | Area of overlap between predicted box and ground-truth box, divided by the area of their union. Averaged over samples | 8 box models (not Molmo2-8B) | Standard localization-accuracy score; tracks how precisely the box matches the target as rotation increases |
| Wrong-side rate | Share of predictions whose center is closer to the centroid of the **opposite-side** structure than to the centroid of the prompted structure | All 9 | Directly tests the core question: whether rotation makes the model point at the mirror-image (wrong-side) structure instead of the prompted one |
| Mean normalized distance | Distance between predicted center (or point) and ground-truth centroid, divided by the image diagonal | All 9 | A continuous, scale-independent measure of positional error that also works for point-only models (Molmo2-8B), where IoU can't be computed |
| Detect rate | Share of calls that return a usable box or point | All 9 | Separates "wrong answer" from "no answer"; a model that stops returning predictions under rotation is a different failure mode than one that mislocalizes |
| Drop vs. 0° | Metric at 0° minus the same metric at angle θ | All metrics | Isolates the effect of rotation itself from each model's baseline accuracy, so results are comparable across models with very different starting performance |

Reported per model, per dataset, per prompt (left, right) and per angle.

---

## 2. 3D Segmentation Under Rotation

Goal: measure how segmentation quality degrades under rotation, and how often the predicted mask lands on the opposite-side (wrong-side) organ instead of the prompted one.

Only **in-domain datasets** are used for each model.

**Research questions:**
- Does rotation cause the predicted mask to overlap the opposite-side organ's ground truth more than the prompted organ's (a laterality flip), rather than just a lower-quality mask on the correct side?
- Is the wrong-side rate asymmetric by rotation direction (e.g., more left→right flips than right→left), which would point to a directional bias rather than symmetric noise?

### 2.1 Datasets per model

**VoxTell**

| # | Dataset | Domain | Modality | Dim. | Paired structures |
|---|---|---|---|---|---|
| 1 | AMOS22-MRI | Abdomen | MRI (abdominal, T2-weighted) | 3D | Kidneys, adrenal glands |
| 2 | Neurite-OASIS | Brain | MRI (T1-weighted structural) | 3D | Hippocampus, thalamus, ventricles, cerebral hemispheres |
| 3 | KiTS23 | Abdomen (kidney) | CT (contrast-enhanced) | 3D | Left/right kidney |
| 4 | TotalSegmentatorV2 | Whole body (thorax, abdomen, bones) | CT | 3D | All bilateral pairs: lungs, kidneys, adrenal glands, ribs |
| 5 | HanSeg-CT | Head and neck | CT | 3D | Parotid and submandibular glands |

**Text3DSAM**

| # | Dataset | Domain | Modality | Dim. | Paired structures |
|---|---|---|---|---|---|
| 1 | CT_HaN-Seg | Head and neck | CT | 3D | Parotid gland |
| 2 | MR_TotalSeg | Whole body (musculoskeletal, abdomen) | MRI (T1-weighted) | 3D | Gluteus maximus (also kidneys) |

**Medical-SAM3**

| # | Dataset | Domain | Modality | Dim. | Paired structures |
|---|---|---|---|---|---|
| 1 | Chest ImaGenome (built on MIMIC-CXR) | Chest (thorax) | Radiography (chest X-ray) | 2D | Left lung, right lung |
| 2 | TRUSTED (renal ultrasound, paired with CT) | Abdomen (kidney) | Ultrasound (3D transabdominal) | 3D | Left kidney, right kidney (48 patients, 96 kidneys) |

### 2.2 Prompts (by dataset)

Each structure is prompted twice, once with **left** and once with **right**. Two templates are used, the same paraphrase-robustness check used for the detection prompts.

#### 2.2.1 Original template

| Dataset | Structure | Left prompt | Right prompt |
|---|---|---|---|
| AMOS22-MRI | Kidney | `"Segment the left kidney in this image."` | `"Segment the right kidney in this image."` |
| Neurite-OASIS | Hippocampus | `"Segment the left hippocampus in this image."` | `"Segment the right hippocampus in this image."` |
| KiTS23 | Kidney | `"Segment the left kidney in this image."` | `"Segment the right kidney in this image."` |
| TotalSegmentatorV2 | Lung | `"Segment the left lung in this image."` | `"Segment the right lung in this image."` |
| HanSeg-CT | Parotid gland | `"Segment the left parotid gland in this image."` | `"Segment the right parotid gland in this image."` |
| CT_HaN-Seg | Parotid gland | `"Segment the left parotid gland in this image."` | `"Segment the right parotid gland in this image."` |
| MR_TotalSeg | Gluteus maximus | `"Segment the left gluteus maximus in this image."` | `"Segment the right gluteus maximus in this image."` |
| Chest ImaGenome | Lung | `"Segment the left lung in this image."` | `"Segment the right lung in this image."` |
| TRUSTED | Kidney | `"Segment the left kidney in this image."` | `"Segment the right kidney in this image."` |

#### 2.2.2 Paraphrase template

Same meaning, reworded, for the paraphrase-robustness check.

| Dataset | Structure | Left prompt (paraphrase) | Right prompt (paraphrase) |
|---|---|---|---|
| AMOS22-MRI | Kidney | `"Find the kidney that is on the left."` | `"Find the kidney that is on the right."` |
| Neurite-OASIS | Hippocampus | `"Find the hippocampus that is on the left."` | `"Find the hippocampus that is on the right."` |
| KiTS23 | Kidney | `"Find the kidney that is on the left."` | `"Find the kidney that is on the right."` |
| TotalSegmentatorV2 | Lung | `"Find the lung that is on the left."` | `"Find the lung that is on the right."` |
| HanSeg-CT | Parotid gland | `"Find the parotid gland that is on the left."` | `"Find the parotid gland that is on the right."` |
| CT_HaN-Seg | Parotid gland | `"Find the parotid gland that is on the left."` | `"Find the parotid gland that is on the right."` |
| MR_TotalSeg | Gluteus maximus | `"Find the gluteus maximus that is on the left."` | `"Find the gluteus maximus that is on the right."` |
| Chest ImaGenome | Lung | `"Find the lung that is on the left."` | `"Find the lung that is on the right."` |
| TRUSTED | Kidney | `"Find the kidney that is on the left."` | `"Find the kidney that is on the right."` |

Both templates are run at every rotation angle, for every bilateral pair listed in 2.1. Only the image rotates.

### 2.3 Experiment settings

| Setting | Value |
|---|---|
| How many models? | 3 |
| How many datasets? | 9 in total (5 + 2 + 2), model-specific |
| Which metrics? | Dice, IoU, HD95, wrong-side rate, centroid shift |
| How many prompts tested? | 4 (left and right, each with the original and paraphrase template) |
| Rotation angles | 0°, 30°, 60°, 90°, 120°, 150°, 180° |

Notes:
- Chest X-ray is 2D, so its rotation is in-plane. The other datasets are 3D volumes.
- Report each metric per rotation angle and as the drop relative to 0°.
- Report the original and paraphrase templates separately, and their difference, to isolate wording effects from the left/right effect.

### 2.4 Task and metrics

**Task:** for each volume (or 2D image for Chest X-ray) rotated by angle θ and each prompt (left or right), the model segments the prompted structure. The predicted mask is compared with the ground-truth mask of that structure.

| Metric | Definition | Why it is used |
|---|---|---|
| Dice | 2 x overlap of predicted and ground-truth mask, divided by the sum of their sizes | Standard segmentation-quality score; gives a single number to track overall mask accuracy as rotation increases |
| IoU | Overlap of predicted and ground-truth mask, divided by their union | Stricter overlap measure than Dice (penalizes false positives/negatives more symmetrically); cross-checks that the Dice trend isn't an artifact of the metric's formula |
| HD95 | 95th-percentile Hausdorff distance between predicted and ground-truth mask boundaries (smaller is better) | Captures boundary/localization error that Dice and IoU can hide, e.g. a mask that is well-overlapped but with a badly placed edge, which matters when checking whether rotation shifts the mask spatially rather than just shrinking it |
| Wrong-side rate | Share of predictions that overlap the **opposite-side** structure's ground truth more than the prompted structure's ground truth | Directly tests the paper's core claim: whether rotation causes the model to segment the mirror-image (wrong-side) structure instead of the prompted one |
| Centroid shift | Distance between predicted centroid and ground-truth centroid, normalized by the volume diagonal | Gives a continuous, scale-independent measure of positional drift, so a side-flip can be detected even before it is severe enough to change the wrong-side rate |
| Drop vs. 0° | Metric at 0° minus the same metric at angle θ | Isolates the effect of rotation itself from each model's/dataset's baseline performance, making results comparable across models and datasets with very different starting Dice/IoU levels |

Empty predictions score Dice 0 and are excluded from HD95 and centroid shift; report how many were empty. Reported per model, per dataset, per prompt and per angle.

---

## 3. Image vs. Word Attribution

Goal: quantify how much the image and each prompt word contribute to the prediction, and whether anatomical terms dominate positional words such as "left" and "right".

**Research questions:**
- Does the organ word's attribution dwarf the side word's ("left"/"right") attribution, which would mean the model's apparent positional behavior is driven by the image/default prior rather than by actually reading the position word?
- When the side word is removed, does the model keep producing a confident but spatially arbitrary mask (rather than becoming uncertain), showing the side word has little causal effect even where it carries some attribution?

### Overview

| Setting | Value |
|---|---|
| Scope | Segmentation model only, **0° rotation only**, bilateral images |
| Models | **VoxTell only** (1 model) |
| Datasets | VoxTell's 5 segmentation datasets (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) |
| Prompts | **2** (left, right), each used in 3 variants: <ul><li>**Original:** `"Segment the left kidney in this image."`</li><li>**Organ-only (side word removed):** `"Segment the kidney in this image."`</li><li>**Paraphrase:** `"Find the kidney that is on the left."`</li></ul> |
| Methods | 6 Captum attribution methods: Integrated Gradients, Layer Integrated Gradients, Occlusion, Feature Ablation, GradientSHAP, and Shapley Value Sampling, run on the same images and prompts so results are comparable |

### Attribution methods compared

| Method | How it works | Image attribution | Text attribution | Output score explained | Expected result |
|---|---|---|---|---|---|
| **Integrated Gradients** | Attributes the prediction score to input features by integrating gradients along a straight path from a baseline to the actual input (Captum `IntegratedGradients`). | Yes, attribution across pixels or patches (baseline: black or blank image) | Yes, via embeddings; attribution across tokens taken on the token embeddings (baseline: empty or padding prompt) | <ul><li>Mean predicted mask probability inside the ground-truth region.</li><li>Usable here because it is a plain differentiable average, so a gradient path can be integrated through it.</li></ul> | <ul><li>The gradient path should show a visible rise at the side word's tokens if the model is sensitive to it.</li><li>A flat gradient profile at the side word next to a sharp spike at the organ word would mean the model's local sensitivity doesn't actually track position.</li></ul> |
| **Layer Integrated Gradients** | Same integration as above, but attributes at a chosen intermediate layer instead of the raw input (Captum `LayerIntegratedGradients`), e.g. the token-embedding layer or a fusion layer. | Yes, applied at an intermediate image-encoder layer | Yes, via embeddings; applied directly at the token-embedding layer | <ul><li>Same mean mask probability score, read from that layer onward.</li></ul> | <ul><li>Localizes *where* the side word's influence appears or disappears: strong attribution at the embedding layer that washes out by a later fusion layer would mean the side signal is lost before it reaches the output.</li></ul> |
| **Occlusion** | Perturbation-based: masks a region of the image or a word directly and measures the resulting change in the prediction, with no gradients needed (Captum `Occlusion`). | Yes, slides an occlusion window over the image | Yes, with suitable input wrappers; occludes (masks) one word at a time | <ul><li>Score drop when a region or word is occluded, relative to the unoccluded score.</li></ul> | <ul><li>Occluding the side word alone should cause a measurable score drop if the model is using it.</li><li>No drop would mean the model ignores the side word regardless of what the gradient-based methods show.</li></ul> |
| **Feature Ablation** | Generalizes occlusion to arbitrary feature groups: each group is replaced by a baseline value and the output difference is recorded (Captum `FeatureAblation`). | Yes, with suitable input wrappers; ablates image regions/patches as groups | Yes, with suitable input wrappers; ablates a whole word as one group | <ul><li>Score drop when a feature group is ablated, relative to the full-input score.</li></ul> | <ul><li>Ablating the side-word group should produce a drop comparable to Shapley Value Sampling's per-word figure.</li><li>A large disagreement with Shapley flags a word-word interaction effect that simple ablation misses but Shapley accounts for.</li></ul> |
| **GradientSHAP** | Approximates Shapley values using gradients: adds noise to interpolate between random baselines and the input, then averages the resulting gradients (Captum `GradientShap`). | Yes, via differentiable baselines over image pixels | Yes, via differentiable embeddings | <ul><li>Same mean mask probability score.</li></ul> | <ul><li>Should broadly agree with Shapley Value Sampling's organ-vs-side ranking, since it approximates the same quantity by a different route.</li><li>Disagreement flags that GradientSHAP's baseline-sampling assumptions don't hold well for this model/task.</li></ul> |
| **Shapley Value Sampling** | Monte Carlo approximation of exact Shapley values: samples random coalitions instead of enumerating every combination (Captum `ShapleyValueSampling`). | Yes, with feature grouping; the image is treated as one group (or a few coarse regions) | Yes, with feature grouping; each word is one group/player | <ul><li>Same mean mask probability score, recomputed per sampled coalition.</li></ul> | <ul><li>If the model genuinely reads position, the side word ("left"/"right") should carry a clearly non-zero value, not just the organ word.</li><li>A side-word value near zero would mean the output is effectively decided before the position word is even considered.</li></ul> |

Notes:
- This mean-probability score is the common output being attributed by all 6 methods.
- Dice is a separate metric, used only for the deletion test below.
- The score also drops when the model moves the mask to the wrong side, the exact failure mode this plan is checking for.

### Task and metrics

**Task:** at 0° rotation, for each image and prompt (left or right), assign an attribution to the image and to each prompt word, using all 6 Captum methods, for the explained score above.

| Metric | Definition | Why it is used |
|---|---|---|
| Modality share | Total absolute attribution of the image divided by the total of the image and all words, for each method | Answers the headline question directly: does the model lean on the image or on the prompt text to produce its answer |
| Per-word attribution | Attribution of each word (organ word, side word), averaged across images | Breaks the text side down further, so "the prompt matters" can be refined into which specific word is doing the work |
| Organ-vs-side ratio | Attribution of the organ word divided by attribution of the side word ("left" or "right") | A single number for the paper's core concern: whether the model is driven far more by *what* the structure is than by *which side* it's on |
| Method agreement | Spearman rank correlation of the word attributions across all 6 methods (pairwise and overall) | Checks that the word-importance ranking isn't an artifact of one particular attribution method, before trusting the organ-vs-side ratio |
| Deletion test | Dice drop after removing the highest-attributed word (and, separately, the highest-attributed image regions), compared with a random removal | A causal check: confirms the attributed word/region actually matters to the output, rather than just correlating with it |
| Left/right check | Whether the side word carries attribution, and whether the attributed image region falls on the matching side (share of image attribution inside the ground-truth region) | Ties the text-side and image-side attributions together on the one question that matters for positional bias: is "left"/"right" grounded in the correct side of the image |

Reported per dataset and per prompt (left, right).

---

## 4. Left/Right Embedding Swap

Goal: test whether the model's output follows the prompt embedding by injecting the embedding of one side into the run of the other and seeing what changes.

**Research questions:**
- If the "right" embedding is injected into a "left" run, does the output mask move to follow the injected side, proving the model's side decision is causally carried by the text embedding rather than by the image or a fixed prior?
- At which fusion layer does this causal effect first appear, pinpointing exactly where in the model the left/right decision (and any bias in it) is made?

### Overview

| Setting | Value |
|---|---|
| Scope | Segmentation model only, **0° rotation only**, bilateral images |
| Models | **VoxTell only** (1 model) |
| Datasets | VoxTell's 5 segmentation datasets (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) |
| Prompt(s) used | **left** and **right** prompts (paired, same image). No bilateral/"both" prompt, matching the left/right-only template used throughout this plan |

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

### Injection details

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

| Metric | Definition | Why it is used |
|---|---|---|
| Dice vs. embedding-side ground truth | Dice between the swapped output and the ground-truth mask of the side whose embedding was injected | Tests the "follows the embedding" hypothesis directly: if the mask matches this ground truth, the output tracked the injected signal, not the original prompt |
| Dice vs. prompt-side ground truth | Dice between the swapped output and the ground-truth mask of the side named in the prompt | The mirror-image comparison; high Dice here instead means the model ignored the injected embedding and stuck with the text/image it was given |
| Swap success rate | Share of cases where the Dice vs. the embedding-side ground truth is higher than the Dice vs. the prompt-side ground truth | Collapses the two Dice scores above into one pass/fail number per case, so results can be aggregated and compared across datasets |
| Centroid shift | Distance the predicted centroid moves from the clean run toward the other side, normalized by the volume diagonal | Captures partial or graded swaps that the binary success-rate metric would miss, e.g. a mask that shifts toward the other side without fully flipping |
| Per-layer swap success | Swap success rate for injection at layer 1, layer 2 and layer 3, shown as a curve | Localizes *where* in the fusion process the side decision is made, not just *whether* it can be flipped at all |

Reported per dataset, for Swap A and Swap B separately.

### Controls
- Inject an embedding from the **same** side (no-op) as a noise floor.
- Inject a random or unrelated-organ embedding to separate "follows the embedding" from "breaks the output".
- Cross-check on the same images with the real left/right prompts.

- **Open points:** Which embedding is swapped (text only, or fused image-text), and whether to patch the whole tensor or only selected tokens.

---

## 5. Attention Grounding Across Layers (KL Divergence)

Goal: find which decoder layer's cross-attention is best grounded on the correct anatomical location, and whether that grounding is systematically worse for one side than the other.

**Research questions:**
- Is there a persistent Left-KL − Right-KL gap across decoder layers, showing the model's attention is better grounded for one side than the other even though both sides are the same type of structure?
- At which layer is attention best grounded overall (lowest KL), and does the left/right gap shrink, grow, or stay flat as the signal passes through layers?

| Setting | Value |
|---|---|
| Models | **VoxTell only** (1 model) |
| How many datasets? | VoxTell's 5 segmentation datasets (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) |
| Prompt(s) used | 2 (left, right), paired on the same image |
| Layers probed | Every decoder layer's text-to-image cross-attention |

### Method
- For each organ, run the model once with its **left** prompt and once with its **right** prompt on the same image, and capture the decoder's cross-attention density at every layer.
- Normalize the ground-truth mask and each layer's attention density into probability distributions over voxels, then compute `KL(ground-truth mask ‖ attention density)` per layer, per side.
- Repeat with one generic, non-spatial prompt per subject ("Segment the anatomy in this image.") and divide its attention map out before computing KL, isolating prompt-specific grounding from wherever the model looks by default.
- Take the per-layer gap (Left KL − Right KL) as the positional-bias signal: a consistently positive or negative gap means attention aligns with one side's true anatomy systematically better than the other's.

| Metric | Definition | Why it is used |
|---|---|---|
| KL divergence (raw) | `KL(ground-truth mask ‖ attention density)`, per layer, per side | Base measure of how well the attention at that layer matches where the organ actually is |
| KL divergence (generic-normalized) | Same, but the attention is divided by a generic non-spatial prompt's attention first | Isolates prompt-specific grounding from the model's default gaze, so the result isn't just "the model looks at the center of the image" |
| Left-right KL gap | Left KL minus Right KL, per layer | The direct positional-bias signal: a persistent sign across layers means one side is mechanistically worse-grounded, not just noisier |
| Best-grounded layer | Layer with the lowest KL, per organ and side | Identifies where in the network attention is most accurate, a candidate layer to target with the embedding-swap intervention |

Reported per organ, per layer, for left and right separately, plus the left-right gap curve across layers.

---

## 6. Uncertainty Under Rotation

Goal: estimate the model's predictive uncertainty under rotation, and test whether it rises with rotation and predicts failure.

**Research questions:**
- Does the model's uncertainty rise specifically when rotation produces a wrong-side (positional) error, or does it stay low even when confidently on the wrong side? Can uncertainty actually catch positional bias, or does the model fail silently?
- Is calibration (Expected Calibration Error, ECE) worse at the rotation angles that produce more wrong-side errors, linking miscalibration directly to the positional-bias failure mode rather than to generic accuracy loss?

| Setting | Value |
|---|---|
| Models | **VoxTell only** (1 model); MC dropout requires VoxTell to have dropout layers, checked first |
| How many datasets? | **2**: AMOS22-MRI, Neurite-OASIS |
| Rotation angles | **7**: 0°, 30°, 60°, 90°, 120°, 150°, 180° (same angles used for the detection and segmentation tasks) |
| Prompt(s) used | **2** (left, right) |

### 6.1 Methods
Only these three methods are used, and all are reported side by side in 6.2.

| Method | What it does | How to compute it | Why it is used | Caveat |
|---|---|---|---|---|
| **TTA variance (primary)** | Runs the model on several rotated views of the same image, rotates each output back to the original frame, and measures how much the predictions disagree | <ul><li>Segmentation: per-pixel variance or entropy of the averaged probability map</li><li>Detection: spread of box positions and variance of scores</li><li>Use a fixed TTA set that does not depend on the true rotation, to avoid leaking the answer</li></ul> | Directly measures rotation sensitivity: if the answer changes a lot across views of the same input, that disagreement signals uncertainty | Only catches inconsistency. A model that is confidently wrong at every angle (e.g. always picks the mirror-image organ) agrees with itself, giving low variance even though it is always wrong |
| **MC dropout (secondary)** | Runs the model 10-20 times per angle on the same input with dropout kept active at inference, and measures how much the repeated predictions disagree | <ul><li>Variance or entropy across the 10-20 passes</li><li>Compare with TTA at every angle</li><li>Check first that the model has dropout layers; if not, note it as a limitation for that model</li></ul> | Catches what TTA misses: uncertainty from the model's own internal randomness rather than from varying the input, so it can flag a confidently-wrong-every-time model | Needs dropout layers active at inference; not every model in this plan may support it |
| **Multi-view side agreement (third; left/right wrong-side rate only)** | Runs K views of the same case, takes the mask-centroid side (left or right of the midline) from each view, and measures how much the views disagree about the side | <ul><li>Prompt paraphrases: "left kidney", "kidney on the patient's left", "L kidney"</li><li>Mirror flip: flip the volume left-right, swap the prompt to the opposite side, un-flip the output</li><li>Small rotations (e.g. ±10°)</li><li>Use a fixed view set that does not depend on the true rotation or true side</li><li>Score: let `p` be the fraction of views that vote "left"; uncertainty = binary entropy of `p`, or `1 - \|2p - 1\|`</li><li>The ground-truth side is never used to compute it</li></ul> | A model that truly understands laterality should give the same side under paraphrased prompts and a mirror flip; disagreement means it is guessing the side. Scored against the per-case `wrong_side` outcome | Same blind spot as TTA: a model consistently wrong across all views gets a low score. Applies to the wrong-side-rate task only |

MC dropout is run alongside TTA so that a model which is wrong at every rotation angle but looks internally consistent is not mistaken for a well-calibrated, low-uncertainty model. Multi-view side agreement adds a laterality-specific signal on top of these two.

### 6.2 What to report

| No. | Item | What to report |
|---|---|---|
| 1 | Uncertainty vs. rotation angle | Mean uncertainty at each angle (does it rise with rotation?) |
| 2 | Failure detection | <ul><li>Does high uncertainty predict low Dice, a missed detection, or a wrong-side prediction?</li><li>AUROC and risk-coverage curves for TTA, MC dropout and multi-view side agreement separately</li><li>Multi-view is scored against `wrong_side`</li></ul> |
| 3 | Calibration | Expected calibration error and reliability diagrams, 0° vs. rotated angles |
| 4 | Link to attention grounding | Compare uncertainty with the layer-wise KL-divergence attention-grounding results |

---

## 7. Prompt Wording Sensitivity

Goal: test whether VoxTell's left/right behavior depends on the structure's name (by rephrasing prompts as location descriptions) or on real visual evidence (by prompting for absent structures).

**Research questions:**
- When the structure is described by location instead of name, does accuracy or side-choice change, revealing whether "left"/"right" understanding is actually tied to the structure's name rather than to genuine spatial-language comprehension?
- When prompted with a side word for a structure that is absent, does the model hallucinate a mask preferentially on one particular side, revealing a positional prior that acts independently of real visual evidence?

| Setting | Value |
|---|---|
| How many models? | **VoxTell only** (1 model) |
| How many datasets? | VoxTell's 5 segmentation datasets (AMOS22-MRI, Neurite-OASIS, KiTS23, TotalSegmentatorV2, HanSeg-CT) |
| Rotation | 0° only *(proposed)* |
| Metrics | Dice, IoU, HD95 for positive prompts; false-positive rate for negative prompts |
| How many targets (organs/lesions)? | All bilateral pairs across these 5 segmentation datasets |
| How many prompt types? | 2 (location wording, negative prompts) + canonical reference |

### 7.1 Anatomical-location wording

**What it is:** the same target, asked for by **where it is** instead of by its name.

**Question:** can VoxTell find the structure from location words alone?

For every row below, run **both** prompts (normal and location-wording) on the same image, for both left and right, and compare.

| Dataset | Normal prompt (reference) | Location-wording prompt *(proposed)* | Metrics used | Expected result |
|---|---|---|---|---|
| AMOS22-MRI, KiTS23 (kidney) | `"Segment the left kidney in this image."` | `"Segment the organ on the left side of the back of the abdomen."` | <ul><li>Dice on the location prompt</li><li>IoU on the location prompt</li><li>Drop vs. the normal prompt</li><li>Wrong-side rate</li></ul> | Small drop: model understands the location words. Large drop: model was really relying on the word "kidney" |
| AMOS22-MRI (adrenal gland) | `"Segment the left adrenal gland in this image."` | `"Segment the small gland above the left kidney."` | <ul><li>Dice on the location prompt</li><li>IoU on the location prompt</li><li>Drop vs. the normal prompt</li><li>Wrong-side rate</li></ul> | Small drop: model understands the location words. Large drop: model was really relying on the word "adrenal gland" |
| Neurite-OASIS (hippocampus) | `"Segment the left hippocampus in this image."` | `"Segment the structure in the left medial temporal lobe."` | <ul><li>Dice on the location prompt</li><li>IoU on the location prompt</li><li>Drop vs. the normal prompt</li><li>Wrong-side rate</li></ul> | Small drop: model understands the location words. Large drop: model was really relying on the word "hippocampus" |
| TotalSegmentatorV2 (lung) | `"Segment the left lung in this image."` | `"Segment the air-filled organ in the left side of the chest."` | <ul><li>Dice on the location prompt</li><li>IoU on the location prompt</li><li>Drop vs. the normal prompt</li><li>Wrong-side rate</li></ul> | Small drop: model understands the location words. Large drop: model was really relying on the word "lung" |
| HanSeg-CT (parotid gland) | `"Segment the left parotid gland in this image."` | `"Segment the salivary gland in front of the left ear."` | <ul><li>Dice on the location prompt</li><li>IoU on the location prompt</li><li>Drop vs. the normal prompt</li><li>Wrong-side rate</li></ul> | Small drop: model understands the location words. Large drop: model was really relying on the word "parotid gland" |

### 7.2 Negative prompts (target absent)

**What it is:** a prompt for a structure that is **not in the image**.

**Question:** does VoxTell return an empty mask (correct), or does it invent a mask (hallucination)?

For every row below, run the negative prompt on its image, **plus** the matched positive prompt (the organ that actually is present, as a control), on the same dataset.

| Dataset | Negative prompt *(proposed)* | Why the target is absent | Metrics used | Expected result |
|---|---|---|---|---|
| AMOS22-MRI, KiTS23 (abdomen) | `"Segment the left hippocampus in this image."` | Hippocampus is in the brain, not in an abdominal scan | <ul><li>False-positive rate</li><li>Predicted mask size on negatives</li><li>Model confidence on negatives</li><li>True-positive rate from the matched positive prompt (e.g. kidney), as the reference to read the false-positive rate against</li></ul> | Low false-positive rate: model correctly returns nothing. High rate: model hallucinates a mask for an organ that isn't there |
| Neurite-OASIS (brain) | `"Segment the left kidney in this image."` | Kidney is not in a brain scan | <ul><li>False-positive rate</li><li>Predicted mask size on negatives</li><li>Model confidence on negatives</li><li>True-positive rate from the matched positive prompt (e.g. hippocampus), as the reference to read the false-positive rate against</li></ul> | Low false-positive rate: model correctly returns nothing. High rate: model hallucinates a mask for an organ that isn't there |
| TotalSegmentatorV2 | `"Segment the left hippocampus in this image."` | Restricted to TotalSegmentatorV2 scans whose field of view excludes the head, confirmed from each scan's metadata | <ul><li>False-positive rate</li><li>Predicted mask size on negatives</li><li>Model confidence on negatives</li><li>True-positive rate from the matched positive prompt (e.g. lung), as the reference to read the false-positive rate against</li></ul> | Low false-positive rate: model correctly returns nothing. High rate: model hallucinates a mask for an organ that isn't there |
| HanSeg-CT (head and neck) | `"Segment the left kidney in this image."` | Kidney is not in a head-and-neck scan | <ul><li>False-positive rate</li><li>Predicted mask size on negatives</li><li>Model confidence on negatives</li><li>True-positive rate from the matched positive prompt (e.g. parotid gland), as the reference to read the false-positive rate against</li></ul> | Low false-positive rate: model correctly returns nothing. High rate: model hallucinates a mask for an organ that isn't there |
