# Exploring Positional Bias: Types of Analysis Across Vision-Language Tasks (General Version)

This is the **general, task-agnostic** version of the positional-bias exploration — not scoped
to any one codebase. Use [`positional-bias-analysis-exploration.md`](positional-bias-analysis-exploration.md)
for the version tailored to this repo's medical-imaging VLM work (VoxTell / Medical-SAM3 /
Text3DSAM, rotation on AMOS22-MRI / Neurite-OASIS); use this one as the broader reference when
planning what to borrow from other domains (NLP long-context, general VQA benchmarks, etc.).

**Positional bias**, broadly: a model's output quality, confidence, or behavior systematically
depends on *where* something is located — in the image/volume, in the input sequence, in the
output sequence, or in the context window — rather than on the content itself.

---

## 1. What "position" can mean (axes of positional bias)

Before picking analyses, it helps to separate the different senses of "position" that can be
studied. A benchmarking framework should state which axis (or axes) it targets.

1. **Spatial position in 2D image** — center vs. periphery, quadrant (TL/TR/BL/BR), edge/border
   proximity, distance from image center, absolute pixel coordinates.
2. **Spatial position in 3D volume** — slice/depth position (superior/inferior, anterior/
   posterior), axial/coronal/sagittal plane, laterality (left/right), position within an organ,
   distance from volume center or from a reference anatomical landmark.
3. **Orientation / rotation position** — rotation angle, viewpoint, flip (mirrored) vs. canonical
   pose.
4. **Sequence/order position in input** — token order, frame order in video, order of objects
   listed in a prompt, order of multiple-choice options.
5. **Sequence/order position in output** — where a finding appears in a generated report
   (first sentence vs. last), primacy/recency in generated lists.
6. **Position within a multi-item context** — "lost in the middle" effects when many
   images/objects/regions are given at once (position within the context, not within one image).
7. **Relative position between entities** — bias tied to *relations* (above/below/left-of) rather
   than absolute coordinates — relevant to visual grounding and spatial VQA.

---

## 2. Analysis types by task family

### 2.1 Segmentation (2D and 3D)
- **Spatial performance heatmaps**: bin Dice/IoU/surface-distance by location (grid cells in 2D,
  voxel-coordinate bins or slice index in 3D) and visualize where performance drops.
- **Center-vs-edge stratification**: compare metrics for structures near the image/volume border
  vs. center.
- **Slice-position stratification (3D)**: performance as a function of normalized slice index
  (e.g., 0–100% through the volume) — tests whether early/late slices are under-served.
- **Organ/structure-relative position**: performance vs. the target's position *within* its
  containing organ (e.g., tumor near organ boundary vs. organ core).
- **Rotation/orientation robustness**: apply controlled rotations and measure Dice degradation
  as a function of rotation angle — is there a "preferred" orientation the model was implicitly
  trained on?
- **Laterality bias**: compare left-side vs. right-side structures for asymmetric performance
  (a common, under-reported bias in medical imaging models).
- **Boundary-shift analysis**: does the predicted mask systematically shift toward image center
  or toward a canonical position, regardless of true position (a "central bias" prior)?
- **Translation/shift perturbation**: translate the same structure to different positions
  (synthetically) and measure how much prediction quality depends purely on placement.

### 2.2 Detection / Localization
- **Bounding-box center distribution analysis**: plot predicted vs. ground-truth box centers to
  detect a pull toward the image center (a well-documented bias in object detectors).
- **Position-conditioned precision/recall**: recall and false-positive rate as a function of
  spatial bin, distance-from-center, or scale-normalized position.
- **Edge-effect analysis**: detection rate for objects that are partially cropped or near
  borders vs. fully-contained objects.
- **Anchor/prior bias probing**: if the architecture uses anchors or positional priors, check
  whether detections cluster around anchor positions independent of true object locations.
- **Multi-object scenes**: does detection accuracy depend on an object's position in *listing
  order* within the scene graph or prompt (if position is provided as text)?

### 2.3 Report Generation (radiology / captioning)
- **Finding-order bias**: does the model report certain finding types earlier or later
  regardless of true clinical priority — and does *sentence position* correlate with
  correctness (models are often more accurate in the first 1–2 sentences)?
- **Primacy/recency analysis**: measure hallucination rate and omission rate as a function of
  position in the generated report.
- **Anatomical-region coverage bias**: which body regions/organs get mentioned proportionally
  more/less regardless of actual finding presence — is there a bias toward structures near
  image center (commonly imaged/centered organs)?
- **Input-order sensitivity**: if multiple images/slices are fed in sequence, does report
  content depend on which slice was shown first/last ("lost-in-the-middle" for multi-image
  input)?
- **Template/position anchoring**: does the model default to boilerplate phrasing tied to
  section position (e.g., "no acute findings" always in the same report slot) independent of
  content?

### 2.4 Visual Question Answering (VQA)
- **Spatial-question stratification**: separate questions that reference position ("what is on
  the left") from position-agnostic questions, and compare accuracy.
- **Answer-position bias (multiple-choice VQA)**: does accuracy or model preference correlate
  with the position/order of the correct answer among choices (A/B/C/D bias), independent of
  content — a classic bias in MCQA-style benchmarks.
- **Object-position vs. answer-correctness**: for "where is X" or counting/existence questions,
  does accuracy depend on where the referenced object actually sits in the image?
- **Question-order bias in multi-turn/multi-question settings**: does the position of a question
  in a sequence of questions affect the answer (context contamination / recency effects).
- **Center-bias probing with counterfactual placement**: paste the same object at different
  image locations (synthetic augmentation) and ask identical questions to isolate pure
  positional effects from content effects.

### 2.5 Visual Grounding / Referring Expression Comprehension
- **Grounding accuracy vs. absolute position**: IoU between predicted and ground-truth boxes
  binned by target location.
- **Relational-expression bias**: for referring expressions using relative terms (leftmost,
  topmost, nearest to X), measure whether models default to a "central" or "largest" object
  as a fallback, ignoring the positional qualifier.
- **Distractor-position sensitivity**: when multiple similar objects exist, does grounding
  accuracy depend on the position of the *distractor* relative to the target (e.g., closer
  distractors cause more confusion)?
- **Expression-length vs. position interplay**: do longer/more complex referring expressions
  reduce positional bias (more reliance on language) or increase it (more reliance on
  shortcuts)?

### 2.6 Classification (organ/lesion presence, staging, etc.)
- **Position-stratified confusion matrices**: per spatial bin, compare precision/recall/F1.
- **Saliency-position alignment**: check whether saliency/attention maps concentrate on the
  correct object location or on a fixed "default" location regardless of input (Grad-CAM /
  attention rollout stratified by true target position).

### 2.7 Multi-image / Video / Long-context tasks
- **Frame-position bias**: accuracy on a query as a function of which frame contains the
  relevant evidence (first frame vs. middle vs. last frame) — analogous to "lost in the
  middle" in NLP long-context work.
- **Temporal-order shuffling**: shuffle frame order and measure whether performance depends on
  the *position* frames were shown in rather than their content.
- **Context-length interaction**: does positional bias grow, shrink, or stay constant as the
  number of images/slices in context increases?

### 2.8 Cross-cutting / task-agnostic analysis techniques
These can be layered onto *any* of the tasks above:

- **Counterfactual/synthetic position perturbation**: programmatically move, rotate, flip, or
  translate the same content and re-run the model — isolates positional effect from confound
  of "different content at different positions" in natural data.
- **Position-stratified metric curves**: the general pattern of "metric vs. position bin"
  used throughout section 2 — should be a first-class primitive in the benchmarking framework.
- **Statistical bias testing**: fit a regression of performance metric ~ position features
  (coordinates, angle, slice index, order index) and test significance/effect size, rather than
  only visual inspection of heatmaps.
- **Attention/saliency-based diagnosis**: visualize where the model "looks" vs. where the
  target actually is, to distinguish a genuine positional prior in attention from a
  downstream-only effect.
- **Center-of-mass / prior-shift analysis**: compare the distribution of predicted positions
  (centroid of masks/boxes/answer references) against the distribution of ground-truth
  positions — a shifted or over-concentrated predicted distribution indicates a positional
  prior.
- **Training-data position-distribution audit**: correlate observed test-time positional bias
  with the positional distribution of the *training* data (e.g., if lesions are rarely
  annotated near volume edges in training, is edge performance worse at test time?).
- **Symmetry/equivariance testing**: test invariance under transformations that should not
  matter (rotation, flip, translation) — quantify how far the model is from the ideal
  equivariant/invariant behavior.
- **Cross-dataset position-bias transfer**: does a positional bias observed on one dataset
  persist on another — distinguishes dataset-artifact bias from model-inherent bias.
- **Bias-vs-difficulty confound control**: check whether "harder" cases (smaller structures,
  low contrast) happen to cluster at certain positions, to avoid mistaking a difficulty
  confound for a pure positional bias.

---

## 3. Toward a benchmarking framework

A framework could be structured along two orthogonal dimensions:

1. **Position axis** (from Section 1): spatial-2D, spatial-3D/slice, rotation/orientation,
   sequence/order, multi-item context, relative-position.
2. **Task family** (from Section 2): segmentation, detection, report generation, VQA, grounding,
   classification, multi-image/video.

For each (axis, task) cell that's relevant, define:
- A **perturbation protocol** (how position is systematically varied — synthetic placement,
  rotation, shuffling, re-ordering).
- A **stratification scheme** (how outputs are binned by position for reporting).
- A **bias metric** (e.g., performance gap between best/worst position bin, regression slope of
  metric vs. position, KL-divergence between predicted and true position distributions).
- A **significance/robustness check** (bootstrap CI, regression p-value) so bias claims are
  statistically grounded, not just visual.

### Suggested phased scope
- **Phase 1 (core)**: 3D segmentation — slice-position stratification, laterality bias, rotation
  robustness.
- **Phase 2**: extend the same perturbation/stratification infrastructure to detection and
  classification (reuse position-stratified metric curves + counterfactual placement).
- **Phase 3**: add report generation (sentence-position bias, finding-order bias) and VQA
  (answer-position bias, spatial-question stratification) — these need new metrics (text-position
  based, not purely geometric).
- **Phase 4**: visual grounding and multi-image/video — relational-expression bias, frame-position
  ("lost in the middle") bias.
- **Cross-cutting infrastructure (build once, reuse everywhere)**: counterfactual perturbation
  engine, position-stratified metric plotting, regression-based bias testing, training-data
  position-distribution audit tooling.

---

## 4. Open questions to resolve before finalizing scope

- Which axis is the primary target — spatial (2D/3D) or sequence/order?
- Should the framework be **task-general** (one shared perturbation/metric library across
  segmentation, VQA, grounding, report generation) or **task-specific** benchmarks that share
  only reporting conventions?
- Do you want **synthetic perturbation** (controlled, causal) as the primary method, **natural
  stratification** (observational, using existing dataset position variation) as primary, or
  both in parallel (recommended — synthetic isolates causality, natural stratification checks
  real-world relevance)?
- Which datasets/tasks beyond segmentation should be included to cover report generation, VQA,
  and grounding?
