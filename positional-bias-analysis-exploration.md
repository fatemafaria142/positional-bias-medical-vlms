# Exploring Positional Bias in Medical Imaging VLMs: Types of Analysis

**Scope**: positional bias in **2D and 3D medical imaging**, grounded in this repo's current
work — text-prompted segmentation VLMs (**VoxTell**, **Medical-SAM3**, **Text3DSAM**) evaluated
under controlled **rotation** on **AMOS22-MRI** (3D CT/MRI organs) and **Neurite-OASIS**
(3D brain MRI structures), with per-(organ, angle) Dice heatmaps and inference-time curves
already in `scripts/` and `output/`.

This document surveys the **types of analysis** that can be used to study positional bias,
organized by task family, so a benchmarking framework can be scoped deliberately rather than
defaulting to "just rotation + segmentation."

**Positional bias**, in this context: a model's output quality, confidence, or behavior
systematically depends on *where* an anatomical structure sits in the image/volume, or on its
*orientation*, rather than purely on the structure's true content/shape.

---

## 0. What the codebase already establishes

Reading the pipeline (`scripts/rotate_multiangle_samples.py`, `scripts/dice_heatmap_organ_angle.py`,
`scripts/summarize_dice_and_inference.py`, `predictions_voxtell_*_rotation_*.csv`) shows the
current design is:

- **Task**: 3D text-prompted segmentation (VoxTell; extendable to Medical-SAM3, Text3DSAM).
- **Perturbation**: controlled rotation at fixed angles (0°, 30°, 60°, 90°, 120°, 150°, 180°).
- **Stratification**: per-organ, per-angle Dice; per-angle inference time.
- **Datasets**: AMOS22-MRI (abdominal/multi-organ CT/MRI) and Neurite-OASIS (brain MRI, with
  explicit laterality — `left_hippocampus`, `right_kidney`, etc.).
- **Annotators/splits**: three parallel subject pools (faria, hridoy, nipu) pooled for
  aggregate stats.

Everything below is framed as **extensions along two axes**: (a) more *types of positional
perturbation/stratification* beyond rotation, and (b) more *task types* beyond segmentation —
so you can decide how far to widen the benchmark.

---

## 1. Axes of "position" relevant to 2D/3D medical imaging

1. **Orientation / rotation** *(already implemented)* — rotation angle, flip (mirrored/reflected
   volume), canonical vs. non-canonical pose relative to standard radiological orientation
   (axial/coronal/sagittal).
2. **Slice / depth position (3D only)** — where a structure sits along the scan axis
   (superior↔inferior, anterior↔posterior); normalized slice index (0–100% through volume).
3. **In-plane spatial position (2D and per-slice in 3D)** — center vs. periphery, distance from
   image/volume center, quadrant, proximity to field-of-view border/crop edge.
4. **Laterality** — left vs. right structures (already present as distinct organ labels in your
   Neurite-OASIS predictions, e.g. left/right hippocampus) — a natural, low-effort bias axis to
   formalize since the labels already encode it.
5. **Translation (synthetic)** — shifting the same structure to different coordinates while
   holding shape/content fixed, to isolate pure positional effect from confounds like organ
   size or scan quality that correlate with real position in natural data.
6. **Scale/zoom position interplay** — whether positional bias interacts with structure size
   (e.g., small structures near the periphery suffer disproportionately more than large ones).
7. **Relative position between structures** — for multi-organ scenes, whether performance on
   one organ depends on the position of a neighboring/overlapping organ (relevant once you go
   beyond single-organ metrics).

Rotation and (once added) slice-position + laterality are the highest-leverage axes for this
codebase, since the data already carries the labels needed (organ name incl. laterality, angle,
slice-bearing volumes) — no new annotation needed, only new aggregation scripts.

---

## 2. Analysis types, by task family

### 2.1 Segmentation (your current core task)
- **Organ × angle Dice heatmap** *(done)* — extend with **organ × slice-position** and
  **organ × distance-from-volume-center** heatmaps using the same pooling pattern as
  `dice_heatmap_organ_angle.py`.
- **Laterality gap**: paired left/right Dice comparison per organ (e.g., left vs. right
  hippocampus, left vs. right kidney) — directly answerable from existing prediction CSVs by
  splitting on the organ-name prefix.
- **Slice-position stratification**: Dice vs. normalized slice index within volume — tests
  whether structures near the top/bottom of a scan are segmented worse.
- **Rotation robustness curve** *(done: mean Dice by angle)* — extend to **per-organ** robustness
  curves (some organs may degrade faster with rotation than others) and to **inference-time
  vs. angle** interaction already tracked.
- **Boundary-shift / centroid-bias analysis**: compare predicted mask centroid vs. ground-truth
  centroid across angles — does the model's error have a systematic directional pull (e.g.,
  toward canonical/un-rotated pose) rather than random noise?
- **Failure-mode clustering at extreme angles**: at 90°/180°, does Dice failure look like
  "missed structure entirely" vs. "shifted/rotated mask" — different clinical implications.
- **Cross-dataset bias transfer**: does the rotation-Dice curve shape on AMOS22-MRI match
  Neurite-OASIS, or is bias magnitude dataset/anatomy-specific (abdominal organs vs. brain
  structures may tolerate rotation differently)?
- **Cross-model comparison**: run the same rotation protocol through Medical-SAM3 and
  Text3DSAM (both already vendored in-repo) to test whether positional bias is a VoxTell
  artifact or a shared property of text-prompted segmentation VLMs.

### 2.2 Detection / Localization (natural extension of segmentation)
- **Bounding-box-from-mask center bias**: derive boxes from existing masks and check if
  predicted-box centers drift toward volume/image center under rotation, independent of true
  organ position.
- **Position-conditioned recall**: recall for organs near volume boundary vs. center, using the
  same stratification bins as the segmentation heatmap.

### 2.3 Report Generation (if extending to VLM radiology-report models)
- **Finding-order vs. rotation**: does reporting order/completeness for a structure change once
  the volume is rotated, even when the structure is still clearly present?
- **Laterality-confusion in text**: does the generated report swap left/right descriptors under
  rotation/flip — a concrete, checkable positional-bias failure mode for text output.
- **Sentence-position accuracy decay**: accuracy of findings mentioned early vs. late in a
  generated report, independent of rotation — a baseline positional bias orthogonal to
  image-space rotation.

### 2.4 Visual Question Answering (VQA) on medical images
- **Spatial questions under rotation**: ask "is the mass on the left or right kidney" before and
  after rotation/flip — measures whether the model's *language* answer tracks the true
  post-rotation anatomy or a stale prior.
- **Existence/counting questions vs. slice position**: does "is structure X present in this
  slice" accuracy depend on slice position within the volume.
- **Answer-choice-order bias** (if using MCQ-style medical VQA benchmarks): position of the
  correct choice among options, independent of rotation — a separate, well-known VQA bias worth
  measuring as a control.

### 2.5 Visual Grounding (referring expression → region/mask, in medical imaging)
- **Grounding accuracy vs. rotation angle**: point/box/mask IoU for a text query like "the right
  kidney" as rotation increases — a natural pairing with your current segmentation setup, since
  VoxTell-style models are already text-prompted.
- **Laterality-referring-expression bias**: does grounding for expressions using explicit
  laterality ("left", "right") degrade faster under flip/rotation than expressions using organ
  name alone — isolates whether the bias lives in the language grounding of "left/right" or in
  the visual segmentation itself.

### 2.6 Classification (organ presence / abnormality presence)
- **Presence-classification accuracy vs. slice/rotation position**: simpler signal than full
  segmentation, useful as a lightweight secondary benchmark using the same perturbed volumes.

---

## 3. Cross-cutting analysis techniques (apply to any task above)

- **Controlled perturbation protocol** *(rotation already implemented)* — extend the same
  `rotate_multiangle_samples.py`-style approach to translation, flip, and slice-crop
  perturbations for causal isolation of positional effect from confounds.
- **Position-stratified metric heatmaps/curves** *(pattern already implemented for organ ×
  angle)* — reusable primitive: swap the stratification axis (angle → slice index, angle →
  distance-from-center, angle → laterality) while keeping the same pooling/plotting code.
- **Statistical regression testing**: fit metric ~ angle (+ organ, + dataset) with mixed effects
  (organ/subject as random effects, since you already pool 3 annotators) to get a significance
  and effect-size estimate per organ, not just a visual heatmap read.
- **Centroid/shift bias analysis**: compare predicted vs. ground-truth centroid displacement as
  a function of angle — distinguishes "random noise" bias from "systematic pull toward a
  canonical pose" bias.
- **Training-data orientation-distribution audit**: check whether AMOS22-MRI/Neurite-OASIS
  training data is itself orientation-imbalanced (most scans near canonical pose) — a likely
  root cause worth confirming rather than assuming.
- **Equivariance gap metric**: for a perfectly rotation-equivariant segmenter, Dice would be
  angle-invariant; define a single scalar "equivariance gap" (e.g., AUC of Dice-drop vs. angle)
  per model/organ/dataset for easy cross-model comparison — good candidate for a benchmark
  leaderboard metric.
- **Inference-time-vs-bias interaction** *(already tracked: inference time by angle)* — check
  whether slower inference at certain angles correlates with degraded accuracy (e.g., model
  doing more "work"/uncertainty at hard orientations) — an interesting secondary finding, not
  just a compute artifact.

---

## 4. Toward a benchmarking framework (2D/3D medical imaging)

Suggested structure: a **(position-axis × task-family)** grid, built incrementally on your
existing rotation-segmentation infrastructure.

| | Segmentation | Detection | Grounding | VQA | Report Gen |
|---|---|---|---|---|---|
| **Rotation/orientation** | done (VoxTell, extend to Medical-SAM3/Text3DSAM) | reuse masks→boxes | natural next task (text-prompted models already grounding-capable) | spatial-laterality Qs | laterality-confusion check |
| **Slice position (3D)** | next: reuse heatmap pattern | derivable | — | presence Qs by slice | — |
| **Laterality** | next: near-zero extra cost, labels already exist | — | left/right referring expr. | left/right Qs | left/right in text |
| **Translation (synthetic)** | new perturbation script | reuse | — | — | — |

### Suggested phasing
- **Phase 1 (near-zero new infra)**: laterality-gap analysis and slice-position stratification
  on existing segmentation predictions — reuses current CSVs and heatmap script pattern.
- **Phase 2**: cross-model rotation study — run Medical-SAM3 and Text3DSAM through the same
  rotation protocol as VoxTell to see if bias is model-specific or systemic.
- **Phase 3**: extend perturbation engine to translation/flip; add equivariance-gap metric as a
  standard leaderboard scalar.
- **Phase 4**: add visual grounding (natural fit — same models, text-prompted) and VQA
  (spatial/laterality questions) as new task types under the same rotation protocol.
- **Phase 5**: report generation, if/when a report-generation medical VLM is added to the
  comparison set — reuse the laterality-confusion and sentence-position bias analyses.

---

## 5. Open questions to resolve before finalizing scope

- Should Phase 1 (laterality + slice-position) be prioritized since it requires no new
  perturbation code, only new aggregation over existing predictions?
- Is cross-model comparison (VoxTell vs. Medical-SAM3 vs. Text3DSAM) a goal, or is VoxTell the
  sole subject and the other repos are just reference implementations?
- Should synthetic translation/flip perturbations be added, or is rotation the sole
  causally-controlled axis for this benchmark?
- Do you want a single **equivariance-gap** leaderboard metric summarizing all bias types, or
  separate reported metrics per axis (rotation-gap, laterality-gap, slice-position-gap)?
- Is extending beyond segmentation (to grounding/VQA/report generation) in scope for this
  benchmark, or a separate follow-up paper?
