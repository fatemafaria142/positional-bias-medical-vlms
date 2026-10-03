# Positional Bias in Medical VLMs — Research Summary

## The overarching research question

**Trust the prompt or trust the pixels?** When a vision-language model is given a
text prompt that names a spatial position ("the *left* kidney", "the *upper*
lobe"), does the model's output actually move to the correct anatomical
location — i.e. is it grounded in the image — or does it fall back on a
shortcut: ignoring the word, always predicting the same side, or following the
wording even when it contradicts the true anatomy?

## Key term: the "positional bias word"

Throughout this project, a **positional bias word** (also "position word") is
the part of a text prompt that names a spatial location or laterality —
*left, right, upper, lower, anterior, posterior* — as opposed to the
**entity word**, which names *what* to find (an organ, lesion, or structure),
e.g. "kidney", "tumor". Positional bias, as studied here, is any systematic
failure of a model's output to move correctly with the positional bias word
in the prompt.

## Research questions

**RQ1 — Existence of positional bias.** Does the model's predicted region
change when the positional bias word in the prompt changes (e.g. "left" →
"right"), all else held fixed — or does the output stay the same regardless
of which positional bias word is used ("no-shift" bias)?

**RQ2 — Direction/correctness of the shift.** When the output *does* shift in
response to a changed positional bias word, does it shift to the
anatomically **correct** location, or merely to a different but still
arbitrary location (i.e. sensitivity to the positional bias word without
actual grounding)?

**RQ3 — Consistency across phrasing.** Across different ways of expressing
the same positional bias word (e.g. "the right kidney" vs. "the kidney on the
right side"), does the model consistently land on the same, correct
anatomical location?

**RQ4 — Positional bias word vs. pixels under conflict.** When the positional
bias word in the prompt and the true anatomical location disagree (e.g. after
a mirror-flip that leaves ground-truth labels uncorrected), does the model
follow the **positional bias word** in the prompt, or the **true anatomy** in
the image?

**RQ5 — Robustness of grounding under geometric transformation.** As the
image is progressively rotated (0°–180°), does the model's ability to
correctly resolve the positional bias word degrade gradually, break down at a
specific angle, or remain stable?

**RQ6 — Relative contribution of the positional bias word vs. the entity
word.** Of the two components of a segmentation prompt — the entity word
("kidney") and the positional bias word ("right") — how much does each
contribute to (a) *what* region is predicted, and (b) *where* (which side) it
is predicted, as measured by word-ablation and attribution methods
(Shapley/LIME/KernelSHAP)?

**RQ7 — Mechanistic basis of the positional bias word's effect.** Inside the
model, is the positional bias word's effect implemented as genuine spatial
grounding — i.e. image-text cross-attention that concentrates on the
anatomically named side — or as a shortcut/memorized prior that does not
require real visual evidence (tested via noising/masking the named region and
via layer-wise embedding analysis)?

**RQ8 — Where and how is the positional bias word decision made?** At which
layer(s) of the model does the representation of the positional bias word
diverge into a left-vs-right decision — early and distributed, or late and
localized to a single layer/module?

**RQ9 — Does positional bias word grounding vary by anatomy?** Is the model's
sensitivity to the positional bias word uniform across organ types, or does
it depend on organ size/contrast (e.g. large organs like the kidney vs. small
ones like the adrenal gland)?

These nine questions were investigated progressively across the project's
phases — first at the level of raw model outputs (RQ1–RQ3, 2D benchmark),
then under adversarial geometric conflict (RQ4–RQ5, rotation/mirror-flip
probes), and finally at the level of internal model mechanism (RQ6–RQ9,
VoxTell internals investigation). See "What was found" below for the answers
each phase produced.

## What was done — the arc of the project

### Phase 1 — 2D benchmark across many zero-shot grounding models (superseded)
The project began with a broad 2D localization benchmark: **15 candidate
models** (CLIPSeg, Moondream2, PaliGemma 2, Florence-2, EVF-SAM, SEEM,
Grounding DINO, OWLv2, Kosmos-2, CogVLM, Qwen2.5-VL-7B, InternVL2.5,
DeepSeek-VL2, Molmo-7B-D, Molmo2-8B) tested with position-varying prompts
("locate {organ}" / left / right / upper / lower) on **5 medical imaging
datasets** (LGG brain tumor MRI, MosMedData+ lung CT, Kvasir-SEG polyp
endoscopy, QaTa-COV19 chest X-ray, ISIC 2018 skin lesion). Scope was later
trimmed to 3 datasets (LGG, Kvasir, ISIC) and **11 models were actually run**
end-to-end.

This phase, its datasets, and the PaliGemma-centric pipeline were later
**removed from the repo** once the project pivoted to 3D, but the results are
preserved in git history.

### Phase 2 — Pivot to 3D volumetric segmentation
2D box/point-prediction models turned out to be a poor testbed for medical
spatial reasoning (see findings below), so the project moved to **full 3D,
text-guided segmentation** on two volumetric datasets:
- **AMOS22 MRI** — 10 subjects, 13 abdominal organs.
- **Neurite OASIS** — 10 subjects, 35 brain structures.

A model-candidate evaluation compared four free-text 3D segmentation
approaches: MedCLIP-SAMv2, Text3DSAM, Medical-SAM3, and VoxTell.

### Phase 3 — Model selection: VoxTell
- Text3DSAM essentially failed on AMOS22 MRI (mean Dice ≈ **0.10**).
- **VoxTell** scored mean Dice **0.828** on AMOS22 MRI (13 organs) and
  **0.739** on Neurite OASIS (35 structures) — the only candidate producing
  non-trivial segmentation on OASIS's fine brain structures.
- VoxTell became the sole model used for the rest of the investigation.

### Phase 4 — Adversarial bias-stress-test datasets
Two 3D probe datasets were built specifically to stress-test positional
grounding:
- **Rotation set** — a proper 180° anatomical rotation about the
  superior/inferior axis (determinant +1), with ground-truth left/right labels
  correctly recomputed, testing whether VoxTell follows true anatomical content
  rather than a fixed voxel-array location.
- **Mirror-flip set** — a reflection (determinant −1) where left/right ground
  truth labels are deliberately **not** relabeled, so the prompt and the true
  anatomy actively disagree (verified per-subject, e.g. every left/right label
  flips for subject `amos_0544`).
- This was later extended into a **multi-angle rotation sweep** (0°, 30°, 60°,
  90°, 120°, 150°, 180°) across 6 subjects × 2 datasets, cross-checked by three
  independent annotators (`faria/`, `nipu/`, `hridoy/`).

### Phase 5 — Mechanistic / model-internals investigation
The centerpiece of the current work: a 16-row investigation plan
(`voxtell-model-internals-investigation-plan.md`) asking, for a prompt like
*"segment the right kidney"*, what actually drives VoxTell's output — image
anatomy, the word "right", the word "kidney", or a memorized prior?

- **Rows 1–11** (word ablation, LIME/SHAP/Shapley attribution, image-text
  alignment) — run on AMOS22 MRI (10 subjects × 4 bilateral organs = 40
  instances) and on OASIS.
- **Rows 12–16** ("mechanistic" experiments — image-flip, noise-replacement,
  embedding blend/patch, cross-attention, layer-wise divergence) — run as a
  3-subject pilot, explicitly labeled as leads rather than established
  results.

## What was found

### 2D phase — most models don't really use spatial words
- Only **point-output** models (Molmo2-8B, Moondream2-point) showed real
  positional sensitivity: wrong-side rate of 5–18%, well below chance.
- **9 of 11 models** sat at ~48–53% wrong-side rate — statistically
  indistinguishable from a coin flip. In other words, most zero-shot grounding
  models do not meaningfully use "left/right/upper/lower" wording at all.
- Several models (Grounding DINO, Florence-2, PaliGemma 2) had **85–100%
  "no-shift" rates**: the output didn't move regardless of how the prompt was
  worded.
- Qwen2.5-VL-7B had near-zero IoU across datasets — a basic localization
  failure, independent of bias.
- ISIC was flagged as a weak bias-test dataset because lesions tend to fill
  most of the frame, leaving little room to detect a spatial shift.

### 3D phase — VoxTell benchmark
- Mean Dice: **0.828** (AMOS22 MRI, 13 organs), **0.739** (Neurite OASIS, 35
  structures).
- Large, high-contrast organs (liver, kidneys, spleen) scored 0.90–0.97; small
  or thin structures (adrenal glands, duodenum) scored 0.66–0.72.
- 4 OASIS structures (choroid plexus L/R, vessel L/R) scored a **consistent
  0.0** — total, repeatable misses, not noise.
- Left/right Dice gaps were small (mean |L−R| ≈ 0.019–0.027), with a slight
  right > left tendency on bilateral OASIS pairs.

### Mechanistic phase — VoxTell is genuinely grounded, not shortcut-driven
From rows 1–11 (AMOS22 MRI, 40 bilateral-organ instances):
- The position word acts as a genuine **selector**: baseline Dice 0.944
  (kidney) / 0.646 (adrenal gland), and predicted side matched ground-truth
  side in **40/40** cases.
- Removing the position word (e.g. "segment the kidney" instead of "segment
  the right kidney") roughly **doubles** the predicted volume — both kidneys
  get segmented (Dice drops 0.944 → 0.631, consistent with the both-sides
  prediction) — and the centroid shifts toward the midline, not toward a fixed
  wrong side.
- Removing the **organ word** is far more destructive (Dice → 0.000, 5–15%
  empty masks) than removing the position word: organ identity drives *what*
  gets segmented, position drives *which side*.
- Shapley/LIME attribution agree: organ word dominates for accuracy metrics,
  position word dominates for the centroid-shift (laterality) metric.
- No evidence of a fixed, memorized left/right shortcut — the predicted side
  reliably tracks the prompt's stated side.

From rows 13–16 (3-subject mechanistic pilot — caveated as leads, not final
results):
- **Row 13** (organ replaced by noise): the kidney mask needs real image
  evidence — 5/6 cases go empty when the kidney region is noised, i.e. no
  memorized shape prior. The adrenal gland relies partly on surrounding
  anatomical context (mask survives at ~80% size when only the gland itself is
  noised, but dies when the whole image is blanked).
- **Row 14** (embedding blend/patch): the side decision is controlled by the
  text embedding via a sharp switch (not gradual) between blend ratios
  0.25→0.75. No single decoder layer can flip the side alone, but layers 1–3
  together can — the side decision is distributed early in the network, not
  decided late.
- **Row 15** (cross-attention): for "left kidney", attention concentrates
  2.4×–8.6× on the named kidney vs. 0.6×–2.6× on the other side, starting from
  layer 1 — real spatial grounding. The adrenal gland shows much weaker
  enrichment, only emerging from layer 4 onward.
- **Row 16** (layer-wise embedding divergence): left/right text
  representations already differ at the text encoder's input. For the kidney,
  the left/right gap widens through the decoder; for the adrenal gland it
  shrinks — yet masks still land on the correct side for all 3 subjects.

**Overall conclusion:** across rows 1–16, VoxTell shows genuine positional
grounding for bilateral organs — more strongly for large organs (kidney) than
small ones (adrenal gland) — with no evidence of a left/right shortcut or
memorized-prior bias. This is the opposite conclusion from the earlier 2D
phase, where most models ignored positional wording entirely.

## Key files

| File | Content |
|---|---|
| `positional-bias-benchmark-plan.md` | Current benchmark/dataset planning doc |
| `voxtell-model-internals-investigation-plan.md` | The 16-row mechanistic investigation plan |
| `voxtell-groundedness-results-amos.md` | Rows 1–11 results on AMOS22 MRI |
| `voxtell-groundedness-results-oasis.md` | Rows 1–11 results on Neurite OASIS |
| `voxtell-mechanistic-results-amos-rows13-16.md` | Rows 13–16 pilot results (latest work) |
| `voxtell-rotation-sweep-results.md` | Rotation-sweep (0–180°) Dice and side-tracking results |
| `voxtell_amos22_mri_groundedness_investigation_results/voxtell-amos-investigation-report.md` | Consolidated AMOS investigation report |
| `PAPER_METHODS_ATTRIBUTION.md`, `ATTRIBUTION_METHODS_EXPLAINED.md` | Method write-ups / in-progress paper draft |
| `VOXTELL_MODEL_COMPONENTS.md`, `model_layer_summary.md` | VoxTell architecture notes used for the internals work |
| `samples/sample_rotation/README.md`, `samples/sample_mirror/README.md` (git history) | Design docs for the rotation/mirror-flip probe datasets |

Earlier-phase docs (2D benchmark plan, `Results/Findings/FINDINGS.md`,
`VOXTELL_RESULTS_ANALYSIS.md`, `MODEL_CANDIDATES.md`) were removed from the
working tree as the project moved on, but remain recoverable from git
history.

## Open threads / unfinished work

- **Rows 13–16 are a 3-subject pilot only.** The results explicitly note this
  is not a real distribution and should be treated as leads to confirm on more
  subjects, not established findings. Scaling this up is the natural next
  step.
- Row 14's "layers 1–3 jointly control side" result and row 13's
  adrenal-gland context-dependence both have unresolved confounds — e.g. it's
  not yet possible to separate "a learned position prior" from "reading
  neighboring organs for context."
- `PAPER_METHODS_ATTRIBUTION.md` still has bracketed placeholders awaiting
  final run values — a paper draft in progress.
- ~~The **multi-angle rotation sweep**... no markdown analysis/results doc yet~~
  — done, see `voxtell-rotation-sweep-results.md`. Headline finding: grounding
  holds (100% correct bilateral side) through ~60° rotation, then breaks
  sharply at 90°, converging to **100% wrong-side** (not merely empty)
  predictions by 150–180° on 5/7 bilateral pairs tested — evidence for a
  fixed-pose/location prior that only takes over once rotation exceeds the
  range the rest of the project (rows 1–11) tested.
- **New: image-region-wise attribution suite (plan rows 17–21), implemented but not yet run
  on Kaggle.** Rows 1–16 answered "image vs. word" only piecemeal (row 13's organ-noising,
  row 15's scalar attention-enrichment ratio). Rows 17–21 add: an exact voxel-grid occlusion
  Shapley map (17, the direct image analogue of rows 5–11's word table), a per-layer
  attention/KL-divergence diagnostic (18, from the VGMED/VGRefine medical-MLLM paper, arXiv
  2603.14323), a cheap gradient/Grad-CAM approximation to row 17 (19), a deeper repeat of row 1
  at a genuine intermediate spatial fused-feature layer (20), and a joint image-vs-word Shapley
  that finally puts "does the image or the word matter more" on one common scale (21). Code is
  in `scripts/voxtell_mechanistic.py`/`scripts/voxtell_experiments.py`, wired into all four
  mechanistic Kaggle notebooks, and verified locally (CPU, no GPU) against tiny stand-in models —
  but not yet run on the real model. Row 20's hook point is the one unverified piece; the
  notebook prints a shape sanity-check on the first subject before trusting its numbers. This is
  now the main next step for the "image vs. word contribution" and deeper memorization questions.
- OASIS groundedness results (`voxtell-groundedness-results-oasis.md`) exist
  alongside the AMOS results but haven't been cross-summarized against them in
  a single findings doc.

## Phase-wise: what I did and what I found

**Phase 1 — 2D zero-shot grounding benchmark.**
Did: Ran 11 zero-shot grounding models (Molmo2, Moondream2, PaliGemma 2,
Florence-2, Grounding DINO, OWLv2, Qwen2.5-VL, etc.) on 3 2D medical datasets
with left/right/upper/lower prompt variants.
Found: 9/11 models were at ~50% wrong-side rate (chance level) or had 85–100%
"no-shift" outputs — most 2D grounding models don't use positional words at
all; only point-output models showed real sensitivity.

**Phase 2–3 — 3D model candidate selection.**
Did: Benchmarked MedCLIP-SAMv2, Text3DSAM, Medical-SAM3, and VoxTell for
free-text 3D segmentation on AMOS22 MRI and Neurite OASIS.
Found: Text3DSAM failed (Dice ≈ 0.10); VoxTell was the only model with strong,
usable Dice (0.828 AMOS / 0.739 OASIS), so it became the sole model for the
rest of the study.

**Phase 4 — Rotation / mirror-flip adversarial probes.**
Did: Built a 180° true-rotation set (correct GT relabeling), a mirror-flip set
(GT deliberately left unrelabeled so prompt/anatomy conflict), and a
multi-angle 0–180° rotation sweep across 3 annotators.
Found: Datasets and predictions are generated and verified (e.g. label-flip
confirmed per subject), but no results/analysis write-up exists yet for this
phase — this is the main open thread.

**Phase 5 — Mechanistic investigation (organ word vs. position word).**
Did: Ran 15 experiments on VoxTell (word ablation, SHAP/LIME attribution,
embedding swap, organ-region noising, embedding blend/patch, cross-attention,
layer-wise divergence) on AMOS22 MRI (rows 1–11, full sample) and a 3-subject
pilot (rows 13–16).
Found: Organ word drives *what* is segmented (Dice → 0 if removed); position
word drives *which side* (removing it merges both sides); cross-attention and
layer-wise analysis confirm this is genuine spatial grounding, not a
memorized left/right shortcut — strongest for large organs (kidney), weaker
for small ones (adrenal gland).
