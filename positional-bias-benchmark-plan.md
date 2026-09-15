# Spatial Reasoning & Positional-Bias Benchmark Plan
---

## 1. Segmentation

3D, text-prompted segmentation models, active in this repo.

### Dataset coverage

- **VoxTell**: AMOS22-MRI, Neurite-OASIS
- **Medical-SAM3**: AMOS22-MRI, Neurite-OASIS (out-of-domain), plus two in-domain datasets with
  left/right *and* upper/lower structure (see note below): Chest X-ray via Chest ImaGenome, and
  kidney ultrasound via TRUSTED
- **Text3DSAM**: AMOS22-MRI, Neurite-OASIS (out-of-domain), plus in-domain datasets (CT_HaN-Seg,
  CT_TotalSeg_cardiac, MR_TotalSeg)

Two public datasets cover both left/right and upper/lower structure, within Medical-SAM3's two
largest confirmed training modalities:

- **Chest X-ray, via Chest ImaGenome** (a scene-graph annotation layer on MIMIC-CXR): defines 29
  anatomical structures including left/right lung, and subdivides each lung into upper, mid, and
  lower lung zones, plus lung base and apical zone — giving left/right *and* upper/lower in one
  radiography dataset, the largest single modality in Medical-SAM3's training corpus.
- **Kidney ultrasound, via TRUSTED** (Tridimensional Renal UltraSound TomodEnsitometrie Dataset):
  paired 3D transabdominal ultrasound and CT from 48 patients (96 kidneys), with left/right kidney
  segmentation and 7 radiographer-annotated anatomical landmarks per kidney, including the upper
  and lower pole — giving left/right *and* upper/lower in the ultrasound modality.

Both are inherently different in dimensionality from the volumetric structures tested for
VoxTell/Text3DSAM (Chest X-ray is 2D; TRUSTED's ultrasound arm is 3D but a different acquisition
and anatomy than AMOS22-MRI/Neurite-OASIS), so these belong alongside the out-of-domain test as
additional, separate comparison points, not a replacement for it.

**Table 1.** Positional-bias analyses for the 3D segmentation models, and their readiness status
per model.

| No. | Analysis | What it is | VoxTell | Medical-SAM3 | Text3DSAM |
|---|---|---|---|---|---|
| 1 | **Rotation-Equivariance** | Mean Dice by rotation angle, plotted per organ (curve or heatmap). Shows which organs degrade faster with rotation than others. | Ready now | Pending | Ready now |
| 2 | **Equivariance-Gap Scalar** | One number per organ, derived from the Rotation-Equivariance curve above (area between the Dice-vs-angle curve and a flat baseline at 0° Dice). Turns the curve into a single value so models can be ranked directly. | Ready now | Pending | Ready now |
| 3 | **Laterality Dice Gap** | Paired left/right Dice per organ (e.g., left vs. right hippocampus, left vs. right kidney). Comes straight from existing CSVs by splitting on the organ-name prefix. | Ready now | Ready now | Ready now |
| 4 | **Wrong-Side Rate** | Share of predictions where the mask centroid lands on the wrong side (left vs. right) vs. ground truth, regardless of Dice score. | Ready now | Ready now | Ready now |
| 5 | **Centroid-Displacement Direction** | Predicted mask centroid vs. ground-truth centroid, across angles. Shows if errors pull one consistent direction (e.g., toward the un-rotated pose) or scatter randomly. | Ready now | Pending | Ready now |
| 6 | **Slice-Position Stratification** | Dice vs. normalized slice index in the volume. Tests if top/bottom-of-scan structures score worse than center-of-scan ones. | Ready now | Ready now | Ready now |
| 7 | **Boundary/Surface-Distance vs. Angle** | Boundary-precision metrics (Normalized Surface Dice, 95th-percentile Hausdorff Distance) tracked by angle, to catch boundary drift even when Dice looks stable. | Ready now | Ready now (baseline only) | Ready now |
| 8 | **Failure-Mode Classification** | At 90°/180°, does the model miss the structure entirely, or produce a shifted/rotated mask? These are different failure types with different clinical weight. | Ready now | Pending | Ready now |
| 9 | **Cross-Dataset Comparison** | Does the same rotation-Dice pattern hold on AMOS22-MRI and Neurite-OASIS, or is the bias specific to one anatomy/modality? | Ready now | Pending | Ready now |

### VoxTell root-cause investigation

Why does VoxTell degrade at certain rotation angles, and does it show a real positional bias
rather than a resampling artifact? Five steps, in order, each building on the last.

**Table 2.** Five-step diagnostic plan for identifying why VoxTell shows rotation-related
positional bias.

| No. | Step | Question it answers | What to do | Status |
|---|---|---|---|---|
| 1 | **Rule out the resampling confound** | Is the Dice drop caused by rotation resampling/field-of-view loss, not by the model itself? | For each rotated volume, compute the fraction of ground-truth mask voxels that fall outside the resampled volume's valid region. Regress Dice-drop against this fraction. A strong correlation means part of the "bias" is a preprocessing artifact, not a model failure. | Ready now, from existing sweep data |
| 2 | **Test for a learned positional shortcut** | Does the model's error pull toward where the organ would sit at 0° (a canonical-pose prior), or does it track the rotated anatomy correctly? | Centroid-displacement-direction analysis: plot predicted centroid vs. ground-truth centroid across angles, check if the direction of drift is consistent (toward canonical pose) or random. | Ready now, from existing sweep data |
| 3 | **Test whether the bias is size or organ dependent** | Does bias hit small/peripheral organs harder than large/central ones (a shortcut explanation), or is it uniform (an architecture explanation)? | Stratify the Dice-vs-angle curves by organ size and compare degradation slopes. | Ready now, from existing sweep data |
| 4 | **Check for a threshold vs. smooth decay** | Does Dice degrade gradually (partial rotation-augmentation coverage during training), or fall off a cliff at one angle (a hard augmentation-range boundary)? | Re-run the sweep at finer angle steps (e.g., 10-15°) for at least a few organs, and look for a knee in the curve. | Pending, needs a new finer-grained sweep |
| 5 | **Confirm the cause in the model itself** | Which of the above hypotheses is actually true, not just consistent with the data? | Read VoxTell's backbone architecture (standard CNN/U-Net vs. any rotation-equivariant components), its training augmentation config (rotation range and probability actually used), and whether its fusion/decoder uses absolute positional embeddings. | Pending, needs source/config review, not new inference |

### Aggregate views

**Table 3.** Aggregate visualizations already produced, or pending, per 3D model.

| No. | Aggregate view | VoxTell | Medical-SAM3 | Text3DSAM |
|---|---|---|---|---|
| 1 | Mean Dice by organ × rotation angle (heatmap) | Ready now | Pending | Ready now |

### Visual grounding with relational language (bilateral left/right only)

None of the three models were trained on relational referring expressions, only anatomy/organ
name prompts. "Left X" / "right X" is itself a relational expression, relative to body midline,
so this is a relational-grounding test scoped to bilateral structures only: for each bilateral
organ pair, does the model correctly ground "left X" vs. "right X" to the correct side. Ground
truth side comes from the anatomical mask, not the prompt. The two Medical-SAM3 in-domain rows
(Table 4) extend this to "upper X" / "lower X" as well, using the same logic against each
dataset's upper/lower zone or pole annotations.

**Table 4.** Bilateral (left/right) and, where the dataset supports it, upper/lower structures to
check, per 3D model and dataset.

| No. | Model | Dataset | Structures to check | Subjects to check |
|---|---|---|---|---|
| 1 | VoxTell | AMOS22-MRI | Kidney, adrenal gland | All AMOS22-MRI subjects with both left and right instances annotated |
| 2 | VoxTell | Neurite-OASIS | Hippocampus, and any other paired brain structure in the manifest | All Neurite-OASIS subjects with both left and right instances annotated |
| 3 | Text3DSAM | CT_HaN-Seg | Carotid artery, cochlea, anterior eyeball segment, posterior eyeball segment, lacrimal gland, submandibular gland, optic nerve, parotid gland | All CT_HaN-Seg subjects with both left and right instances annotated |
| 4 | Text3DSAM | CT_TotalSeg_cardiac | Subclavian artery, common carotid artery, brachiocephalic vein, iliac artery, iliac vena | All CT_TotalSeg_cardiac subjects with both left and right instances annotated |
| 5 | Text3DSAM | MR_TotalSeg | Kidney, adrenal gland, iliac artery, iliac vena, humerus, femur, hip, gluteus maximus, gluteus medius, gluteus minimus, autochthon, iliopsoas | All MR_TotalSeg subjects with both left and right instances annotated |
| 6 | Medical-SAM3 (out-of-domain) | AMOS22-MRI, Neurite-OASIS | Same structures as the VoxTell rows above (kidney, adrenal gland, hippocampus), for direct comparison on the exact same out-of-domain data | All AMOS22-MRI / Neurite-OASIS subjects with both left and right instances annotated |
| 7 | Medical-SAM3 (in-domain) | Chest X-ray, via Chest ImaGenome (MIMIC-CXR) | Left lung, right lung (bilateral); left/right upper, mid, and lower lung zones (upper/lower) | All subjects with left/right lung and lung-zone annotations in Chest ImaGenome |
| 8 | Medical-SAM3 (in-domain) | Kidney ultrasound, via TRUSTED | Left kidney, right kidney (bilateral); upper pole, lower pole of each kidney (upper/lower) | All 48 TRUSTED subjects (96 kidneys), using the 7 annotated anatomical landmarks per kidney |

### Worked examples

One example per row, so the experiment is concrete rather than abstract.

- **VoxTell, AMOS22-MRI, kidney**: send the prompt `"left kidney"` on a subject's volume. The
  model returns a mask. Check whether that mask's centroid actually sits on the left side of the
  body (voxel x-axis vs. volume midline), using the ground-truth kidney mask's own side label as
  the answer key. Then repeat with `"right kidney"` on the same subject and volume, expecting the
  mask to land on the opposite side.
- **VoxTell, Neurite-OASIS, hippocampus**: same idea, prompts `"left hippocampus"` and
  `"right hippocampus"` on the same brain MRI volume, checked against the manifest's
  `gt_side_axis2` label already recorded for that structure.
- **Text3DSAM, CT_HaN-Seg, parotid gland**: prompts `"left parotid gland"` and
  `"right parotid gland"` on the same head-and-neck CT volume. This dataset is in-domain for
  Text3DSAM, so a wrong-side answer here is a stronger finding than the same error on an
  out-of-domain dataset, it can't be blamed on domain shift.
- **Text3DSAM, MR_TotalSeg, gluteus maximus**: prompts `"left gluteus maximus"` and
  `"right gluteus maximus"`, a structure with a large, unambiguous left/right size, useful as an
  easy-case sanity check before trusting results on smaller structures like the adrenal gland.
- **Medical-SAM3, AMOS22-MRI, kidney**: same prompts and check as the VoxTell example above, run
  on the same subjects. Since Medical-SAM3 has never seen CT/MRI at all in training, a wrong-side
  result here is expected to be worse than VoxTell's, and tells you how the model behaves on a
  modality it has no exposure to.
- **Medical-SAM3, Chest X-ray (Chest ImaGenome), lung**: prompts `"left lung"` and `"right lung"`
  on the same CXR image. This is in-domain for Medical-SAM3 (radiography is its largest training
  modality), so compare this result against the AMOS22-MRI/Neurite-OASIS result above: a much
  lower wrong-side rate here than out-of-domain would confirm the earlier failures are a
  modality-gap effect, not a general inability to ground laterality language.
- **Medical-SAM3, Chest X-ray (Chest ImaGenome), lung zones**: same idea but for upper/lower
  instead of left/right, prompts `"right upper lung zone"` and `"right lower lung zone"` on the
  same CXR image, checked against Chest ImaGenome's own zone bounding boxes. Tests whether the
  model's absolute/relative position grounding generalizes beyond left/right to a different
  relation on the same in-domain modality.
- **Medical-SAM3, kidney ultrasound (TRUSTED), kidney**: prompts `"left kidney"` and
  `"right kidney"` on the same ultrasound scan, checked against TRUSTED's own left/right kidney
  segmentation. A second in-domain, different-modality data point alongside the Chest X-ray
  result, to see whether an in-domain finding generalizes across modalities or is specific to one.
- **Medical-SAM3, kidney ultrasound (TRUSTED), kidney pole**: prompts `"upper pole of the kidney"`
  and `"lower pole of the kidney"` on the same scan, checked against TRUSTED's annotated pole
  landmarks. Tests upper/lower grounding in a modality (ultrasound) distinct from Chest X-ray's
  radiography.

### How to analyze it, metrics

Reuses the metrics already defined earlier in this section, applied specifically to this
left/right-only prompt pair, per structure, per subject.

**Table 5.** Metrics for scoring the 3D bilateral left/right grounding experiment.

| No. | Metric | How it's used for this experiment |
|---|---|---|
| 1 | **Wrong-Side Rate** | For each (subject, structure) pair, send both the "left X" and "right X" prompts, take the predicted mask's centroid side, and compare to the prompted side. Count how often the predicted side is the *opposite* of what was asked for. This is the core pass/fail signal for relational grounding. |
| 2 | **Laterality Dice Gap** | Compare mean Dice when the model is prompted with the correct-matching side (e.g., "left kidney" scored against the true left kidney mask) between the left and right prompts. A large gap means the model segments one side much better than the other, even when it gets the side right. |
| 3 | **Centroid-Displacement Direction** | When the model gets the side wrong, check whether the predicted centroid still lands *near* the correct-side structure's true location, or somewhere else entirely. Distinguishes "swapped the label" from "genuinely lost track of the anatomy." |
| 4 | **Failure-Mode Classification** | For wrong-side cases, classify whether the predicted mask is a real structure on the wrong side (a true label swap) or a missing/empty prediction (the model didn't find the relational cue at all). |

---

## 2. Localization

2D localization models: bbox / point output.

### Dataset coverage

All 8 models (Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B,
Molmo2-8B, North-Micro-Vision-Instruct) ran on all 3 datasets:

- **LGG** (brain)
- **Kvasir** (colon)
- **ISIC** (skin)

**Metrics for the benchmark**, from the results already scored across 3 datasets (LGG, Kvasir,
ISIC) for these 8 models. "Ready now" means computable today from the existing result CSVs, no
new inference needed.

**Table 6.** Positional-bias metrics for the 2D localization models, and which are already
computed vs. ready to compute.

| No. | Metric | Definition | Models | Status |
|---|---|---|---|---|
| 1 | Detect Rate | Share of P0–P4 calls that returned any usable prediction | All 8 (Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B, Molmo2-8B, North-Micro-Vision-Instruct) | Already computed |
| 2 | Wrong-Side Rate (per direction) | Share of prompted answers landing on the side opposite the one named in the prompt, broken down left/right/upper/lower | All 8 | Already computed (pooled); per-direction ready now |
| 3 | No-Shift Rate | Share of P1–P4 answers that stayed near the model's own P0 answer; didn't move | All 8 | Already computed |
| 4 | Mean Normalized Distance | Distance from predicted point to ground-truth centroid, normalized by image diagonal | All 8 | Already computed |
| 5 | Mean IoU | Overlap between predicted box and ground-truth box | 7 of 8, all except Molmo2-8B (point-only, no box to compute overlap from) | Already computed |
| 6 | Center-Bias Rate | Share of P0 answers landing near image center regardless of true target location | All 8 | Ready now |
| 7 | Wrong-Side Rate vs. Target Eccentricity | Wrong-Side Rate stratified by how far the true target sits from image center | All 8 | Ready now |
| 8 | Cross-Dataset Consistency | Whether Wrong-Side Rate holds steady across LGG, Kvasir, and ISIC | All 8 | Ready now |

---

## 3. VQA

Open-ended visual question answering, for the localization models that can generate free text.

### Model eligibility for VQA and MCQ experiments

Not all 8 localization models can generate free text, so not all of them can be used for a
spatial-VQA or answer-position-bias experiment. Split by actual capability, not by architecture
family alone. This eligibility applies to both the VQA experiments here and the MCQ experiments
in Section 4.

**Table 7.** Which of the 8 localization models can be used for VQA and Multiple-Choice
experiments.

| No. | Model | Usable for VQA | Usable for Multiple-Choice (answer-position bias) | Why |
|---|---|---|---|---|
| 1 | Qwen2.5-VL-3B | Yes | Yes | Full instruction-tuned VLM, handles open QA and MCQ natively |
| 2 | InternVL2.5 | Yes | Yes | Full instruction-tuned VLM, handles open QA and MCQ natively |
| 3 | Molmo2-8B | Yes | Yes | Chat-capable VLM, pointing is one capability among several, not its only mode |
| 4 | North-Micro-Vision-Instruct | Yes | Yes | Explicitly built for general VQA/OCR/grounding, likely the best fit of the 8 |

**Usable**: Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct (4 models).
**Excluded**: Grounding DINO and OWLv2, since both are pure detectors that only output boxes and
confidence scores, with no free-text generation at all. Kosmos-2 and LocateAnything-3B are also
excluded: both can generate text, but neither has a confirmed general instruction-following/MCQ
capability. Kosmos-2's released focus was grounding/captioning, not instruction-following QA, and
LocateAnything-3B is built around structured coordinate-output task templates (referring
expressions, pointing) with no confirmed open-ended QA behavior. Both remain usable for the
Section 2 localization/grounding experiments, where their capability is confirmed.

### Bilateral-structure datasets for VQA and MCQ

LGG, Kvasir, and ISIC don't have paired left/right structures, they're single-lesion tasks.
Instead of pulling in a separate dataset (CHAOS/ACDC), this reuses the same three in-domain 3D
datasets already used for the Section 1 bilateral visual-grounding experiment, one representative
2D slice per subject extracted from each volume, since these VLMs take 2D images, not volumes:

- **CT_HaN-Seg**: head-and-neck CT, e.g. left/right parotid gland
- **CT_TotalSeg_cardiac**: cardiac/vascular CT, e.g. left/right common carotid artery
- **MR_TotalSeg**: whole-body MR, e.g. left/right kidney

Reusing these datasets, rather than CHAOS/ACDC, means the same bilateral structures get tested at
two task levels, 3D segmentation grounding (Section 1) and 2D VQA/MCQ (here), on the exact same
underlying data.

### VQA prompts

Same order-counterbalancing concern as the MCQ experiment: phrasing a question as "...the left X
or the right X?" always mentions "left" first, so a model with any order-of-mention bias (a
tendency to pick whichever option was named first) would look identical to a laterality bias.
Each question is run twice per image, once phrased left-then-right and once phrased
right-then-left, to separate the two.

**Table 8.** Example VQA prompts for the bilateral left/right experiment, by dataset, with the
mention order swapped between runs.

| No. | Dataset | Bilateral structure | Example VQA prompt (left-then-right) | Example VQA prompt (right-then-left) | Models |
|---|---|---|---|---|---|
| 1 | CT_HaN-Seg (2D slice) | Left parotid gland, right parotid gland | "Is the gland marked in this image the left parotid gland or the right parotid gland?" | "Is the gland marked in this image the right parotid gland or the left parotid gland?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Left common carotid artery, right common carotid artery | "Is the artery marked in this image the left common carotid artery or the right common carotid artery?" | "Is the artery marked in this image the right common carotid artery or the left common carotid artery?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | MR_TotalSeg (2D slice) | Left kidney, right kidney | "Is the kidney marked in this image the left kidney or the right kidney?" | "Is the kidney marked in this image the right kidney or the left kidney?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

### Existing medical VQA datasets with laterality questions (alternative to slice extraction)

Instead of extracting 2D slices from the 3D volumes above, these established medical VQA
benchmarks already contain real, human-written laterality/position questions, no new
preprocessing or synthetic question-writing needed. Found via HuggingFace and web search.

**Table 9.** Existing public medical VQA/MCQ datasets with laterality coverage, found via
HuggingFace and web search.

| No. | Dataset | Images | QA pairs | Modality / body region | Laterality + upper/lower coverage | Answer format | Compatible with your 4 VLMs? | Link |
|---|---|---|---|---|---|---|---|---|
| 1 | GEMeX | 151,025 | 1,605,575 | Chest X-ray, built on Chest ImaGenome anatomical regions | Chest ImaGenome defines left/right lung zones plus left/right upper, mid, and lower lung zones | **Natively single-choice and multiple-choice**, plus open/closed-ended, plus region-grounding (bounding box) per answer | Yes, directly — native 2D chest X-ray, no preprocessing needed | [med-vqa.com/GEMeX](https://www.med-vqa.com/GEMeX) · [arXiv:2411.16778](https://arxiv.org/abs/2411.16778) |
| 2 | SPARC-Rad | 300 (from TCIA healthy-control studies) | 300 | CT, MRI, and radiography, across abdomen, chest, breast, neuro, and musculoskeletal | Manually written by radiology trainees to test anatomical identification, localization, laterality, regional recognition, and inter-structure spatial relationships | Closed-ended, graded via LLM-as-judge; MCQ-style answer choices not separately confirmed | Partial — radiography cases work directly; CT/MRI cases need a 2D slice extracted per case first, same as Table 8's approach | [arXiv:2608.00100](https://arxiv.org/abs/2608.00100) |

**GEMeX is the stronger fit of the two**: it natively supports MCQ, works with your 4 VLMs
without any preprocessing, is large-scale (1.6M QA pairs to filter down), and its
Chest-ImaGenome basis already covers both left/right and upper/lower in one dataset. SPARC-Rad is
more directly on-topic (laterality and spatial relationships are explicitly what it was built to
test) but is much smaller (300 pairs), needs slice extraction for its CT/MRI cases, and its exact
answer format needs confirming before treating it as a drop-in MCQ source.

Both are genuinely out-of-domain for your 4 usable 2D VLMs (neither was part of their training
data), keeping the zero-shot framing consistent with your other 2D datasets.

### How to analyze it, metrics

Reuses the same metrics style as the Section 1 bilateral-grounding experiment, adapted to a
text answer instead of a mask. Rows tagged "VQA" apply here; rows tagged "MCQ" apply to the
Section 4 experiment instead.

**Table 10.** Metrics for scoring the 2D VQA and Multiple-Choice bilateral experiments.

| No. | Metric | Applies to | How it's used |
|---|---|---|---|
| 1 | Correct-Side Rate | VQA, MCQ | Share of answers where the stated side (left/right) matches the ground-truth side of the structure actually shown. The core accuracy signal. |
| 2 | Wrong-Side Rate | VQA, MCQ | Share of answers where the stated side is the *opposite* of the ground truth, same concept as the Wrong-Side Rate used elsewhere in this roster, just scored from text instead of a mask centroid. |
| 3 | No-Answer / Refusal Rate | VQA | Share of responses that don't commit to either side (hedging, "unclear," or an unrelated answer). A separate failure mode from getting it wrong. |
| 4 | Per-Structure Accuracy | VQA, MCQ | Correct-Side Rate broken down by structure (parotid gland vs. carotid artery vs. kidney), to see which anatomy the model handles best or worst. |
| 5 | Answer-Position Bias (Letter Preference) | MCQ only | Compare accuracy when the correct answer is option A vs. when it's option B. If accuracy drops (or the model just picks "A" more often) regardless of which side is actually correct, that's a letter-position bias, separate from genuine anatomical understanding. |
| 6 | Content vs. Letter Bias Disentanglement | MCQ only | Cross-tabulate true side (left/right) against chosen letter (A/B) against correctness. Separates "the model is biased toward right-side anatomy" from "the model is biased toward option A," which look identical in raw accuracy but are different problems. |
| 7 | Order-of-Mention Bias | VQA only | Compare accuracy, and raw pick rate, between the left-then-right and right-then-left phrasings of the same question (Table 8). If the model just tends to pick whichever side was named first, that's an order-of-mention bias, separate from genuine laterality understanding. |

---

## 4. MCQ

Multiple-choice framing of the same bilateral left/right questions, run for the same 4 usable
VLMs (Table 7), added to test answer-position bias.

### Multiple-Choice prompts (answer-position bias)

Same question as the VQA prompts in Section 3, phrased as MCQ. Run each pair twice per image,
once with the correct answer as option A and once as option B, to separate genuine
bilateral-grounding accuracy from a simple preference for one letter position.

**Table 11.** Example Multiple-Choice prompts for the bilateral left/right experiment, by
dataset.

| No. | Dataset | Bilateral structure | Example MCQ prompt | Models |
|---|---|---|---|---|
| 1 | CT_HaN-Seg (2D slice) | Left parotid gland, right parotid gland | "Which gland is marked in this image? A) Left parotid gland B) Right parotid gland" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Left common carotid artery, right common carotid artery | "Which artery is marked in this image? A) Left common carotid artery B) Right common carotid artery" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | MR_TotalSeg (2D slice) | Left kidney, right kidney | "Which kidney is marked in this image? A) Left kidney B) Right kidney" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

### How to analyze it, metrics

Uses Table 10 in Section 3, specifically the rows tagged "MCQ" or "VQA, MCQ": Correct-Side Rate,
Wrong-Side Rate, Per-Structure Accuracy, Answer-Position Bias, and Content vs. Letter Bias
Disentanglement.

### Distance / depth order-bias experiment

Same idea as the laterality MCQ above, applied to the Distance and Depth reasoning types (Table
14) instead of left/right. 3D segmentation models and the two pure detectors can't be probed
this way, since they only output a mask/box, never a distance or depth judgment in text; this
experiment is limited to the 4 usable 2D VLMs from Table 7 (Qwen2.5-VL-3B, InternVL2.5,
Molmo2-8B, North-Micro-Vision-Instruct).

Uses the same 2D slices as the sections above (CT_HaN-Seg, CT_TotalSeg_cardiac, MR_TotalSeg),
each showing two distinguishable structures whose relative distance or depth is already known
from the ground-truth 3D coordinates (e.g., which of two organs' centroids is closer to a marked
reference point, or which sits in a more anterior slice). Each question is run twice per image,
once with the correct answer as option A and once as option B, exactly like the laterality MCQ.

**Table 12.** Example Distance/Depth MCQ prompts, order-swapped to test position bias.

| No. | Dataset | Structure pair | Reasoning type | Example MCQ prompt | Models |
|---|---|---|---|---|---|
| 1 | MR_TotalSeg (2D slice) | Kidney vs. liver | Distance | "Which structure's center is closer to the marked point? A) Kidney B) Liver" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Common carotid artery vs. subclavian artery | Distance | "Which vessel is closer to the marked point? A) Common carotid artery B) Subclavian artery" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | CT_HaN-Seg (2D slice) | Parotid gland vs. submandibular gland | Depth | "Which gland is more anterior, closer to the front of the face? A) Parotid gland B) Submandibular gland" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

**Table 13.** Metrics for scoring the Distance/Depth order-bias experiment.

| No. | Metric | How it's used |
|---|---|---|
| 1 | Correct-Answer Rate | Share of answers matching the ground-truth distance/depth relationship, computed from the two structures' 3D centroids. |
| 2 | Answer-Position Bias (Letter Preference) | Compare accuracy, and raw pick rate, when the correct answer is option A vs. option B, same as the laterality MCQ bias test. |
| 3 | Content vs. Letter Bias Disentanglement | Cross-tabulate the true closer/more-anterior structure against the chosen letter against correctness, to separate a genuine bias toward one structure from a bias toward one letter. |

---

## 5. Spatial Reasoning

Which of the important spatial-reasoning skills each model, across all four sections above, can
actually be tested on.

**Table 14.** VLMs / segmentation models vs. spatial reasoning types.

| No. | Model | Relative position (left/right/above/below) | Absolute position (top/bottom/center) | Distance (near/far) | Depth (in front/behind) | Topological (inside/outside/overlap) |
|---|---|---|---|---|---|---|
| 1 | VoxTell (3D seg) | Yes | Yes | No | No | Yes |
| 2 | Medical-SAM3 (3D seg) | Yes | Yes | No | No | Yes |
| 3 | Text3DSAM (3D seg) | Yes | Yes | No | No | Yes |
| 4 | Qwen2.5-VL-3B | Yes | Yes | Yes | Yes | Yes |
| 5 | InternVL2.5 | Yes | Yes | Yes | Yes | Yes |
| 6 | Molmo2-8B | Yes | Yes | Yes | Yes | Yes |
| 7 | North-Micro-Vision-Instruct | Yes | Yes | Yes | Yes | Yes |
| 8 | Kosmos-2 | Yes | Yes | No | No | Yes |
| 9 | LocateAnything-3B | Yes | Yes | No | No | Yes |
| 10 | Grounding DINO | Yes | Yes | No | No | Yes |
| 11 | OWLv2 | Yes | Yes | No | No | Yes |

**Table 15.** What each spatial reasoning type means, and how to evaluate it.

| No. | Reasoning type | What it means | How to evaluate it |
|---|---|---|---|
| 1 | Relative position | The location of one object described with respect to another object, e.g. "left of", "above", "between" | Prompt with the relation (e.g. "left kidney"), get the model's box/mask/point, compare its position to the reference object using the ground-truth layout. Metric: Wrong-Side Rate |
| 2 | Absolute position | The location of an object described with respect to the whole image/volume, e.g. "top", "bottom", "center" | Prompt with the absolute term, check if the returned box/mask/point centroid actually falls in that region of the image/volume (e.g. top third, center zone). Metric: Region-Match Accuracy |
| 3 | Distance | A qualitative near/far judgment between two objects | For VLMs: ask "which is closer/farther" or "is X near Y", compare to ground-truth centroid distance. For segmentation/detection models: not directly askable, compute centroid-to-centroid distance from the two output masks/boxes instead. Metric: Accuracy vs. ground-truth ranking, or correlation between predicted and true distance |
| 4 | Depth | Front/behind or closer/farther in the camera or scan viewing axis | For VLMs: ask "is X in front of Y" (2D images) or "which is closer to the viewer". For 3D volumes: reframe as slice-position (which axial/coronal/sagittal slice) since there's no camera depth. Metric: Accuracy vs. ground-truth ordering along that axis |
| 5 | Topological relationship | How two regions relate structurally, e.g. inside, outside, overlapping, touching, containing | Prompt with the relation (e.g. "tumor inside kidney"), check output mask/box overlap or containment against the ground-truth masks. Metric: IoU / containment ratio, or binary correct/incorrect on the relation |

**Table 16.** Metrics for evaluating each spatial reasoning type.

| No. | Reasoning type | Metric | What it measures |
|---|---|---|---|
| 1 | Relative position | Wrong-Side Rate | Share of answers where the predicted side (left/right/above/below) is the opposite of the ground truth |
| 2 | Relative position | Laterality Dice Gap | Difference in mean Dice/accuracy between the two sides of a relation, when the model does get the side right |
| 3 | Absolute position | Region-Match Accuracy | Share of predictions whose centroid falls in the correct absolute region of the image/volume (e.g. top third, center zone) |
| 4 | Absolute position | Center-Bias Rate | Share of predictions landing near the image/volume center regardless of the true target location |
| 5 | Distance | Correct-Answer Rate | Share of near/far judgments matching the ground-truth centroid-distance ranking |
| 6 | Distance | Distance Correlation | Correlation between the model's predicted distance ordering and the true ground-truth distance, for models that output a number |
| 7 | Depth | Correct-Answer Rate | Share of front/behind or closer/farther judgments matching the ground-truth ordering along the viewing/scan axis |
| 8 | Topological relationship | IoU / Containment Ratio | Overlap or containment between the predicted and ground-truth masks/boxes, for inside/outside/overlap/touching claims |
| 9 | Topological relationship | Binary Correct/Incorrect Rate | Share of topological relation claims (e.g. "tumor inside kidney") that match ground truth, for models that only state the relation in text |
| 10 | Any type (bias check) | Answer-Position Bias (Letter Preference) | For MCQ-framed questions, compares accuracy/pick-rate when the correct answer is option A vs. option B, to separate content understanding from letter-position bias |

---
