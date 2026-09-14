# Positional-Bias Model Roster: Localization + Segmentation (2D & 3D)
---

## 1. Segmentation models (3D): text-prompted, active in this repo

### Dataset coverage

- **VoxTell**: AMOS22-MRI, Neurite-OASIS
- **Medical-SAM3**: AMOS22-MRI, Neurite-OASIS (out-of-domain)
- **Text3DSAM**: AMOS22-MRI, Neurite-OASIS (out-of-domain), plus in-domain datasets (CT_HaN-Seg,
  CT_TotalSeg_cardiac, MR_TotalSeg)


"Ready now" means the metric can be computed today, either the full rotation sweep already ran,
or the metric only needs existing result files with no new inference. "Pending" means the
rotation sweep for that model hasn't run yet, so that metric can't be computed until it does.

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

| No. | Step | Question it answers | What to do | Status |
|---|---|---|---|---|
| 1 | **Rule out the resampling confound** | Is the Dice drop caused by rotation resampling/field-of-view loss, not by the model itself? | For each rotated volume, compute the fraction of ground-truth mask voxels that fall outside the resampled volume's valid region. Regress Dice-drop against this fraction. A strong correlation means part of the "bias" is a preprocessing artifact, not a model failure. | Ready now, from existing sweep data |
| 2 | **Test for a learned positional shortcut** | Does the model's error pull toward where the organ would sit at 0° (a canonical-pose prior), or does it track the rotated anatomy correctly? | Centroid-displacement-direction analysis: plot predicted centroid vs. ground-truth centroid across angles, check if the direction of drift is consistent (toward canonical pose) or random. | Ready now, from existing sweep data |
| 3 | **Test whether the bias is size or organ dependent** | Does bias hit small/peripheral organs harder than large/central ones (a shortcut explanation), or is it uniform (an architecture explanation)? | Stratify the Dice-vs-angle curves by organ size and compare degradation slopes. | Ready now, from existing sweep data |
| 4 | **Check for a threshold vs. smooth decay** | Does Dice degrade gradually (partial rotation-augmentation coverage during training), or fall off a cliff at one angle (a hard augmentation-range boundary)? | Re-run the sweep at finer angle steps (e.g., 10-15°) for at least a few organs, and look for a knee in the curve. | Pending, needs a new finer-grained sweep |
| 5 | **Confirm the cause in the model itself** | Which of the above hypotheses is actually true, not just consistent with the data? | Read VoxTell's backbone architecture (standard CNN/U-Net vs. any rotation-equivariant components), its training augmentation config (rotation range and probability actually used), and whether its fusion/decoder uses absolute positional embeddings. | Pending, needs source/config review, not new inference |

### Aggregate views

| No. | Aggregate view | VoxTell | Medical-SAM3 | Text3DSAM |
|---|---|---|---|---|
| 1 | Mean Dice by organ × rotation angle (heatmap) | Ready now | Pending | Ready now |

### Visual grounding with relational language (bilateral left/right only)

None of the three models were trained on relational referring expressions, only anatomy/organ
name prompts. "Left X" / "right X" is itself a relational expression, relative to body midline,
so this is a relational-grounding test scoped to bilateral structures only: for each bilateral
organ pair, does the model correctly ground "left X" vs. "right X" to the correct side. Ground
truth side comes from the anatomical mask, not the prompt.

| No. | Model | Dataset | Bilateral structures to check | Subjects to check |
|---|---|---|---|---|
| 1 | VoxTell | AMOS22-MRI | Kidney, adrenal gland | All AMOS22-MRI subjects with both left and right instances annotated |
| 2 | VoxTell | Neurite-OASIS | Hippocampus, and any other paired brain structure in the manifest | All Neurite-OASIS subjects with both left and right instances annotated |
| 3 | Text3DSAM | CT_HaN-Seg | Carotid artery, cochlea, anterior eyeball segment, posterior eyeball segment, lacrimal gland, submandibular gland, optic nerve, parotid gland | All CT_HaN-Seg subjects with both left and right instances annotated |
| 4 | Text3DSAM | CT_TotalSeg_cardiac | Subclavian artery, common carotid artery, brachiocephalic vein, iliac artery, iliac vena | All CT_TotalSeg_cardiac subjects with both left and right instances annotated |
| 5 | Text3DSAM | MR_TotalSeg | Kidney, adrenal gland, iliac artery, iliac vena, humerus, femur, hip, gluteus maximus, gluteus medius, gluteus minimus, autochthon, iliopsoas | All MR_TotalSeg subjects with both left and right instances annotated |
| 6 | Medical-SAM3 | Datasets from its own training corpus (SAM 3 fine-tuned on 33 datasets, 10 modalities, not the same CVPR-BiomedSegFM subsets Text3DSAM used) | Not yet identified in this roster, needs its own bilateral-structure inventory, do not reuse the Text3DSAM list above | Pending, identify Medical-SAM3's actual training datasets first |

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

### How to analyze it, metrics

Reuses the metrics already defined earlier in Section 1, applied specifically to this
left/right-only prompt pair, per structure, per subject.

| No. | Metric | How it's used for this experiment |
|---|---|---|
| 1 | **Wrong-Side Rate** | For each (subject, structure) pair, send both the "left X" and "right X" prompts, take the predicted mask's centroid side, and compare to the prompted side. Count how often the predicted side is the *opposite* of what was asked for. This is the core pass/fail signal for relational grounding. |
| 2 | **Laterality Dice Gap** | Compare mean Dice when the model is prompted with the correct-matching side (e.g., "left kidney" scored against the true left kidney mask) between the left and right prompts. A large gap means the model segments one side much better than the other, even when it gets the side right. |
| 3 | **Centroid-Displacement Direction** | When the model gets the side wrong, check whether the predicted centroid still lands *near* the correct-side structure's true location, or somewhere else entirely. Distinguishes "swapped the label" from "genuinely lost track of the anatomy." |
| 4 | **Failure-Mode Classification** | For wrong-side cases, classify whether the predicted mask is a real structure on the wrong side (a true label swap) or a missing/empty prediction (the model didn't find the relational cue at all). |

---

## 2. Localization models (2D): bbox / point output

### Dataset coverage

All 8 models (Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B,
Molmo2-8B, North-Micro-Vision-Instruct) ran on all 3 datasets:

- **LGG** (brain)
- **Kvasir** (colon)
- **ISIC** (skin)

**Metrics for the benchmark**, from the results already scored across 3 datasets (LGG, Kvasir,
ISIC) for these 8 models. "Ready now" means computable today from the existing result CSVs, no
new inference needed.

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
| 9 | Significance vs. Chance | Statistical test confirming Wrong-Side Rate is really different from the 50% chance baseline | All 8 | Ready now |

### Model eligibility for VQA and MCQ experiments

Not all 8 localization models can generate free text, so not all of them can be used for a
spatial-VQA or answer-position-bias experiment. Split by actual capability, not by architecture
family alone.

| No. | Model | Usable for VQA | Usable for Multiple-Choice (answer-position bias) | Why |
|---|---|---|---|---|
| 1 | Qwen2.5-VL-3B | Yes | Yes | Full instruction-tuned VLM, handles open QA and MCQ natively |
| 2 | InternVL2.5 | Yes | Yes | Full instruction-tuned VLM, handles open QA and MCQ natively |
| 3 | Molmo2-8B | Yes | Yes | Chat-capable VLM, pointing is one capability among several, not its only mode |
| 4 | North-Micro-Vision-Instruct | Yes | Yes | Explicitly built for general VQA/OCR/grounding, likely the best fit of the 8 |
| 5 | Grounding DINO | No | No | Pure detector, outputs only boxes and confidence scores, no free-text generation at all |
| 6 | OWLv2 | No | No | Same, zero-shot detector only, no QA capability |
| 7 | Kosmos-2 | Yes, pilot-test first | Yes, pilot-test first | Real multimodal LLM, can generate free text, but its released focus was grounding/captioning, not instruction-following QA, unclear how reliably it follows an MCQ prompt |
| 8 | LocateAnything-3B | Yes, pilot-test first | Yes, pilot-test first | Built around structured coordinate-output task templates (categories, referring expressions, pointing), unclear if it retains general open-ended instruction-following for arbitrary questions |

**Solid, no pilot needed**: Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct
(4 models).
**Plausible, run a small pilot first**: Kosmos-2, LocateAnything-3B (2 models).
**Excluded**: Grounding DINO, OWLv2 (2 models), structurally incapable of producing a QA-style
answer.

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

#### Table 1: VQA

| No. | Dataset | Bilateral structure | Example VQA prompt | Models |
|---|---|---|---|---|
| 1 | CT_HaN-Seg (2D slice) | Left parotid gland, right parotid gland | "Is the gland marked in this image the left parotid gland or the right parotid gland?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Left common carotid artery, right common carotid artery | "Is the artery marked in this image the left common carotid artery or the right common carotid artery?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | MR_TotalSeg (2D slice) | Left kidney, right kidney | "Is the kidney marked in this image the left kidney or the right kidney?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

#### Table 2: Multiple-Choice (answer-position bias)

Same question, phrased as MCQ. Run each pair twice per image, once with the correct answer as
option A and once as option B, to separate genuine bilateral-grounding accuracy from a simple
preference for one letter position.

| No. | Dataset | Bilateral structure | Example MCQ prompt | Models |
|---|---|---|---|---|
| 1 | CT_HaN-Seg (2D slice) | Left parotid gland, right parotid gland | "Which gland is marked in this image? A) Left parotid gland B) Right parotid gland" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Left common carotid artery, right common carotid artery | "Which artery is marked in this image? A) Left common carotid artery B) Right common carotid artery" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | MR_TotalSeg (2D slice) | Left kidney, right kidney | "Which kidney is marked in this image? A) Left kidney B) Right kidney" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

### How to analyze it, metrics

Reuses the same metrics style as the Section 1 bilateral-grounding experiment, adapted to a
text answer instead of a mask.

| No. | Metric | Applies to | How it's used |
|---|---|---|---|
| 1 | Correct-Side Rate | VQA, MCQ | Share of answers where the stated side (left/right) matches the ground-truth side of the structure actually shown. The core accuracy signal. |
| 2 | Wrong-Side Rate | VQA, MCQ | Share of answers where the stated side is the *opposite* of the ground truth, same concept as the Wrong-Side Rate used elsewhere in this roster, just scored from text instead of a mask centroid. |
| 3 | No-Answer / Refusal Rate | VQA | Share of responses that don't commit to either side (hedging, "unclear," or an unrelated answer). A separate failure mode from getting it wrong. |
| 4 | Per-Structure Accuracy | VQA, MCQ | Correct-Side Rate broken down by structure (parotid gland vs. carotid artery vs. kidney), to see which anatomy the model handles best or worst. |
| 5 | Answer-Position Bias (Letter Preference) | MCQ only | Compare accuracy when the correct answer is option A vs. when it's option B. If accuracy drops (or the model just picks "A" more often) regardless of which side is actually correct, that's a letter-position bias, separate from genuine anatomical understanding. |
| 6 | Content vs. Letter Bias Disentanglement | MCQ only | Cross-tabulate true side (left/right) against chosen letter (A/B) against correctness. Separates "the model is biased toward right-side anatomy" from "the model is biased toward option A," which look identical in raw accuracy but are different problems. |
| 7 | Significance vs. Chance | VQA, MCQ | Binomial test of Correct-Side Rate against the 50% chance baseline (a 2-option question), so a result like "58% correct" can be labeled as a real effect or noise. |

### Existing medical VQA datasets with laterality questions (alternative to slice extraction)

Instead of extracting 2D slices from the 3D volumes above, these established medical VQA
benchmarks already contain real, human-written laterality/position questions, no new
preprocessing or synthetic question-writing needed. Found via HuggingFace and web search.

| No. | Dataset | Images | QA pairs | Modality / body region | Laterality coverage | Answer format | Access |
|---|---|---|---|---|---|---|---|
| 1 | VQA-RAD | 315 | 3,515 | X-ray/CT/MRI, head, chest, abdomen | Confirmed, e.g. "Which side of the heart border is obscured?" answer "right" | Open-answer, not natively MCQ | Public |
| 2 | SLAKE | 642 | 7,000+ | CT/MRI/X-ray, brain, neck, chest, abdomen, pelvic cavity, 39 organs | Has an explicit "position" question category; specific bilateral-organ questions not yet confirmed | Open-answer, not natively MCQ; bilingual (English/Chinese) | Public |
| 3 | PMC-VQA | 149,000 | 227,000 | Broad, pulled from PubMed Central figures, many modalities | Not confirmed, would need filtering for laterality questions | Open-answer | Public, HuggingFace (`xmcmic/PMC-VQA`) |
| 4 | OmniMedVQA | 118,010 | 127,995 | 12 modalities, 20+ human anatomical regions, 73 source datasets | Not confirmed for left/right specifically, but broad anatomical coverage makes it likely present, worth filtering | **Natively MCQ**, each question ships with incorrect distractor options already built in | Public, HuggingFace (`foreverbeliever/OmniMedVQA`) |
| 5 | GEMeX | 151,025 | 1,605,575 | Chest X-ray, built on Chest ImaGenome anatomical regions | Chest ImaGenome defines left/right lung zones, so laterality questions are plausible, needs filtering to confirm exact wording | **Natively single-choice and multiple-choice**, plus region-grounding (bounding box) per answer | Public (med-vqa.com/GEMeX) |
| 6 | Medical-Diff-VQA | 164,324 image pairs | 700,703 | Chest X-ray (MIMIC-CXR derived) | Confirmed, has a "location" question category with a worked example: "what abnormality is seen in the left lung?" | Open-answer | Requires PhysioNet credentialed access |
| 7 | MIMIC-CXR-VQA / MIMIC-Ext-MIMIC-CXR-VQA | MIMIC-CXR-derived | Large-scale | Chest X-ray | Location-based questions, likely includes left/right lung | Open-answer | Requires PhysioNet credentialed access |

**Two stand out as the best fit**, since they solve the MCQ-format gap that VQA-RAD/SLAKE have:

- **OmniMedVQA** is already MCQ-native, no distractor-writing needed, directly usable for
  Table 2 as-is once laterality questions are filtered out of its 128k QA items.
- **GEMeX** is the strongest overall candidate: natively supports both VQA (open/closed-ended)
  and MCQ (single/multi-choice) in one dataset, is built on an anatomical-region ontology that
  already defines left/right lung zones, and even ships region-grounding boxes per answer, useful
  for cross-checking against your Section 2 localization results on the same images.

All are genuinely out-of-domain for your 8 2D VLMs (none were trained on medical VQA data), so
this keeps the zero-shot framing consistent with your other 2D datasets. The two PhysioNet-gated
datasets (Medical-Diff-VQA, MIMIC-CXR-VQA) need credentialed access approval before download,
worth factoring into timeline if you go that route.
