# Multiple Approaches for Identifying and Characterizing Positional Bias in Medical Vision-Language Models
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
largest confirmed training modalities. Both are inherently different in dimensionality from the
volumetric structures tested for VoxTell/Text3DSAM (Chest X-ray is 2D; TRUSTED's ultrasound arm
is 3D but a different acquisition and anatomy than AMOS22-MRI/Neurite-OASIS), so these belong
alongside the out-of-domain test as additional, separate comparison points, not a replacement for
it.

**Table 1.** Medical-SAM3 in-domain datasets covering left/right and upper/lower structure.

| No. | Dataset | Modality | Description | Left/right coverage | Upper/lower coverage | Link |
|---|---|---|---|---|---|---|
| 1 | Chest X-ray, via Chest ImaGenome | Radiography, largest single modality in Medical-SAM3's training corpus | A scene-graph annotation layer on MIMIC-CXR, defining 29 anatomical structures per image | Left lung, right lung | Each lung subdivided into upper, mid, and lower lung zones, plus lung base and apical zone | [PhysioNet: chest-imagenome](https://physionet.org/content/chest-imagenome/1.0.0/) |
| 2 | Kidney ultrasound, via TRUSTED | Ultrasound (paired with CT) | Tridimensional Renal UltraSound TomodEnsitometrie Dataset: paired 3D transabdominal ultrasound and CT from 48 patients (96 kidneys), with 7 radiographer-annotated anatomical landmarks per kidney | Left kidney, right kidney | Upper pole, lower pole of each kidney | [Scientific Data paper](https://www.nature.com/articles/s41597-025-04467-1) · [arXiv:2310.12646](https://arxiv.org/abs/2310.12646) · [Figshare dataset](https://springernature.figshare.com/articles/dataset/TRUSTED_The_Paired_3D_Ultrasound_and_CT_Human_Data_for_Kidney_Segmentation_and_Registration_Research/27981050) |

**Table 2.** Positional-bias analyses for the 3D segmentation models, and their readiness status
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

Scoped to VoxTell only (Medical-SAM3 and Text3DSAM root-cause work, and report generation, are
out of scope for now). Why does VoxTell degrade at certain rotation angles and get left/right
wrong, and does it show a real positional bias rather than a resampling artifact or a decorative
modifier it never actually uses? Eight steps, in order, each building on the last: steps 1-6 test
the rotation-angle behavior, steps 7-8 test whether the model uses positional language at all and,
if not, where in the pipeline (text-image alignment vs. decoder/fusion) that signal is lost.

**Table 3.** Eight-step diagnostic plan for identifying why VoxTell shows positional bias.

| No. | Step | Question it answers | What to do | Status |
|---|---|---|---|---|
| 1 | **Rule out the resampling confound** | Is the Dice drop caused by rotation resampling/field-of-view loss, not by the model itself? | For each rotated volume, compute the fraction of ground-truth mask voxels that fall outside the resampled volume's valid region. Regress Dice-drop against this fraction. A strong correlation means part of the "bias" is a preprocessing artifact, not a model failure. | Ready now, from existing sweep data |
| 2 | **Test for a learned positional shortcut** | Does the model's error pull toward where the organ would sit at 0° (a canonical-pose prior), or does it track the rotated anatomy correctly? | Centroid-displacement-direction analysis: plot predicted centroid vs. ground-truth centroid across angles, check if the direction of drift is consistent (toward canonical pose) or random. | Ready now, from existing sweep data |
| 3 | **Test whether the bias is size or organ dependent** | Does bias hit small/peripheral organs harder than large/central ones (a shortcut explanation), or is it uniform (an architecture explanation)? | Stratify the Dice-vs-angle curves by organ size and compare degradation slopes. | Ready now, from existing sweep data |
| 4 | **Check for a threshold vs. smooth decay** | Does Dice degrade gradually (partial rotation-augmentation coverage during training), or fall off a cliff at one angle (a hard augmentation-range boundary)? | Re-run the sweep at finer angle steps (e.g., 10-15°) for at least a few organs, and look for a knee in the curve. | Pending, needs a new finer-grained sweep |
| 5 | **Confirm the cause in the architecture** | Does VoxTell's backbone or fusion/decoder have any component that would explain rotation or laterality sensitivity? | Check the backbone type (standard CNN vs. rotation-equivariant) and whether the fusion/decoder uses absolute positional embeddings. | **Done.** Backbone: plain ResEncL CNN, no rotation-equivariant components, so rotation robustness comes only from augmentation, not architecture. Decoder: no positional embeddings anywhere; position is only implicit in the CNN feature map. |
| 6 | **Confirm the cause in the training data pipeline** | Does VoxTell's training augmentation config explain the rotation-angle threshold and the left/right errors specifically? | Check the rotation-augmentation range and whether left-right mirroring is included. | **Done.** VoxTell's Appendix A.1 states "standard nnU-Net data augmentation strategies were applied," but does not itself give a numeric rotation range. nnU-Net's default 3D rotation augmentation (`3d_fullres`/`3d_lowres`) samples per-axis rotation angles of roughly ±30° (±0.52 rad); the 2D default is narrower, ±15° to ±30° (sources: [nnU-Net issue #38](https://github.com/MIC-DKFZ/nnUNet/issues/38), [nnU-Net issue #2400](https://github.com/MIC-DKFZ/nnUNet/issues/2400)), so the ~30°/axis figure here is the nnU-Net default, not a number confirmed directly in VoxTell's own appendix text, and should be treated as inferred until checked against VoxTell's actual training config/plans file. Left-right mirroring was deliberately excluded to protect laterality (this part *is* stated explicitly in the appendix). Since rotation augmentation stayed on, a large rotation near 180 degrees can visually mimic a mirror flip the model was never trained to tell apart, a likely cause of wrong-side errors at large angles. |
| 7 | **Test whether the model uses position at all, or just the organ name** | Does VoxTell actually ground "left kidney" using positional language, or does it segment "kidney" from the name alone and ignore "left"/"right" (i.e., position is decorative, not used)? | Prompt with position-only phrasing that drops the organ name (e.g., "the structure on the left side of the abdomen" instead of "left kidney"), on subjects where that's unambiguous. If output collapses (empty mask, or defaults to a fixed organ regardless of region), that shows the model isn't actually parsing spatial language, it's pattern-matching organ names and treating left/right as a near-ignored modifier. | Pending, needs new inference with name-free prompts |
| 8 | **Test whether the failure is in text-image alignment, not the decoder** | When the model gets left/right wrong, is it because the text encoder fails to distinguish "left X" from "right X" as different embeddings (an alignment failure), or because both prompts produce distinct embeddings but the image decoder ignores that signal (a decoder/fusion failure)? | Compare the text-embedding distance between "left X" and "right X" prompts (cosine distance in the text encoder's output space) against the mask-output difference (Dice or centroid distance between the two predicted masks) for the same subject. If the text embeddings are nearly identical, the alignment step itself doesn't distinguish laterality, pointing upstream (contrastive/text-image alignment) rather than downstream (decoder). If the text embeddings are clearly distinct but the masks come out nearly identical anyway, the fusion/decoder is dropping the positional signal despite having it available. | Pending, needs access to VoxTell's text encoder output and a small embedding-comparison script |

Source for steps 5-6: VoxTell CVPR 2026 paper, Appendix A, Implementation Details (arXiv:2511.11450v1,
Sections A.1 Training Configuration and A.2 Model Architecture).

### Possible causes of VoxTell's positional bias

Consolidated from Table 3 above: every failure cause identified so far, confirmed or hypothesized.

**Table 3a.** Why VoxTell fails, with reason and explanation.

| No. | Status | Data already collected? | Cause | Reason | Explanation | How to convert this hypothesis into real data |
|---|---|---|---|---|---|---|
| 1 | Confirmed | Yes, existing rotation sweep covers this, just needs a finer pass near 180° | **Mirror-flip confusion at large angles** | Left-right mirroring was excluded from training augmentation, but rotation augmentation was still applied. | A rotation near 180° visually resembles a mirror flip the model never learned to distinguish from a true rotation. This is a likely cause of wrong-side (left/right) errors at large rotation angles. | - Run the existing rotation sweep at fine angle steps (10 to 15°) around 150 to 180° specifically.<br>- Plot Wrong-Side Rate vs. angle.<br>- If wrong-side errors spike sharply near 180° rather than rising gradually with all angles, that is direct evidence for the mirror-confusion mechanism.<br>- Uses the sweep infrastructure you already have, just needs the fine-grained run near 180°. |
| 2 | Confirmed | Yes, this is a code/config fact, not sweep data | **No architectural safeguard for rotation** | The backbone is a plain ResEncL CNN with no rotation-equivariant components. | Rotation robustness comes only from data augmentation, not the architecture itself. If augmentation coverage runs out (e.g. beyond about 30° per axis), nothing else in the model protects against unseen rotations. | - Read VoxTell's released model code/config (backbone class definition) directly.<br>- Cross-check that no rotation-equivariant layer (e.g. e3nn, steerable CNN) appears anywhere in the network definition.<br>- No new sweep run needed, this is a static architecture check. |
| 3 | Confirmed | Yes, this is a code/config fact, not sweep data | **No positional embeddings in the decoder** | The fusion/decoder has no explicit absolute positional embeddings anywhere in the pipeline. | Position is only implicit in the CNN feature map, so any positional signal (e.g. "left" vs. "right") must survive purely through learned convolutional features rather than an explicit spatial encoding. | - Inspect the decoder/fusion module's forward pass in the released code.<br>- Look for any positional-encoding tensor (sinusoidal, learned, or coordinate-concat).<br>- No sweep data needed, this is a static code check. |
| 4 | Hypothesized, pending | Yes, reuses the existing rotation sweep data, just needs a new regression/plot | **Resampling/field-of-view artifact** | Rotating a volume can push ground-truth mask voxels outside the resampled volume's valid region. | Part of the apparent Dice drop at rotation may not be a true model failure at all, but a preprocessing artifact from resampling/field-of-view loss, not genuine positional bias. | - For every rotated volume in your existing sweep, compute the fraction of ground-truth mask voxels that fall outside the resampled volume's valid region.<br>- Regress Dice-drop against that fraction across angles and subjects.<br>- A strong positive correlation (report R², include a scatter plot) means part of the "bias" is this artifact, not the model.<br>- Uses data you already have (Table 3, step 1). |
| 5 | Hypothesized, pending | Yes, reuses existing sweep predictions, just needs centroid extraction | **Canonical-pose shortcut** | The model may have learned to expect organs in their "standard" unrotated position during training. | Predictions could drift back toward where the organ would sit at 0° rather than tracking the true rotated anatomy, a learned prior rather than genuine spatial reasoning. | - For each angle, compute predicted-centroid minus ground-truth-centroid as a vector.<br>- Check whether that vector consistently points back toward the 0°-pose location of the organ across subjects, rather than scattering randomly.<br>- Already-collected sweep predictions are enough (Table 3, step 2), just needs the centroid extraction plus a direction-consistency test. |
| 6 | Hypothesized, pending | Yes, reuses existing per-organ Dice-vs-angle CSVs | **Size/organ-dependent bias** | Bias may not be uniform across organs. | If small/peripheral organs degrade faster than large/central ones, that points to a shortcut-based failure; if degradation is uniform, that points to an architecture-wide explanation instead. | - Compute each organ's mean voxel volume from ground-truth masks.<br>- Compute each organ's Dice-degradation slope (Dice at 0° minus Dice at each angle, or a fitted slope).<br>- Plot degradation slope vs. organ volume across all organs and subjects, and check for a negative correlation.<br>- Reuses existing per-organ Dice-vs-angle CSVs (Table 3, step 3), no new inference needed. |
| 7 | Hypothesized, pending | No, needs a new finer-grained sweep plus the actual training config | **Hard augmentation-range cutoff** | Training rotation augmentation has a fixed range. | nnU-Net's default 3D rotation augmentation is plus or minus 30° per axis (2D default plus or minus 15° to 30°), per [nnU-Net issue #38](https://github.com/MIC-DKFZ/nnUNet/issues/38) and [issue #2400](https://github.com/MIC-DKFZ/nnUNet/issues/2400). This is the nnU-Net default, not a number VoxTell's own appendix states explicitly, so it needs confirming against VoxTell's actual config. If real, Dice may fall off a cliff right at that boundary rather than decaying smoothly, since training never exposed the model past that range. | - Find VoxTell's actual `nnUNetPlans`/trainer config (rotation range parameter), if released, to replace the assumed 30° default with a confirmed number.<br>- Re-run the rotation sweep at finer steps (10 to 15°) spanning that boundary.<br>- Look for a knee or cliff in the Dice-vs-angle curve right at the configured limit, vs. smooth decay elsewhere.<br>- This is Table 3, step 4, and needs a new finer-grained sweep you do not have yet. |
| 8 | Hypothesized, pending | No, needs new inference with different prompts | **Positional language is decorative, not used** | The model may be pattern-matching on the organ name alone. | If prompts naming only position (e.g. "the structure on the left side") without the organ name cause the output to collapse, that shows the model ignores "left"/"right" and just segments whatever organ name it recognizes. | - Run new inference with position-only prompts that drop the organ name (e.g. "the structure on the left side of the abdomen") on subjects where that phrasing is anatomically unambiguous.<br>- Compare the output to the normal "left X" prompt.<br>- An empty mask, wrong organ, or a fixed default regardless of stated side confirms the hypothesis.<br>- This is Table 3, step 7, currently pending, not yet run. |
| 9 | Hypothesized, pending | No, needs text-encoder access and a new script | **Text-image alignment failure vs. decoder/fusion failure** | Wrong-side errors could originate at two different points in the pipeline. | If "left X" and "right X" produce nearly identical text embeddings, the text encoder itself fails to distinguish laterality (an upstream alignment failure). If the embeddings are clearly distinct but the masks still come out nearly identical, the decoder/fusion step is dropping the positional signal despite having it available (a downstream failure). | - Extract the text embedding for "left X" and "right X" prompts, paired per organ.<br>- Compute cosine distance between them.<br>- Separately compute Dice or centroid distance between the two predicted masks for the same subject/organ pair.<br>- Plot embedding-distance vs. mask-distance across organs: near-zero embedding distance means an alignment failure, large embedding distance but near-zero mask distance means a decoder/fusion failure.<br>- Needs a small new script plus text-encoder access (Table 3, step 8, currently pending). |

### Aggregate views

**Table 4.** Aggregate visualizations already produced, or pending, per 3D model.

| No. | Aggregate view | VoxTell | Medical-SAM3 | Text3DSAM |
|---|---|---|---|---|
| 1 | Mean Dice by organ × rotation angle (heatmap) | Ready now | Pending | Ready now |

### Visual grounding with relational language (bilateral left/right only)

**Segmentation vs. grounding, the difference this test is built around:**
- Segmentation quality asks: *"is the predicted mask shaped correctly?"* → scored with Dice
  against the ground-truth mask, regardless of what the prompt said.
- Visual grounding asks a different question: *"did the mask land where the words actually
  pointed?"* → a model can have great Dice on "kidney" in general and still put the mask on the
  wrong side when the word "left" or "right" is the only thing that changes.
- A model can nail one and fail the other. Good shape, wrong side = a grounding failure hiding
  behind an otherwise-healthy Dice score.

**The core setup, in one line:**
- We send the exact same image to the model twice.
- Each time, we send a different prompt, and the only thing that changes between the two prompts
  is the position word:
  - `"Segment the left kidney in this image."`
  - `"Segment the right kidney in this image."`
- Why: since the image never changes, the only thing that can explain a difference in where the
  two masks land is whether the model actually read "left" vs. "right." This isolates the
  model's use of positional language from everything else (image content, mask quality, dataset).

**Why "left/right" counts as relational language:**
- None of the three models were trained on longer relational phrases like "the organ to the left
  of the spine," only on plain anatomy names like "kidney."
- But "left"/"right" only mean anything relative to the body's own midline, so `"left kidney"` is
  itself a minimal relational phrase, even though it looks like just a name.
- This is the one relational test these models can be fairly given today.

**The 5-step recipe (worked example: VoxTell, AMOS22-MRI, kidney):**
1. Take one subject's MRI volume.
2. Prompt with `"Segment the left kidney in this image."` → get a predicted mask.
3. Prompt with `"Segment the right kidney in this image."` on the *same* volume → get a second
   predicted mask.
4. For each mask, check which side of the body's midline it actually sits on, and compare it to
   the *true* side from the ground-truth annotation (not from the prompt).
5. Pass = "left" prompt lands on the true left kidney, and "right" prompt lands on the true right
   kidney. Fail = one or both land on the opposite side (e.g., always the anatomically-right
   kidney, no matter which word it was given).

**Scope of this section:**
- Applies to the 3D segmentation models only: VoxTell, Medical-SAM3, Text3DSAM.
- Table 5 below repeats this exact recipe once per dataset, just swapping in the model, structure,
  and prompt wording.
- For the two Medical-SAM3 in-domain datasets, the identical recipe is also run with `"upper X"` /
  `"lower X"` instead of `"left X"` / `"right X"` (lung zones for Chest ImaGenome, kidney pole for
  TRUSTED), not a separate table row, just the same row's structure with the direction words
  swapped.

**Table 5.** One worked example per dataset, following the 5-step recipe above. Rows 1-7 are the
left/right pairs; rows 8-9 repeat the identical recipe with upper/lower instead, for the two
Medical-SAM3 in-domain datasets.

| No. | Model | Dataset | Structure we prompt | First we ask (full prompt) | Then we ask (full prompt) | What makes this one worth checking |
|---|---|---|---|---|---|---|
| 1 | VoxTell | AMOS22-MRI | Kidney | `"Segment the left kidney in this image."` | `"Segment the right kidney in this image."` | The main baseline case for VoxTell |
| 2 | VoxTell | Neurite-OASIS | Hippocampus | `"Segment the left hippocampus in this image."` | `"Segment the right hippocampus in this image."` | Same check, but on brain MRI instead of abdominal MRI |
| 3 | Text3DSAM | CT_HaN-Seg | Parotid gland | `"Segment the left parotid gland in this image."` | `"Segment the right parotid gland in this image."` | This dataset is in-domain for Text3DSAM, so a wrong answer here can't be blamed on an unfamiliar dataset |
| 4 | Text3DSAM | CT_TotalSeg_cardiac | Common carotid artery | `"Segment the left common carotid artery in this image."` | `"Segment the right common carotid artery in this image."` | Same in-domain check as row 3, on a blood vessel instead of a gland |
| 5 | Text3DSAM | MR_TotalSeg | Gluteus maximus | `"Segment the left gluteus maximus in this image."` | `"Segment the right gluteus maximus in this image."` | A large, easy-to-see structure, a sanity check before trusting results on tiny structures |
| 6 | Medical-SAM3 | Chest X-ray, via Chest ImaGenome | Lung | `"Segment the left lung in this image."` | `"Segment the right lung in this image."` | Chest X-ray is Medical-SAM3's largest training modality, so this is its in-domain case |
| 7 | Medical-SAM3 | Kidney ultrasound, via TRUSTED | Kidney | `"Segment the left kidney in this image."` | `"Segment the right kidney in this image."` | A second, different in-domain modality (ultrasound) for the same model |
| 8 | Medical-SAM3 | Chest X-ray, via Chest ImaGenome | Lung zone | `"Segment the upper lung zone in this image."` | `"Segment the lower lung zone in this image."` | Same dataset as row 6, upper/lower instead of left/right |
| 9 | Medical-SAM3 | Kidney ultrasound, via TRUSTED | Kidney pole | `"Segment the upper kidney pole in this image."` | `"Segment the lower kidney pole in this image."` | Same dataset as row 7, upper/lower instead of left/right |

*Prompt template source: the exact wording VoxTell is prompted with in this repo's own dataset manifests (e.g. `faria/samples/amos22_mri/manifest_amos22_mri.csv`, `prompt_text` column: `"Segment the right kidney in this image."`). The same template is applied here to Text3DSAM and Medical-SAM3 for consistency across the three models.

For every row, the model's answer is checked the same way: does the predicted mask's location
actually match the true left/right side of that structure, as recorded in that dataset's own
ground-truth annotations, not just whatever side the prompt happened to name.

### How to analyze it, metrics

Reuses the metrics already defined earlier in this section, applied specifically to this
left/right-only prompt pair, per structure, per subject.

**Table 6.** Metrics for scoring the 3D bilateral left/right grounding experiment.

| No. | Metric | How it's used for this experiment |
|---|---|---|
| 1 | **Wrong-Side Rate** | For each (subject, structure) pair, send both the "left X" and "right X" prompts, take the predicted mask's centroid side, and compare to the prompted side. Count how often the predicted side is the *opposite* of what was asked for. This is the core pass/fail signal for relational grounding. |
| 2 | **Laterality Dice Gap** | Compare mean Dice when the model is prompted with the correct-matching side (e.g., "left kidney" scored against the true left kidney mask) between the left and right prompts. A large gap means the model segments one side much better than the other, even when it gets the side right. |
| 3 | **Centroid-Displacement Direction** | When the model gets the side wrong, check whether the predicted centroid still lands *near* the correct-side structure's true location, or somewhere else entirely. Distinguishes "swapped the label" from "genuinely lost track of the anatomy." |
| 4 | **Failure-Mode Classification** | For wrong-side cases, classify whether the predicted mask is a real structure on the wrong side (a true label swap) or a missing/empty prediction (the model didn't find the relational cue at all). |

**Explain**

- **Wrong-Side Rate**
  - Send both prompts ("left X" and "right X") on the same subject, take each predicted mask's
    centroid, and note which side of the body it falls on.
  - Compare that side to the side named in the prompt: if they don't match, count it as a
    wrong-side error. This is the main pass/fail number for the whole experiment.

- **Laterality Dice Gap**
  - For the cases where the model gets the side right, compute Dice separately for the left
    prompt's mask and the right prompt's mask, each against its matching ground-truth side.
  - Subtract the two Dice scores: a large gap means the model is much worse at segmenting one
    side than the other, even though it isn't confusing left and right.

- **Centroid-Displacement Direction**
  - Only looked at when the side is wrong: take the predicted centroid and measure how far it is
    from the true location of the correct-side structure.
  - If it's close, the model just mislabeled the side (mask is basically in the right place); if
    it's far, the model lost track of the anatomy entirely, not just the label.

- **Failure-Mode Classification**
  - For each wrong-side case, check whether the model still produced a real, structure-shaped
    mask, just on the wrong side, or produced an empty/near-empty prediction.
  - This separates "the model swapped left and right" from "the model didn't understand the
    prompt at all," which need different fixes.

---

## 2. Localization

2D localization models: bbox / point output.

### Dataset coverage

All 8 models (Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B,
Molmo2-8B, North-Micro-Vision-Instruct) ran on all 3 datasets:

- **LGG** (brain)
- **Kvasir** (colon)
- **ISIC** (skin)

**Florence-2** is added as a 9th localization model, not yet run. It needs the same three-dataset
sweep as the existing 8 before any row in Table 7 can be filled in for it.

**Metrics for the benchmark**, from the results already scored across 3 datasets (LGG, Kvasir,
ISIC) for the existing 8 models. "Ready now" means computable today from the existing result
CSVs, no new inference needed. All rows below are pending for Florence-2 specifically, since it
has not been run yet; once it has, it slots into the "Models" column alongside the other 8.

**Table 7.** Positional-bias metrics for the 2D localization models, and which are already
computed vs. ready to compute.

| No. | Metric | Definition | Models | Status |
|---|---|---|---|---|
| 1 | Detect Rate | Share of P0–P4 calls that returned any usable prediction | All 8 (Grounding DINO, OWLv2, Kosmos-2, Qwen2.5-VL-3B, InternVL2.5, LocateAnything-3B, Molmo2-8B, North-Micro-Vision-Instruct); pending for Florence-2 | Already computed for the 8; pending for Florence-2 |
| 2 | Wrong-Side Rate (per direction) | Share of prompted answers landing on the side opposite the one named in the prompt, broken down left/right/upper/lower | All 8; pending for Florence-2 | Already computed (pooled) for the 8; per-direction ready now; pending for Florence-2 |
| 3 | No-Shift Rate | Share of P1–P4 answers that stayed near the model's own P0 answer; didn't move | All 8; pending for Florence-2 | Already computed for the 8; pending for Florence-2 |
| 4 | Mean Normalized Distance | Distance from predicted point to ground-truth centroid, normalized by image diagonal | All 8; pending for Florence-2 | Already computed for the 8; pending for Florence-2 |
| 5 | Mean IoU | Overlap between predicted box and ground-truth box | 7 of 8, all except Molmo2-8B (point-only, no box to compute overlap from); pending for Florence-2 | Already computed for the 7; pending for Florence-2 |
| 6 | Center-Bias Rate | Share of P0 answers landing near image center regardless of true target location | All 8; pending for Florence-2 | Ready now for the 8; pending for Florence-2 |
| 7 | Wrong-Side Rate vs. Target Eccentricity | Wrong-Side Rate stratified by how far the true target sits from image center | All 8; pending for Florence-2 | Ready now for the 8; pending for Florence-2 |
| 8 | Cross-Dataset Consistency | Whether Wrong-Side Rate holds steady across LGG, Kvasir, and ISIC | All 8; pending for Florence-2 | Ready now for the 8; pending for Florence-2 |

---

## 3. VQA

Open-ended visual question answering, for the localization models that can generate free text.

### Model eligibility for VQA and MCQ experiments

Not all 9 localization models can generate free text, so not all of them can be used for a
spatial-VQA or answer-position-bias experiment. Split by actual capability, not by architecture
family alone. This eligibility applies to both the VQA experiments here and the MCQ experiments
in Section 4.

**Table 8.** Which of the 9 localization models can be used for VQA and Multiple-Choice
experiments.

| No. | Model | Usable for VQA | Usable for Multiple-Choice (answer-position bias) | Why |
|---|---|---|---|---|
| 1 | Qwen2.5-VL-3B | Yes | Yes | Full instruction-tuned VLM, handles open QA and MCQ natively |
| 2 | InternVL2.5 | Yes | Yes | Full instruction-tuned VLM, handles open QA and MCQ natively |
| 3 | Molmo2-8B | Yes | Yes | Chat-capable VLM, pointing is one capability among several, not its only mode |
| 4 | North-Micro-Vision-Instruct | Yes | Yes | Explicitly built for general VQA/OCR/grounding, likely the best fit of the 9 |

**Usable**: Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct (4 models).
**Excluded**: Grounding DINO and OWLv2, since both are pure detectors that only output boxes and
confidence scores, with no free-text generation at all. Kosmos-2, LocateAnything-3B, and
Florence-2 are also excluded: all three can generate text, but none has a confirmed general
instruction-following/MCQ capability. Kosmos-2's released focus was grounding/captioning, not
instruction-following QA. LocateAnything-3B is built around structured coordinate-output task
templates (referring expressions, pointing) with no confirmed open-ended QA behavior. Florence-2
is a task-token-driven model (separate tokens for captioning, open-vocabulary detection, phrase
grounding, OCR, and similar tasks), not a chat-style model that takes arbitrary natural-language
questions, so it has no confirmed free-form VQA or MCQ behavior either. All three remain usable
for the Section 2 localization/grounding experiments, where their capability is confirmed.

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

### External multimodal VQA datasets (left/right + upper/lower)

Alongside the sliced-3D approach above, these existing Hugging Face multimodal VQA datasets were
checked directly (by sampling their actual `question`/`answer` text) and confirmed to contain
**both** left/right and upper/lower positional language already, without needing any slicing or
relabeling. Useful as ready-made, larger-scale supplementary data for the VQA experiment.

**Table 9.** External multimodal VQA datasets confirmed to cover both left/right and upper/lower
positional language.

| No. | Dataset | HF Link | Modality | Size | Left/Right | Upper/Lower |
|---|---|---|---|---|---|---|
| 1 | SLAKE | [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) | MRI, CT | 14,028 QA pairs (9,840 train / 2,100 val / 2,090 test) | Yes | Yes |
| 2 | Medical-CXR-VQA | [X-iZhang/Medical-CXR-VQA](https://huggingface.co/datasets/X-iZhang/Medical-CXR-VQA) | Chest X-ray | ~78K QA pairs | Yes | Yes |
| 3 | M3D-VQA | [GoodBaiBai88/M3D-VQA](https://huggingface.co/datasets/GoodBaiBai88/M3D-VQA) | 3D medical volumes (CT) | Large-scale (part of M3D suite) | Yes | Yes |
| 4 | MedPix-VQA | [mmoukouba/MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) | Mixed (MedPix teaching cases) | Varies | Yes | Yes |
| 5 | Kvasir-VQA | [SimulaMet-HOST/Kvasir-VQA](https://huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA) | Endoscopy (GI) | 58,849 QA pairs, 6,500 images | Yes | Yes |
| 6 | GEMeX-VQA (open-ended) | [BoKelvin/GEMeX-VQA](https://huggingface.co/datasets/BoKelvin/GEMeX-VQA), `open_ended_question.jsonl` | Chest X-ray | Same corpus as the MCQ subset in Table 12 | Yes* | Yes* |

*GEMeX-VQA (open-ended) positional coverage was confirmed on the MCQ subset of this same dataset
(Section 4, Table 12), not independently re-sampled for the open-ended split.

M3D-VQA is the only dataset here that's natively 3D/volumetric, making it the closest match in
dimensionality to the Section 1 segmentation datasets. Of the others, note that SLAKE has a
dedicated `location` field and Kvasir-VQA uses a structured 9-way grid answer
(upper/center/lower x left/center/right), both give cleaner, lower-noise positional labels than
free-text mentions, even where their raw match rate is lower.

### VQA prompts

Same order-counterbalancing concern as the MCQ experiment: phrasing a question as "...the left X
or the right X?" always mentions "left" first, so a model with any order-of-mention bias (a
tendency to pick whichever option was named first) would look identical to a laterality bias.
Each question is run twice per image, once phrased left-then-right and once phrased
right-then-left, to separate the two.

**Table 10.** Example VQA prompts for the bilateral left/right experiment, by dataset, with the
mention order swapped between runs.

| No. | Dataset | Bilateral structure | Example VQA prompt (left-then-right) | Example VQA prompt (right-then-left) | Models |
|---|---|---|---|---|---|
| 1 | CT_HaN-Seg (2D slice) | Left parotid gland, right parotid gland | "Is the gland marked in this image the left parotid gland or the right parotid gland?" | "Is the gland marked in this image the right parotid gland or the left parotid gland?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Left common carotid artery, right common carotid artery | "Is the artery marked in this image the left common carotid artery or the right common carotid artery?" | "Is the artery marked in this image the right common carotid artery or the left common carotid artery?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | MR_TotalSeg (2D slice) | Left kidney, right kidney | "Is the kidney marked in this image the left kidney or the right kidney?" | "Is the kidney marked in this image the right kidney or the left kidney?" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

### How to analyze it, metrics

Reuses the same metrics style as the Section 1 bilateral-grounding experiment, adapted to a
text answer instead of a mask. Rows tagged "VQA" apply here; rows tagged "MCQ" apply to the
Section 4 experiment instead.

**Table 11.** Metrics for scoring the 2D VQA and Multiple-Choice bilateral experiments.

| No. | Metric | Applies to | How it's used |
|---|---|---|---|
| 1 | Correct-Side Rate | VQA, MCQ | Share of answers where the stated side (left/right) matches the ground-truth side of the structure actually shown. The core accuracy signal. |
| 2 | Wrong-Side Rate | VQA, MCQ | Share of answers where the stated side is the *opposite* of the ground truth, same concept as the Wrong-Side Rate used elsewhere in this roster, just scored from text instead of a mask centroid. |
| 3 | No-Answer / Refusal Rate | VQA | Share of responses that don't commit to either side (hedging, "unclear," or an unrelated answer). A separate failure mode from getting it wrong. |
| 4 | Per-Structure Accuracy | VQA, MCQ | Correct-Side Rate broken down by structure (parotid gland vs. carotid artery vs. kidney), to see which anatomy the model handles best or worst. |
| 5 | Answer-Position Bias (Letter Preference) | MCQ only | Compare accuracy when the correct answer is option A vs. when it's option B. If accuracy drops (or the model just picks "A" more often) regardless of which side is actually correct, that's a letter-position bias, separate from genuine anatomical understanding. |
| 6 | Content vs. Letter Bias Disentanglement | MCQ only | Cross-tabulate true side (left/right) against chosen letter (A/B) against correctness. Separates "the model is biased toward right-side anatomy" from "the model is biased toward option A," which look identical in raw accuracy but are different problems. |
| 7 | Order-of-Mention Bias | VQA only | Compare accuracy, and raw pick rate, between the left-then-right and right-then-left phrasings of the same question (Table 10). If the model just tends to pick whichever side was named first, that's an order-of-mention bias, separate from genuine laterality understanding. |

---

## 4. MCQ

Multiple-choice framing of the same bilateral left/right questions, run for the same 4 usable
VLMs (Table 8), added to test answer-position bias.

### Multiple-Choice prompts (answer-position bias)

Same question as the VQA prompts in Section 3, phrased as MCQ. Run each pair twice per image,
once with the correct answer as option A and once as option B, to separate genuine
bilateral-grounding accuracy from a simple preference for one letter position.

### External multimodal MCQ datasets (left/right + upper/lower)

Same idea as the Section 3 external VQA table (Table 9), but for existing multimodal MCQ
datasets: checked directly against their actual `question`/`answer`/`choices` text and confirmed
to contain both left/right and upper/lower positional language already.

**Table 12.** External multimodal MCQ datasets confirmed to cover both left/right and upper/lower
positional language.

| No. | Dataset | HF Link | Modality | Size | Left/Right | Upper/Lower |
|---|---|---|---|---|---|---|
| 1 | GEMeX-VQA (MCQ subset) | [BoKelvin/GEMeX-VQA](https://huggingface.co/datasets/BoKelvin/GEMeX-VQA), `multi_choice_question.jsonl` | Chest X-ray | Subset of GEMeX-VQA | Yes | Yes |
| 2 | MedXpertQA (MM subset) | [TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) (config `MM`) | Radiology images + text | 4,460 questions total, 17 specialties, 11 body systems | Yes | Yes |
| 3 | PMC-VQA | [RadGenome/PMC-VQA](https://huggingface.co/datasets/RadGenome/PMC-VQA) (also mirrored at [xmcmic/PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA)) | Mixed medical figures (from PMC Open Access papers) | 227K QA pairs, 149K images | Yes | Yes |

GEMeX-VQA (MCQ subset) has the strongest signal of the three: position is a first-class,
structured MCQ answer choice (e.g. "Right lung" / "Left lung base" / "Cardiac region"), not just
incidental phrasing, making it the best candidate for a controlled answer-position-bias MCQ run
alongside the sliced-3D prompts below.

**Table 13.** Example Multiple-Choice prompts for the bilateral left/right experiment, by
dataset.

| No. | Dataset | Bilateral structure | Example MCQ prompt | Models |
|---|---|---|---|---|
| 1 | CT_HaN-Seg (2D slice) | Left parotid gland, right parotid gland | "Which gland is marked in this image? A) Left parotid gland B) Right parotid gland" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Left common carotid artery, right common carotid artery | "Which artery is marked in this image? A) Left common carotid artery B) Right common carotid artery" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | MR_TotalSeg (2D slice) | Left kidney, right kidney | "Which kidney is marked in this image? A) Left kidney B) Right kidney" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

### How to analyze it, metrics

Uses Table 11 in Section 3, specifically the rows tagged "MCQ" or "VQA, MCQ": Correct-Side Rate,
Wrong-Side Rate, Per-Structure Accuracy, Answer-Position Bias, and Content vs. Letter Bias
Disentanglement.

### Distance / depth order-bias experiment

Same idea as the laterality MCQ above, applied to the Distance and Depth reasoning types (Table
15) instead of left/right. 3D segmentation models and the two pure detectors can't be probed
this way, since they only output a mask/box, never a distance or depth judgment in text; this
experiment is limited to the 4 usable 2D VLMs from Table 8 (Qwen2.5-VL-3B, InternVL2.5,
Molmo2-8B, North-Micro-Vision-Instruct).

Uses the same 2D slices as the sections above (CT_HaN-Seg, CT_TotalSeg_cardiac, MR_TotalSeg),
each showing two distinguishable structures whose relative distance or depth is already known
from the ground-truth 3D coordinates (e.g., which of two organs' centroids is closer to a marked
reference point, or which sits in a more anterior slice). Each question is run twice per image,
once with the correct answer as option A and once as option B, exactly like the laterality MCQ.

**Table 14.** Example Distance/Depth MCQ prompts, order-swapped to test position bias.

| No. | Dataset | Structure pair | Reasoning type | Example MCQ prompt | Models |
|---|---|---|---|---|---|
| 1 | MR_TotalSeg (2D slice) | Kidney vs. liver | Distance | "Which structure's center is closer to the marked point? A) Kidney B) Liver" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 2 | CT_TotalSeg_cardiac (2D slice) | Common carotid artery vs. subclavian artery | Distance | "Which vessel is closer to the marked point? A) Common carotid artery B) Subclavian artery" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |
| 3 | CT_HaN-Seg (2D slice) | Parotid gland vs. submandibular gland | Depth | "Which gland is more anterior, closer to the front of the face? A) Parotid gland B) Submandibular gland" | Qwen2.5-VL-3B, InternVL2.5, Molmo2-8B, North-Micro-Vision-Instruct |

**Table 15.** Metrics for scoring the Distance/Depth order-bias experiment.

| No. | Metric | How it's used |
|---|---|---|
| 1 | Correct-Answer Rate | Share of answers matching the ground-truth distance/depth relationship, computed from the two structures' 3D centroids. |
| 2 | Answer-Position Bias (Letter Preference) | Compare accuracy, and raw pick rate, when the correct answer is option A vs. option B, same as the laterality MCQ bias test. |
| 3 | Content vs. Letter Bias Disentanglement | Cross-tabulate the true closer/more-anterior structure against the chosen letter against correctness, to separate a genuine bias toward one structure from a bias toward one letter. |

---

## 5. Mapping Spatial Reasoning to Positional Bias

Which of the important spatial-reasoning skills each model, across all four sections above, can
actually be tested on.

**Table 16.** Spatial reasoning types testable per model.

| No. | Model | Relative position (left/right/above/below another *object*, e.g. "tumor above the kidney") | Absolute position (top/bottom/center of the *image/volume itself*, no second object needed) | Distance (near/far) | Depth (in front/behind) | Topological (inside/outside/overlap) |
|---|---|---|---|---|---|---|
| 1 | VoxTell (3D seg) | Unconfirmed, only shown with the organ name included in the prompt, not yet tested with position-only phrasing | Unconfirmed, only shown with the organ name included in the prompt, not yet tested with position-only phrasing | No | No | Yes |
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

**Table 17.** Spatial reasoning types, definitions, and metrics.

| No. | Reasoning type | What it means | Metric | How it's evaluated / what it measures |
|---|---|---|---|---|
| 1 | Relative position | The location of one object described with respect to another object, e.g. "left of", "above", "between" | Wrong-Side Rate | Prompt with the relation (e.g. "left kidney"), get the model's box/mask/point, compare its position to the reference object using the ground-truth layout. Share of answers where the predicted side is the opposite of the ground truth |
| 2 | Relative position | (same as above) | Laterality Dice Gap | Difference in mean Dice/accuracy between the two sides of a relation, when the model does get the side right |
| 3 | Relative position | (same as above) | Name-Free Grounding Rate | Prompt with position-only phrasing that drops the organ name (e.g. "the structure on the left side of the abdomen"). Share of cases where the model still lands on the correct-side structure without being told the organ name, distinguishing real spatial-language use from organ-name pattern-matching where "left/right" is decorative |
| 4 | Relative position | (same as above) | Text-Image Alignment Gap | Cosine distance between the text-encoder embeddings of "left X" vs. "right X", compared against the mask-output difference (Dice/centroid distance) between the two predicted masks for the same subject. Localizes a laterality failure to the text-image alignment step (embeddings barely differ) vs. the decoder/fusion step (embeddings differ but the output doesn't) |
| 5 | Absolute position | The location of an object described with respect to the whole image/volume, e.g. "top", "bottom", "center" | Region-Match Accuracy | Prompt with the absolute term, check if the returned box/mask/point centroid actually falls in that region of the image/volume (e.g. top third, center zone) |
| 6 | Absolute position | (same as above) | Center-Bias Rate | Share of predictions landing near the image/volume center regardless of the true target location |
| 7 | Absolute position | (same as above) | Name-Free Grounding Rate | Same idea as the relative-position version, applied to absolute-position phrasing (e.g. "the structure near the top of the volume" with no organ name). Tests whether "top/bottom" grounding also survives without the organ-name crutch |
| 8 | Distance | A qualitative near/far judgment between two objects | Correct-Answer Rate | For VLMs: ask "which is closer/farther" or "is X near Y", compare to ground-truth centroid distance. For segmentation/detection models: not directly askable, compute centroid-to-centroid distance from the two output masks/boxes instead |
| 9 | Distance | (same as above) | Distance Correlation | Correlation between the model's predicted distance ordering and the true ground-truth distance, for models that output a number |
| 10 | Depth | Front/behind or closer/farther in the camera or scan viewing axis | Correct-Answer Rate | For VLMs: ask "is X in front of Y" (2D images) or "which is closer to the viewer". For 3D volumes: reframe as slice-position (which axial/coronal/sagittal slice) since there's no camera depth |
| 11 | Topological relationship | How two regions relate structurally, e.g. inside, outside, overlapping, touching, containing | IoU / Containment Ratio | Prompt with the relation (e.g. "tumor inside kidney"), check output mask/box overlap or containment against the ground-truth masks |
| 12 | Topological relationship | (same as above) | Binary Correct/Incorrect Rate | Share of topological relation claims (e.g. "tumor inside kidney") that match ground truth, for models that only state the relation in text |

---
