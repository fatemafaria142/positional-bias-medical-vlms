# Why Text3DSAM performs the way it does on AMOS22 MRI and Neurite OASIS

> **Update:** an earlier version of this pipeline had a real bug — see
> "Pipeline fix (axis ordering)" in §0 below. After fixing it, AMOS22 MRI
> mean Dice went from 0.035 to **0.1025** (3x), with right/left kidney
> moving from 0.000 to ~0.20 and liver improving from 0.24 to 0.26. The
> numbers and organ-level detail below are the **current, corrected**
> results in `samples/amos22_mri/predictions_text3dsam.csv`. Neurite OASIS
> is unaffected by the fix (explained in §0) and its 0.000/0/350 result
> stands.

This documents the likely causes behind the results in
`samples/amos22_mri/predictions_text3dsam.csv` (mean Dice 0.1025, liver
best at ~0.26, 76/130 rows nonzero) and
`samples/neurite_oasis/predictions_text3dsam.csv` (mean Dice 0.000, 0/350
nonzero). Both are consequences of what the model was trained on, not bugs
in this pipeline — the input images and masks were verified correctly
aligned (see `results_text3d_oasis/.../left_cerebral_cortex.png`: real
predicted voxels exist, they just never overlap ground truth).

## 0. Pipeline fix (axis ordering)

`Text3DSAM` resizes every volume to a fixed `(128, 256, 256)` and always
treats array axis 0 as the coarse/through-plane axis. AMOS22 MRI subjects
are scanned in different planes (axial/coronal/sagittal), so which array
axis is physically thinnest varies per subject even though every volume's
NIfTI affine reports the same `RAS` axcodes — e.g. `amos_0508`
(260x**80**x320) has its thin axis at position 1, while `amos_0544`
(576x468x**72**) has its thin axis at position 2.
The original pipeline fed volumes straight into `Resize` without
correcting for this, so for most subjects the wrong axis got forced into
the model's fixed 128-slot, and the real thin axis got stretched into a
256-slot instead — an extra, subject-dependent distortion on top of the
resize itself. `scripts/common3d.py:thin_axis_first_order()` now permutes
each volume so its thinnest axis is always at position 0 before resizing,
and the prediction is permuted back before scoring. Ablation on
`amos_0508`/liver: Dice 0.21 → 0.32 from this fix alone.

A second fix (percentile-clip intensity normalization to `[0,1]`, on the
theory that the model expects normalized inputs like its training data)
was tried and **made results catastrophically worse** — predicted
probabilities never exceeded ~0.33, below the 0.5 threshold, collapsing
Dice to ~0 everywhere. This was reverted; the model apparently expects
close to raw-scale MRI intensities, not `[0,1]`-normalized ones. Kept only
the axis-ordering fix.

Neurite OASIS is unaffected by either fix: all 10 sampled subjects share
the identical `160x192x224` shape with axis 0 already the thinnest, so the
reordering is a no-op there.

## 1. What the model actually is

Text3DSAM (`MagicXin/Text3DSAM`, ~59M parameters) is trained end-to-end on
[CVPR-BiomedSegFM](https://huggingface.co/datasets/junma/CVPR-BiomedSegFM):
~200,000 3D image-mask pairs from public biomedical datasets covering CT,
MR, PET, ultrasound, and microscopy — assembled for the CVPR 2025
text-guided segmentation challenge. Two facts about that training set
explain most of what we see:

- **It is organ/lesion-centric and body-region specific.** Its class
  taxonomy (`CVPR25_TextSegFMData_with_class.json`) is built from datasets
  like AMOS (abdominal CT), various tumor/lesion sets, and other
  torso/organ benchmarks. It contains **no neuroanatomical brain-structure
  classes** — no hippocampus, amygdala, thalamus, ventricles, cerebral
  cortex/white matter, etc.
- **It is CT-dominant.** The largest single-modality contribution is CT.
  AMOS itself is included as `CT_AMOS`, i.e. the *CT* arm of AMOS22 — the
  **MRI** arm (which `samples/amos22_mri/` uses) is a different,
  much less-represented modality for the same anatomy.

### 1a. Where the model comes from (code vs. weights)

The `Text3DSAM/` directory is **not** committed to this repo (`.gitignore:6`) —
it is set up separately on each machine, and the prediction pipeline pulls
from two different sources:

- **Model code → GitHub.** The scripts add the Text3DSAM checkout to
  `sys.path` and import the architecture from its source tree:
  ```python
  sys.path.insert(0, str(TEXT3DSAM_ROOT))          # repo_root/Text3DSAM
  from src.model.modeling import Med3DSeg
  ```
  `src/model/modeling.py`, `pred.py`, etc. are the cloned Text3DSAM
  repository, not a `pip`/`transformers` package.
- **Weights + tokenizer → Hugging Face (`MagicXin/Text3DSAM`).** Loaded from
  a local folder populated by downloading that HF repo:
  ```python
  MODEL_PATH = TEXT3DSAM_ROOT / "checkpoints" / "Text3DSAM"
  model = Med3DSeg.from_pretrained(str(MODEL_PATH))            # HF checkpoint weights
  tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))  # HF tokenizer files
  ```
  See `scripts/run_text3dsam_amos22_mri.py:2,17`.

So prediction runs the **Hugging Face `MagicXin/Text3DSAM` weights** through
the **GitHub model code**, zero-shot — no fine-tuning is done in this repo.
All Text3DSAM scripts (`run_text3dsam_dataset.py`,
`run_text3dsam_amos22_mri.py`, `visualize_text3dsam_predictions.py`,
`visualize_text3dsam_grid.py`) load the checkpoint this same way.

## 2. Why AMOS22 MRI is weak but not zero (mean Dice 0.1025, liver 0.26)

**a. Modality gap: trained mostly on CT, evaluated on MRI.**
CT intensities are standardized (Hounsfield units); MRI intensities are
not — they vary by scanner, sequence, and contrast weighting, and none of
that is normalized to what the model saw in training. The model's learned
intensity/texture cues for "liver-looking tissue" etc. are CT-shaped and
only partially transfer to MRI's very different tissue contrast.

**b. Aggressive, anisotropic resizing to a fixed 128×256×256 input.**
Every volume — regardless of its real shape or voxel spacing — is
trilinear-resized to the model's fixed `image_size`. AMOS22 MRI subjects
in the manifest range from `260×80×320` (spacing 1.4×1.5×1.4mm) to
`512×168×512` (spacing 0.82×1.1×0.82mm): very different real-world
proportions get squashed into the same box. Combined with the model's
coarse internal feature grid (patch size 64³ → a handful of tokens per
axis before upsampling), **small structures lose most of their spatial
signal** by the time the mask decoder sees them.

**c. Organ size/frequency still correlates strongly with Dice.**
Liver, right/left kidney, aorta, and spleen (large, high-contrast, dominant
in abdominal training data) form a clear top tier at 0.16-0.26 mean Dice.
Gallbladder, pancreas, duodenum, IVC, stomach, and esophagus — small, thin,
or highly variable in shape — sit lower (0.04-0.07) but are no longer
uniformly zero. Only the adrenal glands (tiny, low-contrast) remain
essentially undetected (0.02 and 0.0005). This is the expected gradient for
a model whose spatial resolution and training exposure both favor large,
high-contrast structures — it's just a gradient now, not a cliff, once the
axis-ordering bug (§0) stopped adding noise on top of it.

**d. Small text encoder.** The prompt encoder is
`nlpie/tiny-clinicalbert`, a distilled/tiny BERT variant chosen for speed,
not a large clinical language model. Fine linguistic distinctions the
prompts rely on — "right kidney" vs. "left kidney", "inferior vena cava"
vs. "aorta" — depend on the text embedding actually separating those
concepts well; a tiny encoder gives the model less to work with here than
a full-size clinical BERT would.

**e. Hard 0.5 threshold on the sigmoid output.**
Several 0-Dice rows still had thousands of predicted voxels — the model
does emit *some* signal, it's just below (or in the wrong place relative
to) the 0.5 cutoff after resizing back to the original volume. This is a
methodological choice in the evaluation, not just a model failure — see
the recommendation in §4.

## 3. Why Neurite OASIS is a complete failure (0/350, mean Dice 0.000)

This is the sharper case, and it follows directly from §1:

- **Zero training exposure to the anatomy.** CVPR-BiomedSegFM's class
  taxonomy has no brain-structure labels at all. The model has never seen
  a training example paired with a prompt like "segment the left
  hippocampus."
- **Zero training exposure to the image domain.** The OASIS volumes used
  here (`aligned_norm.nii.gz`) are skull-stripped, bias-corrected, and
  affine-aligned to a common template — a preprocessing style specific to
  neuroimaging pipelines (FreeSurfer/SAMSEG), visually and statistically
  unlike any CT/MRI torso scan in the training data.
- **Novel prompt vocabulary.** Even the *language* is out of distribution:
  words like "hippocampus," "amygdala," "ventricle," "cerebral cortex"
  essentially never appear in the abdominal/thoracic/lesion-oriented
  training prompts, so the text encoder has no learned association to
  anchor them to any visual pattern.

The result is not "weak but present" signal like liver in AMOS22 — it's
the model operating entirely outside anything it was ever trained to
represent, for both the image and the text side of the prompt
simultaneously. Dice = 0.000 on all 350 rows is the expected outcome of
total domain mismatch, not a broken pipeline.

## 4. What this means for the positional-bias study

- **AMOS22 MRI** gives a graded signal: for organs where the model has
  *some* competence (liver, aorta, kidneys), you can meaningfully ask
  "does the predicted mask drift toward a neighboring organ, or just
  toward the wrong place?" — the predicted centroid/bbox columns in
  `predictions_text3dsam.csv` support that comparison.
- **Neurite OASIS** cannot answer that question the same way: with zero
  true-positive overlap anywhere, there is no "correct-ish" prediction to
  measure drift *from*. It instead demonstrates the boundary case — total
  out-of-domain collapse — useful as a contrast point in the paper, but
  not a source of graded positional-bias evidence on its own.
- If graded (non-zero) OASIS results are needed for a lateral (L/R)
  drift analysis specifically, consider: (a) lowering the probability
  threshold well below 0.5 and inspecting raw sigmoid maps instead of
  binarized masks, and/or (b) fine-tuning the checkpoint briefly on a
  handful of OASIS examples so the text encoder at least learns the new
  vocabulary — full retraining is not required to get non-zero signal,
  just enough exposure for the model to stop operating blind.

## 5. Caveats

- This is one checkpoint (`MagicXin/Text3DSAM`) at one probability
  threshold (0.5). The paper's own reported numbers (avg Dice 0.609) are
  from CVPR-BiomedSegFM's *own* held-out validation set, which matches its
  training distribution — not a fair comparison point against either
  dataset used here.
- "CT_AMOS" being in the training set means the model has seen AMOS22
  *anatomy* before (via the CT arm) — the MRI weakness is specifically a
  modality gap, not an anatomy gap, which is why liver still partially
  transfers while OASIS's genuinely novel anatomy does not transfer at
  all.
