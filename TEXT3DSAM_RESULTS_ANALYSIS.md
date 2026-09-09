# Why Text3DSAM performs the way it does on AMOS22 MRI and Neurite OASIS

This documents the likely causes behind the results in
`samples/amos22_mri/predictions_text3dsam.csv` (mean Dice 0.035, liver best
at ~0.24) and `samples/neurite_oasis/predictions_text3dsam.csv` (mean Dice
0.000, 0/350 nonzero). Both are consequences of what the model was trained
on, not bugs in this pipeline — the input images and masks were verified
correctly aligned (see `results_text3d_oasis/.../left_cerebral_cortex.png`:
real predicted voxels exist, they just never overlap ground truth).

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

## 2. Why AMOS22 MRI is weak but not zero (mean Dice 0.035, liver 0.24)

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

**c. Organ size/frequency correlates almost exactly with Dice.**
Liver (the largest organ, present and dominant in nearly every abdominal
dataset in training) is the clear best performer. Spleen, pancreas, and
adrenal glands — small, thin, or highly variable in shape — score exactly
0 across all 10 subjects. This is the expected failure mode of a model
whose spatial resolution and training exposure both favor large,
high-contrast structures.

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
