# Paper Summary: UNISELF (Zhang et al., Medical Image Analysis, 2026)

**Full citation:** Jinwei Zhang, Lianrui Zuo, Blake E. Dewey, Samuel W. Remedios, Yihao Liu,
Savannah P. Hays, Dzung L. Pham, Ellen M. Mowry, Scott D. Newsome, Peter A. Calabresi,
Shiv Saidha, Aaron Carass, Jerry L. Prince. "UNISELF: A unified network with instance
normalization and self-ensembled lesion fusion for multiple sclerosis lesion segmentation."
*Medical Image Analysis* 109 (2026) 103954.

**Note on affiliations:** all listed authors are from Johns Hopkins University, Vanderbilt
University, or the Uniformed Services University of the Health Sciences — none are listed
as UMass Amherst. Worth confirming this is the intended paper/professor before referencing
it directly in an email.

## What the paper is about

UNISELF is a deep learning model for **segmenting multiple sclerosis (MS) lesions** in brain
MRI. It is a discriminative segmentation method (U-Net based), **not** a generative model —
there is no text/report generation, VQA, or image synthesis involved. The core problem it
addresses: existing MS lesion segmentation models struggle to be simultaneously (a) accurate
on their training domain, (b) generalizable to new hospitals/scanners, and (c) robust when
some MRI contrasts (T1w, T2w, PDw, FLAIR) are missing at test time — especially when trained
on a small, single-site dataset (as is common in practice, since large multi-site labeled MS
data is rare).

## Core contribution

Two technical additions on top of a standard U-Net backbone, both applied only at test time:

1. **Self-ensembled lesion fusion.** Instead of standard test-time augmentation (averaging
   predictions from a few flipped/rotated versions of the image), UNISELF runs 24 augmented
   versions (3 anatomical planes × 8 flip/rotation combinations) through the same trained
   network, builds a voxel-wise "confidence map" (how many of the 24 predictions called this
   voxel a lesion), then applies a **two-threshold "detect confidently, then grow into
   uncertain neighbors"** step — conceptually similar to Canny edge detection's
   high/low-threshold hysteresis.
2. **Test-time instance normalization (TTIN).** Instead of using normalization statistics
   fixed during training (standard batch normalization), UNISELF recomputes normalization
   statistics **per individual test scan** at inference time. This corrects for feature
   mismatches caused by scanner/site differences and by missing input contrasts.

**Result:** trained on only 5 subjects (single site), UNISELF matches or beats the best
published ISBI-2015-challenge methods in-domain, and clearly outperforms all benchmarks
(Tiramisu, ModDrop++, contrast dropout, domain generalization baselines, nnU-Net,
WMH-SynthSeg) on out-of-domain test sets — including a private 9-scanner clinical dataset,
and under simulated imaging artifacts (motion, noise, ghosting, bias field, blur,
anisotropy) and missing-FLAIR scenarios.

## Limitations (stated by the authors, Section 7.7)

- The final model is selected from the last training epoch only (no k-fold
  cross-validation), which risks overfitting.
- The two fusion thresholds (τ1, τ2) are fixed across the entire brain — likely suboptimal
  for small cortical lesions, which may need more sensitive (lower) thresholds.
- Multi-orientation inputs are processed by a single shared network; the authors suggest a
  Mixture-of-Experts architecture as a possible improvement.
- The private multi-site test set has only **automatically generated "silver-standard"
  labels** (from two other AI tools, LST-AI and HD-MS-Lesions), not real expert
  delineations — a meaningfully weaker ground truth than their other test sets.
- Still trained on very limited labeled data overall; the authors suggest self-supervised
  pretraining on large unlabeled MRI or training with synthetic MRI containing lesions as
  future directions.

## What's yet to explore (authors' own "Broader applications," Section 7.6)

- Extending the detection-and-growth idea to **separate confluent lesions** for paramagnetic
  rim lesion detection (a distinct MS biomarker).
- Applying the same self-ensemble/TTIN approach to **MRI reconstruction**, not just
  segmentation.
- More rigorous multi-site validation of any such extensions.

## Relevance to hallucination/faithfulness/consistency research questions

This paper does **not** directly address hallucination, faithfulness, or report generation —
it's a segmentation-only, non-generative method. The one directly relevant detail: in
Section 7.2, the authors explicitly note that generative approaches to imputing missing MRI
contrasts (image-to-image synthesis) **"risk hallucinated or low-fidelity content,"** and
that this is part of their motivation for avoiding synthesis entirely in UNISELF. This shows
the lab is already aware of and concerned about hallucination as a failure mode — just in the
context of image synthesis (fabricating a missing MRI contrast), not text report generation.

**Takeaway for the professor email:** don't claim this paper is about report generation or
hallucination — it isn't. Instead, use the Section 7.2 observation as an honest bridge: his
own work already treats hallucination as a real risk in generative MRI methods; your proposal
extends that same concern to a different generative task (MRI report generation) that his
lab hasn't covered yet.
