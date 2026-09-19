# VoxTell Model-Internals Investigation Plan: Image vs. Text vs. Fusion, Memorization vs. Learning

## 0. The core question

> **When VoxTell segments "the right kidney," what is actually driving the output — the anatomy
> it sees in the image, the word "right," the word "kidney," or a memorized shape/location prior
> from training?**

Every row in the master table below produces evidence toward one of two broad readings: either the
model **genuinely grounds position** (it reads the position word and locates the correct side in
the image), or it relies on a **shortcut or memorized prior** (position is decorative; the model
leans on organ identity, image content, or a fixed pose prior instead). The **Analysis** column in
each row spells out, in full, what result points which way — no shorthand codes.

---

## Plain-language summary (read this first if the table below feels dense)

**The big question, in one sentence:** when VoxTell draws an outline for "the right kidney," is it
actually listening to the word "right," or is it just guessing based on what a kidney usually looks
like and where it usually sits?

**What's a "hook"?** A few rows below need something called a "hook." Think of it as a small window
you open into the model while it's working, so you can peek at what it's doing *inside*, not just
look at the final drawing it hands back. Without a hook, we can only ever see the finished outline —
never the model's "thought process" that led to it.

**Image-Text Alignment (row 1):** Does the model's internal "spotlight" actually land on the right
body part when it hears the word "right"? This isn't about the final drawing — it's about checking,
inside the model, whether the word and the correct spot in the image are actually connected to each
other before any drawing happens.

**Fusion (rows 2-4) — does the word actually change what gets drawn?**
- **Row 2:** If we only swap the word "left" for "right," does the final drawing actually move? We
  check whether "how different the two words look to the model" lines up with "how different the two
  drawings turn out."
- **Row 3 — the strongest test:** We secretly swap in the "right" word's internal signal while the
  model thinks it's still processing "left," and watch whether the drawing follows. If the drawing
  doesn't budge, the model isn't really using the word at all — no matter how different the two words
  looked to it.
- **Row 4:** We look inside the model (not the final picture) to find exactly where — if anywhere —
  the "left" version and the "right" version start to look different from each other.

**Word Attribution (rows 5-11) — which word matters most?** The idea: remove one word from the
prompt at a time (e.g. drop "left," or drop "kidney") and watch how the answer changes. The word
that changes the answer the most is the word that actually matters.
- **Rows 5-7** are three different tools for doing this removal-and-measure process. Row 5 tries
  every possible combination of words removed (most thorough, and cheap enough here since the
  prompts are short). Rows 6 and 7 are faster, approximate shortcuts for the same idea.
- **Rows 8-11** are four different ways to *score* what changed when a word was removed:
  - **Row 8:** Did the answer get worse compared to the real, correct answer?
  - **Row 9:** Did the answer change at all, compared to what the model said before — even without
    knowing the real answer?
  - **Row 10:** Did removing the word flip the drawing from the left side to the right side (or vice
    versa)? This is the most direct test for the actual left/right question.
  - **Row 11:** Did the model just give up and draw nothing at all?

**How to read a "genuine grounding" vs. "shortcut/memorized" result:** every row's Analysis column
ends in one of two directions. "Genuine grounding" means the evidence says the model really is
reading and using the word. "Shortcut or memorized" means the evidence says the model is leaning on
something else instead — what the organ usually looks like, or where it usually sits — and the word
itself isn't doing much work.

---

## Master analysis table

All setups assume access to `predictor.text_backbone` (`Qwen/Qwen3-Embedding-4B`),
`predictor.embed_text_prompts(text_prompts)`, and `predictor.predict_single_image(img, organs)`
(from `voxtell-amos-rotation-multiangle-kaggle.ipynb`).

**What "needs hook" means:** a *hook* here is a PyTorch `forward_hook` — a small callback attached
to a specific internal layer of the model that lets you intercept or inspect the data flowing
through that layer while the model runs, without modifying VoxTell's own code. `predict_single_image`
only gives you the final mask; if a row needs to see something *inside* the model (the fused
feature map, the text embedding at the exact point it enters the decoder), it needs a hook attached
to the right internal module first. Whether such a clean attachment point even exists in VoxTell's
code is currently **unverified** — see the **Prerequisite** note below the table for how to check.

| ID | Category | Use Case | Setup — What We Do | Reasoning — Why This Test | Metric — What We Measure | Analysis — What We Learn |
|---|---|---|---|---|---|---|
| 1 | Image-Text Alignment | Fused-representation correspondence *(needs hook)* | For a matched (image, correct prompt) pair, check whether the fused feature map's high-activation region spatially overlaps the ground-truth structure location; repeat for a deliberately mismatched prompt on the same image, checking whether it still does (wrongly) or goes diffuse. | Tests whether alignment is spatially real *inside the network*, not just inferable from comparing output masks — the most direct internal check that a text cue actually maps to the right image region. | Spatial overlap (e.g. IoU between high-activation region and GT bounding box), matched vs. mismatched. | • High overlap only for matched pairs → real alignment exists inside the network — supports genuine grounding.<br>• High overlap even for mismatched pairs → the image dominates regardless of what the prompt says — supports the shortcut/memorization reading.<br>• Low overlap even for matched pairs → no real correspondence exists to align in the first place — also supports the shortcut/memorization reading. |
| 2 | Fusion | Embedding-distance vs. output-distance regression | Self-contained: for each bilateral organ/subject pair, (a) embed `"left X"` and `"right X"` via `embed_text_prompts` and compute cosine distance, and (b) run both prompts through `predict_single_image` on the same image and compute the Dice/centroid distance *between the two output masks*. Regress (a) against (b) across all pairs. | A correlational check — if text distinctions never translate into output differences at any scale, that's fusion ignoring the signal outright, not just occasionally. | Regression slope (text-embedding distance → output-mask distance). | • Positive slope → text separation predicts output separation — supports genuine grounding.<br>• Flat/near-zero slope → output doesn't track text difference at all — supports the shortcut/memorization reading. |
| 3 | Fusion | Embedding-swap ablation *(needs hook, strongest causal test)* | Compute image features for `"left X"`, substitute in the text embedding computed for `"right X"` before it reaches fusion. | The strongest causal test available — directly manipulates the one variable in question (which text embedding reaches fusion) instead of just observing correlation. | Whether the output mask shifts toward the right-side location. | • Shifts toward "right" → fusion causally uses the text embedding — supports genuine grounding.<br>• Doesn't move → fusion provably ignores it, regardless of how distinct the embeddings are — supports the shortcut/memorization reading. |
| 4 | Fusion | Fused feature-map spatial diff *(needs hook)* | Log the fused feature map for left- vs. right-prompt runs on the same image; compute a spatial difference map. | Localizes precisely *where* (or whether) left/right information diverges inside the network — pinpoints the failure point more precisely than output-level tests alone. | Where left/right first diverge (or fail to), spatially. | • Early divergence localized near the correct side → supports genuine grounding.<br>• No meaningful divergence anywhere → supports the shortcut/memorization reading. |
| 5 | Word Attribution — Method | Exact word-ablation / brute-force Shapley *(recommended default)* | Enumerate all 2^n subsets of the prompt's content words, run each through `predict_single_image`, compute the exact Shapley value per word from the resulting output values. | VoxTell prompts are short (6-9 words), so exact enumeration is cheap (2^8 = 256 runs) — no need for LIME/KernelSHAP's approximation when exactness is affordable. `shap.explainers.Exact`/`Permutation` with a `shap.maskers.Text` masker can also do this for you. | n/a — this is the estimation method itself. | • Produces exact, not approximate, per-word importance scores — the most defensible version of "which word matters" you can report. |
| 6 | Word Attribution — Method | LIME (`lime.lime_text.LimeTextExplainer`) | Fits a local linear surrogate model on randomly perturbed (word-masked) prompt versions, weighted by similarity to the original. | Faster to set up with a well-known library if exactness isn't required; note LIME's classifier-oriented API needs adapting for a continuous (non-classification) target. | n/a. | • Approximate per-word importance.<br>• Less precise than row 5 at this prompt length — use only if quick iteration matters more than exactness. |
| 7 | Word Attribution — Method | KernelSHAP / PartitionExplainer | Approximate Shapley values via sampled coalitions. | Standard choice for longer text where exact enumeration is infeasible — not needed here given prompt length, but useful if this method is later extended to longer, paragraph-style prompts. | n/a. | • Same output type as row 5 but approximate.<br>• Only worth using if you extend beyond short templated prompts. |
| 8 | Word Attribution — Target Metric | `f_dice_gt` | Dice(predicted mask for perturbed prompt, ground-truth mask). | Ties word importance to actual correctness — the metric that matters clinically. | Dice vs. GT, per perturbation. | • Which words matter for *getting the answer right* — the accuracy-facing view. |
| 9 | Word Attribution — Target Metric | `f_dice_self` | Dice(predicted mask for perturbed prompt, predicted mask for the *original, unperturbed* prompt). | Doesn't require ground truth, so it also works on unlabeled subjects — isolates "does the word change the output at all" from "is the output correct." | Dice vs. original prediction, per perturbation. | • Which words matter for *changing the output at all* — the stability/sensitivity view. |
| 10 | Word Attribution — Target Metric | `f_centroid_shift` | Signed displacement of the predicted mask's centroid along the left-right axis, relative to the original prompt's centroid. | The single most on-point metric for the core laterality question — directly measures whether removing a word moves the mask to the other side. | Signed centroid displacement, per perturbation. | • Which word, when removed, moves the predicted mask's side.<br>• This is the single most on-point metric for "does it take the anatomy or the L/R word." |
| 11 | Word Attribution — Target Metric | `f_volume` | Predicted mask voxel count. | Captures a different failure mode than moving to the wrong side — the model "giving up" entirely. | Voxel count, per perturbation. | • Detects collapse to empty/near-empty prediction when a word is removed. |

---