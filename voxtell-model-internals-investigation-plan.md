# VoxTell Model-Internals Investigation Plan: Image vs. Text vs. Fusion, Memorization vs. Learning

## 0. The core question

> **When VoxTell segments "the right kidney," what is actually driving the output — the anatomy
> it sees in the image, the word "right," the word "kidney," or a memorized shape/location prior
> from training?**

Every row in the master table below produces evidence toward one of two broad readings: **H1 —
genuine grounding** (the model reads position words and locates the correct side in the image) vs.
**H2/H3 — shortcut or memorization** (position is decorative; the model leans on organ identity,
image content, or a fixed pose prior instead). The **Analysis** column in each row states what
result points which way.

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
| AL4 | Image-Text Alignment | Fused-representation correspondence *(needs hook)* | For a matched (image, correct prompt) pair, check whether the fused feature map's high-activation region spatially overlaps the ground-truth structure location; repeat for a deliberately mismatched prompt on the same image, checking whether it still does (wrongly) or goes diffuse. | Tests whether alignment is spatially real *inside the network*, not just inferable from comparing output masks — the most direct internal check that a text cue actually maps to the right image region. | Spatial overlap (e.g. IoU between high-activation region and GT bounding box), matched vs. mismatched. | High overlap only for matched pairs → real alignment exists (H1). High overlap even for mismatched pairs → image dominates regardless of prompt (H2/H3). Low overlap even for matched pairs → no real correspondence exists to align in the first place (H2/H3). |
| F2 | Fusion | Embedding-distance vs. output-distance regression | Self-contained: for each bilateral organ/subject pair, (a) embed `"left X"` and `"right X"` via `embed_text_prompts` and compute cosine distance, and (b) run both prompts through `predict_single_image` on the same image and compute the Dice/centroid distance *between the two output masks*. Regress (a) against (b) across all pairs. | A correlational check — if text distinctions never translate into output differences at any scale, that's fusion ignoring the signal outright, not just occasionally. | Regression slope (text-embedding distance → output-mask distance). | Positive slope → text separation predicts output separation (H1). Flat/near-zero slope → output doesn't track text difference at all (H2/H3). |
| F3 | Fusion | Embedding-swap ablation *(needs hook, strongest causal test)* | Compute image features for `"left X"`, substitute in the text embedding computed for `"right X"` before it reaches fusion. | The strongest causal test available — directly manipulates the one variable in question (which text embedding reaches fusion) instead of just observing correlation. | Whether the output mask shifts toward the right-side location. | Shifts toward "right" → fusion causally uses the text embedding (H1). Doesn't move → fusion provably ignores it, regardless of how distinct the embeddings are (H2/H3). |
| F4 | Fusion | Fused feature-map spatial diff *(needs hook)* | Log the fused feature map for left- vs. right-prompt runs on the same image; compute a spatial difference map. | Localizes precisely *where* (or whether) left/right information diverges inside the network — pinpoints the failure point more precisely than output-level tests alone. | Where left/right first diverge (or fail to), spatially. | Early divergence localized near the correct side → H1. No meaningful divergence anywhere → H2/H3. |
| F5 | Fusion | Wrong-organ / absent-organ prompt-invariance | Fixed image, run the correct prompt plus: wrong organ name, wrong side, and an organ genuinely absent from the volume. | Tests whether the decoder is prompt-driven at all, or defaults to "segment whatever's salient" regardless of what's asked — a blunt but decisive shortcut/memorization check. | Whether output tracks the prompt or stays roughly constant across the four prompts. | Output changes appropriately per prompt → H1. Output stays "something plausible" regardless of prompt → H2/H3 (organ-agnostic shortcut, or "there's always something here" prior). |
| M1 | Word Attribution — Method | Exact word-ablation / brute-force Shapley *(recommended default)* | Enumerate all 2^n subsets of the prompt's content words, run each through `predict_single_image`, compute the exact Shapley value per word from the resulting output values. | VoxTell prompts are short (6-9 words), so exact enumeration is cheap (2^8 = 256 runs) — no need for LIME/KernelSHAP's approximation when exactness is affordable. `shap.explainers.Exact`/`Permutation` with a `shap.maskers.Text` masker can also do this for you. | n/a — this is the estimation method itself. | Produces exact, not approximate, per-word importance scores — the most defensible version of "which word matters" you can report. |
| M2 | Word Attribution — Method | LIME (`lime.lime_text.LimeTextExplainer`) | Fits a local linear surrogate model on randomly perturbed (word-masked) prompt versions, weighted by similarity to the original. | Faster to set up with a well-known library if exactness isn't required; note LIME's classifier-oriented API needs adapting for a continuous (non-classification) target. | n/a. | Approximate per-word importance; less precise than M1 at this prompt length — use only if quick iteration matters more than exactness. |
| M3 | Word Attribution — Method | KernelSHAP / PartitionExplainer | Approximate Shapley values via sampled coalitions. | Standard choice for longer text where exact enumeration is infeasible — not needed here given prompt length, but useful if this method is later extended to longer, paragraph-style prompts. | n/a. | Same output type as M1 but approximate; only worth using if you extend beyond short templated prompts. |
| G1 | Word Attribution — Target Metric | `f_dice_gt` | Dice(predicted mask for perturbed prompt, ground-truth mask). | Ties word importance to actual correctness — the metric that matters clinically. | Dice vs. GT, per perturbation. | Which words matter for *getting the answer right* — the accuracy-facing view. |
| G2 | Word Attribution — Target Metric | `f_dice_self` | Dice(predicted mask for perturbed prompt, predicted mask for the *original, unperturbed* prompt). | Doesn't require ground truth, so it also works on unlabeled subjects — isolates "does the word change the output at all" from "is the output correct." | Dice vs. original prediction, per perturbation. | Which words matter for *changing the output at all* — the stability/sensitivity view. |
| G3 | Word Attribution — Target Metric | `f_centroid_shift` | Signed displacement of the predicted mask's centroid along the left-right axis, relative to the original prompt's centroid. | The single most on-point metric for the core laterality question — directly measures whether removing a word moves the mask to the other side. | Signed centroid displacement, per perturbation. | Which word, when removed, moves the predicted mask's side — the most direct test for "does it take the anatomy or the L/R word." |
| G4 | Word Attribution — Target Metric | `f_volume` | Predicted mask voxel count. | Captures a different failure mode than moving to the wrong side — the model "giving up" entirely. | Voxel count, per perturbation. | Detects collapse to empty/near-empty prediction when a word is removed. |
| S1 | Synthesis | Organ-name contribution (e.g. "kidney") | Cross-reference F5's pass/fail with the organ-name word's attribution score from the M1-M3/G1-G4 word-attribution rows. | Combines a causal output-level test (F5) with a quantified attribution score to triangulate the organ name's real contribution, not just one anecdotal signal. | F5 outcome + attribution magnitude on the organ-name word. | Output changes correctly when organ name changes (F5) *and* high attribution score → organ name genuinely drives the output. |
| S2 | Synthesis | Position-word contribution (e.g. "left"/"right") | Cross-reference F3's causal shift with the position word's attribution score from the M1-M3/G1-G4 rows, especially on `f_centroid_shift` (G3). | Combines the strongest causal fusion test (F3) with the quantified word-attribution score to answer whether position words genuinely move the output, not just look different in isolation. | F3 outcome + attribution magnitude on the position word, esp. G3. | Output actually moves (F3) *and* high attribution on `f_centroid_shift` → position word contributes (H1). Both weak together → strongest case that position is decorative (H2). |
| S3 | Synthesis | Image-text alignment (as currently scoped) | Read AL4 on its own — this is the only alignment test currently in scope. | Flags that the broader alignment test suite (matched-vs-mismatched confidence ranking, alignment-vs-rotation-angle) was trimmed out; AL4 alone is a real but narrower signal than the full suite would give. | AL4's spatial-overlap result. | High overlap only for matched pairs → real alignment exists inside the network. Otherwise → no real correspondence, or the image dominates regardless of prompt. |

---

## Supporting notes (not tabular — kept as prose)

**Perturbation/masking rule for M1-M3 (word removal):** don't just delete a token — it breaks
grammar in ways that could confound the result (removing "left" from "Segment the left kidney in
this image" leaves a still-grammatical sentence, but removing "kidney" doesn't). Use role-aware
templated ablation instead:
- Remove the position word → `"Segment the kidney in this image."`
- Remove the organ name, replace with a generic noun → `"Segment the left structure in this image."`
- Remove a function word (`"the"`, `"in"`, `"this"`, `"image"`) → drop cleanly; these are the
  low-importance control rows you'd expect near-zero attribution for, and double as a sanity check
  that the attribution method isn't just picking up sentence-length artifacts.

**Output of the word-attribution rows (M1-M3, G1-G4):** one attribution table per (dataset, organ)
— rows = words in the template, columns = the four target functions (G1-G4), cell = Shapley/LIME
importance score. Rank the position word against the organ-name word directly, per organ, per
dataset, and run across many organs/subjects to report a distribution (e.g. "position word
attribution was in the bottom quartile of all content words in 80% of cases"), not just one number.

**Prerequisite for AL4, F3, F4 (rows marked "needs hook"):** `predict_single_image` is the only
documented entry point. These three rows need a hook inside the installed `voxtell` package
(`voxtell.inference.predictor` and whatever segmentation network class it wraps) to intercept the
text embedding and/or fused feature map — unverified, scope it by reading the installed package
source directly (`python -c "import voxtell; print(voxtell.__file__)"`, then walk from there)
before writing any ablation code against it. If no such hook point exists cleanly (e.g. fusion
happens inside a compiled/traced graph), those three rows fall out of scope and F2/F5 (no hook
needed) are what's left to answer "does fusion listen to text" observationally, not causally.

---

## Execution order

1. **F2** — cheapest, no hook needed, self-contained (computes its own embedding-distance and
   output-distance inputs).
2. **M1-M3 + G1-G4 (word attribution)** — independent of the hook; produces the per-word
   contribution tables (feeds S1, S2).
3. **F5** — one new multi-prompt inference pass per subject, no hook needed.
4. **AL4, F3, F4** — blocked on the hook-discovery prerequisite above; run last, only if the hook
   point is found.
5. **S1, S2, S3** — write-up step, combines the rows above; no new inference.
