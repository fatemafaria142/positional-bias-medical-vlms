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
| 1 | Image-Text Alignment | Fused-representation correspondence *(needs hook)* | • Pick one image together with its correct prompt (e.g. "segment the right kidney") — the "matched" pair.<br>• Run inference and, using the hook, capture the internal fused feature map — the model's internal representation right before it produces the final mask.<br>• Check whether the region of that internal map with the strongest activation spatially overlaps the true, ground-truth location of the right kidney.<br>• Repeat with a deliberately wrong prompt on the exact same image (wrong side, or a different organ entirely) — the "mismatched" pair.<br>• Check whether the high-activation region still (wrongly) lands on the same spot, or spreads out and goes diffuse instead. | • This checks alignment at the internal representation level, not just by comparing final output masks — a more direct test of whether the model's internal encoding of the prompt is genuinely tied to the right spot in the image.<br>• Output-level comparisons can be fooled by coincidence; looking at the actual internal activation region removes that ambiguity.<br>• This is the most direct internal evidence available for whether the text cue and the image region are truly linked inside the network. | Spatial overlap (e.g. Intersection-over-Union) between the high-activation region and the ground-truth bounding box, computed separately for the matched and mismatched prompts. | • High overlap only for the matched (correct) prompt, low overlap for the mismatched prompt → a real internal link exists between the word and the image region — supports the model genuinely reading and using the word.<br>• High overlap for both the matched and mismatched prompts → the image itself is driving the activation regardless of what the prompt actually says — supports the model relying on a shortcut or memorized prior instead of the word.<br>• Low overlap even for the matched (correct) prompt → there is no real internal correspondence between the word and the image region to begin with — also supports the shortcut/memorized reading. |
| 2 | Fusion | Embedding-distance vs. output-distance regression | • For an organ that exists on both sides of the body (e.g. kidney), take one image and create two prompts differing only in the position word: "segment the left kidney" and "segment the right kidney."<br>• Compute the text embedding for each prompt and measure how different those two embeddings are (cosine distance) — this shows how differently the model's language understanding treats the two words.<br>• Separately, run both prompts through the model on the same image and get two predicted masks; measure how different those two masks are from each other (Dice or centroid distance) — this shows how differently the model's actual answer changed.<br>• Repeat across every bilateral organ and subject available, then plot the text-embedding difference against the mask difference and fit a line through all the points. | • This is a correlational check: if the model's internal understanding of "left" vs. "right" never translates into a different final answer, no matter how differently it treated the two words internally, that signals the connection between understanding the word and acting on it is broken somewhere.<br>• Running this across many organs and subjects, instead of one example, turns an anecdote into a pattern — which is what actually counts as evidence.<br>• This test doesn't need any special internal model access, so it can be run today with what's already available. | The slope of the best-fit line relating text-embedding difference to output-mask difference, across all organ/subject pairs. | • A clearly positive slope, where more text difference reliably predicts more output difference → the model's differing understanding of "left" and "right" is actually translating into differing action — supports genuine reading and use of the word.<br>• A flat or near-zero slope, where text difference and output difference are unrelated → no matter how differently the model represents the two words internally, the final answer doesn't track it — supports the shortcut/memorized reading. |
| 3 | Fusion | Embedding-swap ablation *(needs hook, strongest causal test)* | • Take one image and the prompt "segment the left kidney." Run it through the model as normal, up to the point where the text embedding is about to be combined with the image information.<br>• Instead of letting the real "left kidney" text embedding continue through, secretly substitute in the text embedding that was computed for "right kidney" instead — keeping the image and everything else in the pipeline exactly the same.<br>• Let the model finish producing its output mask using this substituted embedding, and see where that mask ends up.<br>• This requires a hook at the exact point the text embedding is consumed by the rest of the model, so the substitution can happen mid-pipeline rather than by simply sending a different prompt from scratch. | • This is a true cause-and-effect test, not just an observation: it directly forces one variable (which text embedding is used) to change while holding everything else fixed, then watches what happens.<br>• It removes any lingering doubt from the correlational tests — even after showing that "left" and "right" produce very different text embeddings, this is the only test proving whether the model's downstream processing actually pays attention to that difference.<br>• It's the strongest evidence available in this whole plan, because a positive result here can't be explained away by coincidence or some other confounding factor. | Whether the resulting mask — built using the secretly-substituted "right kidney" embedding — shifts toward the true location of the right kidney, rather than staying at the left kidney's location. | • The output mask shifts toward the right kidney's true location → the model's downstream processing is causally driven by the text embedding it receives — strong support for genuine reading and use of the word.<br>• The output mask stays at, or near, the left kidney's location despite the substitution → the model's downstream processing isn't actually using the text embedding to decide where to segment, no matter how different the two embeddings looked — strong support for the shortcut/memorized reading. |
| 4 | Fusion | Fused feature-map spatial diff *(needs hook)* | • Run the model once with "segment the left kidney" and once with "segment the right kidney," on the exact same image.<br>• Using a hook, capture the internal fused representation — the combined image-plus-text signal, right before the final mask is produced — for both runs.<br>• Compute a difference map between the two internal representations, subtracting one from the other, point by point, to see where (if anywhere) they diverge.<br>• Identify where in the network, and where spatially in the image, that divergence first appears. | • This pinpoints exactly where the "left" and "right" signals stop looking identical inside the model, which is more precise than only comparing final masks — it can catch a case where the distinguishing information exists briefly inside the model but gets washed out before reaching the final output.<br>• Knowing *where* the divergence happens, or fails to happen, is more useful for diagnosing and eventually fixing the model than only knowing *whether* it happens at all. | The location and magnitude of the difference between the two internal representations. | • A clear, early divergence that localizes near the correct anatomical side → the internal representation is genuinely tracking the word — supports genuine grounding.<br>• No meaningful divergence anywhere in the internal representation, despite the prompts being different → the internal processing is treating the two prompts as functionally identical — supports the shortcut/memorized reading. |
| 5 | Word Attribution — Method | Exact word-ablation / brute-force Shapley *(recommended default)* | • Take a short prompt like "Segment the left kidney in this image," and identify its individual words.<br>• Systematically try every possible combination of those words present or removed (using a grammatically sensible removal rule, not a raw deletion), running the model once per combination.<br>• Because these prompts are short (around 6 to 9 words), the total number of combinations is small — 2 raised to the power of the word count, e.g. 256 combinations for 8 words — so trying every single one is affordable rather than needing to approximate.<br>• From the full set of results, compute the exact contribution of each individual word using the standard game-theory approach (Shapley values): how much does the model's answer change, on average, whenever a given word is present versus absent, across every possible combination of the other words. | • Because the prompts are so short, there's no need to guess or approximate which combinations matter most — every single one can be tried, so the resulting importance scores are exact rather than estimated.<br>• This avoids any risk that an approximation method misses an important interaction between two words, e.g. the position word only mattering when the organ name is also present.<br>• This is the most rigorous, most defensible version of "which word matters" that can be reported, since it doesn't depend on a sampling method's particular choices. | n/a — this row describes the estimation method itself; the actual measurement comes from whichever scoring function is applied to each combination's output (accuracy against ground truth, change relative to the model's own original answer, sideways shift, or collapse to an empty prediction). | • Produces an exact, word-by-word importance score for the whole prompt, with no approximation error — the most defensible number to report when ranking whether the organ name or the position word actually drives the output. |
| 6 | Word Attribution — Method | LIME (`lime.lime_text.LimeTextExplainer`) | • Generate a set of randomly perturbed versions of the original prompt, each with a random subset of words masked out.<br>• Run the model on each perturbed version and record the resulting output.<br>• Fit a simple, local linear model relating "which words were present" to "what the output was," weighting each perturbed example by how similar it was to the original prompt.<br>• Read the fitted model's coefficients as the importance score for each word. | • This is a well-known, widely available method (via the `lime` library) that doesn't require writing custom combinatorial code from scratch.<br>• It's a reasonable choice when a quick, approximate answer is worth more than spending the extra time to compute the exact version.<br>• Its underlying design assumes a classification-style output (predicting a category), so it needs some adaptation to work with a continuous score like a Dice value or a centroid shift. | n/a — this row describes the estimation method itself. | • Produces an approximate, word-by-word importance score.<br>• Less precise than trying every possible word combination directly, especially at this short a prompt length where the exact version is already affordable — best used only when speed of setup matters more than precision. |
| 7 | Word Attribution — Method | KernelSHAP / PartitionExplainer | • Sample a subset of possible word combinations, rather than trying every single one, and run the model on each sampled combination.<br>• Use those samples, weighted according to the standard Shapley-value sampling scheme, to estimate each word's contribution without needing to try every combination. | • This is the standard approach for text too long to try every possible word combination exhaustively.<br>• It isn't strictly necessary here, since these prompts are short enough that trying every combination directly is already affordable.<br>• Worth keeping in mind if this same approach is ever extended to longer, more free-form prompts in the future. | n/a — this row describes the estimation method itself. | • Produces the same type of word-by-word importance score as trying every combination directly, but as an approximation rather than an exact value.<br>• Only worth choosing over the exact approach if the prompts being analyzed are too long to enumerate every combination. |
| 8 | Word Attribution — Target Metric | `f_dice_gt` | • For each word-removal combination being tested, run the model and get its predicted mask.<br>• Compare that predicted mask against the actual, correct ground-truth mask for that subject, using the Dice similarity score.<br>• Repeat across all word combinations to see how removing each word changes this accuracy score. | • This ties a word's importance directly to whether the model gets the right answer, which is the outcome that actually matters in a real clinical or practical setting.<br>• It requires having a verified ground-truth mask available for the subject being tested, which limits it to labeled data. | Dice similarity score between the predicted mask for a given word combination and the ground-truth mask. | • A word whose removal causes a large drop in this score is a word that matters for getting the correct, clinically accurate answer.<br>• A word whose removal barely changes this score is a word the model isn't actually relying on to get the answer right, even if the word carries clear meaning to a person reading the prompt. |
| 9 | Word Attribution — Target Metric | `f_dice_self` | • For each word-removal combination, run the model and get its predicted mask.<br>• Compare that predicted mask against the model's own original prediction, the one made using the full, unmodified prompt — not against the ground truth.<br>• Repeat across all word combinations to see how much removing each word shifts the model's own answer, regardless of whether that answer was correct to begin with. | • This doesn't need a ground-truth label at all, so it can be computed even on subjects that were never manually annotated, widening how much data this analysis can run on.<br>• It isolates a different question from the accuracy-based version: not "does removing this word make the answer wrong," but "does removing this word change the answer at all."<br>• A word can matter a lot by this measure even on a subject with no ground truth available, which helps this analysis scale up. | Dice similarity score between the predicted mask for a given word combination and the model's own original, full-prompt prediction. | • A word whose removal causes a large change here is a word the model's output is sensitive to, whether or not the resulting answer happens to be correct.<br>• A word whose removal barely changes this score is a word the model appears to ignore entirely when producing its output. |
| 10 | Word Attribution — Target Metric | `f_centroid_shift` | • For each word-removal combination, run the model and get its predicted mask, then compute the center point (centroid) of that mask.<br>• Measure how far that center point has moved along the left-right axis of the body, compared to the center point of the model's original, full-prompt prediction.<br>• Record this as a signed value, so it's clear whether the shift moved toward the left side or the right side, not just how far it moved. | • Left/right laterality is the exact question at the heart of this whole investigation, so a metric that directly measures side-to-side movement is more on-point than a general accuracy or similarity score.<br>• This is the clearest possible way to answer: if the position word is taken away, does the model's answer actually move to the other side, or does it stay put. | Signed left-right displacement of the predicted mask's center point, relative to the model's original, full-prompt prediction. | • Removing the position word causes a clear, consistent shift in the predicted side → the position word is genuinely responsible for telling the model which side to segment.<br>• Removing the position word causes little to no shift → the model isn't relying on the position word to decide which side to segment — it's likely relying on the organ name alone, or on some other default behavior. |
| 11 | Word Attribution — Target Metric | `f_volume` | • For each word-removal combination, run the model and get its predicted mask, then count how many voxels (3D pixels) make up that mask.<br>• Compare this voxel count against the voxel count of the model's original, full-prompt prediction. | • A model can fail in more than one way when a word is removed — it might shift to the wrong location (caught by the centroid-shift measure above), or it might simply produce no usable answer at all. This measure is specifically designed to catch that second failure mode.<br>• Without this measure, a mask that collapses to near-empty could be missed or misread as "no change," when it's actually a distinct and important kind of failure. | Voxel count of the predicted mask for a given word combination, compared to the voxel count of the model's original, full-prompt prediction. | • A large drop toward zero voxels when a word is removed → the model effectively gave up and stopped producing a usable answer without that word — a different failure mode from simply shifting to the wrong side.<br>• A voxel count that stays roughly stable when a word is removed → the model is still confidently producing an answer of similar size, regardless of whether that answer moved to a different, possibly wrong, location. |

---

## Supporting notes (not tabular — kept as prose)

**Perturbation/masking rule for rows 5-7 (word removal):** don't just delete a token — it breaks
grammar in ways that could confound the result (removing "left" from "Segment the left kidney in
this image" leaves a still-grammatical sentence, but removing "kidney" doesn't). Use role-aware
templated ablation instead:
- Remove the position word → `"Segment the kidney in this image."`
- Remove the organ name, replace with a generic noun → `"Segment the left structure in this image."`
- Remove a function word (`"the"`, `"in"`, `"this"`, `"image"`) → drop cleanly; these are the
  low-importance control rows you'd expect near-zero attribution for, and double as a sanity check
  that the attribution method isn't just picking up sentence-length artifacts.

**Output of the word-attribution rows (5-11):** one attribution table per (dataset, organ) — rows =
words in the template, columns = the four target functions (rows 8-11), cell = Shapley/LIME
importance score. Rank the position word against the organ-name word directly, per organ, per
dataset, and run across many organs/subjects to report a distribution (e.g. "position word
attribution was in the bottom quartile of all content words in 80% of cases"), not just one number.

**Prerequisite for rows 1, 3, 4 (marked "needs hook"):** `predict_single_image` is the only
documented entry point. These three rows need a hook inside the installed `voxtell` package
(`voxtell.inference.predictor` and whatever segmentation network class it wraps) to intercept the
text embedding and/or fused feature map — unverified, scope it by reading the installed package
source directly (`python -c "import voxtell; print(voxtell.__file__)"`, then walk from there)
before writing any ablation code against it. If no such hook point exists cleanly (e.g. fusion
happens inside a compiled/traced graph), those three rows fall out of scope and row 2 (no hook
needed) is what's left to answer "does fusion listen to text" observationally, not causally.

---

## Execution order

1. **Row 2** — cheapest, no hook needed, self-contained (computes its own embedding-distance and
   output-distance inputs).
2. **Rows 5-11 (word attribution)** — independent of the hook; produces the per-word contribution
   tables.
3. **Rows 1, 3, 4** — blocked on the hook-discovery prerequisite above; run last, only if the hook
   point is found.

---

## Implementation status

All 11 rows are implemented, in two Kaggle notebooks (one per dataset):
`voxtell-amos-groundedness-investigation-kaggle.ipynb`,
`voxtell-oasis-groundedness-investigation-kaggle.ipynb`, generated by
`scripts/build_groundedness_investigation_notebooks.py`.

**Hook prerequisite resolved:** rows 1, 3, 4 need **no `forward_hook`** — verified directly
against the installed `voxtell` v0.1.0 source. `predictor.predict_sliding_window_return_logits(
image, text_embeddings)` already returns the raw spatial pre-sigmoid logits (exactly "the
internal representation right before the final mask"), and `text_embeddings` is a plain tensor
argument from `predictor.embed_text_prompts()` — so row 3's causal embedding-swap test is just
calling the public API with a different embedding, no monkeypatching of voxtell's own code.

**Scope decisions made along the way:**
- 3 subjects per dataset, no rotation (`rot000` only) — reuses the existing
  `manifest_{dataset}.csv` per person, filtered to unrotated rows; no new manifest created.
  - AMOS22 MRI: `amos_0508` (nipu), `amos_0530` (faria), `amos_0555` (hridoy)
  - Neurite-OASIS: `OASIS_OAS1_0014_MR1` (faria), `OASIS_OAS1_0123_MR1` (hridoy),
    `OASIS_OAS1_0062_MR1` (nipu)
- Two separate notebook runs (one per dataset), not combined.
- Word attribution (rows 5-11): **exact Shapley everywhere** (row 5's own "recommended
  default") over role-aware word slots (position / organ / the / in / this / image) — not raw
  token deletion, and not the cheaper templated-ablation shortcut, since the exact
  2^n-per-organ enumeration turned out cheap enough (~10-25 min/dataset) once measured against
  real per-subject inference timing from the existing rotation CSVs.
- **Every** ablation combo's mask is saved (not just the baseline) — no experiment left out.
- Rows 6 (LIME) and 7 (SHAP) are both implemented with the real libraries (required for
  comparison against other papers), not skipped as redundant with row 5:
  - **Row 7 (SHAP)** reuses the exact combo data already computed — `shap.KernelExplainer`
    over the same role/metric lookup table, zero extra GPU cost. Verified locally that it
    matches the hand-rolled exact Shapley values (expected: SHAP internally enumerates the
    full 2^n space at this feature count).
  - **Row 6 (LIME)** is genuinely new inference — `lime.lime_text.LimeTextExplainer` with its
    own literal word-deletion perturbations (100 samples/organ, batched per organ into one
    model call each).
- Output is **CSV-only** — no PNGs rendered on Kaggle. Local visualization
  (`scripts/plot_groundedness_results.py`) decodes the RLE-encoded masks and plots overlays
  against the original images (already present locally in `<person>/samples/<dataset>/`), and
  draws word/role-importance bar charts (single-method or a 3-way exact-Shapley/SHAP/LIME
  comparison chart) — needs only numpy/pandas/matplotlib, no GPU/torch/voxtell.

**Output: 9 CSVs per dataset (18 total across both notebooks), all under `results/`:**

| # | File | Row(s) | Contents |
|---|---|---|---|
| 1 | `baseline_predictions_{dataset}.csv` | *supports* 1, 3, 4 | Real (unablated) prediction per subject x organ: `gt_mask_rle_b64`/`pred_mask_rle_b64`, bbox, centroid, side, Dice. Not itself a numbered row's deliverable — it's the shared foundational data rows 1/3/4 are computed from (see the row-by-row map below), plus general "does the model even segment correctly" sanity plots. |
| 2 | `ablation_masks_{dataset}.csv` | 8-11 | Every 2^n word-ablation combo's mask + all four target metrics, per subject x organ x combo |
| 3 | `word_attribution_{dataset}.csv` | 5 | Exact Shapley value per role x organ x subject x metric |
| 4 | `shap_attribution_{dataset}.csv` | 7 | SHAP (KernelExplainer) value per role x organ x subject x metric, plus a `matches_exact_shapley` check |
| 5 | `lime_attribution_{dataset}.csv` | 6 | LIME weight per word x organ x subject |
| 6 | `row1_image_text_alignment_{dataset}.csv` | 1 | Matched vs. mismatched IoU |
| 7 | `row2_embedding_output_regression_{dataset}.csv` | 2 | Embedding-distance vs. output-distance points |
| 8 | `row3_causal_embedding_swap_{dataset}.csv` | 3 | Causal side-agreement |
| 9 | `row4_fused_feature_diff_{dataset}.csv` | 4 | Spatial diff map summary |

**Row-by-row result location** (every row 1-11 traced to its exact file/column, so coverage is
never ambiguous):

| Row | Task | Where the result is |
|---|---|---|
| 1 | Image-text alignment (matched vs. mismatched IoU) | `row1_image_text_alignment_{dataset}.csv` — `iou_matched`, `iou_mismatched`, `genuine_grounding` |
| 2 | Embedding-distance vs. output-distance regression | `row2_embedding_output_regression_{dataset}.csv` — raw points; the fitted slope itself is printed in the notebook's summary section, not a CSV column (it's one fitted number, not a per-row value) |
| 3 | Causal embedding-swap test | `row3_causal_embedding_swap_{dataset}.csv` — `left_prompt_predicts_left_side`, `right_prompt_predicts_right_side`, `causally_grounded` |
| 4 | Fused feature-map spatial diff | `row4_fused_feature_diff_{dataset}.csv` — `diff_voxels`, `diff_fraction_of_union`, `diff_centroid_axis0` |
| 5 | Exact Shapley (method) | `word_attribution_{dataset}.csv` — `shapley_value` per role |
| 6 | LIME (method) | `lime_attribution_{dataset}.csv` — `lime_weight` per word |
| 7 | KernelSHAP (method) | `shap_attribution_{dataset}.csv` — `shap_value` per role |
| 8 | `f_dice_gt` (target metric) | `metric == 'f_dice_gt'` rows in `word_attribution`/`shap_attribution`; `f_dice_gt` column in `ablation_masks_{dataset}.csv` |
| 9 | `f_dice_self` (target metric) | `metric == 'f_dice_self'` rows in `word_attribution`/`shap_attribution`; `f_dice_self` column in `ablation_masks_{dataset}.csv` |
| 10 | `f_centroid_shift` (target metric) | `metric == 'f_centroid_shift'` rows in `word_attribution`/`shap_attribution`; `f_centroid_shift` column in `ablation_masks_{dataset}.csv` |
| 11 | `f_volume` (target metric) | `metric == 'f_volume'` rows in `word_attribution`/`shap_attribution`; `f_volume` column in `ablation_masks_{dataset}.csv` |

Rows 8-11 don't get their own CSV because they aren't independent experiments in the master
table — they're the four *scoring functions* applied to rows 5-7's method output (run exact
Shapley/SHAP/LIME, then score each ablation combo four different ways), so they live as
`metric` values/columns inside the same files as rows 5-7, not as standalone files.