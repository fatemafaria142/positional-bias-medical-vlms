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

## Research questions at a glance

One core question and 16 row-level questions (17 in total). Each row-level question is answered
by the row of the same number in the master table below. Rows 1–11 have been run. **Rows 12–16 are
implemented in the two "mechanistic" notebooks but have not been run yet** that cover what rows 1–11 cannot answer: whether "left" means
anatomy or image coordinates, whether the model uses a location prior instead of the image, a real
embedding intervention, cross-attention evidence, and where in the network left/right diverge.

| ID | Group | Research question |
|---|---|---|
| **Core** | Overall | When VoxTell segments "the right kidney", what drives the output: the image anatomy, the word "right", the word "kidney", or a memorized shape/location prior? |
| 1 | Image-Text Alignment | Is the link between the prompt and the image region real? Does the matched prompt land on the correct region while a mismatched prompt does not? |
| 2 | Fusion | Does a larger left/right difference in the text embeddings lead to a larger difference in the output masks? |
| 3 | Fusion | If the "right kidney" embedding is substituted for "left kidney", does the mask move to the right kidney? |
| 4 | Fusion | Where inside the model, if anywhere, do the left and right representations diverge? |
| 5 | Word Attribution: Method | What is each word's exact importance (brute-force Shapley value)? |
| 6 | Word Attribution: Method | What importance does LIME assign to each word? |
| 7 | Word Attribution: Method | What importance does KernelSHAP assign to each word? |
| 8 | Word Attribution: Metric | Does removing a word reduce accuracy against the ground truth (`f_dice_gt`)? |
| 9 | Word Attribution: Metric | Does removing a word change the model's own answer (`f_dice_self`)? |
| 10 | Word Attribution: Metric | Does removing the position word move the mask to the opposite side (`f_centroid_shift`)? |
| 11 | Word Attribution: Metric | Does removing a word make the mask collapse to nothing (`f_volume`)? |
| 12 *(notebook ready, not yet run)* | Image | If the image is flipped left-right and the prompt stays the same, does the mask follow the patient's anatomy or stay on the same side of the image? |
| 13 *(notebook ready, not yet run)* | Image | Does the model rely on the image evidence, or on a location prior? Does it still segment where the organ usually is when the organ is erased, the image is blank, or the image comes from another subject? |
| 14 *(notebook ready, not yet run)* | Fusion | In a real intervention, where does the mask move when the "left" representation is blended with, or partly replaced by, the "right" one, and at which decoder layer is the side decided? |
| 15 *(notebook ready, not yet run)* | Image-Text Alignment | Does the cross-attention of "left kidney" concentrate on the left kidney rather than the right kidney or another organ? |
| 16 *(notebook ready, not yet run)* | Fusion | At which decoder layer do the left and right representations first diverge, compared with the gap between two different organs? |

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
| 12 *(notebook ready, not yet run)* | Image — Anatomy vs. coordinates | Left-right image flip, prompt fixed | • Take the unrotated image and its ground truth and flip both along the right-left axis (the flipped ground truth keeps its original labels, so the anatomically-left kidney now sits on the image's right).<br>• Run the same prompts ("left kidney", "right kidney") on the original and the flipped image.<br>• Score each mask by which ground-truth organ it overlaps: the organ with the named anatomical side (anatomy) or the organ at the named image side (coordinates).<br>• Repeat across all subjects and organ pairs, and with the existing rotation sweep if available. | • Rows 1–11 show the word selects a side but not the frame of reference: an anatomy-based model follows the organ, a coordinate-based one follows the image side.<br>• A flip keeps image content identical, so only the frame of reference changes. | Dice and coverage against the anatomically-named organ vs. the image-side organ, per prompt, original vs. flipped. | • Mask follows the anatomically-named organ after the flip → "left" is grounded in anatomy — supports genuine grounding.<br>• Mask stays on the same image side and now overlaps the other organ → "left" is an image-coordinate rule — supports the shortcut reading.<br>• Mixed or degraded results → partial reliance on both; report the split. |
| 13 *(notebook ready, not yet run)* | Image — Evidence vs. location prior | Organ erasure, blank image, other-subject image | • Erase the target organ from the image (fill its ground-truth region with local background intensity) and prompt for it.<br>• Run the same prompt on a blank or noise image, and on a different subject's image.<br>• Record where any predicted mask lands, and how large it is.<br>• Compare against a baseline that puts each organ at the dataset-mean position for its side. | • Only the text was ablated so far, so the image's role is untested.<br>• A model that reads the image finds nothing where the organ is missing. A model that reproduces a prior still segments the usual place. | Predicted volume, centroid distance to the dataset-mean location for that side, and overlap with the erased region. | • Little or no mask when the organ is erased or the image is blank → the model depends on image evidence — supports genuine grounding.<br>• A mask of normal size at the usual location regardless of the image → a memorized location/shape prior — supports the shortcut/memorized reading.<br>• Matches the dataset-mean-position baseline → the position word is being converted into a fixed prior. |
| 14 *(notebook ready, not yet run)* | Fusion — Real embedding intervention | Embedding blend and per-layer patching | • Blend the text embeddings of "left X" and "right X" (e.g. weights 0, 0.25, 0.5, 0.75, 1, plus −0.5 and 1.5 to extrapolate) and run each on the same image.<br>• Separately, run "left X" but, at decoder layer *k* (0–5), replace that layer's update (its change to the query state) with the update computed for "right X" on the same patch. Do this for each single layer, and for all layers from *k* onward, in both directions (L→R and R→L).<br>• As a control, patch the whole query state at a layer, which should exactly reproduce the source prompt's mask.<br>• Score each output by side and by Dice against each side's ground truth. | • Passing the right embedding to the model is the same as prompting "right", so a real test has to intervene inside the model or interpolate.<br>• Patching the update, not the full state, matters: the decoder is pre-norm and the state only carries the text through the residual stream, so a full-state patch is trivially the source prompt.<br>• The blend gives a dose-response curve, and the per-layer patch shows which layers carry the side. | Side call and Dice per blend weight and per patched layer, plus the weight or layer at which the side flips. | • Smooth, monotonic side shift with blend weight and a clear set of layers that flip the side → the side is causally carried by the text representation and decided in identifiable layers — supports genuine grounding.<br>• The side flips abruptly or only at extreme weights → a threshold-like or brittle representation.<br>• Patching layers does not move the mask → the side is decided elsewhere (image side or later decoder) — supports a non-text shortcut.<br>• The control fails to reproduce the source mask → a bug in the intervention; results are not valid. |
| 15 *(notebook ready, not yet run)* | Image-Text Alignment — Cross-attention maps | Per-layer cross-attention of the text query over image tokens | • Record the cross-attention weights of every decoder layer (averaged over heads) for all four prompts of a subject.<br>• Map each weight from the image-token grid (12×12×12 per patch, ordered height-width-depth) back into voxel space, with the same Gaussian patch weighting as inference.<br>• For each prompt and layer, compute the attention density inside each of the four ground-truth regions relative to the volume average.<br>• Also record the attention centre of mass along the right-left axis against each organ's centroid. | • Rows 1 and 4 used final masks or logits, so the internal word-region link is untested.<br>• The cross-attention is where the text query reads the image, so it is the most direct evidence for alignment. | Attention enrichment (density in a region ÷ mean density) for matched, wrong-side and other-organ regions, per layer; centre of mass nearer to the named organ or not. | • Higher enrichment on the named organ than on the other side or other organs, in the late layers → the text query attends to the right place — supports genuine grounding.<br>• Equal enrichment on both sides → attention is side-blind; any side selection happens after attention.<br>• High enrichment on the organ regardless of the side word → attention finds the organ only (organ identity), leaving the side to another mechanism. |
| 16 *(notebook ready, not yet run)* | Fusion — Per-layer divergence | Left-vs-right query state distance at each decoder layer | • From the same run as row 15, record the query state after every decoder layer (plus the input state) for all four prompts.<br>• Per patch, compute the cosine and L2 distance between "left X" and "right X" at each layer, averaged over patches that overlap the organs and separately over all patches.<br>• Use the distance between "left X" and a different organ ("left Y") as a reference scale.<br>• Also compare the difference between the two prompts' attention maps per layer. | • The notebooks only saved the final fused vector, so where the left/right difference is created, kept or lost is unknown.<br>• A reference organ pair shows whether the side difference is small or large relative to what the layers already distinguish. | Left/right distance per layer, as a ratio to the input distance and to the organ-vs-organ reference distance. | • The left/right distance grows in the layers after attention and stays large → the fusion builds a side-specific state — supports genuine grounding.<br>• The distance is present at the input and shrinks through the layers → the side signal is washed out before the mask is produced.<br>• The distance is always small compared with the organ-vs-organ reference → side is a minor component of the representation. |

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
4. **Rows 12–16 (planned extensions)** — none has been run.
   - **Rows 12 and 13** (image flip; organ erasure / blank / other-subject image) need only
     preprocessed volumes and the public API, no changes inside the model. Do these first: they
     address the part of the core question that rows 1–11 cannot.
   - **Rows 14–16** (embedding intervention, cross-attention maps, per-layer divergence) need a
     custom sliding-window loop that reads and patches the decoder layers of
     `predictor.network.transformer_decoder`. Rows 15 and 16 share one recording run; row 14 reuses
     it for its patched variants. Verify the loop first by matching the predictor's own logits, and
     by checking that the full-state patch control reproduces the source prompt.

---

## Implementation status

Rows 1–11 are implemented, in four Kaggle notebooks, generated by
`scripts/build_groundedness_investigation_notebooks.py` (each has a paired `-kaggle.ipynb` /
`-v1.ipynb` file, byte-identical to each other):

| Notebook (`-kaggle`/`-v1` pair) | Dataset | Subjects | Organs/subject | Prompts/organ | **Total prompts** |
|---|---|---|---|---|---|
| `voxtell-amos-groundedness-investigation` | AMOS22 MRI | 10 | 4 (L/R kidney, L/R adrenal) | 25 | **1,000** |
| `voxtell-oasis-groundedness-investigation-batch1` | Neurite-OASIS | 5 | 10 (5 pairs × L/R) | 25 | **1,250** |
| `voxtell-oasis-groundedness-investigation-batch2` | Neurite-OASIS | 5 (disjoint from batch1 — together, all 10 available OASIS subjects) | 10 | 25 | **1,250** |
| `voxtell-oasis-groundedness-investigation-batch3` | Neurite-OASIS | 3 (0123, 0057, 0151 — the largest total mask over the 8 organs below; same subjects as some in batch1/2 but **different organs**) | 16 (8 new pairs × L/R: cerebellum white matter, thalamus, putamen, caudate, hippocampus, ventral DC, amygdala, pallidum) | 25 | **1,200** |

**Prompts/organ = 25** breaks down the same way in every notebook:
- 4 — main ablation (`ablation_masks_*.csv`: both present, position-only-removed,
  organ-only-removed, both removed)
- 20 — LIME (row 6: 2 roles × 10 samples)
- 1 — internal-layer embedding capture (section 13)

(Not counted above: a one-time smoke-test overhead of `4 × organs/subject` per notebook, run
once before the main loop — 16 for AMOS, 40 for each of OASIS batches 1 and 2, 64 for batch 3.)

The two OASIS notebooks' output CSVs are suffixed differently (`_neurite_oasis.csv` for batch 1,
`_neurite_oasis_batch2.csv` for batch 2) even though both read from the same on-disk
`samples/neurite_oasis/` folder — otherwise downloading both notebooks' results into the same
local folder would silently overwrite one batch's files with the other's.

**Rows 12–16 -- implemented, not yet run.** Two Kaggle notebooks, generated by
`scripts/build_mechanistic_notebooks.py` from the tested modules `scripts/voxtell_mechanistic.py` and
`scripts/voxtell_experiments.py`:

| Notebook | Dataset | Subjects | Organs | Rows |
|---|---|---|---|---|
| `voxtell-amos-mechanistic-experiments-kaggle` | AMOS22 MRI | 10 | L/R kidney, L/R adrenal gland | 12, 13, 14, 15, 16 |
| `voxtell-oasis-mechanistic-experiments-kaggle` | Neurite-OASIS | 10 | L/R cerebral white matter, L/R cerebellum cortex (the structures the model segments) | 12, 13, 14, 15, 16 |

- Row 12: flip along the right-left axis, prompt unchanged; scored as `follows_named_organ` vs. `follows_image_side`.
- Row 13: erase the named organ / both organs (filled from the surrounding background) / replace the image by noise.
- Row 14: five embedding blends (alpha -0.5 ... 1.5), single-layer patches of the query update (all 6 layers, both
  directions), patches from a layer onward, and a whole-state control that must reproduce the source prompt's mask.
- Row 15: cross-attention of each prompt over every organ region, per decoder layer (enrichment and attention centre).
- Row 16: cosine / relative L2 distance between the left and right query states per layer, with an organ-vs-organ reference.
- Checked locally on a tiny stand-in model (custom sliding-window loop == the predictor's own loop; whole-state patch
  reproduces the source; every CSV and the summary cell run end to end). **Not run on the real model yet.**
- The notebooks verify the custom loop against the real predictor first (section 3) and stop if it does not match.

**Hook prerequisite resolved:** rows 1, 3, 4 need **no `forward_hook`** — verified directly
against the installed `voxtell` v0.1.0 source. `predictor.predict_sliding_window_return_logits(
image, text_embeddings)` already returns the raw spatial pre-sigmoid logits (exactly "the
internal representation right before the final mask"), and `text_embeddings` is a plain tensor
argument from `predictor.embed_text_prompts()` — so row 3's causal embedding-swap test is just
calling the public API with a different embedding, no monkeypatching of voxtell's own code.

**Scope decisions made along the way:**
- 10 AMOS22 MRI subjects and 10 Neurite-OASIS subjects (split across 2 disjoint 5-subject
  notebooks), no rotation (`rot000` only) — reuses the existing `manifest_{dataset}.csv` per
  person, filtered to unrotated rows; no new manifest created. See the notebook/subject table
  above for exact counts.
- Word attribution (rows 5-11): **exact Shapley** (row 5's own "recommended default") over
  just **two** role-aware word slots — `position` and `organ` — not the full 6-slot space
  (the/in/this/image stay fixed; out of scope for this positional-bias-focused investigation).
  4 combos/organ (2²), computed exhaustively — no approximation needed, same for SHAP (row 7).
- The organ-ablated prompt's "organ removed" state replaces the real organ name with a real,
  nearby anatomical landmark from that SAME subject's scan (e.g. `"structure near the liver"`),
  picked by nearest fresh ground-truth centroid — not a vague generic placeholder. Landmark
  candidates are ANY other organ in the scan (bilateral or not), excluding only the target's
  own left/right pair. Verified against real `.nii.gz` volumes (not just manifest data) for
  both datasets — see section 8 of each notebook for the rationale.
- **Every** ablation combo's mask is saved (not just the baseline) — no experiment left out.
- Rows 6 (LIME) and 7 (SHAP) are both implemented with the real libraries (required for
  comparison against other papers), not skipped as redundant with row 5:
  - **Row 7 (SHAP)** reuses the exact combo data already computed — `shap.KernelExplainer`
    over the same role/metric lookup table, zero extra GPU cost. Verified locally that it
    matches the hand-rolled exact Shapley values (expected: SHAP internally enumerates the
    full 2^n space at this feature count).
  - **Row 6 (LIME)** is genuinely new inference — `lime.lime_text.LimeTextExplainer`, explaining
    `position` and `organ` separately (10 samples each), same landmark-based ablated prompts.
- Row 4 uses the raw, continuous, pre-sigmoid logit difference between the left- and
  right-prompt baseline predictions (not a thresholded-mask XOR) — catches divergence that
  would otherwise wash out at the 0.5 threshold, computed inline during the main run (section
  10) since it needs the raw logit volume before it's freed.
- Output is **CSV-only** — no PNGs rendered on Kaggle. Local visualization
  (`scripts/plot_groundedness_results.py`) decodes the RLE-encoded masks and plots overlays
  against the original images (already present locally in `<person>/samples/<dataset>/`), and
  draws word/role-importance bar charts (single-method or a 3-way exact-Shapley/SHAP/LIME
  comparison chart) — needs only numpy/pandas/matplotlib, no GPU/torch/voxtell.

**Output: 13 CSVs per notebook (52 total across all four notebooks), all under `results/`:**

| # | File | Row(s) | Contents |
|---|---|---|---|
| 1 | `baseline_predictions_{dataset}.csv` | *supports* 1, 3, 4 | Real (unablated) prediction per subject x organ: `gt_mask_rle_b64`/`pred_mask_rle_b64`, bbox, centroid, side, Dice. Not itself a numbered row's deliverable — it's the shared foundational data rows 1/3/4 are computed from (see the row-by-row map below), plus general "does the model even segment correctly" sanity plots. |
| 2 | `ablation_masks_{dataset}.csv` | 8-11 | Every 2^n word-ablation combo's mask + all four target metrics, per subject x organ x combo |
| 3 | `word_attribution_{dataset}.csv` | 5 | Exact Shapley value per role x organ x subject x metric |
| 4 | `shap_attribution_{dataset}.csv` | 7 | SHAP (KernelExplainer) value per role x organ x subject x metric, plus a `matches_exact_shapley` check |
| 5 | `lime_attribution_{dataset}.csv` | 6 | LIME weight per word x organ x subject |
| 6 | `row1_image_text_alignment_{dataset}.csv` | 1 | Matched vs. mismatched IoU |
| 7 | `row2_embedding_output_regression_{dataset}.csv` | 2 | Embedding-distance (cosine) vs. output-distance (centroid, not Dice — see notebook comment) points |
| 8 | `row2_regression_fit_{dataset}.csv` | 2 | Fitted slope/intercept/R² from those points, plus a plain-language interpretation |
| 9 | `row3_causal_embedding_swap_{dataset}.csv` | 3 | Causal side-agreement |
| 10 | `row4_fused_feature_diff_{dataset}.csv` | 4 | Raw pre-sigmoid **logit** diff (not a mask XOR) between left/right baseline predictions, per bilateral pair |
| 11 | `internal_representations_{dataset}.csv` | *supports* 4/mechanistic | Raw text + fused (post cross-attention) query embeddings per subject x organ, via a real `forward_hook` |
| 12 | `sidedness_analysis_{dataset}.csv` | *supports* 4/mechanistic | Cosine similarity between different organ-pairs' (right-left) embedding-difference vectors, in text and fused space |
| 13 | `sidedness_summary_{dataset}.csv` | *supports* 4/mechanistic | One row: mean cosine similarity per space, delta, plain-language interpretation of whether fusion learned a shared "sidedness direction" |

**Row-by-row result location** (every row 1-11 traced to its exact file/column, so coverage is
never ambiguous):

| Row | Task | Where the result is |
|---|---|---|
| 1 | Image-text alignment (matched vs. mismatched IoU) | `row1_image_text_alignment_{dataset}.csv` — `iou_matched`, `iou_mismatched`, `genuine_grounding` |
| 2 | Embedding-distance vs. output-distance regression | `row2_embedding_output_regression_{dataset}.csv` — raw (cosine-distance, centroid-distance) points; `row2_regression_fit_{dataset}.csv` — the fitted `slope`/`intercept`/`r_squared` plus a plain-language `interpretation` |
| 3 | Causal embedding-swap test | `row3_causal_embedding_swap_{dataset}.csv` — `left_prompt_predicts_left_side`, `right_prompt_predicts_right_side`, `causally_grounded` |
| 4 | Fused feature-map spatial diff | `row4_fused_feature_diff_{dataset}.csv` — `logit_diff_mean_abs`, `logit_diff_max_abs`, `logit_diff_argmax_crop_vox_0/1/2`, `frac_voxels_class_flip` (raw pre-sigmoid logit diff, not a mask XOR) |
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