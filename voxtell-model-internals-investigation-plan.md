# VoxTell Model-Internals Investigation Plan: Image vs. Text vs. Fusion, Memorization vs. Learning

## 0. The core question and the four hypotheses to distinguish

> **When VoxTell segments "the right kidney," what is actually driving the output — the anatomy
> it sees in the image, the word "right," the word "kidney," or a memorized shape/location prior
> from training?**

Every use case below is designed to produce evidence for or against one of four competing
explanations. Keep these four in view — each table row states which one it moves the needle on:

| Hypothesis | Statement | What would prove it |
|---|---|---|
| **H1 — Genuine grounding** | The model reads the position word, locates the correct side in the image, and segments accordingly. | Text encoder separates left/right (Section 1); that separation causally changes the output (Section 4); image-text alignment holds up under mismatched pairs (Section 3). |
| **H2 — Organ-name-only shortcut** | The position word ("left"/"right") is close to decorative. The model segments "kidney" from the name and picks a side based on image content or a fixed prior, not the word. | Position-word embedding distance is small relative to organ-name distance (Section 1, sharpened by Section 5's exact word-attribution); swapping only the position word barely changes output (Section 4); name-free position-only prompts fail (Section 1, T4). |
| **H3 — Training-distribution memorization** | The model relies on a memorized canonical shape/location prior rather than learning to track the anatomy actually present, so it falls back to a fixed pose expectation instead of the real one. | Canonical-pose-pull coefficient (Section 2, I2) drifts away from tracking true anatomy; image-text alignment (Section 3, AL2/AL3) is weak or inconsistent; cross-subject shape-template check (Section 2, I3) shows suspiciously templated output. |
| **H4 — Surface-token reactivity** | The text encoder reacts to *any* token change, not specifically to spatial semantics — so "left vs. right" distance looks meaningful only because it's *a* token change, not because it's a *position* token change. | Nonsense-token substitution shifts the embedding about as much as the real left/right swap (Section 1, T5), and Section 5's word-attribution scores don't concentrate on the position word specifically. |

These aren't mutually exclusive — e.g., H2 and H3 can both be true at once (the model is organ-name-driven *and* pose-memorized). The tables below are built so each row's result maps cleanly onto one or more of these.

**What's already confirmed** (VoxTell CVPR 2026 paper, Appendix A — via `positional-bias-benchmark-plan.md`):
- Backbone: plain ResEncL CNN, no rotation-equivariant components.
- Decoder: no positional embeddings anywhere; position is only implicit in the CNN feature map.
- Rotation augmentation during training: ~30°/axis. Left-right mirroring deliberately excluded.

**Access confirmed available** (from `voxtell-amos-rotation-multiangle-kaggle.ipynb`):
- `predictor.text_backbone` (`Qwen/Qwen3-Embedding-4B`) and `predictor.embed_text_prompts(text_prompts)` — text embeddings are directly extractable.
- `predictor.predict_single_image(img, organs)` — the only documented entry point into image/fusion/decoder. Anything beneath the final mask needs a `forward_hook` on internal `nn.Module`s inside `voxtell.inference.predictor` — an open discovery task (see Section 7), not yet confirmed hookable.

**Data policy for this plan:** everything below runs off the raw volumes directly
(`faria/samples/<dataset>/volumes/*_image.nii.gz` / `*_label.nii.gz`, with the manifest CSVs used
only for their `prompt_text` / `organ` / `label_path` columns — i.e., as an index into inputs and
ground truth, not as a source of precomputed predictions). No step below reuses the existing
`predictions_voxtell_*_rotation_*.csv` output files; every Dice, centroid, or mask-size number used
here is produced fresh via `predict_single_image` for this analysis specifically.

---

## 0.5. Training-data provenance check — done, and it matters

Every hypothesis in Section 0 assumes the model is generalizing (well or badly) to subjects it
never saw during training. That assumption doesn't hold cleanly for this repo's two evaluation
datasets — checked directly against VoxTell's own paper (Appendix B.1, "Large-Scale Multi-Modality
Dataset," the table of all 158 datasets in VoxTell's training corpus; arXiv:2511.11450v1, PDF pages
23-24):

| # | Finding | Evidence | What it means |
|---|---|---|---|
| P1 | **OASIS is in VoxTell's training corpus** | Appendix B.1 table entry: `OASIS [143] 436 3 MRI Brain Regions`, citing Marcus et al.'s cross-sectional MRI OASIS release — the same underlying source **Neurite-OASIS** (used throughout this repo's evaluation) derives from. Same modality (MRI), same anatomical region (brain). | **High risk of train/eval overlap for every Neurite-OASIS result in this plan and in `positional-bias-benchmark-plan.md`.** A Dice score or correct laterality call on Neurite-OASIS is *not* clean evidence of generalization — the model may simply have seen this dataset's poses during training. This caveat should attach to every OASIS-based finding, not just the ones in this document. |
| P2 | **AMOS is partially in the training corpus — CT only, not confirmed for MRI** | Appendix B.1 table entry: `AMOS [82] 360 16 CT Abdominal Organs`. No separate AMOS-MRI row appears anywhere in the 158-dataset table. | This repo evaluates **AMOS22-MRI**, not the CT arm. Lower direct-overlap risk than OASIS for this specific evaluation subset — but not a clean negative either, since AMOS22's CT and MRI arms were collected from overlapping patient cohorts by the same challenge, so some indirect familiarity (similar anatomy/scanner distribution, if not the literal MRI volumes) can't be ruled out from the paper text alone. |
| P3 | **Practical implication for prioritization** | — | Given P1/P2, treat **AMOS22-MRI as the closer-to-held-out dataset of the two** — prioritize it when a finding needs to be defended as evidence of genuine generalization rather than recall. Any finding that only replicates on Neurite-OASIS and not on AMOS22-MRI should be reported as *consistent with* memorization, not just consistent with a generic pose prior. |

**Why this mattered to check first:** a memorized-prior finding from Section 2/3 (H3) means something
different depending on whether the underlying data was in training at all — "the model falls back
to a canonical pose because it memorized *this dataset's* poses" (now a live possibility for OASIS)
is a stronger and more specific claim than "the model has a generic pose prior." P1-P3 are what let
you tell those two apart, and they should be cited alongside any H3-supporting result that uses
Neurite-OASIS data.

---

## 1. Text-side use cases — what is the text encoder actually encoding?

**Important scope note:** this section only probes the *text encoder in isolation* —
`embed_text_prompts` never touches an image. It answers "does the encoder produce a different
embedding for 'left' vs. 'right'," **not** "does that word actually change the segmentation."
A word can register here and still be ignored downstream, or vice versa in theory. This section
exists to rule out the cheapest failure mode first (if the text side can't even tell "left" from
"right," nothing downstream can recover it) — the actual word-to-segmentation-output question is
answered in **Section 5**, which re-runs real image inference per word perturbation. Read the two
together, don't treat Section 1 alone as an answer to "does this word matter."

Goal here: isolate what token(s) in the prompt the *text encoder* treats as meaningful, before
asking whether that meaning survives into the output. No image inference needed — cheapest phase,
and it gates Sections 3-4. Treat T1/T2 here as the coarse, two-point version of the question;
Section 5 gives you the fine-grained, per-word, *image-grounded* version.

| # | Use case | Setup | What's held fixed | What's varied | Metric | Supports H1/genuine position use | Supports H2 (name-only) / H4 (surface-token) |
|---|---|---|---|---|---|---|---|
| T1 | **Left vs. right, same organ** | Embed `"Segment the left kidney..."` and `"Segment the right kidney..."` via `embed_text_prompts`. | Organ name, sentence template | Position word only | Cosine distance | Distance is clearly non-trivial (well above tokenizer noise floor). | Distance is near zero — position word barely registers. |
| T2 | **Organ-name swap, same side (control/scale reference)** | Embed `"Segment the left kidney..."` vs `"Segment the left spleen..."`. | Position word, template | Organ name only | Cosine distance, compared directly against T1's value | T1 distance ≈ T2 distance (position and organ carry comparable weight). | T2 distance >> T1 distance — organ identity dominates. |
| T3 | **Full-vocabulary clustering** | Embed every organ × side combination in the manifests. PCA/UMAP to 2D. | — | Organ, side | Whether left/right pairs split into position-defined clusters or sit as minor within-organ jitter | Position forms a visible, separate axis of variation across the whole vocabulary. | Points cluster almost entirely by organ; left/right pairs sit right on top of each other. |
| T4 | **Position-only, name-free prompt** | Embed `"the structure on the left side of the abdomen"` vs. the right-side equivalent — no organ name at all. | Everything except side | Side word, no organ name | Cosine distance, compared to T1 | Name-free distance is comparable to or larger than T1. | Name-free distance collapses toward zero — position alone means almost nothing without an organ name. |
| T5 | **Nonsense-token control** | Re-embed T1's pair, but swap "left" for a semantically empty token of similar length, instead of "right." | Organ name, template, token count | Token identity | Compare T5's distance-from-original to T1's left→right distance | T5's nonsense-swap distance is clearly *smaller* than T1's real swap. | T5's distance ≈ T1's — the encoder reacts to any token perturbation, not spatial meaning specifically. |

---

## 2. Image-side use cases — how much does the visual path track the actual anatomy?

Goal: isolate what the *image* side contributes, holding the prompt fixed, so output behavior can
be attributed to image representation rather than a text-side failure. All rows use fresh
`predict_single_image` calls on the on-disk volumes.

| # | Use case | Setup | What's held fixed | What's varied | Metric | Supports H1 (image tracks real anatomy) | Supports H3 (image representation is pose-bound) |
|---|---|---|---|---|---|---|---|
| I1 | **Same prompt, image rotated across the sweep** | Fixed prompt, fresh inference across all on-disk angles for that organ/subject. | Prompt | Rotation angle | Dice vs. angle, computed fresh against each angle's own GT | Degradation is gradual and organ-appropriate, tracking anatomy at each angle. | Sharp degradation that doesn't track anatomy smoothly — confirms the image path itself, not just the text path, is where the sensitivity lives, since the prompt never changed. |
| I2 | **Canonical-pose bias in predicted centroid (quantified)** | From I1's fresh predictions, fit per organ: `predicted_centroid(angle) = a · true_centroid(angle) + (1−a) · canonical_centroid_at_0°`, via regression. | Prompt | Angle | Fitted coefficient `a` per organ | `a` near 1 — predicted centroid tracks the *true*, rotated anatomy at every angle. | `a` trending toward 0 as angle increases — falls back to where the organ *would* sit at 0°, not the anatomy actually in front of it. |
| I3 | **Cross-subject shape-template check** | Fixed organ + side prompt, fresh inference across multiple subjects at low and high rotation angles. | Prompt, angle | Subject | Shape similarity (Dice after alignment) across subjects | Shape similarity only as high as real anatomical similarity predicts. | Shape similarity stays suspiciously high at high angles even as per-subject Dice-vs-GT collapses — templated shape output. |
| I4 | **Zero/mean text-embedding ablation** | Replace the text embedding with an all-zero vector or across-vocabulary mean, keep image/angle fixed, run fresh inference. | Image, angle | Text embedding (real vs. ablated) | Output mask vs. the real-prompt output, same image | Output changes substantially without a real text embedding. | Output barely changes — decoder already leans almost entirely on image content. |

---

## 3. Image-text alignment — does the text embedding actually correspond to what's in the image?

This is a distinct question from Section 4 (fusion/does the *decoder* use the signal). Alignment
asks: independent of what the decoder does with it, is there a coherent correspondence between a
prompt's embedding and the image region it should point to — the kind of thing a retrieval score or
contrastive alignment score would normally measure. VoxTell isn't published as a CLIP-style dual
encoder, so there's no ready-made similarity score; this section builds one from observable
behavior instead.

| # | Use case | Setup | Metric | Supports H1 (aligned) if... | Supports H2/H3 (poorly aligned) if... |
|---|---|---|---|---|---|
| AL1 | **Prompt-ranking retrieval proxy** | Fix one image. Run every organ×side prompt that's anatomically plausible for that volume through `predict_single_image` individually. Rank prompts by an output-confidence proxy (predicted mask voxel count, or the raw probability map's max/mean confidence before thresholding, if the API exposes it). | Whether the *correct* organ+side prompt for a given visible structure ranks at or near the top of that ranking. | The correct prompt reliably produces the highest-confidence, best-localized output relative to mismatched prompts on the same image. | Multiple mismatched prompts (wrong side, or even wrong organ) produce comparably confident output — the model can't discriminate a matching prompt from a non-matching one via output confidence alone. |
| AL2 | **Matched vs. mismatched pair gap** | For each (image, correct-prompt) pair, also run 2-3 deliberately mismatched prompts (wrong side; a different but plausible organ) on the *same* image. | Distribution of confidence/Dice for matched vs. mismatched pairs — a gap (like a CLIP alignment margin). | Clear, consistent gap: matched pairs score higher than mismatched ones across subjects. | Gap is small or inconsistent — alignment isn't reliably discriminating matched from mismatched content. |
| AL3 | **Alignment vs. rotation angle** | Repeat AL2 across the rotation sweep (fresh inference). | Matched-vs-mismatched gap as a function of angle. | Gap holds steady across angles. | Gap shrinks or disappears as rotation increases — alignment itself (not just decoder use of it) degrades with angle, pointing the failure further upstream than fusion. |
| AL4 | **Fused-representation correspondence (if hookable)** | If Section 7's hook point is found: for a matched pair, check whether the fused feature map's high-activation region spatially overlaps the ground-truth structure location; for a mismatched pair, whether it still does (wrongly) or goes diffuse. | Spatial overlap (e.g. IoU between high-activation region and GT bounding box) | High overlap only for matched pairs. | High overlap even for mismatched pairs (image dominates regardless of prompt) — or low overlap even for matched pairs (no real correspondence exists to align in the first place). |

---

## 4. Fusion-side use cases — does the positional signal, once it exists in text, actually reach and move the output?

Goal: causally test whether a text-side distinction (Section 1/5) and an alignment signal (Section
3) actually propagate through fusion into a different mask.

| # | Use case | Setup | Metric | Supports H1 (fusion listens to text) | Supports H2/H3 (fusion ignores or barely uses text) |
|---|---|---|---|---|---|
| F1 | **Paired-prompt output divergence (core test)** | Same image, fresh runs of `"left X"` and `"right X"`. | Dice/centroid distance *between the two output masks* (not vs. ground truth) | Clearly different masks, tracking the two true locations. | Nearly identical masks, despite Section 1 showing the text embeddings *do* differ. |
| F2 | **Embedding-distance vs. output-distance regression** | Plot Section 1's text-embedding cosine distance (x-axis) against F1's mask-difference metric (y-axis), across all bilateral organ/subject pairs. | Regression slope | Positive slope. | Flat/near-zero slope — output doesn't track text difference at all. |
| F3 | **Embedding-swap ablation (causal, strongest test in this table)** | Compute image features for `"left X"`, substitute in the text embedding computed for `"right X"` before fusion (needs the Section 7 hook). | Whether the output mask shifts toward the right-side location | Output shifts toward "right." | Output doesn't move — fusion provably isn't using the swapped embedding. |
| F4 | **Fused feature-map spatial diff (if hookable)** | Log the fused feature map for left- vs. right-prompt runs on the same image; compute a spatial difference map. | Where left/right first diverge (or fail to) | Divergence appears early, localizes near the correct side. | No meaningful divergence anywhere in the fused representation. |
| F5 | **Wrong-organ / absent-organ prompt-invariance** | Fixed image, run correct prompt plus: wrong organ name, wrong side, and an organ genuinely absent from the volume. | Whether output tracks the prompt or stays roughly constant | Output changes appropriately per prompt. | Output stays "something plausible" regardless of prompt. |

---

## 5. Text-to-image contribution — is each word actually related to the segmentation, or not? (LIME / SHAP)

**This is the section that directly answers "is this word related to the image segmentation, or
not."** Unlike Section 1 (text-encoder-only, no image), every score in this section comes from
re-running real inference on the real image with the word perturbed, then measuring the change in
the *actual predicted mask* — so a word's importance here is measured strictly in terms of its
effect on segmentation output, not on how different it looks in embedding space. A word can look
very different in Section 1's embedding space and still score near-zero here (text-side distinction
that never reaches the output — a fusion/decoder failure), or the reverse in principle. This is
where you get a real, per-word, image-grounded contribution score.

This replaces T1/T2's coarse two-point comparison with per-word importance scores. Yes, LIME and
SHAP both apply here, and they're a better fit than they usually are for text, for one specific
reason: VoxTell prompts
are short and templated (typically 6-9 words: `"Segment"`, `"the"`, `"left"`/`"right"`, organ name
(1-2 words), `"in"`, `"this"`, `"image"`). At that length, you don't need LIME's random-sampling
approximation or KernelSHAP's sampling approximation — **exact leave-one-word-out / full Shapley
enumeration is cheap** (a prompt with 8 words has 2^8 = 256 subsets; even at ~1-2s of inference per
call, that's minutes, not hours, per (image, prompt) pair).

**Recommended approach, in order of rigor:**

| Method | What it does | When to use it here |
|---|---|---|
| **Exact word-ablation / brute-force Shapley** (recommended default) | Enumerate all 2^n subsets of the prompt's content words (or at least all leave-one-out and leave-two-out subsets if n is a bit large), run each through the model, compute the exact Shapley value per word from the resulting output values. | Default choice given how short these prompts are — gives exact, not approximate, per-word attribution. No library dependency beyond basic combinatorics; `shap.explainers.Exact` or `shap.explainers.Permutation` (via `shap.Explainer(f, masker)` with a `shap.maskers.Text` masker) will do this enumeration for you if you'd rather not hand-roll it. |
| **LIME (`lime.lime_text.LimeTextExplainer`)** | Fits a local linear surrogate model on randomly perturbed (word-masked) versions of the prompt, weighted by similarity to the original. | Reasonable if you want a quick, well-known library and don't need exactness; less precise than brute-force at this prompt length, and LIME's classifier-oriented API needs adapting for a continuous target (see below). |
| **KernelSHAP / PartitionExplainer** | Approximate Shapley values via sampled coalitions, standard for longer text. | Not really needed here since exact enumeration is affordable — use it only if you extend this to longer, paragraph-style prompts later. |

**Defining the target function** (both LIME and SHAP need a scalar `f(prompt) → number`; segmentation output is a 3D mask, so pick one of these per experiment, all computed fresh from `predict_single_image`, never from the CSVs):

| Target | Definition | What it isolates |
|---|---|---|
| `f_dice_gt` | Dice(predicted mask for perturbed prompt, ground-truth mask) | Which words matter for *getting the answer right* — the accuracy-facing view. |
| `f_dice_self` | Dice(predicted mask for perturbed prompt, predicted mask for the *original, unperturbed* prompt) | Which words matter for *changing the output at all* — the stability/sensitivity view, doesn't need ground truth, so it also works on subjects without labels. |
| `f_centroid_shift` | Signed displacement of the predicted mask's centroid along the left-right axis, relative to the original prompt's centroid | The most direct test for laterality specifically: **which word, when removed, moves the predicted mask's side.** This is the single most on-point metric for your original question ("does it take the anatomy or the L/R word"). |
| `f_volume` | Predicted mask voxel count | Detects collapse to empty/near-empty prediction when a word is removed — a different failure mode (model "gives up") from moving to the wrong side. |

**Perturbation/masking rule for word removal:** don't just delete a token (it breaks grammar in
ways that could confound the result — e.g. removing "left" from "Segment the left kidney in this
image" leaves a still-grammatical sentence, but removing "kidney" doesn't). Use role-aware
templated ablation instead:
- Remove the position word → `"Segment the kidney in this image."`
- Remove the organ name, replace with a generic noun → `"Segment the left structure in this image."`
- Remove a function word (`"the"`, `"in"`, `"this"`, `"image"`) → drop cleanly, these are the
  low-importance control rows you'd expect near-zero attribution for, and they double as a sanity
  check that the attribution method isn't just picking up sentence-length artifacts.

**Output of this section:** one attribution table per (dataset, organ) — rows = words in the
template, columns = the four target functions, cell = Shapley/LIME importance score. Rank the
position word against the organ-name word directly, per organ, per dataset. This is what makes
Section 0's H2 vs. H1 distinction quantitative rather than a single anecdotal T1/T2 comparison —
and running it across many organs/subjects lets you report a distribution (e.g. "position word
attribution was in the bottom quartile of all content words in 80% of cases"), not just one number.

---

## 6. Which word contributes more? — combined direct-answer table

Cross-references Sections 1-5 into one place.

| Contribution source | Evidence from | What a strong result looks like | What a weak result looks like |
|---|---|---|---|
| **Organ name (e.g. "kidney")** | T2, F5, Section 5 (organ-name row of the attribution table) | Large T2 distance; F5 shows output changes correctly when organ name changes; high Shapley/LIME attribution. | — |
| **Position word (e.g. "left"/"right")** | T1, T4, F1, F3, Section 5 (position-word row) | Non-trivial T1/T4 distance *and* F1/F3 show output actually moving *and* high attribution in Section 5, especially on `f_centroid_shift`. | Any of these individually weak is a warning sign; all weak together (T1 small, F1 flat, Section 5 attribution low) is the strongest case that position is decorative. |
| **Image content (anatomy actually visible)** | I1, I2, I4 | I2's `a` stays near 1 across angles; I4 shows output collapses without real text. | I2's `a` falls toward 0 as angle increases, and/or I4 shows output barely changes without text. |
| **Surface token identity (not meaning)** | T5 | T5 distance << T1 distance. | T5 distance ≈ T1 distance. |
| **Image-text alignment itself** | AL1-AL4 | Consistent matched-vs-mismatched gap across conditions. | Gap shrinks or never existed. |

**How to read the combination:** if I2 shows `a` collapsing toward 0 for a given organ, *and* AL2/AL3
show the alignment gap for that organ is weak or degrading, *and* Section 5 shows the position
word's attribution score is consistently low relative to the organ name — that combination is the
strongest possible case for: **the model relies on a memorized spatial/shape prior for that organ,
and position words contribute much less than organ identity or the image's own default pose
expectation.**

---

## 7. Open discovery task before Section 3's AL4 and Section 4's F3/F4 can run

`predict_single_image` is the only documented entry point. AL4, F3, F4 need a hook inside the
installed `voxtell` package (`voxtell.inference.predictor` and whatever segmentation network class
it wraps) to intercept the text embedding and/or fused feature map. This is unverified — scope it
by reading the installed package source directly (`python -c "import voxtell; print(voxtell.__file__)"`,
then walk from there) before writing any ablation code against it. If no such hook point exists
cleanly (e.g. fusion happens inside a compiled/traced graph), those three rows may need to fall back
to input-level surrogates only (F1, F2, F5, AL1-AL3), which is still enough to answer "does fusion
listen to text" and "is text-image alignment real" observationally, just not causally.

---

## Execution order

0. **Section 0.5 (P1-P3, training-data provenance)** — absolute first step, before any inference is
   run. Cheapest step in the whole plan (reading a paper section) and reframes how every later
   result should be interpreted and reported.
1. **Section 1 (T1-T5)** — cheap, no GPU image inference, gates interpretation of Sections 3-4.
2. **Section 5 (word attribution)** — can start as soon as Section 1 is done; needs fresh inference
   but no hook discovery, and directly produces your headline "which word matters" numbers.
3. **Section 2 (I1-I4)** and **Section 3 (AL1-AL3)** — new inference passes across the rotation
   sweep and across matched/mismatched prompt pairs; independent of each other, can run in parallel.
4. **Section 4 (F1, F2, F5)** — needs one new paired-prompt inference pass; independent of the hook.
5. **Section 3's AL4, Section 4's F3/F4** — last, blocked on Section 7's hook discovery.
6. **Section 6** — write-up step, combines everything above; no new inference.
