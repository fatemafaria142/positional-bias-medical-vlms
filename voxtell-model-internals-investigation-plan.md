# VoxTell Model-Internals Investigation Plan: Image vs. Text vs. Fusion, Memorization vs. Learning

## 0. The core question and the four hypotheses to distinguish

Behavioral evidence already shows the pattern: VoxTell holds up from 0° to ~30° rotation, then
degrades sharply. That single fact is doing more work than it looks like — see Section 1. The
question this document is built to answer, precisely:

> **When VoxTell segments "the right kidney," what is actually driving the output — the anatomy
> it sees in the image, the word "right," the word "kidney," or a memorized shape/location prior
> from training? And does its apparent rotation-tolerance reflect real geometric understanding, or
> just the training augmentation range?**

Every use case below is designed to produce evidence for or against one of four competing
explanations. Keep these four in view — each table row states which one it moves the needle on:

| Hypothesis | Statement | What would prove it |
|---|---|---|
| **H1 — Genuine grounding** | The model reads the position word, locates the correct side in the image, and segments accordingly, at least approximately independent of training augmentation range. | Text encoder separates left/right (Section 2); that separation causally changes the output (Section 4); performance degradation doesn't align suspiciously with the training augmentation boundary. |
| **H2 — Organ-name-only shortcut** | The position word ("left"/"right") is close to decorative. The model segments "kidney" from the name and picks a side based on image content or a fixed prior, not the word. | Position-word embedding distance is small relative to organ-name distance (Section 2); swapping only the position word barely changes output (Section 4); name-free position-only prompts fail (Section 2, A4). |
| **H3 — Training-distribution memorization** | The model performs well only inside the augmentation range it was trained on (~30°/axis), because it memorized the training pose distribution rather than learning rotation-robust geometry. | Degradation onset angle matches the augmentation boundary closely, consistently across organs/datasets (Section 1 — headline test). |
| **H4 — Surface-token reactivity** | The text encoder reacts to *any* token change, not specifically to spatial semantics — so "left vs. right" distance looks meaningful only because it's *a* token change, not because it's a *position* token change. | Nonsense-token substitution shifts the embedding about as much as the real left/right swap (Section 2, A5). |

These aren't mutually exclusive — e.g., H2 and H3 can both be true at once (the model is organ-name-driven *and* pose-memorized). The tables below are built so each row's result maps cleanly onto one or more of these.

**What's already confirmed** (VoxTell CVPR 2026 paper, Appendix A — via `positional-bias-benchmark-plan.md`):
- Backbone: plain ResEncL CNN, no rotation-equivariant components.
- Decoder: no positional embeddings anywhere; position is only implicit in the CNN feature map.
- Rotation augmentation during training: **~30°/axis**. Left-right mirroring deliberately excluded.

**Access confirmed available** (from `voxtell-amos-rotation-multiangle-kaggle.ipynb`):
- `predictor.text_backbone` (`Qwen/Qwen3-Embedding-4B`) and `predictor.embed_text_prompts(text_prompts)` — text embeddings are directly extractable.
- `predictor.predict_single_image(img, organs)` — the only documented entry point into image/fusion/decoder. Anything beneath the final mask needs a `forward_hook` on internal `nn.Module`s inside `voxtell.inference.predictor` — an open discovery task (see Section 6), not yet confirmed hookable.

---

## 1. Headline test — does the 0°→30° cliff line up with the training augmentation boundary?

Run this first. It's the cheapest, highest-leverage test you have, and it reframes everything else:
if the empirical breakdown point matches the known training augmentation range, that alone is a
strong, direct argument for H3 (memorization of training pose distribution) over H1 (genuine
rotation-robust geometric understanding) — no internals access required, just the sweep data
you already have.

| # | Use case | Setup | Metric | Supports H1 (genuine) if... | Supports H3 (memorized) if... |
|---|---|---|---|---|---|
| R1 | **Breakdown-angle vs. augmentation-boundary correlation** | From the existing rotation-sweep CSVs (`predictions_voxtell_*_rotation_*.csv`), for each organ compute the angle at which Dice first drops below a fixed threshold (e.g. 90% of its 0° value). | Breakdown angle per organ, plotted against the known ~30°/axis augmentation range. | Breakdown angle is well past 30° and roughly *uniform* across organs regardless of size/frequency in training. | Breakdown angle clusters tightly *at or just past* 30°, consistently across organs and both datasets (AMOS22-MRI, Neurite-OASIS). |
| R2 | **Fine-grained boundary localization** | Re-run the sweep at 10-15° steps (this is already flagged as pending, finer-grained, in the existing plan) specifically in the 20-40° window. | Shape of the Dice-vs-angle curve in that window: smooth decay vs. a knee/cliff. | Smooth, gradual decay with no sharp knee near 30° — consistent with a general robustness limit, not a hard-coded boundary. | A visible knee right around 30° — a hard cutoff shape is the signature of "augmentation covered this range and nothing more," not a learned geometric limit that would decay more gradually. |
| R3 | **Per-organ boundary vs. organ frequency/size** | Cross-reference R1's per-organ breakdown angle against organ size and how often that organ appears across the training-relevant literature/manifests (proxy: voxel count, frequency in manifest). | Correlation coefficient between breakdown angle and organ size/frequency. | No correlation — breakdown angle is architecture/training-range driven uniformly, not tied to how "familiar" an organ is. | Positive correlation — larger/more common organs tolerate rotation slightly better (more augmented exposure), smaller/rarer ones break down earlier — a memorization-frequency signature layered on top of the general 30° boundary. |

**Why this table goes first:** this is the single fastest way to test H3 directly, using data you
already have (the CSVs), before touching model internals at all. If R1/R2 come back cleanly aligned
with 30°, that's your headline finding, and it reframes Sections 2-4 below as "why does the model
fail to generalize past its training range" rather than "does the model understand position at all"
— two related but distinct questions worth keeping separate in your write-up.

---

## 2. Text-side use cases — what is the text encoder actually encoding?

Goal: isolate what token(s) in the prompt the *text encoder* treats as meaningful, before asking
whether that meaning survives into the output. No image inference needed — cheapest phase, and it
gates Sections 3-4 (if the text side doesn't separate two prompts, nothing downstream can use a
distinction that was never made).

| # | Use case | Setup | What's held fixed | What's varied | Metric | Supports H1/genuine position use | Supports H2 (name-only) / H4 (surface-token) |
|---|---|---|---|---|---|---|---|
| T1 | **Left vs. right, same organ** | Embed `"Segment the left kidney..."` and `"Segment the right kidney..."` via `embed_text_prompts`. | Organ name, sentence template | Position word only | Cosine distance | Distance is clearly non-trivial (well above tokenizer noise floor). | Distance is near zero — position word barely registers. |
| T2 | **Organ-name swap, same side (control/scale reference)** | Embed `"Segment the left kidney..."` vs `"Segment the left spleen..."`. | Position word, template | Organ name only | Cosine distance, compared directly against T1's value | T1 distance ≈ T2 distance (position and organ carry comparable weight). | T2 distance >> T1 distance — organ identity dominates, position is a minor perturbation. **This is the direct test of "which word is contributing more."** |
| T3 | **Full-vocabulary clustering** | Embed every organ × side combination in the manifests. PCA/UMAP to 2D. | — | Organ, side | Whether left/right pairs split into position-defined clusters or sit as minor within-organ jitter | Position forms a visible, separate axis of variation across the whole vocabulary. | Points cluster almost entirely by organ; left/right pairs sit right on top of each other. |
| T4 | **Position-only, name-free prompt** | Embed `"the structure on the left side of the abdomen"` vs. the right-side equivalent — no organ name at all. | Everything except side | Side word, no organ name | Cosine distance, compared to T1 | Name-free distance is comparable to or larger than T1 (position alone is meaningful language to the encoder). | Name-free distance collapses toward zero — without an organ name to modulate, "left"/"right" alone means almost nothing to the encoder. |
| T5 | **Nonsense-token control** | Re-embed T1's pair, but swap "left" for a semantically empty token of similar length (e.g. a rare/filler word), instead of "right." | Organ name, template, token count | Token identity (real position word vs. nonsense word) | Compare T5's distance-from-original to T1's left→right distance | T5's nonsense-swap distance is clearly *smaller* than T1's real swap — the encoder reacts specifically to spatial meaning, not just to "a word changed." | T5's distance is roughly the *same size* as T1 — the encoder is reacting to token perturbation in general (H4), and T1's "position sensitivity" is an artifact of that, not evidence of spatial understanding. |

**Read T1-T5 together, in this order:** T2 tells you which word matters more in the raw geometry
(directly answers "does it focus on 'kidney' more than 'right'"). T5 tells you whether even that
weaker position signal is really about *position*, or just about *some word changed*. T4 tells you
whether position can stand on its own without an organ name propping it up.

---

## 3. Image-side use cases — how sensitive is the visual path to rotation, independent of text?

Goal: isolate what the *image* side contributes, holding the prompt fixed, so a later Dice drop
can be attributed to image representation degradation rather than a text-side failure.

| # | Use case | Setup | What's held fixed | What's varied | Metric | Supports H1 (image tracks real anatomy) | Supports H3 (image representation is pose-bound) |
|---|---|---|---|---|---|---|---|
| I1 | **Same prompt, image rotated across the sweep** | Fixed prompt (e.g. `"Segment the right kidney..."`), run across all angles in the existing sweep (this is what the sweep CSVs already contain — reuse, don't re-run). | Prompt | Rotation angle | Dice vs. angle (already available) — but re-read specifically as "how much of Section 1's cliff is explainable by image path alone" | Degradation is gradual and organ-appropriate. | Degradation shape matches R1/R2's cliff — confirms the image path, not the text path, is where the augmentation-boundary sensitivity actually lives (since the prompt never changed across this whole sweep). |
| I2 | **Canonical-pose bias in predicted centroid (quantified)** | Fit, per organ: `predicted_centroid(angle) = a · true_centroid(angle) + (1−a) · canonical_centroid_at_0°`, via regression across the sweep. | Prompt | Angle | Fitted coefficient `a` per organ | `a` near 1 — predicted centroid tracks the *true*, rotated anatomy at every angle. | `a` trending toward 0 as angle increases past ~30° — the model increasingly falls back to where the organ *would* sit at 0°, i.e. a memorized canonical pose, not the anatomy actually in front of it. |
| I3 | **Cross-subject shape-template check** | For a fixed organ + side prompt, compare predicted mask *shape* (centered/aligned) across multiple different subjects at 0°, then again at 60°+. | Prompt, angle | Subject | Shape similarity (e.g. Dice after alignment) across subjects | Shape similarity across subjects is only as high as real anatomical similarity would predict. | Shape similarity is suspiciously high and *doesn't degrade* even as per-subject Dice-vs-ground-truth collapses post-30° — the model is outputting a templated shape regardless of what's actually in the rotated image. |
| I4 | **Zero/mean text-embedding ablation** | Replace the text embedding with an all-zero vector or the across-vocabulary mean, keep image and rotation angle as normal, run inference. | Image, angle | Text embedding (real vs. ablated) | Output mask vs. the real-prompt output, same image | Output changes substantially without a real text embedding — image alone isn't enough to produce a specific organ+side mask. | Output barely changes — the decoder is already leaning almost entirely on image content, i.e. doing organ-agnostic "segment the salient structure" regardless of what was asked, a strong image-only/memorized-prior signature. |

---

## 4. Fusion-side use cases — does the positional signal, once it exists in text, actually reach and move the output?

Goal: causally test whether a text-side distinction (confirmed or refuted in Section 2) propagates
through fusion into a different mask. This is where "is it image-focused or text-focused" gets a
direct, causal answer rather than an inferred one.

| # | Use case | Setup | Metric | Supports H1 (fusion listens to text) | Supports H2/H3 (fusion ignores or barely uses text) |
|---|---|---|---|---|---|
| F1 | **Paired-prompt output divergence (core test)** | Same image, run `predict_single_image` once with `"left X"`, once with `"right X"`. | Dice/centroid distance *between the two output masks* (not vs. ground truth) | Two prompts, same image → clearly different masks, tracking the two true locations. | Two prompts, same image → nearly identical masks, despite Section 2 showing the text embeddings *do* differ — fusion is discarding an available signal. |
| F2 | **Embedding-distance vs. output-distance regression** | Plot Section 2's text-embedding cosine distance (x-axis, per organ pair) against F1's mask-difference metric (y-axis). | Regression slope across all bilateral organ/subject pairs | Positive slope — more text-side separation predicts more output-side separation. | Flat/near-zero slope — output difference doesn't track text difference at all, regardless of how distinct the embeddings are. |
| F3 | **Embedding-swap ablation (causal, strongest test in this table)** | Compute image features for the `"left X"` prompt, but substitute in the *text embedding* computed for `"right X"` before fusion (requires a hook at the point `voxtell.inference.predictor` consumes the text embedding — see Section 6). | Whether the output mask shifts toward the right-side location | Output shifts toward the "right" location — fusion is causally driven by the text embedding, not just correlated with it. | Output doesn't move — fusion provably isn't using the text embedding for this decision, independent of how different the embeddings are. |
| F4 | **Fused feature-map spatial diff (if hookable)** | Once F3's hook point is found, log the fused feature map for left- vs. right-prompt runs on the same image; compute a spatial difference map. | Where in the network left/right first diverge (or fail to) | Divergence appears early and localizes near the correct anatomical side. | No meaningful divergence anywhere in the fused representation — pinpoints that the failure is at/before fusion, not in the decoder's use of an available signal. |
| F5 | **Wrong-organ / absent-organ prompt-invariance** | Fixed 0° image, run the correct prompt plus: wrong organ name, wrong side, and an organ genuinely absent from the volume. | Whether output tracks the prompt or stays roughly constant | Output changes appropriately per prompt — segments the right thing, or returns empty for absent organs. | Output stays "something plausible" regardless of prompt — a "there's always something here" memorized-prior signature, independent of what was actually asked. |

---

## 5. Which word contributes more? — direct answer table

This collapses the "does it focus on anatomy, the position word, or just the organ name" question
into one side-by-side view, cross-referencing Sections 2-4 so it can be read on its own.

| Contribution source | Evidence from | What a strong result looks like | What a weak result looks like |
|---|---|---|---|
| **Organ name (e.g. "kidney")** | T2 (embedding distance), F5 (wrong-organ prompt) | T2 distance is large; F5 shows output changes correctly when organ name changes. | — |
| **Position word (e.g. "left"/"right")** | T1, T4 (embedding distance, with/without organ name); F1, F3 (does it move the output) | T1/T4 distances are non-trivial *and* F1/F3 show the output actually moving. | T1 small and/or F1 flat — position word is present in the prompt but contributes ~nothing to the output. |
| **Image content (anatomy actually visible)** | I1, I2, I4 (rotation sweep, canonical-pose pull, zero-text ablation) | I2's `a` stays near 1 across angles; I4 shows output collapses without real text (image alone insufficient — i.e., image isn't just being segmented on its own, it needs the prompt to know *what*). | I2's `a` falls toward 0 post-30°, and/or I4 shows output barely changes without text — image path is doing most of the work, off a memorized prior, largely independent of the prompt. |
| **Surface token identity (not meaning)** | T5 (nonsense-token control) | T5 distance << T1 distance — the encoder is specifically meaning-sensitive. | T5 distance ≈ T1 distance — apparent "position sensitivity" is really just "a token changed." |

**How to read the combination for your specific claim** ("works 0-30°, blows away after"): if
Section 1 confirms the cliff aligns with the augmentation boundary, *and* I2 shows `a` collapsing
toward 0 past that same boundary, *and* F1 shows left/right masks staying similar even when T1
shows the text embeddings differ — that combination is the strongest possible case for: **the model
has a memorized, training-distribution-bound spatial prior, and position words contribute much less
than organ identity or the image's own default pose expectation.** Any one of those three alone is
suggestive; all three together is a real finding.

---

## 6. Open discovery task before Section 4's F3-F5 can run

`predict_single_image` is the only documented entry point. F3-F5 need a hook inside the installed
`voxtell` package (`voxtell.inference.predictor` and whatever segmentation network class it wraps)
to intercept the text embedding at the point it's consumed by fusion. This is unverified — scope it
by reading the installed package source directly (`python -c "import voxtell; print(voxtell.__file__)"`
then walk from there) before writing any ablation code against it. If no such hook point exists
cleanly (e.g. fusion happens inside a compiled/traced graph), F3-F5 may need to fall back to
input-level surrogates only (F1, F2, F5), which is still enough to answer "does fusion listen to
text" observationally, just not causally.

---

## Execution order

1. **Section 1 (R1-R3)** — first, cheapest, reuses existing CSVs, and reframes everything after it.
2. **Section 2 (T1-T5)** — cheap, no GPU inference, gates interpretation of Section 4.
3. **Section 3 (I1-I3)** reuse existing sweep data; **I4** and **Section 4's F1, F2, F5** need one
   new paired-prompt inference pass — straightforward with the existing `predict_single_image` setup.
4. **Section 4's F3, F4** last — blocked on Section 6's hook discovery.
