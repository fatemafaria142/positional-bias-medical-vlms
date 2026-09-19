# VoxTell Model-Internals Investigation Plan: Memorization vs. Genuine Position Learning

## 0. Framing

`positional-bias-benchmark-plan.md` already establishes the *behavioral* evidence for VoxTell
(Dice-vs-angle degradation, wrong-side rate, architecture/training-config facts) and sketches two
final root-cause steps (Table 3, rows 7-8) that go past behavior into the model's internals. This
document expands those two steps into a full, executable investigation, organized around one
question:

> **When VoxTell gets "left kidney" right, is it because it read "left" and used it — or because
> it memorized a canonical shape/location prior and the word did nothing?**

That question can't be answered from Dice scores alone. Two models can post identical Dice curves
for entirely different reasons: one genuinely parses spatial language and grounds it in the image,
the other learned "kidney" → "the blob usually near here" and treats "left"/"right" as noise it
never needed. This plan is designed to tell those apart, at three levels of the model
(text encoder → image/fusion → decoder output), plus a dedicated memorization-probe layer that
cuts across all three.

**What's already confirmed** (from the existing doc, Table 3 rows 5-6, sourced from the VoxTell
CVPR 2026 paper Appendix A):
- Backbone: plain ResEncL CNN, no rotation-equivariant components.
- Decoder: no positional embeddings anywhere — position is only implicit in the CNN feature map.
- Rotation augmentation: ~30°/axis during training; left-right mirroring deliberately excluded.
- This gives a prior hypothesis (architecture allows a shortcut) that the steps below test directly
  rather than assume.

**Access confirmed available** (from `voxtell-amos-rotation-multiangle-kaggle.ipynb`):
- `VoxTellPredictor` (pip package `voxtell`) exposes `predictor.text_backbone`
  (`Qwen/Qwen3-Embedding-4B`) and `predictor.embed_text_prompts(text_prompts)` directly — text
  embeddings are extractable with no monkey-patching beyond what the notebook already does for the
  fp16 fix.
- `predictor.predict_single_image(img, organs)` is the only documented entry point into the
  image/fusion/decoder path. Anything below the final mask (intermediate CNN features, fusion
  weights, cross-attention if any) will need to be reached by hooking `predictor`'s internal
  `nn.Module`s (`torch.nn.Module.register_forward_hook`) — this needs to be scoped from the actual
  installed source (`voxtell.inference.predictor`, and whatever segmentation network it wraps)
  before Phase B/C can run; treat "what's hookable" as an open discovery task, not an assumption.

---

## Phase A — Text-encoder-level analysis (does the text side even distinguish position?)

Goal: establish whether the *language* signal for laterality/position exists at all before asking
whether downstream stages use it. This is the cheapest phase (no image inference needed) and gates
everything else — if embeddings for "left X" and "right X" are near-identical, no downstream fix
can recover positional grounding from text.

| # | Step | What to do | Signal it produces |
|---|------|------------|---------------------|
| A1 | **Embedding distance: left vs. right** | For every bilateral organ in the manifests (kidney, hippocampus, etc.), embed `"Segment the left X in this image."` and `"Segment the right X in this image."` via `predictor.embed_text_prompts`. Compute cosine distance. | A distance close to 0 = the text encoder barely distinguishes the position word; this alone would explain wrong-side behavior without touching the image side. |
| A2 | **Embedding distance: organ-name swap (control)** | Same setup, but embed `"Segment the left kidney..."` vs `"Segment the left spleen..."` (name changes, position word fixed). | Establishes the *expected* magnitude of a "meaningful" distance, as a scale reference for A1. If name-swap distance >> position-swap distance, that's direct evidence the encoder treats organ identity as the dominant signal and position as secondary. |
| A3 | **Nearest-neighbor prompt geometry** | Embed the full prompt vocabulary used across manifests (all organs × both sides × any phrasing variants). Project to 2D (PCA/UMAP) and check whether "left X" / "right X" pairs cluster by organ (position is a minor perturbation within an organ cluster) or split into separate position-defined clusters. | Visual/quantitative confirmation of A1 at the vocabulary level, not just one organ. |
| A4 | **Position-only, name-free prompts (Table 3 row 7)** | Embed prompts that drop the organ name entirely: `"the structure on the left side of the abdomen"` vs. the equivalent right-side phrasing, for subjects where that's anatomically unambiguous. Compare against A1's organ-included embeddings. | If name-free position embeddings are *further* from each other than name-included ones, the encoder does carry standalone positional semantics, just weaker than organ identity. If they collapse to near-identical, the encoder isn't parsing "left"/"right" as free-standing spatial language at all. |
| A5 | **Synonym/paraphrase robustness** | Re-embed each bilateral prompt with paraphrases that preserve meaning (`"left"` → `"on the left side"`, `"the left-sided kidney"`) and semantically-empty variants (swap "left" for a nonsense token, or drop it and pad with a filler word of the same token length). Compare distances to the true left/right pair from A1. | If a nonsense-token substitution shifts the embedding by roughly as much as the real "left"→"right" swap, that's evidence the encoder is reacting to *any* token perturbation (surface form), not specifically to positional semantics — a memorization/shortcut signature at the text level. |

**Interpretation gate:** if A1/A4 show the text encoder does separate left/right prompts by a
clear margin (well above the A5 nonsense-token noise floor), the failure — if any — is downstream
(Phase B/C). If it doesn't, stop and report the alignment-stage failure; Phase B/C become secondary
(the fusion/decoder literally has nothing to work with).

---

## Phase B — Image/fusion-level analysis (is the positional signal used, or dropped?)

Goal: for cases where Phase A shows the text side *does* encode position, test whether that signal
actually reaches and moves the segmentation output.

| # | Step | What to do | Signal it produces |
|---|------|------------|---------------------|
| B1 | **Output divergence from paired prompts (Table 3 row 8, core test)** | For the same image, run `predict_single_image` once per prompt in a matched pair (`"left X"`, `"right X"`). Compute Dice/centroid distance *between the two predicted masks themselves* (not against ground truth). | If the text embeddings are clearly distinct (Phase A) but the two masks are nearly identical, the fusion/decoder is discarding a signal it was given — a decoder-side failure, not a language-understanding failure. |
| B2 | **Embedding-distance vs. output-distance regression** | Across all bilateral organs/subjects, plot Phase A's text-embedding cosine distance (x-axis) against B1's mask-difference metric (y-axis). | A flat/near-zero slope = text-side distance doesn't predict output difference at all, i.e., text signal isn't propagating regardless of its strength. A positive slope with a nonzero floor tells you the relationship exists but is attenuated somewhere in fusion. |
| B3 | **Embedding-swap ablation** | Run inference normally to get the image features for a "left kidney" query, then manually substitute the *text embedding* with the one computed for "right kidney" before it reaches fusion (requires hooking the point in `voxtell.inference.predictor` where the text embedding is consumed — a discovery task, see Phase C's note). Check if the output mask moves toward the right-kidney location. | Directly causal test, stronger than B1/B2's correlational read: if swapping the embedding does *not* change the output, fusion is provably ignoring it, independent of how distinct the embeddings are. |
| B4 | **Zero/mean-ablate the text embedding** | Replace the text embedding with an all-zero vector or the dataset-mean embedding across all organ prompts, keep the image fixed, run inference. | If output barely changes vs. the real prompt's output, the decoder is leaning almost entirely on image content (i.e., it's doing organ-agnostic saliency segmentation and picking the anatomically "usual" structure), a strong memorization/shortcut signature. |
| B5 | **Feature-map inspection at the fusion point (if hookable)** | Once B3's hook point is found, log the fused feature map (or whatever conditions the decoder) for the left- vs. right-prompt runs on the same image. Compute a spatial difference map. | Localizes *where* in the network positional information starts, or fails, to diverge — useful for pinning "text-image alignment" vs. "decoder ignoring available signal" precisely, rather than inferring it indirectly from B1-B4. |

---

## Phase C — Memorization-specific probes

Goal: directly test for memorized canonical-pose/shape priors, independent of the text-vs-decoder
split above. These are designed to break the "any" alternative explanation (resampling artifact,
architecture limit) already ruled out or scoped in the existing plan, and isolate memorization
specifically.

| # | Step | What to do | Signal it produces |
|---|------|------------|---------------------|
| C1 | **Prompt-invariance test** | For a fixed, un-rotated (0°) image, send the *correct* prompt and several *wrong* prompts (wrong organ name, wrong side, and a prompt for an organ that isn't in the image at all) through `predict_single_image`. | A model relying on memorized shape priors will often still segment "something" regardless of which organ was asked for (especially for a common structure like kidney/spleen), i.e., output barely tracks the prompt. A grounded model returns an empty/near-empty mask, or the *correct* different structure, when the prompt changes. |
| C2 | **Wrong-organ-in-scene test** | Prompt for an organ genuinely absent from the volume (e.g., "segment the gallbladder" on a brain MRI). | Tests whether the model has a "there's always something here" prior vs. correctly abstaining. Ties into C1 but isolates absence rather than substitution. |
| C3 | **Canonical-pose pull, quantified (extends existing centroid-displacement step)** | The existing plan's step 2 already checks whether predicted-centroid drift at rotation angles points toward the 0° canonical pose. Extend this by fitting, per organ, a simple model: `predicted_centroid(angle) = a * true_centroid(angle) + (1-a) * canonical_centroid_at_0deg`. Estimate `a` via regression. | `a` near 1 = model tracks true anatomy regardless of rotation (position genuinely learned). `a` near 0 = model outputs the canonical-pose location almost irrespective of the actual rotated anatomy (a memorized prior, not perception). This turns the earlier qualitative "does it drift toward canonical pose" check into a quantified score per organ, directly comparable across organs/datasets. |
| C4 | **Cross-subject shape consistency (mask-identity check)** | For a given organ + side prompt, compare predicted mask *shape* (not location) across multiple different subjects. Compute shape similarity (e.g. Dice after centering/aligning masks). | Abnormally high shape similarity across subjects with genuinely different anatomy would suggest the decoder outputs a templated shape prior rather than segmenting the actual visible structure — a shape-level memorization signature distinct from the location-level one C3 measures. |
| C5 | **Prompt-order / distractor robustness** | Where the API allows multiple prompts per call (the notebook batches all organs for a subject in one `predict_single_image` call), test whether output for "left kidney" changes depending on what other organ prompts are batched alongside it in the same call, vs. run alone. | If output for the same organ prompt changes materially based on unrelated co-batched prompts, that points to prompt-embedding cross-talk/memorized batch-level shortcuts rather than a well-isolated per-prompt grounding process — worth ruling out since the actual sweep pipeline always batches. |

---

## Phase D — Tie-in with existing behavioral evidence

The mechanistic phases above should be read against, not instead of, the behavioral results already
in `positional-bias-benchmark-plan.md`:

- Table 3 row 1 (resampling confound) and row 4 (threshold vs. smooth decay) stay purely behavioral
  — no internals needed, just re-run at finer angle steps.
- Table 3 rows 2-3 (canonical-pose pull direction, size/organ dependence) get sharpened by C3 here
  (quantified `a` coefficient instead of a qualitative direction check) and cross-referenced against
  organ size, since a memorized-prior organ should show a *higher* `a` if it's a large, commonly-seen
  structure (more chances to memorize a canonical location) and a *lower* `a` for rare/small organs.
- Rows 7-8 in the existing table are exactly Phase A (row 7) and Phase B (row 8) here, expanded into
  concrete, ordered sub-steps with specific code hooks (`embed_text_prompts`, and the fusion-point
  hook to be located).

---

## Execution order and dependencies

1. **Phase A first, always** — cheapest (no GPU image inference, just `embed_text_prompts` calls),
   and gates interpretation of everything downstream.
2. **C1/C2/C4/C5 can run in parallel with Phase A** — they only need `predict_single_image`, already
   working in the existing Kaggle notebooks, no new model access required. These are the fastest
   path to a first memorization signal while the text-encoder analysis runs.
3. **B1/B2/C3 next** — reuse the existing rotation-sweep CSVs (`predictions_voxtell_*_rotation_*.csv`)
   plus one new paired-prompt inference pass (left/right prompts per subject, un-rotated is enough
   to start; rotated volumes extend it for free since the sweep data already exists).
4. **B3/B4/B5 last** — require finding and hooking the actual fusion point inside the installed
   `voxtell` package source, which is a discovery task (read `voxtell.inference.predictor` and
   whatever segmentation network class it instantiates) before any code can be written against it.
   Scope this as its own short investigation before committing to a implementation plan for it.

## Deliverable per phase

Each phase should produce: (a) the raw metric/plot described in its table, (b) one paragraph stating
which of the two hypotheses — "genuine position learning" vs. "memorized shortcut" — the result
supports, and (c) an explicit confidence caveat where the test is correlational (B1, B2, C3, C4)
vs. causal (B3, B4, C1, C2). Causal results should be weighted higher when they conflict with
correlational ones.
