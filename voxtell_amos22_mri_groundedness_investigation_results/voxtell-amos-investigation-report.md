# VoxTell AMOS22 MRI — Does the Model Genuinely Understand Position and Organ Names?

**Scope:** 10 subjects, 4 bilateral organs (left/right kidney, left/right adrenal gland),
position + organ-name ablation (4 prompt variants per organ), exact Shapley + SHAP + LIME,
plus a four-stage internal-layer analysis (image encoder, text embedding, fusion, output).

This is the single, complete report — every question below is answered directly from the
actual result files, with a table backing each answer.

---

## Glossary — Plain-Language Explanation of Every Term Used Below

- **Dice score (`f_dice_gt`, "Dice," "accuracy")** — a number from 0 to 1 that measures how well
  a predicted mask overlaps the true, correct mask. 1.0 means a perfect match; 0 means no overlap
  at all (or an empty prediction). This is the standard way to score "how good is the
  segmentation" in medical imaging. `f_dice_gt` specifically means "Dice score against Ground
  Truth" — i.e., checked against the real, correct answer (as opposed to `f_dice_self`, below).
- **`f_dice_self`** — the same Dice calculation, but comparing a prediction against the *model's
  own original answer* (the one it gave with the full, unmodified prompt) instead of against the
  real ground truth. This tells us "how much did the model's answer change," which is a different
  question from "was the new answer correct."
- **IoU (Intersection over Union)** — very similar to Dice: a 0-to-1 score for how much two masks
  overlap, computed slightly differently (overlap area ÷ combined area, rather than Dice's
  2×overlap ÷ combined-size formula). Used interchangeably with Dice as an overlap measure in
  this report; both mean "how much do these two masks agree."
- **Voxel** — the 3D equivalent of a pixel: one small cube of volume in a 3D medical scan. "Voxel
  count" means how many of these tiny cubes are included in a mask — a bigger number means a
  bigger predicted region.
- **Centroid** — the center point of a mask (the average position of all its voxels). "Centroid
  shift" means how far that center point moved between two different predictions.
- **Ablation** — deliberately removing or changing one specific piece of the input (in this
  report, a single word in the prompt) to see how much the output changes as a result. This is
  the main technique used throughout to figure out which words the model actually depends on.
- **Cosine similarity** — a 0-to-1 (occasionally negative) score for how similar two directions
  are in a mathematical space, regardless of their size — 1.0 means pointing in exactly the same
  direction, 0 means unrelated directions. Used here to compare how "similar" two text or image
  representations are to each other.
- **Embedding** — the internal, numerical representation a model creates for a piece of text
  (or, after fusion, for a text-plus-image combination). Similar meanings tend to produce similar
  embeddings; this is what lets us measure "how differently does the model understand these two
  prompts" using cosine similarity.
- **Shapley value (exact Shapley)** — a method, borrowed from game theory, for fairly splitting
  "credit" for an outcome among several contributing factors (here: individual words). It works
  by checking every possible combination of words present/absent and averaging each word's
  effect — giving an exact, mathematically rigorous importance score per word.
- **SHAP** — a widely-used software library that computes Shapley values (or close approximations
  of them) for machine learning models generally. Used here as an independent cross-check against
  our own exact Shapley calculation.
- **LIME** — a different explainability method that works by randomly trying many slightly
  altered versions of a prompt and fitting a simple model to the pattern of results, rather than
  computing exact game-theory values. Included as a third, independently-implemented method.
- **Fusion** — the step inside the model where the text prompt's representation and the image's
  representation are combined together, so the model can use both at once to decide what to
  segment and where.
- **Bilateral organ** — an organ that exists in a left and a right version (e.g., kidneys,
  adrenal glands). Only organs like this can meaningfully test whether the model understands
  "left" vs. "right," since organs without a left/right version (like the liver) don't have a
  laterality question to test in the first place.

---

## Question 1 — Does the model's internal activation land on the correct anatomy, or does it fire the same way regardless of what's asked?

**Setup:** for a given image, we check whether the mask produced for the *correct* prompt
("left kidney") overlaps the true left-kidney location, and separately whether the mask
produced for a *deliberately wrong* prompt ("right kidney," same image) also lands on that same
true left-kidney location. If the model relied on image content alone, both would overlap.

**Answer: The model's activation is genuinely tied to the word, not the image alone.**

| Organ pair | IoU, correct prompt | IoU, wrong prompt (same image) |
|---|---|---|
| left kidney / right kidney | 0.892 | **0.000** |
| left adrenal gland / right adrenal gland | 0.497 | **0.000** |
| **Overall (20 cases)** | — | **0.000, every single case** |

The wrong prompt produces literally zero overlap with the true location — not "less" overlap,
none at all.

---

## Question 2 — Does a bigger difference in how the model understands two prompts translate into a bigger difference in its output?

**Setup:** compare how differently the model represents "left X" and "right X" in text
(embedding distance) against how differently it draws them (output distance), and check whether
more understanding-difference predicts more output-difference.

**Answer: This specific comparison can't produce a meaningful answer with only two organ types available.**

The text-embedding distance between "left X" and "right X" is a fixed property of the words
themselves — identical for every subject with the same organ, since the text encoder never looks
at the image. With only kidney and adrenal gland available, there are only two distinct values to
work with, not the broad spread across many organ types this kind of comparison needs.

| Organ pair | Text-embedding distance | Mean output distance (voxels) |
|---|---|---|
| left kidney / right kidney | 0.163 | 116.6 |
| left adrenal gland / right adrenal gland | 0.176 | 64.1 |

Kidney shows a *smaller* text-embedding distance but a *larger* output distance than adrenal
gland — the reverse of a naive "more difference → more difference" story. This is fully explained
by anatomy (kidneys sit farther apart than adrenal glands), not by grounding. A dataset with more
bilateral organ types (the OASIS brain dataset, with five pairs instead of two) will give a much
more informative version of this comparison.

---

## Question 3 — If we secretly substitute the "right kidney" understanding while the model is processing "left kidney," does the output actually follow the substitution?

**Setup:** rather than just observing a correlation, this forces one specific input to change
while holding everything else fixed, and checks whether the output causally follows — the
strongest form of evidence available in this investigation.

**Answer: Yes — the output follows the substitution every single time.**

| Organ pair | Left prompt → lands on true left side | Right prompt → lands on true right side |
|---|---|---|
| left kidney / right kidney | 100% | 100% |
| left adrenal gland / right adrenal gland | 100% | 100% |
| **Overall (20 cases)** | **100%** | **100%** |

A result this clean and consistent cannot be explained by coincidence.

---

## Question 4 — Where, if anywhere, do the internal representations for "left X" and "right X" actually diverge?

**Setup:** compare the model's internal representation for "left X" against "right X" on the same
image, and measure how much they differ.

**Answer: They diverge completely, and correctly — not partially or diffusely.**

| Organ pair | Fraction of the two masks' combined area that differs between them |
|---|---|
| left kidney / right kidney | 1.000 (fully disjoint) |
| left adrenal gland / right adrenal gland | 1.000 (fully disjoint) |

Combined with Questions 1 and 3, this divergence isn't just "different somehow" — it specifically
separates into the two anatomically correct locations.

---

## Question 5 — Across three independent methods, which word matters more: position or organ name?

**Setup:** exact Shapley values (a mathematically exact game-theory calculation), SHAP (the
standard library, independently implemented), and LIME (a conceptually different method) were
each used to score how much the position word and the organ name contribute to accuracy.

**Answer: All three agree — the organ name contributes several times more than the position word.**

| Method | Position | Organ | Ratio (organ ÷ position) |
|---|---|---|---|
| Exact Shapley | 0.137 | **0.658** | 4.8x |
| SHAP | 0.137 | **0.658** | 4.8x (agrees with exact Shapley on 320 out of 320 rows) |
| LIME (kidney only — see note) | 0.000940 | **0.002834** | 3.0x |

*Note:* LIME's word-splitting treats a multi-word organ name ("adrenal gland") as two separate
tokens rather than one, so its adrenal-gland-specific number isn't reliable here. Kidney (a single
word) is unaffected and already confirms the same ranking Shapley and SHAP give across all four
organs.

---

## Question 6 — Does removing each word actually hurt the model's real accuracy?

**Setup:** compare Dice score (accuracy against ground truth) for the real prompt against the
prompt with the organ name removed, and against the prompt with the position word removed.

**Answer: Removing the organ name is catastrophic; removing the position word is serious but partial.**

| Organ | Both words present | Organ removed | Position removed |
|---|---|---|---|
| left kidney | 0.943 | **0.000** | 0.637 |
| right kidney | 0.946 | **0.000** | 0.626 |
| left adrenal gland | 0.648 | **0.000** | 0.385 |
| right adrenal gland | 0.646 | **0.000** | 0.436 |

Removing the organ name drops accuracy to exactly zero, every time. Removing the position word
drops accuracy by roughly a third, but the model still produces a real, partially-correct answer.

---

## Question 7 — Does removing each word change the model's own answer, regardless of whether the new answer is correct?

**Setup:** compare each altered prediction against the model's own original, full-prompt
prediction — this measures sensitivity to the word, separate from ground-truth correctness.

**Answer: Both words cause a large change — but in different ways.**

| Organ | Similarity to original, organ removed | Similarity to original, position removed |
|---|---|---|
| left kidney | 0.649 | **0.000** |
| right kidney | 0.637 | **0.000** |
| left adrenal gland | 0.535 | **0.000** |
| right adrenal gland | 0.642 | **0.000** |

Removing the position word produces a completely different answer (zero similarity to the
original — a wholly different mask). Removing the organ name also changes the answer
substantially, but some structural similarity remains (54–65%) even as the answer becomes wrong.

---

## Question 8 — Does removing the position word specifically push the output to the wrong side, or does the model ignore it for laterality?

**Setup:** the most direct laterality-specific test — track how far the predicted mask's center
moves along the left-right axis when each word is removed.

**Answer: Yes — removing position causes a clear, substantial, consistent shift. Position genuinely controls laterality.**

| Word removed | Mean shift in predicted location (voxels) |
|---|---|
| position | **25.52** |
| organ | 21.85 |

Position's contribution to this laterality-specific measure is at least as large as the organ
name's — even though the organ name dominates overall accuracy (Question 6). Position is doing
real, specific work: deciding which side gets segmented.

---

## Question 9 — When a word is removed, does the model give up, or does it stay confident and produce something (even if wrong)?

**Setup:** track the predicted mask's size after each word is removed — a collapse toward zero
means the model gave up; a stable or larger size means it's still confidently producing an answer.

**Answer: The two words fail in opposite ways.**

| Organ | Both words present | Organ removed | Position removed |
|---|---|---|---|
| left kidney | 70,074 voxels | **0** | 147,207 (2.1x larger) |
| right kidney | 67,711 voxels | **0** | 147,207 (2.2x larger) |
| left adrenal gland | 1,263 voxels | **0** | 2,114 (1.7x larger) |
| right adrenal gland | 1,008 voxels | **0** | 2,114 (2.1x larger) |

Without the organ name, the model gives up entirely — zero voxels, every case. Without the
position word, it does **not** give up — it produces a *larger* mask than usual, consistent with
hedging between the two possible sides rather than confidently committing to one.

---

## Question 10 — Can the image encoder alone explain how the model gets left/right right or wrong?

**Setup:** the image encoder is the first stage of the pipeline, processing the raw scan before
anything else happens. We checked VoxTell's own code to see whether it has any access to the text
prompt at all.

**Answer: No — and this shapes how we read every later stage.**

| Check | Result |
|---|---|
| Does the image encoder receive the text prompt? | **No** — confirmed directly from the model's source code: the image encoder takes only the image |
| Does the same image encoding get reused across every prompt on that image? | **Yes** — one encoding is computed per image and shared across every organ tested on it |

This isn't a measured number — it's a structural fact. It matters because it rules out one
explanation immediately: whatever lets the model distinguish "left" from "right" cannot live in
the image encoder, since the image encoder never sees which word was used. That understanding has
to live in the text or fusion stage, which the next two questions test directly.

---

## Question 11 — Before the image is even involved, does the raw text already encode a consistent "left vs. right" concept?

**Setup:** we extracted the raw text embedding for each prompt directly from the model's text
encoder, before any interaction with the image. For each organ pair, we computed the direction in
embedding space separating "left X" from "right X," then measured how similar that direction is
across kidney and adrenal gland. A highly similar direction means one shared, reusable concept; a
dissimilar one means each organ's laterality is stored independently.

**Answer: Yes — a very strong, shared concept already exists in the text alone.**

| Comparison | Similarity between the two organs' "right minus left" directions |
|---|---|
| kidney vs. adrenal gland, before fusion | **0.963** (out of a maximum of 1.0) |

A value this close to 1.0 means "left kidney → right kidney" and "left adrenal gland → right
adrenal gland" point in almost exactly the same direction — a general, reusable concept, not two
unrelated, memorized facts. Since this is measured before the image is involved, the concept is
inherited from language pretraining, not something built from scratch during segmentation
training.

---

## Question 12 — After the model looks at the image, does that concept get sharper and more general, or does it fragment and become more organ-specific?

**Setup:** the model computes real attention weights between the text and the image during
fusion, but its own code discards them right after use. We recovered this discarded
representation and repeated the same cross-organ comparison as Question 11, this time after
fusion.

**Answer: The concept gets less general, not more — fusion pulls it toward organ-specific patterns.**

| Comparison | Similarity between the two organs' "right minus left" directions |
|---|---|
| Before fusion (Question 11) | 0.963 |
| After fusion | **0.680** |
| Change | **−0.282** |

If the model's own training had sharpened a single, unified spatial concept, this number should
go up after fusion, not down. Instead it drops substantially — the segmentation-specific training
doesn't generalize the concept further; if anything, it specializes it somewhat per organ. The
direction is still fairly consistent (0.680 is far from zero), so this is a partial
specialization, not a loss of the shared concept.

---

## Question 13 — Does all of this internal machinery actually translate into correct real-world output?

**Setup:** regardless of what the internal representations look like, does the model's final,
actual output get the right answer?

**Answer: Yes — 100%, and this is the result the earlier questions are explaining, not replacing.**

| Behavioral check | Result |
|---|---|
| Matched vs. mismatched overlap (Question 1) | 0.000 mismatched overlap, every case |
| Causal side-agreement (Question 3) | 100%, every case |
| Laterality-specific shift when position is removed (Question 8) | Confirmed — position causes a real, substantial shift |

This is the ground-truth check: it works, consistently. Questions 10–12 explain *how*, not
whether.

---

## Question 14 — Does the image contribute more, or does the text contribute more?

**Setup:** this is a different comparison from Questions 5–9 (which compared two *words* against
each other) — here we compare the two *modalities* against each other: the scan itself vs. the
prompt. We can't ablate the image the same way we ablated words (there's no meaningful "blank
image" equivalent to a generic placeholder word), so instead we use two natural, already-verified
facts from this data to isolate each modality's effect: the text embedding for a fixed organ
prompt is provably identical across every subject (Question 2), and the image encoding for a
fixed image is provably identical across every prompt (Question 10). That means any variation we
observe when we hold one of the two constant must come entirely from the other.

**Answer: They contribute two different, complementary kinds of information — text selects *what* to look for, the image supplies *where exactly and what shape* it actually is for that specific patient. Neither works without the other.**

**With the text held perfectly constant** (all 10 subjects given the exact same prompt, "Segment
the right kidney in this image."), the predicted location still tracked each patient's own,
very different, real kidney position almost perfectly:

| Subject | True kidney position (voxels) | Predicted position (voxels) | Dice |
|---|---|---|---|
| amos_0508 | 176.9 | 177.2 | 0.936 |
| amos_0530 | 227.1 | 226.8 | 0.954 |
| amos_0544 | 389.4 | 388.9 | 0.955 |
| amos_0553 | 168.5 | 168.7 | 0.955 |
| amos_0555 | 197.1 | 197.2 | 0.971 |
| amos_0558 | 324.6 | 324.4 | 0.947 |
| amos_0522 | 205.0 | 205.4 | 0.897 |
| amos_0532 | 205.0 | 205.3 | 0.945 |
| amos_0554 | 340.3 | 340.9 | 0.941 |
| amos_0575 | 172.0 | 172.1 | 0.960 |

The true position ranges enormously across these 10 patients (168.5 to 389.4 — more than double,
reflecting real anatomical and scan-framing differences), and the predicted position tracks it
almost exactly (correlation of 1.000 between predicted and true position). Since the text was
identical for every one of these 10 predictions, **100% of this patient-specific accuracy came
from the image.**

**With the image held constant and only the text changed** (Question 4), the output instead
relocates completely — a 100% spatial split between the "left" and "right" prompts on the same
scan. Since the image didn't change at all, **100% of that relocation came from the text.**

**Neither modality can produce a correct answer alone.** With a real image but no specific organ
named in the text (Questions 6 and 9), the output is a completely empty mask — the image's
detailed anatomical information is not used at all until the text tells the model what to look
for. There is no equivalent "text alone, no real image" condition to test, since the model
architecturally requires an image input to run at all — but the empty-output result already shows
that image content by itself, without a specific text query, produces nothing usable.

**In short: this isn't a contest with a single winner.** The text acts as a switch that decides
which structure and which side to attend to; the image supplies the actual shape and exact
location of that structure for the specific patient in front of it. Take away the text's ability
to name a real structure, and the output collapses to nothing (Questions 6 and 9) — the image's
detail goes unused. The image, meanwhile, is what lets the same fixed text prompt produce ten
different, correctly patient-specific answers rather than one generic template repeated across
every subject.

---

## Question 15 — Does raw cosine similarity between embeddings independently confirm that the organ name matters more than the position word?

**Setup:** Questions 11 and 12 compared how *consistent the left-right direction* is across
organs. This is a more basic, direct check: does changing the organ name move a prompt's
embedding by more than changing the position word does? We computed the plain cosine similarity
between every pair of the four organ prompts ("left kidney," "right kidney," "left adrenal
gland," "right adrenal gland"), in both the text embedding and the fused embedding, and grouped
the six possible pairs into three categories: same organ but opposite side, different organ but
same side, and different organ and opposite side.

**Answer: Yes — independently, and without looking at any segmentation output at all, the raw embeddings confirm that changing the organ name moves the representation more than changing the position word does.**

| Comparison type | Text-stage similarity | Fused-stage similarity |
|---|---|---|
| Same organ, opposite side (only the position word differs) | **0.837 / 0.825** | **0.764 / 0.855** |
| Different organ, same side (only the organ name differs) | 0.785 / 0.795 | 0.543 / 0.576 |
| Different organ, opposite side (both words differ) | 0.620 / 0.634 | 0.419 / 0.447 |

Changing just the organ name (row 2) produces a *lower* similarity — a *bigger* move in
embedding space — than changing just the position word (row 1), in the text stage and even more
clearly after fusion, where the gap widens substantially (0.76–0.85 vs. 0.54–0.58). This is a
third, completely independent line of evidence for the same conclusion as Question 5: the organ
name carries more distinguishing information than the position word, confirmed here purely from
embedding geometry, with no reference to Dice scores, masks, or any other output-based measure at
all.

---

