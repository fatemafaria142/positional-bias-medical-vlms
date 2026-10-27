# RQ1–RQ7 — Plain-English Overview

One idea runs through all seven research questions: **VoxTell is a text-promptable 3D
segmentation model, and the only thing that should change between "segment the left kidney"
and "segment the right kidney" is one word.** Does it actually track that word correctly, and
if it gets confused, *where* in the model does that confusion live? Each RQ pokes at a
different layer of that question — from raw text embeddings, through the decoder, to the final
pixels, to what happens when the prompt is made deliberately ambiguous.

RQ1–RQ4 are the original planned questions. RQ5–RQ7 are follow-ups added once the
mechanistic tooling existed, reusing the same code rather than new methodology.

All seven currently run **only on bilateral (left/right) organ pairs** — 16 pairs in
Neurite-OASIS brain MRI (see table at the bottom). An organ with no left/right counterpart
(e.g. brain stem) is filtered out before any RQ even sees it.

---

## RQ1 — Do the model's 32 fusion channels tell left from right?

**Plain question:** Feed the model the exact same scan, once with "left X" and once with
"right X" — do the internal channels that fuse text and image actually look different, or do
they look the same regardless of which side you asked for?

**What it does:** Extracts the 32-channel fused representation at 4 decoder stages (+ the
finest stage's direct output) for both prompts, and compares them (cosine similarity,
correlation, L2 distance). Also checks the final predicted mask against *both* the correct
organ's ground truth and the mirrored organ's ground truth.

**Why it matters:** If the channels look nearly identical for "left" vs. "right", the model
may not really be using the side word at that stage — a first clue toward a laterality
weakness, though not proof on its own (see RQ2–RQ4).

**Key output:** `dice_other_side_organ` — how much the prediction overlaps the *wrong* side.
This is the single most direct laterality-error number in the whole investigation.

---

## RQ2 — Does the text encoder already know left from right, before VoxTell even touches it?

**Plain question:** Before any of VoxTell's own processing, do the raw "left X" / "right X"
text embeddings (from the frozen Qwen3 encoder) already look different? And, as a basic sanity
check — does the image side of the model ever secretly "see" the prompt?

**What it does:** Cosine similarity / L2 distance between raw text embeddings for "left" vs.
"right". Separately, confirms the image encoder produces bit-identical output regardless of
prompt (it structurally can't see the text — this is a sanity check, not a finding).

**Why it matters:** Tells you whether a laterality weakness starts in the text encoder itself,
or gets introduced later by VoxTell's own layers (RQ3's question).

---

## RQ3 — Does the decoder preserve or wash out left/right information as it processes the prompt?

**Plain question:** As the text embedding flows through the 6-layer decoder, does the gap
between "left" and "right" get bigger, smaller, or stay the same?

**What it does:** Measures cosine/L2 distance between the left- and right-prompt internal
states at every one of the 6 decoder layers, plus KL divergence between their attention
patterns. Compares against a "different organ, same side" reference pair, so you can tell
laterality-specific drift from ordinary organ-to-organ drift.

**Why it matters:** A line that falls toward layer 6 suggests the decoder is *suppressing* the
side information it started with; a flat or rising line suggests it's preserved or amplified.

---

## RQ4 — If you forcibly swap "left" features for "right" at one layer, does the output flip?

**Plain question:** This is the first *causal* test (RQ1–RQ3 just look; this one intervenes).
At each of the 6 layers, what happens if you surgically replace that layer's own contribution
with the opposite-side prompt's contribution, leaving everything else alone?

**What it does:** Three conditions per layer — unpatched baseline, a positive-control full-state
swap (which flips the output by construction, just proving the plumbing works), and the real
test: an update-only patch. Scores whether the prediction's laterality call actually flips
toward the swapped-in side.

**Why it matters:** This is the closest thing to "proof" in the investigation — it directly
ranks which of the 6 layers *causally* controls the left/right decision, rather than just
correlating with it.

---

## RQ5 — Which specific region of the image matters most (not just "the image matters")?

**Plain question:** *Which part* of the image does the model rely on?

**What it does:** Tiles a box around the organ into 8 blocks, tries all 256 present/absent
combinations (replacing absent blocks with noise), and computes an exact Shapley value per
block — the most expensive experiment here (256 full predictions × organ × subject), so it
only runs on a small subject subset.

**Why it matters:** Produces an actual spatial heatmap of "where in the image the decision comes
from," rather than a single scalar.

---

## RQ6 — Is a laterality mistake caused by the position word, the organ word, or something else?

**Plain question:** Lumping the *entire* prompt into one "text" player hides what matters. This breaks the
sentence apart: is it specifically the word "left"/"right" doing the work, or the organ name, or
even just the surrounding sentence filler ("Segment the ... in this image")?

**What it does:** Four-player exact Shapley decomposition — image, organ word, position word,
"other words" (sentence scaffolding) — 16 coalitions, scored both against ground truth and
against the fully-informed prediction.

**Why it matters:** Pinpoints *which word* is actually responsible when the model gets left/right
wrong, rather than just knowing "text in general" was involved.

---

## RQ7 — When the prompt is deliberately confusing, which side does the model default to?

**Plain question:** Every other RQ uses a clean, unambiguous prompt. This one asks: what if the
prompt contradicts itself ("segment the left kidney... actually, make it the right one")? Does
the model land on left/right roughly 50/50, or does it systematically favor one side regardless
of which word came first?

**What it does:** Six deliberately conflicting/ambiguous prompt variants (correction-style,
injected irrelevant text, system-override-shaped, side word buried in a secondary clause).
Scores which side (if any) the output resolves to, and compares against this subject's own
clean left/right predictions.

**Why it matters:** This is the only RQ that can surface a genuine systematic bias — clean,
unambiguous prompts (used everywhere else) can never reveal whether the model secretly prefers
one side when genuinely unsure.

---

## How They Fit Together

| Question type | RQs |
|---|---|
| **Look** (representations, no intervention) | RQ1, RQ2, RQ3 |
| **Intervene** (causal) | RQ4 |
| **Attribute** (Shapley credit) | RQ5, RQ6 |
| **Where** (spatial grounding) | RQ5 |
| **Stress-test** (ambiguous input) | RQ7 |

A typical results narrative: RQ1/RQ4 show *what* happens (does it flip?), RQ4/RQ7 show
*how often*, and RQ5/RQ6 show *why* (which ingredient — image, organ word, or position word —
is responsible).

---

## Organ Coverage

Every RQ above runs once per bilateral organ pair. Structures with no left/right counterpart
(brain midline structures) are excluded — they carry no laterality signal.

**Neurite-OASIS (brain MRI) — 16 pairs, currently used for all RQs:**
cerebral white matter, cerebral cortex, lateral ventricle, inferior lateral ventricle,
cerebellum white matter, cerebellum cortex, thalamus, caudate, putamen, pallidum, hippocampus,
amygdala, accumbens, ventral DC, vessel, choroid plexus.

**AMOS22 (abdominal CT/MRI) — only 2 pairs, not yet staged:**
kidney, adrenal gland. (AMOS annotates 15 organs total, but only these two have a left/right
counterpart — the other 13, like liver, spleen, pancreas, have no laterality signal under the
current bilateral-only design.)

For the full technical detail behind each RQ (exact formulas, CSV column names, visualization
specs), see `RQ1_to_RQ7_Methodology.md`.
