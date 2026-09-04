# Prompt-Wise Shift Analysis - 5 Questions to Navigate These Results

This file drills one level deeper than `FINDINGS.md`: instead of one
Wrong-Side/No-Shift number averaged over all four positional prompts
(P1–P4), it breaks every model out **by individual prompt** (P1=left,
P2=right, P3=upper, P4=lower). That breakout surfaces a pattern the
averaged view hides completely - several models that looked like
"chance-level" (~50% wrong-side) are not guessing randomly at all, they
have a **fixed favorite region** and the P1/P2 (or P3/P4) numbers just
average out to ~50% because they're near-mirror opposites of each other.

Regenerate the underlying numbers with `scripts/aggregate_findings.py`'s
`score_plan_format`/`load_prefixed` helpers (see the script at the end of
this file's methodology note) any time results change.

**Read this file through five questions, in order:**

1. [Does the model's answer actually move when told a direction?](#q1-does-the-answer-actually-move-shift-rate-by-prompt)
2. [When it does move, does it move the right way?](#q2-when-it-moves-does-it-move-the-right-way-wrong-side-rate-by-prompt)
3. [Why do some "50% wrong-side" models turn out not to be guessing at all?](#q3-whats-the-main-reason-some-50-wrong-side-models-arent-actually-guessing)
4. [Is left/right easier to follow than upper/lower?](#q4-is-leftright-easier-to-follow-than-upperlower)
5. [Which model can you actually trust with a positional prompt?](#q5-which-model-can-you-actually-trust-with-a-positional-prompt)

---

## Q1: Does the answer actually move? (Shift Rate by Prompt)

**What this answers:** RQ1 - for each individual prompt, what fraction of
that model's answers moved more than 5% of image width/height away from
its own P0 (unprompted) answer on the same image? A model stuck at 0%
across the board never reacts to being told a position at all, regardless
of whether that position is correct.

*Rows sorted by average shift rate across P1–P4, highest first.*

### LGG

| Model | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|
| NorthMicroVision | 86.0% | 88.0% | 62.0% | 76.0% |
| Moondream2 (point) | 54.0% | 56.0% | 76.0% | 74.0% |
| LocateAnything-3B | 49.0% | 63.3% | 47.9% | 39.1% |
| Molmo2-8B | 28.2% | 45.0% | 40.9% | 68.6% |
| CLIPSeg | 33.3% | 25.0% | 54.2% | 65.2% |
| Kosmos-2 | 34.0% | 30.0% | 20.0% | 30.0% |
| Qwen2.5-VL-7B | 16.7% | 12.0% | 14.0% | 14.0% |
| OWLv2 | 14.3% | 16.7% | 11.9% | 11.9% |
| Moondream2 (detect) | 20.4% | 12.8% | 6.1% | 14.9% |
| Florence-2 | 2.0% | 2.0% | 0.0% | 0.0% |
| Grounding DINO | 2.0% | 0.0% | 0.0% | 0.0% |
| PaliGemma 2 | 0.0% | 0.0% | 0.0% | 2.0% |

### KVASIR

| Model | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|
| Moondream2 (point) | 90.0% | 60.0% | 58.0% | 68.0% |
| NorthMicroVision | 74.0% | 57.4% | 45.8% | 72.0% |
| Grounding DINO | 75.5% | 65.9% | 26.5% | 24.0% |
| Molmo2-8B | 50.0% | 40.0% | 42.0% | 38.8% |
| CLIPSeg | 70.7% | 63.6% | 25.0% | 10.0% |
| LocateAnything-3B | 40.8% | 36.0% | 40.0% | 40.0% |
| Qwen2.5-VL-7B | 16.7% | 14.6% | 10.4% | 22.9% |
| Kosmos-2 | 18.0% | 14.0% | 10.0% | 12.0% |
| OWLv2 | 20.0% | 20.0% | 7.7% | 0.0% |
| Florence-2 | N/A** | N/A** | 0.0% | 0.0% |
| Moondream2 (detect) | 0.0% | 0.0% | 0.0% | 0.0% |
| PaliGemma 2 | 0.0% | 0.0% | 0.0% | 0.0% |

\* Florence-2's P1/P2 shift rate is undefined on Kvasir - its detect rate collapses to 54% there (see `FINDINGS.md`), and the surviving detections happened to leave no valid P1/P2 pair to compare against P0 for some samples.

### ISIC

| Model | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|
| Molmo2-8B | 84.0% | 78.0% | 90.0% | 72.0% |
| Moondream2 (point) | 90.0% | 84.0% | 68.0% | 52.0% |
| NorthMicroVision | 78.0% | 76.0% | 44.0% | 56.0% |
| LocateAnything-3B | 40.0% | 42.0% | 28.0% | 26.0% |
| Grounding DINO | 30.0% | 20.0% | 26.0% | 22.0% |
| CLIPSeg | 41.5% | 20.5% | 10.0% | 20.0% |
| Qwen2.5-VL-7B | 33.3% | 13.0% | 11.9% | 22.7% |
| Kosmos-2 | 14.0% | 12.0% | 12.0% | 4.0% |
| Moondream2 (detect) | 33.3% | 0.0% | 0.0% | 0.0% |
| Florence-2 | 8.0% | 8.0% | 2.0% | 6.0% |
| PaliGemma 2 | 4.0% | 4.0% | 6.0% | 4.0% |
| OWLv2 | 3.3% | 3.6% | 3.0% | 2.9% |

**Reading this table:** Florence-2, Grounding DINO (on LGG), and PaliGemma 2 are frozen - under 5% shift on nearly every prompt, on every dataset. Whatever they output for P0, they output again for P1–P4. Molmo2-8B, Moondream2-point, and NorthMicroVision are the only three that reliably shift (>40–90%) on most prompts across all three datasets.

---

## Q2: When it moves, does it move the *right* way? (Wrong-Side Rate by Prompt)

**What this answers:** RQ3 - of the answers that got compared to a prompt at all, what fraction landed on the side *opposite* the one the prompt named? Reading this next to Q1 separates "doesn't move" from "moves, but the wrong way."

*Rows sorted by directional imbalance (max − min across P1–P4), highest first - the more imbalanced, the more evidence of a fixed bias rather than noise (see Q3).*

### LGG

| Model | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 0.0% | 98.0% | 0.0% | 100.0% |
| Moondream2 (detect) | 75.5% | 4.3% | 65.3% | 23.4% |
| OWLv2 | 59.2% | 40.8% | 81.2% | 19.1% |
| NorthMicroVision | 2.0% | 12.0% | 0.0% | 56.0% |
| Kosmos-2 | 24.0% | 66.0% | 40.0% | 66.0% |
| Florence-2 | 66.0% | 36.0% | 68.0% | 32.0% |
| Grounding DINO | 64.0% | 36.0% | 66.0% | 32.0% |
| PaliGemma 2 | 38.0% | 62.0% | 66.0% | 36.0% |
| CLIPSeg | 37.0% | 57.1% | 41.7% | 60.9% |
| Moondream2 (point) | 22.0% | 0.0% | 8.0% | 2.0% |
| Molmo2-8B | 17.8% | 8.7% | 2.0% | 0.0% |
| LocateAnything-3B | 16.3% | 16.3% | 27.1% | 30.4% |

### KVASIR

| Model | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 0.0% | 96.0% | 0.0% | 100.0% |
| PaliGemma 2 | 6.0% | 94.0% | 96.0% | 4.0% |
| OWLv2 | 90.0% | 10.0% | 69.2% | 42.9% |
| Kosmos-2 | 82.0% | 12.0% | 82.0% | 16.0% |
| Florence-2 | N/A** | N/A** | 18.9% | 79.6% |
| Grounding DINO | 22.4% | 17.1% | 18.4% | 64.0% |
| Moondream2 (detect) | 14.3% | 0.0% | 40.0% | 0.0% |
| CLIPSeg | 58.5% | 43.2% | 62.5% | 30.0% |
| LocateAnything-3B | 44.9% | 28.0% | 38.0% | 28.0% |
| Molmo2-8B | 28.0% | 14.0% | 16.0% | 14.3% |
| NorthMicroVision | 20.0% | 25.5% | 27.1% | 14.0% |
| Moondream2 (point) | 4.0% | 0.0% | 12.0% | 6.0% |

### ISIC

| Model | P1 (left) | P2 (right) | P3 (upper) | P4 (lower) |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 6.4% | 98.0% | 0.0% | 100.0% |
| Florence-2 | 12.0% | 88.0% | 12.0% | 92.0% |
| Kosmos-2 | 82.0% | 18.0% | 82.0% | 12.0% |
| PaliGemma 2 | 16.0% | 84.0% | 68.0% | 34.0% |
| Grounding DINO | 12.0% | 78.0% | 18.0% | 60.0% |
| NorthMicroVision | 4.0% | 6.0% | 2.0% | 58.0% |
| OWLv2 | 53.3% | 42.9% | 33.3% | 68.6% |
| Moondream2 (detect) | 12.5% | 0.0% | 33.3% | 33.3% |
| Moondream2 (point) | 2.0% | 0.0% | 32.0% | 0.0% |
| LocateAnything-3B | 24.0% | 42.0% | 24.0% | 52.0% |
| CLIPSeg | 53.7% | 52.3% | 40.0% | 58.0% |
| Molmo2-8B | 6.0% | 4.0% | 8.0% | 2.0% |

---

## Q3: What's the *main reason* some "50% wrong-side" models aren't actually guessing?

`FINDINGS.md` reported wrong-side rates of ~48–53% for nine of eleven
models and called that "chance level." The per-prompt table above shows
that framing is **half right**: the *average* is near 50%, but for
several models the two halves of that average are extreme and opposite,
not two independent coin flips. That's a qualitatively different failure
mode - a **fixed positional anchor** the model defaults to almost
regardless of what the prompt asks for.

**Clearest example - Qwen2.5-VL-7B, identical pattern on all 3 datasets:**

| Prompt | Asks for | Wrong-side rate | What this means |
|---|---|---|---|
| P1 | left | 0–6% | Almost always answers "left" - happens to match |
| P2 | right | 96–98% | Almost always **still** answers "left" - now wrong |
| P3 | upper | 0% | Almost always answers "upper" - happens to match |
| P4 | lower | 100% | Almost always **still** answers "upper" - now wrong |

Qwen2.5-VL-7B is not ignoring the image and not guessing - it is
outputting close to the same top-left-ish box on every single call, no
matter what the prompt says. Averaging P1+P2 (or P3+P4) produces ~50%
"wrong-side," which is why the aggregate table in `FINDINGS.md` couldn't
distinguish this from genuine randomness. Only the per-prompt view does.

**Second example - Kosmos-2**, opposite anchor (defaults toward the
*right* side): wrong-side rate is consistently ~80% on P1 (asked left)
and ~15% on P2 (asked right), on both ISIC and Kvasir. Florence-2 and
PaliGemma 2 show the same left/right or upper/lower split pattern with
different anchor sides, on different datasets.

**Practical takeaway:** for these anchor-biased models, "wrong-side
rate" alone under-communicates the failure. The real story is *"this
model has a favorite answer and repeats it"* - closer to a no-shift
failure (Q1) than to an active-but-incorrect one, even though its Q2
number looks alarming for exactly one of the two prompts in each pair.

---

## Q4: Is left/right easier to follow than upper/lower?

**What this answers:** RQ1/RQ3 aggregated by *axis* instead of by model,
averaged across all 11 models. Is either axis systematically harder for
the field as a whole?

| Dataset | Avg Wrong-Side - Left/Right (P1+P2) | Avg Wrong-Side - Upper/Lower (P3+P4) | Avg Shift - Left/Right | Avg Shift - Upper/Lower |
|---|---|---|---|---|
| LGG | 35.8% | 38.5% | 28.8% | 30.4% |
| Kvasir | 32.3% | 36.6% | 37.6% | 23.0% |
| ISIC | 33.2% | 38.4% | 34.2% | 24.5% |

| Prompt (all models, all datasets pooled) | Avg Wrong-Side Rate | Avg Shift Rate |
|---|---|---|
| P1 - left | 30.7% | 35.9% |
| P2 - right | 36.9% | 31.0% |
| P3 - upper | 36.1% | 25.0% |
| P4 - lower | **39.6%** | 27.0% |

**Finding:** upper/lower prompts are harder than left/right prompts on
**all three datasets**, both by wrong-side rate (higher on UD in every
row) and, on Kvasir and ISIC, by shift rate too (lower on UD). **P4
("in the lower region")** is the single hardest prompt across the whole
study - highest average wrong-side rate (39.6%) of all four. **P1
("on the left side")** is the easiest and most-followed prompt overall,
with the lowest wrong-side rate and the highest shift rate. A plausible reason: "left"
is the first positional word most of these models see during P1–P4 and
the most common in web image-caption data ("on the left, ..."),
while vertical framing ("upper region", "lower region") is rarer and
more clinical-sounding phrasing that these models were less exposed to
during pretraining/grounding fine-tuning.

---

## Q5: Which model can you actually trust with a positional prompt?

Combining Q1 (does it move), Q2 (does it move correctly), and Q3 (is
that movement genuine or a coincidental anchor), only two models pass
all three bars on all three datasets:

| Model | Shifts on most prompts? (Q1) | Low wrong-side on *every* prompt, not just some? (Q2) | Anchor-biased? (Q3) |
|---|---|---|---|
| **Molmo2-8B** | Yes (28–90%) | Yes - every P1–P4 cell stays under ~18% wrong-side on every dataset | No |
| **Moondream2 (point)** | Yes (52–90%) | Yes - every cell under ~32%, mostly under 12% | No |
| Everything else | Mixed or near-zero | No - at least one prompt per dataset exceeds 50%, often 80%+ | Yes, for most of them |

**Bottom line:** if the downstream use case needs a model whose answer
can be trusted to genuinely respond to "left/right/upper/lower" wording
rather than defaulting to a fixed region, **Molmo2-8B** and
**Moondream2's `/point` endpoint** are the only two of the 11 tested
models that qualify on every dataset and every direction. Every
bbox/mask detector tested (CLIPSeg, Florence-2, Grounding DINO, Kosmos-2,
OWLv2, PaliGemma 2, Qwen2.5-VL-7B, LocateAnything-3B, NorthMicroVision,
Moondream2's own `/detect` endpoint) has at least one direction, on at
least one dataset, where wrong-side rate exceeds 50% - i.e., worse than
a coin flip specifically for that direction, on top of frequently just
not moving at all.

---

*Methodology: shift rate and wrong-side rate are computed per (dataset,
model, prompt_id) exactly as defined in `FINDINGS.md`'s metric table,
just grouped by individual prompt instead of pooled across P1–P4.
Regenerate with `python scripts/aggregate_prompt_shift.py`, which writes
`Results/Findings/PROMPT_SHIFT_DATA.csv` on top of
`scripts/aggregate_findings.py`'s `score_plan_format()`/`load_prefixed()`
functions.*
