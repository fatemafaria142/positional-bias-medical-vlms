# Positional Bias: Model-by-Model Summary Table

This table answers the same 6 questions for all 11 models, so every row
can be read the same way. Three separate results tables are given, one
per dataset tested (LGG brain scans, Kvasir colon endoscopy, ISIC skin
images), since a model's behavior can differ noticeably by domain.

**Ranking rule for a positional-bias study: movement comes first.** A
model that never moves when told a position is not a neutral or "average"
result, it is the worst possible outcome for this research question,
since it means the model is not engaging with positional language at
all. Its correct-side percentage can still land near 50% purely by luck
(whatever fixed answer it always gives happens to be on the right side
about half the time), which would wrongly rank it in the middle of the
pack if sorted by accuracy alone. So each table below is sorted first by
**Bias Category** (Responsive, then Weakly Responsive, then Position-Blind),
and only within a category by correct-side accuracy. A model has to
actually move before its correct-side number is treated as meaningful.

**A note on accuracy:** these numbers were recalculated from scratch. An
earlier draft mixed two different definitions of "did the answer move"
across models, which understated how often Qwen2.5-VL-7B and a few others
actually moved their answer. That is fixed now, and every model below is
scored by the exact same method (see Table 5).

## Table 1. Question Legend: How Each Model Is Scored

*A quick legend for the 6 questions used to score every model, plus the Bias Category shown in Tables 2 to 4.*

| # | Question |
|---|---|
| Q1 | Given Prompt P0, the plain "locate the target" prompt with no position mentioned, does the model return a usable answer at all? This is the baseline check, before position enters the picture: can it find the target when just asked to, with nothing about left, right, upper, or lower? |
| Q2 | Does its answer move at all once the prompt adds a position (Prompts P1 through P4)? This just checks movement, not whether the movement went the correct way. This is the most important question for a positional-bias study, since a model has to move before "did it move correctly" can mean anything. |
| Q3 | For the left/right prompts specifically, does it land on the correct side? Checked only on Prompt P1 ("locate it on the left") and Prompt P2 ("locate it on the right"), and only meaningful for a model that clears Q2. |
| Q4 | For the upper/lower prompts specifically, does it land on the correct side? Checked only on Prompt P3 ("locate it in the upper region") and Prompt P4 ("locate it in the lower region"), and only meaningful for a model that clears Q2. |
| Q5 | How accurate is it at pure localization, ignoring position wording entirely? How close does its answer to Prompt P0 land to the real target, on a 0 (perfect) to 1.4 (opposite corner) scale? |
| Q6 | Why does it get it wrong? The likely underlying cause behind the numbers in Q1 to Q5. |

| Bias Category | Meaning | Movement (Q2) threshold used here |
|---|---|---|
| Responsive | Moves for most positional prompts. The best group for this study, since it is at least trying to use the prompt's position, whether or not it lands correctly. | 40% or more of prompts move the answer |
| Weakly Responsive | Moves sometimes, but not consistently. | 15% to 40% of prompts move the answer |
| Position-Blind | Barely moves at all, effectively ignoring the positional wording no matter what it says. This is the worst outcome for a positional-bias study, regardless of what its correct-side percentage happens to say. | Under 15% of prompts move the answer |

## Table 2. Brain Tumor Localization (LGG Dataset): Positional Bias Results

*Each row is one model, answering all 6 questions above, on the LGG brain scan dataset only. Sorted by Bias Category first (Responsive models rank above Weakly Responsive, which rank above Position-Blind), then by correct-side accuracy within each category.*

| Bias Category | Model | Q1: Usable answer to Prompt P0? | Q2: Moves when told a position? | Q3: Correct on left/right prompts? | Q4: Correct on upper/lower prompts? | Q5: Distance from true target (0 is perfect, 1.4 is worst) | Q6: Likely reason for its mistakes |
|---|---|---|---|---|---|---|---|
| Responsive | Molmo2-8B | Yes, 88% of the time | Yes, 54% of the time | 87% correct | 99% correct | 0.11 | Very few mistakes; occasionally leaves its answer unchanged rather than moving it the wrong way |
| Responsive | NorthMicroVision | Yes, 100% of the time | Yes, 78% of the time | 93% correct | 72% correct | 0.11 | Noticeably weaker on upper/lower prompts than left/right prompts |
| Responsive | LocateAnything-3B | Yes, 94% of the time | Yes, 50% of the time | 84% correct | 71% correct | 0.09 | Errors look spread out and inconsistent, not tied to one particular side |
| Responsive | CLIPSeg | Yes, 78% of the time | Yes, 43% of the time | 53% correct | 49% correct | 0.09 | Moves for most prompts, but the movement carries no real directional signal |
| Weakly Responsive | Qwen2.5-VL-7B | Yes, 100% of the time | Sometimes, 33% of the time | 50% correct (coin flip) | 50% correct (coin flip) | 0.40 (worst on this dataset) | Moves for a third of prompts, but tends to land in the same general corner regardless of which direction was asked, and rarely overlaps the real tumor even when unprompted |
| Weakly Responsive | Kosmos-2 | Yes, 100% of the time | Sometimes, 29% of the time | 55% correct | 47% correct | 0.13 | Tends to default toward one side regardless of the prompt |
| Weakly Responsive | OWLv2 | Yes, 100% of the time | Sometimes, 28% of the time | 50% correct (coin flip) | 50% correct (coin flip) | 0.11 | Moves for about a quarter of prompts, but the movement carries no real directional signal |
| Position-Blind | Moondream2 | Yes, 100% of the time | Rarely, 15% of the time | 59% correct | 55% correct | 0.10 | Its answer barely changes between prompts on this dataset |
| Position-Blind | Grounding DINO | Yes, 100% of the time | Almost never, 1% of the time | 50% correct (coin flip) | 51% correct (coin flip) | 0.12 | Gives nearly the same answer no matter what the prompt says |
| Position-Blind | Florence-2 | Yes, 100% of the time | Almost never, 1% of the time | 49% correct (coin flip) | 50% correct (coin flip) | 0.12 | Gives nearly the same answer no matter what the prompt says |
| Position-Blind | PaliGemma 2 | Yes, 100% of the time | Almost never, 1% of the time | 50% correct (coin flip) | 49% correct (coin flip) | 0.11 | Gives nearly the same answer no matter what the prompt says |

## Table 3. Colon Polyp Localization (Kvasir Dataset): Positional Bias Results

*Each row is one model, answering all 6 questions above, on the Kvasir endoscopy dataset only. Sorted by Bias Category first (Responsive models rank above Weakly Responsive, which rank above Position-Blind), then by correct-side accuracy within each category.*

| Bias Category | Model | Q1: Usable answer to Prompt P0? | Q2: Moves when told a position? | Q3: Correct on left/right prompts? | Q4: Correct on upper/lower prompts? | Q5: Distance from true target (0 is perfect, 1.4 is worst) | Q6: Likely reason for its mistakes |
|---|---|---|---|---|---|---|---|
| Responsive | Moondream2 | Only 8% of the time, see note below | Yes, 58% of the time (on the rare answers it gives) | 92% correct | 85% correct | 0.09 | Almost always fails to return an answer at all on this dataset |
| Responsive | NorthMicroVision | Yes, 100% of the time | Yes, 63% of the time | 77% correct | 80% correct | 0.12 (after excluding a few broken outliers, see note below) | A small number of garbled, out-of-image answers appear on this dataset |
| Responsive | Grounding DINO | Yes, 100% of the time | Yes, 47% of the time | 80% correct | 59% correct | 0.15 | Follows left/right prompts reasonably well, but not upper/lower ones |
| Responsive | Molmo2-8B | Yes, 100% of the time | Yes, 45% of the time | 79% correct | 85% correct | 0.11 | Sometimes leaves its answer unchanged rather than moving it the wrong way |
| Responsive | CLIPSeg | Yes, 100% of the time | Yes, 40% of the time | 49% correct (coin flip) | 54% correct | 0.11 | Moves for most prompts, but the movement carries no real directional signal |
| Weakly Responsive | LocateAnything-3B | Yes, 98% of the time | Sometimes, 39% of the time | 64% correct | 67% correct | 0.08 | Errors look spread out and inconsistent, not tied to one particular side |
| Weakly Responsive | Qwen2.5-VL-7B | Yes, 98% of the time | Sometimes, 30% of the time | 52% correct (coin flip) | 50% correct (coin flip) | 0.45 (worst on this dataset) | Moves for a third of prompts, but tends to land in the same general corner regardless of which direction was asked, and rarely overlaps the real polyp even when unprompted |
| Position-Blind | Kosmos-2 | Yes, 100% of the time | Rarely, 14% of the time | 53% correct | 51% correct | 0.13 | Tends to always point toward the right side, no matter what the prompt asks |
| Position-Blind | OWLv2 | Only 46% of the time, see note below | Rarely, 15% of the time | 50% correct (coin flip) | 44% correct | 0.09 | Often fails to return an answer, and barely changes the ones it does return |
| Position-Blind | PaliGemma 2 | Yes, 100% of the time | Never moves, 0% of the time | 50% correct (coin flip) | 50% correct (coin flip) | 0.13 | Gives the exact same answer no matter what the prompt says |
| Position-Blind | Florence-2 | Yes, 100% of the time | Never moves, 0% of the time | not measured, see note below | 47% correct | 0.12 | Frequently fails to return an answer to left/right prompts specifically, and never changes the ones it does return |

## Table 4. Skin Lesion Localization (ISIC Dataset): Positional Bias Results

*Each row is one model, answering all 6 questions above, on the ISIC dermoscopy dataset only. Sorted by Bias Category first (Responsive models rank above Weakly Responsive, which rank above Position-Blind), then by correct-side accuracy within each category.*

| Bias Category | Model | Q1: Usable answer to Prompt P0? | Q2: Moves when told a position? | Q3: Correct on left/right prompts? | Q4: Correct on upper/lower prompts? | Q5: Distance from true target (0 is perfect, 1.4 is worst) | Q6: Likely reason for its mistakes |
|---|---|---|---|---|---|---|---|
| Responsive | Molmo2-8B | Yes, 100% of the time | Yes, 82% of the time | 95% correct | 95% correct | 0.16 | Very few mistakes on this dataset |
| Responsive | NorthMicroVision | Yes, 100% of the time | Yes, 64% of the time | 95% correct | 70% correct | 0.09 | Noticeably weaker on upper/lower prompts than left/right prompts |
| Weakly Responsive | Moondream2 | Only 18% of the time, see note below | Sometimes, 35% of the time (on the rare answers it gives) | 91% correct | 67% correct | 0.04 | Almost always fails to return an answer at all on this dataset |
| Weakly Responsive | LocateAnything-3B | Yes, 100% of the time | Sometimes, 34% of the time | 67% correct | 62% correct | 0.07 | Errors look spread out and inconsistent, not tied to one particular side |
| Weakly Responsive | Qwen2.5-VL-7B | Yes, 94% of the time | Sometimes, 31% of the time | 47% correct (coin flip) | 49% correct (coin flip) | 0.38 (worst on this dataset) | Moves for a third of prompts, but tends to land in the same general corner regardless of which direction was asked, and rarely overlaps the real lesion even when unprompted |
| Weakly Responsive | Grounding DINO | Yes, 100% of the time | Sometimes, 25% of the time | 55% correct | 61% correct | 0.05 | Its answer barely changes between prompts, and when it does the direction is unreliable |
| Weakly Responsive | CLIPSeg | Yes, 100% of the time | Sometimes, 22% of the time | 47% correct (coin flip) | 51% correct | 0.04 | Moves for about a fifth of prompts, but the movement carries no real directional signal |
| Position-Blind | Kosmos-2 | Yes, 100% of the time | Rarely, 11% of the time | 50% correct (coin flip) | 53% correct | 0.07 | Tends to default toward one side, no matter what the prompt asks |
| Position-Blind | Florence-2 | Yes, 100% of the time | Almost never, 6% of the time | 50% correct (coin flip) | 48% correct (coin flip) | 0.04 | Gives nearly the same answer no matter what the prompt says |
| Position-Blind | OWLv2 | Yes, 88% of the time | Almost never, 6% of the time | 52% correct (coin flip) | 49% correct (coin flip) | 0.03 | Barely changes its answer between prompts |
| Position-Blind | PaliGemma 2 | Yes, 100% of the time | Almost never, 5% of the time | 50% correct (coin flip) | 49% correct (coin flip) | 0.03 | Gives nearly the same answer no matter what the prompt says |

## Table 5. Notes and Caveats: Reading the Results Correctly

*Read these alongside Tables 2 to 4, they explain the numbers that need context.*

| Note | Explanation |
|---|---|
| A shift-rate bug in an earlier version of this table was fixed | Four of the eleven models (Molmo2-8B, OWLv2, Qwen2.5-VL-7B, Moondream2) came with a "did it move" answer already computed by the pipeline that generated them, while the other seven were computed separately for this report. Those two computations turned out to use different thresholds for "did the answer actually move" (one used 5% of the image's diagonal length, the other used 5% of the image's width or height), so the Q2 figures for those four models were not directly comparable to the other seven. All 11 models are now recalculated the same way, from the raw coordinates, using one consistent threshold. This raised Qwen2.5-VL-7B's measured movement noticeably (for example, from about 14% to about 33% on the brain scan dataset); its Q3, Q4, and Q5 numbers were not affected, since those never depended on this threshold. |
| NorthMicroVision's raw distance number needed cleaning | 21 of its 750 answers across all three datasets (about 3%, all on the Kvasir dataset) contain coordinates far outside the actual image, for example a value of 300,000 on an image that is only 622 pixels wide. This happens because the model's text answer isn't always in the same format (sometimes plain pixel numbers, sometimes fractions of the image), and a few of them get misread during conversion to pixel coordinates. Those 21 broken answers were excluded before averaging. Its Q3/Q4 percentages were never affected, since side is judged from which half of the image the center falls in. |
| Some models rarely return a usable answer at all (Moondream2, OWLv2, Florence-2 on Kvasir) | This is a coverage problem, not an accuracy problem. Their Q3/Q4 percentages are only calculated from the answers they do give, so they can look better than how reliable the model actually is in everyday use, since a large share of calls return nothing usable. |
| Florence-2's Q3 on Kvasir is "not measured" | It returned no usable answer at all on any P1 or P2 (left/right) prompt for this dataset, so there is nothing to score its left/right accuracy against. Its P3/P4 (upper/lower) answers did come through, hence Q4 still has a value. |
| A score close to 50% in Q3 or Q4 does not mean "half right, half wrong" | It means the model's choice of side carries no real information from the prompt, the same as flipping a coin. |
| Why Bias Category outranks correct-side accuracy in the sort order | This is a study of whether models respond to positional language, so a model that never moves (Position-Blind) is scored as worse than one that moves but sometimes lands wrong (Responsive or Weakly Responsive), even if the frozen model's coin-flip correct-side number looks similar or higher on paper. A frozen model gives no evidence either way about whether it understands position; a model that moves and gets it wrong at least engages with the instruction, which is a different and more informative failure mode for this research question. |

*Source: raw per-row prediction files in `Results/<dataset>/`. Regenerate the underlying numbers with `scripts/aggregate_findings.py` and `scripts/aggregate_prompt_shift.py`.*
