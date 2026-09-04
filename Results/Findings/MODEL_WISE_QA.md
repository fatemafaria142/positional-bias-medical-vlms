# Model-Wise Positional Bias - 4 Questions Per Model

One card per model, same 4 questions every time, so each can be read (or
presented) on its own without cross-referencing a big table. Numbers are
averaged across all 3 datasets (LGG, Kvasir, ISIC) - see `FINDINGS.md`
and `PROMPT_WISE_SHIFT.md` for the per-dataset and per-prompt breakdowns
these are built from.

**The 4 questions, asked of every model:**

1. **Does its answer shift when the prompt's position changes, and how often?**
2. **When it does answer, does it land on the correct (prompted) side, and how often?**
3. **How good is its raw localization, independent of prompting?**
4. **What's the main reason it fails when it fails?**

Models are ordered best → worst by Question 2 (correct-side rate).

---

## 1. Moondream2 - `/point` endpoint

1. **Shift?** Yes, and frequently - its point moves for **69.2%** of positional prompts.
2. **Correct side?** Yes, most often of all 11 models - **92.7%** correct-side rate.
3. **Raw localization?** Good - mean normalized distance 0.166 (point tier, no IoU applicable).
4. **Reason for the remaining ~7% failures:** Its rare misses cluster on samples where the target sits very close to the image's midline or midheight (mostly on ISIC, where lesions often fill most of the frame - see `FINDINGS.md`'s ISIC caveat) - the "correct side" is a coin flip on those cases regardless of the model.

---

## 2. Molmo2-8B

1. **Shift?** Yes - **56.2%** of positional prompts move its point.
2. **Correct side?** Yes, second-best overall - **89.9%** correct-side rate.
3. **Raw localization?** Good - mean normalized distance 0.125, the best of all 11 models.
4. **Reason for the remaining ~10% failures:** Its no-shift rate (~44%) is higher than Moondream2-point's - on almost half of positional prompts it keeps its P0 answer unchanged, so most residual wrong-side cases are prompt-following gaps (didn't move) rather than moving the wrong way.

---

## 3. NorthMicroVision

1. **Shift?** Yes, the most of any bbox model - **68.0%**.
2. **Correct side?** Mostly - **81.2%** correct-side rate, third overall.
3. **Raw localization?** Cannot be trusted as reported - mean normalized distance is inflated to 5.69 by a handful of hallucinated boxes with coordinates far outside the image (e.g. x > 300,000 on a ~600px-wide Kvasir frame); side classification is unaffected by this, but treat the distance number as broken, not literal.
4. **Reason for the remaining ~19% failures:** A real mix of genuine misjudgment plus the same raw-output parsing instability that produces the distance outliers above - this model occasionally emits ungrounded coordinate text that isn't tied to the image at all.

---

## 4. Moondream2 - `/detect` endpoint

1. **Shift?** Rarely - only **6.6%** of positional prompts move its box.
2. **Correct side?** Yes, when it answers - **74.7%** correct-side rate.
3. **Raw localization?** Good on the calls that succeed - mean IoU 0.504, one of the two highest of all 11 models.
4. **Reason for failure - this is a coverage problem, not a positional one:** Detect rate is only **40.5%** overall and collapses to **12%** on Kvasir specifically - most calls return a degenerate, near-full-image box that gets discarded as "no detection." The 74.7%/0.504 numbers above are computed only on the minority of calls that survive, so they overstate how often this endpoint is actually usable.

---

## 5. LocateAnything-3B

1. **Shift?** Sometimes - **41.1%** of positional prompts move its box.
2. **Correct side?** More often than not - **69.1%** correct-side rate.
3. **Raw localization?** Good - mean normalized distance 0.079, among the best of all 11 models.
4. **Reason for the remaining ~31% failures:** No single dominant pattern (its directional imbalance across P1–P4 is the second-lowest of all 11 models, 0.197) - its errors look like genuine, evenly-spread misjudgment rather than a fixed bias toward one side.

---

## 6. Grounding DINO

1. **Shift?** Rarely - **24.0%** of positional prompts move its box.
2. **Correct side?** Below coin-flip - **59.1%** correct-side rate.
3. **Raw localization?** Reasonable - mean normalized distance 0.105 (no IoU computed for this model in this pipeline).
4. **Reason for failure:** Mostly a **frozen-output problem** - on LGG specifically it's essentially motionless (0–2% shift) and its correct-side rate there is driven mostly by which side its one fixed answer happens to land on, not active reasoning about the prompt.

---

## 7. Kosmos-2

1. **Shift?** Rarely - only **17.5%** of positional prompts move its box, the second-lowest of all 11 models.
2. **Correct side?** Coin-flip - **51.5%** correct-side rate.
3. **Raw localization?** Reasonable - mean normalized distance 0.108.
4. **Reason for failure - a fixed positional anchor, not randomness:** Kosmos-2 has the third-highest directional imbalance of all 11 models (0.607). On ISIC and Kvasir it is wrong on **~82%** of "left" prompts but only **~15%** of "right" prompts - it defaults to the right side of the image almost regardless of what's asked, and rarely moves off that default (see point 1). Averaging left+right prompts together makes this look like ~50% chance performance; per-prompt, it isn't chance at all.

---

## 8. CLIPSeg

1. **Shift?** Somewhat - **35.2%** of positional prompts move its mask centroid, mid-pack.
2. **Correct side?** Coin-flip - **50.7%** correct-side rate.
3. **Raw localization?** Reasonable - mean normalized distance 0.080.
4. **Reason for failure:** Unlike the frozen models above, CLIPSeg *does* react to the appended phrase (it has one of the higher shift rates of the bbox/mask tier) - but that reaction doesn't reliably track the requested direction. Its segmentation mask seems to shift in response to the changed text salience/wording rather than to genuine left/right/upper/lower understanding, so movement and correctness are largely uncorrelated for this model.

---

## 9. PaliGemma 2

1. **Shift?** Essentially never - **1.7%** of positional prompts move its output, the second-lowest of all 11 models.
2. **Correct side?** Coin-flip - **49.7%** correct-side rate.
3. **Raw localization?** Reasonable - mean normalized distance 0.090.
4. **Reason for failure - total prompt-following failure:** With a shift rate this close to zero, PaliGemma 2 outputs almost the same segmentation for P0 through P4 on the same image. Its fixed `segment {X}` calling syntax appears to anchor on the target noun and effectively ignore the appended positional clause; the ~50% correct-side rate is just whichever side that single static answer happens to fall on per image, not a signal that it's reasoning about position at all.

---

## 10. Qwen2.5-VL-7B

1. **Shift?** Rarely - **16.9%** of positional prompts move its box.
2. **Correct side?** Coin-flip on average, but not for the reason it looks like - **49.6%** correct-side rate overall.
3. **Raw localization?** Poor - the worst of all 11 models by far: mean normalized distance 0.410 (next-worst model is ~0.17) and mean IoU 0.005 (near-total non-overlap with the true target box).
4. **Reason for failure - two separate problems stacked together:** (a) It has the single **highest directional imbalance of any model (1.00)** - on all three datasets it is wrong on 96–100% of "right"/"lower" prompts while being wrong on only 0–6% of "left"/"upper" prompts, i.e. it defaults to a fixed top-left-ish answer almost regardless of the prompt (see `PROMPT_WISE_SHIFT.md` Q3 for the full breakdown). (b) Independent of that, its raw grounding is close to random - near-zero IoU means even its "correct-side" hits are frequently not actually on the target, just on the correct half of the image by coincidence.

---

## 11. OWLv2

1. **Shift?** Rarely - **9.2%** of positional prompts move its box, the second-lowest shift rate of all 11 models.
2. **Correct side?** Coin-flip - **48.8%** correct-side rate.
3. **Raw localization?** Good on the calls that succeed - mean IoU 0.501, tied for the highest of all 11 models.
4. **Reason for failure - coverage plus a frozen output:** Detect rate is only **65.9%** overall and drops to **28%** on Kvasir (many predictions are degenerate near-full-image boxes and get discarded, the same pattern as Moondream2-detect above). On the detections that do survive, the box barely moves between P0 and P1–P4 (9.2% shift), so the ~50% correct-side figure again reflects a static answer's luck-of-the-draw side, not genuine positional reasoning.

---

## 12. Florence-2

1. **Shift?** Essentially never - **2.3%** of positional prompts move its box, the lowest of all 11 models.
2. **Correct side?** Coin-flip - **48.3%** correct-side rate, the lowest of all 11 models.
3. **Raw localization?** Reasonable - mean normalized distance 0.093.
4. **Reason for failure - total prompt-following failure:** Florence-2's `<CAPTION_TO_PHRASE_GROUNDING>` task conditions almost entirely on the target noun phrase; the appended positional clause ("...on the left side of this image") barely changes its output tokens at all. With a 2.3% shift rate, it is functionally answering the same P0 question five times regardless of wording - the worst prompt-following performance of any model tested.

---

## Summary table (for quick reference)

| Rank | Model | Shift Rate | Correct-Side Rate | Mean Norm. Dist. | Mean IoU | Main failure reason |
|---|---|---|---|---|---|---|
| 1 | Moondream2 (point) | 69.2% | 92.7% | 0.166 | - | Near-center targets (ISIC) only |
| 2 | Molmo2-8B | 56.2% | 89.9% | 0.125 | - | Occasional no-shift, not wrong-shift |
| 3 | NorthMicroVision | 68.0% | 81.2% | 5.69\* | - | Genuine misjudgment + raw-output parsing noise |
| 4 | Moondream2 (detect) | 6.6% | 74.7%\*\* | 0.078 | 0.504 | Coverage - 40–60% of calls discarded as degenerate |
| 5 | LocateAnything-3B | 41.1% | 69.1% | 0.079 | - | Evenly-spread genuine misjudgment |
| 6 | Grounding DINO | 24.0% | 59.1% | 0.105 | - | Frozen output on LGG |
| 7 | Kosmos-2 | 17.5% | 51.5% | 0.108 | - | Fixed anchor toward the right side |
| 8 | CLIPSeg | 35.2% | 50.7% | 0.080 | - | Reacts to text, not to true position |
| 9 | PaliGemma 2 | 1.7% | 49.7% | 0.090 | - | Total prompt-following failure |
| 10 | Qwen2.5-VL-7B | 16.9% | 49.6% | 0.410 | 0.005 | Fixed top-left anchor + poor grounding overall |
| 11 | OWLv2 | 9.2% | 48.8% | 0.078 | 0.501 | Coverage problem + frozen output |
| 12 | Florence-2 | 2.3% | 48.3% | 0.093 | - | Total prompt-following failure |

\* Distance figure unreliable - see card 3.
\*\* Computed only on the ~40% of calls with a valid (non-degenerate) detection - see card 4.

*Methodology: all rates are the simple mean of that model's per-dataset value in `Results/Findings/FINDINGS.md` across LGG, Kvasir, and ISIC. Directional-imbalance claims (Kosmos-2, Qwen2.5-VL-7B) come from `Results/Findings/PROMPT_WISE_SHIFT.md`. Regenerate the underlying numbers with `python scripts/aggregate_findings.py` and `python scripts/aggregate_prompt_shift.py`.*
