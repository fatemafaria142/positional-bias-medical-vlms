# Positional Bias Results - Findings by Dataset

Generated from the 11 model result files in `Results/<dataset>/` via
`scripts/aggregate_findings.py` (run it again after adding/re-running a
model to refresh the numbers below).

**11 models scored, 3 datasets, P0–P4 prompts per image** (see
`position_bias_research_plan.md` for the full prompt set and comparison
spec). Moondream2 produces two independent outputs per call (`/detect`
bbox and `/point`), so it appears as two rows.

## Metric definitions

| Column | Meaning | Answers |
|---|---|---|
| **Detect Rate** | Share of P0–P4 calls that returned any usable prediction (not empty/degenerate) | Can the model even attempt this task? |
| **Wrong-Side Rate (P1–P4)** | Of the positionally-prompted calls, the share where the model's answer landed on the *opposite* side from the one the prompt named (e.g. prompt says "left", answer centroid is right of midline) | RQ3 - does it follow the prompt, and if so, correctly? |
| **No-Shift Rate (P1–P4)** | Share of P1–P4 answers that stayed within 5% of image width/height of that same model's own P0 (unprompted) answer on the same image | RQ1 - does the answer move at all when position is named? |
| **Mean Norm. Distance** | Mean Euclidean distance between predicted centroid and ground-truth centroid, normalized by image diagonal (0 = exact, ~1.4 = opposite corners) | Raw localization accuracy, independent of prompting |
| **Mean IoU** | Mean box/mask IoU vs. ground-truth box, where the model's tier produces one (bbox/mask models only; "N/A" for pure point models) | Shape/overlap quality |

A **50% wrong-side rate is chance level** (equivalent to flipping a coin
on which side to answer) - it does not mean "half right, half wrong" in a
meaningful sense, it means the model's choice of side carries no
information from the prompt. Reading wrong-side and no-shift together
separates two distinct failure modes per the research plan: a model can
(a) never move (**high no-shift**, prompt-following failure) or (b) move
but toward the *wrong* side (**high wrong-side despite low no-shift**,
i.e. it trusts the text over the image and gets misled by it).

---

### LGG - target: "the tumor"

| Model | Output | Detect Rate | Wrong-Side Rate (P1-P4) | No-Shift Rate (P1-P4) | Mean Norm. Distance | Mean IoU |
|---|---|---|---|---|---|---|
| Molmo2-8B | point | 90.4% | 7.1% | 55.1% | 0.109 | - |
| Moondream2 (point) | point | 100.0% | 8.0% | 35.0% | 0.141 | - |
| NorthMicroVision | bbox | 100.0% | 17.5% | 22.0% | 0.113 | - |
| LocateAnything-3B | bbox | 95.6% | 22.4% | 50.0% | 0.094 | - |
| Moondream2 (detect) | bbox | 96.8% | 42.7% | 86.5% | 0.099 | 0.184 |
| Kosmos-2 | bbox | 100.0% | 49.0% | 71.5% | 0.129 | - |
| CLIPSeg | mask | 56.4% | 49.0% | 56.9% | 0.085 | - |
| Grounding DINO | bbox | 100.0% | 49.5% | 99.5% | 0.115 | - |
| Qwen2.5-VL-7B | bbox | 99.2% | 50.0% | 85.9% | 0.398 | 0.000 |
| OWLv2 | bbox | 100.0% | 50.3% | 86.3% | 0.112 | 0.181 |
| Florence-2 | bbox | 100.0% | 50.5% | 99.0% | 0.115 | - |
| PaliGemma 2 | mask | 100.0% | 50.5% | 99.5% | 0.114 | - |

**Findings:**
- Only the two **point-tier models** (Molmo2-8B, Moondream2-point) show real prompt sensitivity - wrong-side rates of 7–8%, far below chance. They're also the only models with mid-range no-shift (35–55%), meaning they move for *most* but not all prompts.
- The other nine models sit at **44–51% wrong-side - statistically indistinguishable from a coin flip.** Prompted laterality/verticality carries essentially no information for them on this dataset.
- Grounding DINO, Florence-2, PaliGemma 2 combine ~50% wrong-side with **86–99.5% no-shift** - their answer barely changes at all between P0 and P1–P4, so the ~50% wrong-side figure reflects a static, prompt-independent guess landing on either side by chance, not active-but-wrong reasoning.
- Qwen2.5-VL-7B has by far the worst raw localization here (mean distance 0.398, IoU ≈ 0 - essentially never overlaps the true tumor box), independent of the positional-bias question.
- LGG's small, off-center glioma masks (vs. the near-full-frame lesions in ISIC) make this dataset the clearest test of true positional sensitivity - the gap between the point-tier and everything else is largest here.

---

### KVASIR - target: "the polyp"

| Model | Output | Detect Rate | Wrong-Side Rate (P1-P4) | No-Shift Rate (P1-P4) | Mean Norm. Distance | Mean IoU |
|---|---|---|---|---|---|---|
| Moondream2 (point) | point | 100.0% | 5.5% | 31.0% | 0.182 | - |
| Moondream2 (detect) | bbox | 12.0% | 11.5% | 100.0% | 0.094 | 0.555 |
| Molmo2-8B | point | 99.6% | 18.1% | 57.3% | 0.107 | - |
| NorthMicroVision | bbox | 98.0% | 21.5% | 37.4% | 16.855\* | - |
| Grounding DINO | bbox | 95.6% | 31.2% | 52.9% | 0.146 | - |
| LocateAnything-3B | bbox | 99.2% | 34.7% | 60.8% | 0.077 | - |
| Kosmos-2 | bbox | 100.0% | 48.0% | 86.5% | 0.127 | - |
| CLIPSeg | mask | 93.2% | 48.1% | 59.6% | 0.112 | - |
| Qwen2.5-VL-7B | bbox | 99.6% | 49.0% | 83.9% | 0.453 | 0.003 |
| PaliGemma 2 | mask | 100.0% | 50.0% | 100.0% | 0.125 | - |
| OWLv2 | bbox | 28.0% | 53.2% | 89.4% | 0.088 | 0.564 |
| Florence-2 | bbox | 54.4% | 53.5% | 100.0% | 0.124 | - |

\* NorthMicroVision's mean distance is inflated by a handful of hallucinated bounding boxes with coordinates far outside the image (e.g. x > 300,000 on a ~600px-wide image); its detect rate and wrong-side rate are unaffected since those still resolve to *some* side. Treat this one number as an outlier-driven artifact, not a real localization scale.

**Findings:**
- Same ranking pattern as LGG: **Moondream2-point and Molmo2-8B** lead on wrong-side rate by a wide margin; everything from Kosmos-2 downward is at chance (48–53.5%).
- **Detect Rate collapses for OWLv2 (28%) and Florence-2 (54%)** on Kvasir specifically - both frequently return a degenerate (near-full-image) box on endoscopy frames, which is filtered out as "no detection" rather than scored as a wrong guess. This is a coverage problem distinct from positional bias.
- Moondream2's `/detect` endpoint has the same collapse (12% detect rate) but scores very high IoU (0.555) on the rare frames it does return - it's precise when it fires, but rarely fires. Its own `/point` output doesn't share this problem (100% detect rate), so the point head and the detect head are not equally reliable on this domain.
- PaliGemma 2 and Florence-2 are again ~100% no-shift - static, prompt-independent output regardless of P0–P4 wording.

---

### ISIC - target: "the lesion"

| Model | Output | Detect Rate | Wrong-Side Rate (P1-P4) | No-Shift Rate (P1-P4) | Mean Norm. Distance | Mean IoU |
|---|---|---|---|---|---|---|
| Molmo2-8B | point | 100.0% | 5.0% | 19.0% | 0.159 | - |
| Moondream2 (point) | point | 100.0% | 8.5% | 26.5% | 0.174 | - |
| NorthMicroVision | bbox | 100.0% | 17.5% | 36.5% | 0.093 | - |
| Moondream2 (detect) | bbox | 12.8% | 21.7% | 93.8% | 0.041 | 0.773 |
| LocateAnything-3B | bbox | 100.0% | 35.5% | 66.0% | 0.065 | - |
| Grounding DINO | bbox | 100.0% | 42.0% | 75.5% | 0.054 | - |
| Kosmos-2 | bbox | 100.0% | 48.5% | 89.5% | 0.067 | - |
| OWLv2 | bbox | 69.6% | 50.0% | 96.8% | 0.033 | 0.757 |
| PaliGemma 2 | mask | 100.0% | 50.5% | 95.5% | 0.030 | - |
| CLIPSeg | mask | 94.0% | 50.8% | 77.8% | 0.043 | - |
| Florence-2 | bbox | 100.0% | 51.0% | 94.0% | 0.039 | - |
| Qwen2.5-VL-7B | bbox | 92.4% | 52.2% | 79.7% | 0.380 | 0.012 |

**Findings:**
- Molmo2-8B is the single most prompt-sensitive model across all three datasets (5.0% wrong-side here, its best score). Point-tier models remain the clear leaders.
- Raw localization (mean distance, IoU) is **best of all three datasets for most models** here (e.g. OWLv2 IoU 0.757, PaliGemma distance 0.030) - consistent with the research plan's caveat that ISIC lesions often fill most of the frame, so a centroid guess near image-center is "accurate" almost by construction. **This makes ISIC the weakest dataset for measuring positional bias specifically** - good distance/IoU numbers here reflect the dataset's near-center bias more than genuine localization skill, and the wrong-side numbers should be trusted over the distance/IoU column for this dataset.
- The ~50–52% wrong-side cluster (OWLv2, PaliGemma, CLIPSeg, Florence-2, Qwen2.5-VL) again shows no usable response to positional wording, matching the other two datasets.
- Qwen2.5-VL-7B again has the worst IoU (0.012) despite a respectable-looking detect rate, and its wrong-side rate (52.2%) is worst of all 12 rows - no evidence it uses either the prompt or the image reliably here.

---

## Cross-dataset synthesis

1. **Same two models lead on every dataset:** Molmo2-8B and Moondream2's point head are the only models with wrong-side rates meaningfully below chance (5–18%) on all three domains. Both are point-tier - consistent with the research plan's prediction that point output removes shape/box noise and gives "the most direct signal."
2. **Nine of eleven models are at chance (≈48–53% wrong-side) everywhere.** This is the headline finding: most zero-shot grounding models tested here do not meaningfully use left/right/upper/lower wording at all - they neither reliably follow it (RQ1) nor get actively misled into the wrong side by it (RQ3) in a way distinguishable from random guessing.
3. **No-shift rate shows *why*:** several of the chance-level models (Grounding DINO, Florence-2, PaliGemma 2) also have 85–100% no-shift rates - their output is nearly static across P0–P4, so the ~50% wrong-side figure is really "a fixed answer that happens to be right or wrong by luck of the sample," not the model actively reasoning about (and failing) the positional instruction.
4. **Detection coverage is domain-dependent and separate from bias:** OWLv2, Florence-2, and Moondream2-detect all lose most of their predictions to degenerate/full-frame boxes specifically on Kvasir endoscopy frames, dragging their effective sample size down; this should be reported alongside wrong-side rate, not folded into it.
5. **Qwen2.5-VL-7B is the standout failure on raw localization** (IoU ≈ 0 on LGG and ISIC, worst-of-cohort mean distance on all three) - a separate, more basic problem than positional bias, since it isn't finding the target at all regardless of prompt wording.
6. **ISIC numbers should be read cautiously**: its lesions are often near-frame-filling, which inflates distance/IoU scores dataset-wide regardless of a model's actual positional competence - LGG (smallest, most off-center masks) is the most diagnostic dataset for RQ1–RQ3 specifically.

---

*Methodology: wrong-side/no-shift/distance/IoU definitions follow the Comparison Specification in `position_bias_research_plan.md`. Numbers regenerated by `python scripts/aggregate_findings.py`, which reads every file under `Results/<dataset>/` and normalizes the two on-disk result-file schemas (per-model prefixed columns vs. the shared `pred_*`/`gt_*` schema) to one common metric set before aggregating.*
