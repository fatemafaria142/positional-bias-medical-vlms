# Which experiments support the title? Language vs. image region

Title: *Disentangling the Relative Contributions of Language and Region to Positional Hallucinations in VLMs*

Key: ✅ tests it directly · ◐ partly or indirectly · ❌ does not test it · ➖ baseline only

| # | Task | Branch | What is changed | Language | Image region | Result | Strength |
|---|---|---|---|---|---|---|---|
| 1 | 2D VLM positional prompt sweep (P0–P4) | `faria/` and `nipu/medical-positional-bias` | Only the position wording | ✅ | ❌ | 9 of 11 models are at chance (about 50% wrong-side). Moondream2-point 92.7% and Molmo2 89.9% correct-side. | Strong for Language |
| 2 | 2D VLM prompt vs true location (wrong-side rate, no-shift rate) | Same | Prompt names a side that may not match the target | ✅ | ◐ (target location is used as truth, image not altered) | Grounding DINO, Florence-2 and PaliGemma 2 barely move (85–100% no-shift), so they ignore the prompt. | Medium |
| 3 | VoxTell baseline on AMOS and OASIS | `faria/voxtell3d-medical-segmentation` | Nothing | ➖ | ➖ | Dice 0.83 (AMOS), 0.74 (OASIS). | Needed as the starting point |
| 4 | Text3DSAM baseline | `faria/3d-segmentation-datasets`, `faria/text3dsam-multi-dataset-experiments` | Nothing | ➖ | ➖ | Dice 0.10 (AMOS), 0.00 (OASIS). | Not usable as evidence |
| 5 | MedSAM3 | `faria/voxtell3d-medical-segmentation` | Nothing | ➖ | ➖ | Predictions and notebooks exist; results not reviewed. | Unreviewed |
| 6 | Mirror-flip test | `faria/voxtell3d-medical-segmentation` | Image flipped left/right, ground-truth label left uncorrected | ❌ | ◐ | Dice charts exist; no write-up reviewed. | Weak |
| 7 | **Rotation sweep (0–180°)** | `faria/voxtell3d-rotation-medical-segmentation` | Only the image rotates | ❌ (prompt fixed) | ✅ | Dice 0.83 → 0.07 (AMOS), 0.74 → 0.05 (OASIS). At 150–180°, 46–65% of masks land on the opposite-side organ (author's own calculation from the saved masks; not yet a script in the repo). | **Strongest for Region** |
| 8 | Row 1: image-text alignment | Same | Prompt names the wrong side | ✅ | ◐ | The wrong prompt gives 0.000 overlap with the true location in all 20 AMOS cases. | Strong |
| 9 | Row 2: embedding distance vs output distance | Same | Left/right word | ✅ | ❌ | Inconclusive: only two organ types exist for the regression. | Weak |
| 10 | Row 3: embedding swap | Same | Swaps the text embedding for the other side's | ✅ | ❌ | The mask follows the swap 100% of the time. | **Strongest causal test for Language** |
| 11 | Row 4: fused feature diff | Same | Left vs right prompt | ✅ | ❌ | Left and right representations diverge clearly inside the model. | Medium |
| 12 | Rows 5–7: exact Shapley, SHAP, LIME | Same | Words are removed from the prompt | ✅ | ❌ | The organ word contributes several times more than the position word; all three methods agree. | Strong |
| 13 | Rows 8–11: metric variants (Dice, self-Dice, centroid shift, volume) | Same | Words are removed | ✅ | ❌ | Removing the organ word is catastrophic. Removing the position word gives a both-hemisphere mask. | Strong |
| 14 | Sidedness-direction analysis | Same | Nothing (reads internal layers) | ✅ | ❌ | Fusion makes the left/right concept more organ-specific, not more general. | Medium |
| 15 | Row 13: organ replaced by noise | Same | Only the organ's voxels | ❌ | ✅ | Kidney needs its own voxels. Adrenal mask keeps about 80% of its size even when the gland is noise. | Medium (3 subjects) |
| 16 | Row 14: embedding blend and layer patching | Same | Text embedding | ✅ | ❌ | The side flips sharply between alpha 0.25 and 0.75. Layers 1–3 together set the side. | Medium |
| 17 | Row 15: attention enrichment | Same | Nothing (reads attention) | ◐ | ✅ | Attention on the named kidney is 2.4–8.6x the volume average. Adrenal is weak. | Medium (3 subjects) |
| 18 | Row 16: layer distance | Same | Nothing | ✅ | ❌ | Left/right distance is always smaller than organ-vs-organ distance. | Weak |

## Totals

- **Language:** tasks 1, 2, 8–14 and 16, with task 18 as a weak extra. Well covered.
- **Image region:** tasks 7 and 15, and partly 6 and 17. Only task 7 (rotation sweep) is strong.
- **Neither:** tasks 3–5 are baselines.

## Caveats

- Rows 13–16 used only 3 AMOS subjects; the write-up calls them a pilot.
- OASIS segments only 2 of its 5 structure pairs correctly, so pooled OASIS numbers mostly measure failure.
- The rotation opposite-side result is a separate calculation from the saved masks, not yet in the repo as a script.
- No task compares language and region on one scale, so "Relative Contributions" in the title is still open.
