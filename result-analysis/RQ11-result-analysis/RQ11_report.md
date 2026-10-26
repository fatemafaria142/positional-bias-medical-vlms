# RQ11: When the Prompt Gives Conflicting or No Information About the Side, Does the Model Lean to the Left or to the Right?

## What this research question asks

In the earlier questions the prompt was always clear, such as "left hippocampus" or "right hippocampus". Here we make the prompt confusing on purpose. In some prompts we name one side and then ask for the other side, and in one prompt we name no side at all. Then we check whether the predicted mask ends on the left organ or on the right organ. If the model has a habit, it will lean to the same side when it is confused.

**Setting.** VoxTell on the Neurite-OASIS dataset, for 16 bilateral organ pairs and the first 50 subjects, which gives 800 subject-organ cases for each of 4 prompt variants. The variants are written below, with X standing for the organ.

- Left named first, then right: "Segment the left X in this image. Ignore that, segment the right X instead."
- Right named first, then left: "Segment the right X in this image. Ignore that, segment the left X instead."
- No side named, with a distractor sentence: "Segment the X in this image. This scan was acquired under standard protocol."
- Side named only in an added sentence: "Segment the X in this image. Actually, make it the right one."

## Procedure

What happens in one test.

1. Take one scan with, for example, a left hippocampus and a right hippocampus.
2. Give the model a prompt that contradicts itself, such as "Segment the left hippocampus in this image. Ignore that, segment the right hippocampus instead."
3. The model returns one mask.
4. We check where the mask landed, which is on the true left hippocampus, on the true right one, on both, on neither, or empty.

This test is repeated for every organ, every subject and every prompt variant.

## Definitions

- **Mask on the left organ or on the right organ.** The predicted mask is compared with the true left organ and the true right organ. It is on one side when it covers that organ best and not the other. The remaining cases, where the mask covers both organs, covers neither organ or is empty, are counted as mask on neither side.
- **Lean.** The share of masks on the right organ minus the share of masks on the left organ, in percentage points. A positive value leans to the right and a negative value leans to the left.
- **95% interval.** Obtained by resampling the 50 subjects 2,000 times, because the 16 organs of one subject are not independent.

## Tables

**Table 1. Where the predicted mask lands for each prompt type, over 800 subject-organ cases.** Read each row as one prompt type. The three share columns add up to 100%. A positive lean means the masks go more often to the right organ, and a negative lean means they go more often to the left organ.

| Prompt type | Example prompt (X is the organ) | Mask on the LEFT organ, % of cases (95% interval) | Mask on the RIGHT organ, % of cases (95% interval) | Mask on neither side, % of cases | Lean, right minus left (percentage points) | Leans to |
|---|---|---|---|---|---|---|
| Left named first, then right | Segment the left X in this image. Ignore that, segment the right X instead. | 0.0% (0.0 to 0.0) | 58.6% (56.1 to 61.0) | 41.4% | +58.6 | Right |
| Right named first, then left | Segment the right X in this image. Ignore that, segment the left X instead. | 36.6% (34.5 to 38.6) | 0.0% (0.0 to 0.0) | 63.4% | -36.6 | Left |
| No side named, distractor sentence | Segment the X in this image. This scan was acquired under standard protocol. | 10.1% (8.4 to 11.8) | 4.2% (3.1 to 5.4) | 85.6% | -5.9 | Left |
| Side named only in an added sentence | Segment the X in this image. Actually, make it the right one. | 0.0% (0.0 to 0.0) | 70.8% (69.4 to 72.1) | 29.2% | +70.8 | Right |

**Table 2. Where the predicted mask lands for each organ and each prompt type, over 50 subjects.** Each cell gives the percentage of subjects whose mask is on the left organ, then the percentage whose mask is on the right organ. The organs are sorted from the strongest lean to the right to the strongest lean to the left, averaged over the four prompt types.

| Organ | Left named first, then right (left organ / right organ) | Right named first, then left (left organ / right organ) | No side named, distractor sentence (left organ / right organ) | Side named only in an added sentence (left organ / right organ) |
|---|---|---|---|---|
| Thalamus | 0% / 72% | 12% / 0% | 0% / 14% | 0% / 98% |
| Cerebral white matter | 0% / 66% | 0% / 0% | 0% / 0% | 0% / 100% |
| Hippocampus | 0% / 100% | 44% / 0% | 0% / 0% | 0% / 100% |
| Amygdala | 0% / 100% | 52% / 0% | 2% / 10% | 0% / 100% |
| Caudate | 0% / 94% | 62% / 0% | 0% / 12% | 0% / 98% |
| Cerebellum cortex | 0% / 100% | 88% / 0% | 0% / 20% | 0% / 100% |
| Cerebellum white matter | 0% / 50% | 34% / 0% | 0% / 2% | 0% / 100% |
| Pallidum | 0% / 100% | 92% / 0% | 8% / 4% | 0% / 100% |
| Putamen | 0% / 98% | 100% / 0% | 22% / 4% | 0% / 100% |
| Inferior lateral ventricle | 0% / 6% | 2% / 0% | 10% / 2% | 0% / 76% |
| Accumbens | 0% / 54% | 0% / 0% | 86% / 0% | 0% / 100% |
| Ventral DC | 0% / 98% | 100% / 0% | 34% / 0% | 0% / 60% |
| Choroid plexus | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| Cerebral cortex | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| Lateral ventricle | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| Vessel | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |

## Figures

**Figure 1. Side of the predicted mask for each organ and each prompt.** Each tile gives the percentage of the 50 subjects with the mask on the left organ (L) and on the right organ (R).

**Figure 2. Share of masks on the left organ and on the right organ for each prompt.** The shares are out of the 800 subject-organ cases, and the rest of the cases have no side.

**Figure 3. Ground truth and predicted masks of the cerebral white matter of one subject for the four prompts.** L and R are the Dice of the mask with the true left and the true right organ.

**Figure 4. Ground truth and predicted masks of the cerebellum cortex of one subject for the four prompts.** L and R are the Dice of the mask with the true left and the true right organ.

**Figure 5. Ground truth and predicted masks of the cerebellum white matter of one subject for the four prompts.** L and R are the Dice of the mask with the true left and the true right organ.

**Figure 6. Ground truth and predicted masks of the hippocampus of one subject for the four prompts.** L and R are the Dice of the mask with the true left and the true right organ.

## Results

1. **The bias is to the right when the prompt contradicts itself.** When the last side named is right, 58.6% of the masks land on the right organ. When the last side named is left, only 36.6% land on the left organ, and the two 95% intervals (56.1 to 61.0 and 34.5 to 38.6) do not overlap (Table 1, Figure 2). The model never keeps the side named first, with 0.0% in both prompts. So a "right" instruction is followed more often than a "left" instruction, by 22.0 points.
2. **The bias is to the right when the side appears only in an added sentence.** For "Actually, make it the right one", 70.8% of the masks land on the right organ and none land on the left organ (Table 1, Figure 2). This is the highest share on the right of all four prompts.
3. **The bias is to the left when no side is named.** The mask lands on the left organ in 10.1% of the cases and on the right organ in 4.2%, and the 95% intervals (8.4 to 11.8 and 3.1 to 5.4) do not overlap (Table 1, Figure 2). This is a small bias, because 85.6% of the masks land on neither side.
4. **The right bias in the contradicting prompts is strongest in these organs.** Thalamus, cerebral white matter, accumbens and inferior lateral ventricle follow a final "right" in 72%, 66%, 54% and 6% of the subjects, but a final "left" in only 12%, 0%, 0% and 2% (Table 2, Figure 1). Hippocampus (100% against 44%), amygdala (100% against 52%) and caudate (94% against 62%) also follow "right" much more often than "left".
5. **Some organs show no right bias.** Putamen (98% against 100%), ventral DC (98% against 100%) and pallidum (100% against 92%) follow a final "right" and a final "left" almost equally (Table 2, Figure 1).
6. **The left bias with no side named comes mainly from three organs.** The mask lands on the left organ in 86% of the subjects for accumbens, 34% for ventral DC and 22% for putamen, while the right share is 0%, 0% and 4% for the same organs (Table 2, Figure 1). The largest right shares are 20% for cerebellum cortex, 14% for thalamus and 12% for caudate.
7. **Four organs show no bias at all.** Cerebral cortex, choroid plexus, lateral ventricle and vessel have 0% on both sides for every prompt (Table 2, Figure 1), so the test says nothing about their side.

## Conclusion

The prompt is biased to the right when it contradicts itself or adds the side in a later sentence. A final "right" is followed in 58.6% of the cases against 36.6% for a final "left", and an added "make it the right one" is followed in 70.8%. The prompt is biased slightly to the left when no side is named, with 10.1% on the left organ against 4.2% on the right organ, mainly because of accumbens, ventral DC and putamen. The size of the bias depends on the organ, from none for putamen, ventral DC and pallidum to a very strong right bias for thalamus, cerebral white matter and accumbens.
