# RQ6: Which Word of the Prompt Carries the Credit, the Organ Word, the Position Word, the Other Words or the Image?

## What this research question asks

A prompt such as "Segment the left hippocampus in this image." has several parts. The image shows the organ, the organ word says which organ to find, the position word says on which side, and the other words are the rest of the sentence. We take each of these parts away in turn and measure how much the Dice score falls. Then we divide the Dice score of the full prompt fairly between the four parts. This tells us which part the model relies on most, and whether the words around the organ name help or hurt.

## Hypothesis

The organ word is expected to carry most of the credit. The position word is expected to carry a smaller credit that helps only once the organ is named, and the other words are expected to carry almost none.

## Procedure

1. For every organ, the four parts are switched on and off in all 16 combinations, and each combination is one prompt variant that the model segments.
2. The Dice score of each variant is measured against the true organ of the prompted side.
3. The exact Shapley value of each part is calculated from the 16 Dice scores, and the exact interaction of each pair of parts.

## Definitions

- **Players.** The four parts that are switched on or off are the image, the organ word, the position word and the other words. With four players there are 16 combinations.
- **Image off.** The voxels of the organ itself are replaced with Gaussian noise that has the mean and spread of the scan, and the rest of the scan is unchanged.
- **Organ word off.** The organ name is replaced with the generic word "structure", for example "Segment the left structure in this image."
- **Position word off.** The word "left" or "right" is dropped, for example "Segment the hippocampus in this image."
- **Other words off.** The sentence is cut down to its content words, for example "Left hippocampus." instead of "Segment the left hippocampus in this image."
- **Dice score.** The overlap between the predicted mask and the true organ, from 0 (no overlap) to 1 (perfect match).
- **Shapley value.** The share of the Dice score that belongs to one player, calculated exactly over all 16 combinations. It is the average gain in Dice when that player is switched on, over every order in which the players can be added. It is given in Dice units. A positive value means that the player raises the Dice score, and a negative value means that it lowers the Dice score.
- **Total credit.** The sum of the four Shapley values. It equals the Dice of the full prompt minus the Dice of the prompt with all four players off, which is practically 0, so the total credit is practically the Dice of the full prompt.
- **Interaction.** The extra Dice that two players give when they are on together, beyond what each gives alone. A positive value means that the two players help each other, and a negative value means that they get in each other's way.

## Tables

**Table 1. Shapley value of each player in Dice units.**

| Player | Credit, mean ± SD | Credit, median | Share of the total credit | Cases with positive credit | Cases with negative credit |
|---|---|---|---|---|---|
| Image | 0.118 ± 0.118 | 0.079 | 20.3% | 84% | 3% |
| Organ word | 0.422 ± 0.198 | 0.456 | 72.9% | 88% | 0% |
| Position word | 0.105 ± 0.065 | 0.112 | 18.1% | 86% | 1% |
| Other words | −0.066 ± 0.114 | −0.024 | −11.4% | 13% | 74% |
| Total credit | 0.579 ± 0.352 | 0.755 | 100.0% | 83% | not applicable |

**Table 2. Shapley value of each player for each organ in Dice units, mean over the subjects and over the left and right organ, sorted by total credit.**

| Organ | Image | Organ word | Position word | Other words | Total credit |
|---|---|---|---|---|---|
| Cerebellum white matter | 0.344 | 0.439 | 0.113 | −0.018 | 0.878 |
| Cerebral white matter | 0.361 | 0.434 | 0.101 | −0.026 | 0.870 |
| Putamen | 0.163 | 0.560 | 0.168 | −0.027 | 0.864 |
| Cerebellum cortex | 0.081 | 0.635 | 0.155 | −0.039 | 0.832 |
| Pallidum | 0.009 | 0.665 | 0.143 | 0.003 | 0.819 |
| Thalamus | 0.208 | 0.496 | 0.187 | −0.093 | 0.797 |
| Caudate | 0.119 | 0.546 | 0.137 | −0.019 | 0.783 |
| Hippocampus | 0.041 | 0.614 | 0.128 | −0.010 | 0.772 |
| Accumbens | 0.119 | 0.453 | 0.129 | 0.011 | 0.712 |
| Ventral DC | 0.186 | 0.410 | 0.150 | −0.035 | 0.712 |
| Amygdala | 0.178 | 0.490 | 0.099 | −0.075 | 0.691 |
| Inferior lateral ventricle | 0.038 | 0.431 | 0.077 | −0.027 | 0.519 |
| Cerebral cortex | 0.003 | 0.303 | 0.048 | −0.340 | 0.014 |
| Lateral ventricle | 0.033 | 0.283 | 0.046 | −0.360 | 0.003 |
| Choroid plexus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Vessel | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **Mean of 16 organs** | 0.118 | 0.422 | 0.105 | −0.066 | 0.579 |

**Table 3. Interaction between each pair of players in Dice units.**

| Pair of players | Interaction, mean ± SD | Interaction, median |
|---|---|---|
| Image and organ word | 0.230 ± 0.215 | 0.177 |
| Image and position word | 0.015 ± 0.069 | 0.000 |
| Image and other words | 0.006 ± 0.067 | 0.000 |
| Organ word and position word | 0.217 ± 0.124 | 0.213 |
| Organ word and other words | −0.122 ± 0.207 | −0.048 |
| Position word and other words | −0.029 ± 0.062 | −0.012 |

**Table 4. Mean total credit and Shapley value of each player for each subject, summarized over the subjects.** Each subject first gets the mean over all of its organs, and the table then gives the mean, the standard deviation, the lowest and the highest of these subject means.

| Player | Mean over subjects | SD over subjects | Lowest subject | Highest subject |
|---|---|---|---|---|
| Image | 0.118 | 0.010 | 0.095 | 0.134 |
| Organ word | 0.422 | 0.023 | 0.336 | 0.454 |
| Position word | 0.105 | 0.006 | 0.079 | 0.114 |
| Other words | −0.066 | 0.005 | −0.081 | −0.059 |
| Total credit | 0.579 | 0.032 | 0.435 | 0.614 |

## Figures

**Figure 1. Shapley value of the image, the organ word, the position word and the other words for each organ.**

- **What it shows.** The organ word has the largest value for all 14 organs that have a nonzero credit, from +0.283 for lateral ventricle to +0.665 for pallidum, and the mean of the 16 organs is +0.422 for the organ word, +0.118 for the image, +0.105 for the position word and −0.066 for the other words.
- **What to notice.** The image value is +0.361 for cerebral white matter and +0.344 for cerebellum white matter but only +0.009 for pallidum, and the other words are −0.340 for cerebral cortex and −0.360 for lateral ventricle, which leaves their total credit at +0.014 and +0.003. Choroid plexus and vessel are 0.000 for every part.
- **Why it matters.** The balance between the four parts depends on the organ, and for four organs the total credit is close to zero.

**Figure 2. Shapley value of the image, the organ word, the position word and the other words for the left cerebellum white matter, cerebellum cortex, putamen and cerebral white matter of one example scan.**

- **What it shows.** The organ word has the largest value in all four organs (+0.47, +0.66, +0.69 and +0.43), and the final Dice score is 0.90, 0.86, 0.90 and 0.90.
- **What to notice.** The image has +0.32 for the cerebellum white matter and +0.39 for the cerebral white matter but only +0.05 for the cerebellum cortex and +0.11 for the putamen, and the other words are between −0.02 and −0.04.
- **Why it matters.** The image helps the two white matter organs much more than the cortex and the putamen, where the words carry almost all of the credit.

**Figure 3. Shapley value of the image, the organ word, the position word and the other words for the left pallidum, thalamus, ventral DC and hippocampus of one example scan.**

- **What it shows.** The organ word has the largest value in all four organs (+0.70, +0.51, +0.51 and +0.66), and the final Dice score is 0.86, 0.77, 0.77 and 0.80.
- **What to notice.** The image has only +0.02 for the pallidum and +0.04 for the hippocampus, and the other words of the thalamus have −0.15, the largest negative value of the four organs.
- **Why it matters.** For the pallidum and the hippocampus almost the whole Dice score comes from the words and not from the image.

**Figure 4. Shapley value of the image, the organ word, the position word and the other words for the left caudate, accumbens, amygdala and inferior lateral ventricle of one example scan.**

- **What it shows.** The organ word has the largest value in all four organs (+0.56, +0.47, +0.53 and +0.41), and the final Dice score falls from 0.86 for the caudate to 0.76, 0.68 and 0.48.
- **What to notice.** The other words of the accumbens are +0.04, the only positive value for the other words in this figure, while those of the amygdala are −0.06 and those of the inferior lateral ventricle are −0.04.
- **Why it matters.** The inferior lateral ventricle gets the lowest Dice score of these four organs although its organ word value of +0.41 is not much lower than that of the others.

**Figure 5. Shapley value of the image, the organ word, the position word and the other words for the left cerebral cortex, lateral ventricle, vessel and choroid plexus of one example scan.**

- **What it shows.** The final Dice score is 0.03 for the cerebral cortex and 0.00 for the other three organs.
- **What to notice.** The organ word has +0.32 for the cerebral cortex and +0.35 for the lateral ventricle, and the other words have −0.35 for both, so the two cancel. The vessel and the choroid plexus are 0.00 for every part.
- **Why it matters.** For these organs a good organ word credit is lost because the other words take it away again.

**Figure 6. Shapley value of the image, the organ word, the position word and the other words for the right cerebral white matter, cerebellum white matter, putamen and caudate of one example scan.**

- **What it shows.** The organ word has the largest value in all four organs (+0.47, +0.45, +0.59 and +0.62), and the final Dice score is 0.93, 0.91, 0.90 and 0.90.
- **What to notice.** The image has +0.36 for the cerebral white matter and +0.35 for the cerebellum white matter, and the position word has +0.23 for the putamen, the largest position word value of the four organs.
- **Why it matters.** The two white matter organs again rely on the image, as on the left side, and the putamen relies more on the position word than the other three organs do.

**Figure 7. Shapley value of the image, the organ word, the position word and the other words for the right pallidum, thalamus, cerebellum cortex and hippocampus of one example scan.**

- **What it shows.** The organ word has the largest value in all four organs (+0.68, +0.61, +0.65 and +0.68), and the final Dice score is 0.85, 0.87, 0.82 and 0.86.
- **What to notice.** The image has only +0.01 for the pallidum and +0.04 for the hippocampus, and the thalamus has the largest position word value (+0.19).
- **Why it matters.** The pallidum and the hippocampus rely on the words on the right side too, as on the left side.

**Figure 8. Shapley value of the image, the organ word, the position word and the other words for the right amygdala, accumbens, ventral DC and inferior lateral ventricle of one example scan.**

- **What it shows.** The organ word has the largest value in all four organs (+0.54, +0.50, +0.37 and +0.34), and the final Dice score falls from 0.80 for the amygdala to 0.77, 0.69 and 0.49.
- **What to notice.** The ventral DC has the largest position word value of the four organs (+0.20), and the amygdala has the largest image value (+0.22) and the lowest other words value (−0.07).
- **Why it matters.** The ventral DC is the organ of this group that depends most on the position word, and the amygdala the one that depends most on the image.

**Figure 9. Shapley value of the image, the organ word, the position word and the other words for the right cerebral cortex, lateral ventricle, choroid plexus and vessel of one example scan.**

- **What it shows.** The final Dice score is 0.02 for the cerebral cortex and 0.00 for the other three organs.
- **What to notice.** The organ word has +0.31 for the cerebral cortex and +0.28 for the lateral ventricle, and the other words have −0.35 for both. The choroid plexus and the vessel are 0.00 for every part.
- **Why it matters.** The right organs behave like the left organs of Figure 5, so the loss through the other words does not depend on the side.

## Key findings

1. **The organ word carries most of the credit.** It gets 0.422 Dice units, which is 72.9% of the total credit of 0.579, against 0.118 (20.3%) for the image, 0.105 (18.1%) for the position word and −0.066 (−11.4%) for the other words (Table 1, Figure 1). The organ word has positive credit in 88% of the cases, and the image in 84%. This means that the model relies mainly on the name of the organ, and much less on the image or on the side word.
2. **The other words slightly hurt.** Their credit is −0.066, and it is negative in 74% of the cases and positive in only 13% (Table 1). It is −0.340 for cerebral cortex and −0.360 for lateral ventricle (Table 2, Figure 1). This means that the sentence around the organ name takes away part of the Dice score, most strongly for these two organs.
3. **The position word gets its credit mainly together with the organ word.** Its credit is the smallest of the three parts with a positive credit (0.105, Table 1). The interaction of the organ word and the position word is 0.217, while that of the image and the position word is only 0.015 (Table 3). This means that the position word helps mostly once the organ is named, and hardly together with the image alone.
4. **The image and the organ word help each other, and the other words get in the way.** The interaction of the image and the organ word is 0.230, the largest of the six pairs, and the organ word and the other words have an interaction of −0.122 (Table 3). This means that the model needs the real organ in the image together with the organ name, and that the other words take away part of what the organ word gives.
5. **The credit depends on the organ.** The image gets almost as much credit as the organ word for cerebral white matter (0.361 against 0.434) and cerebellum white matter (0.344 against 0.439), but almost none for pallidum (0.009) and hippocampus (0.041) (Table 2, Figure 1, Figure 2, Figure 3). This means that how the prompt is used differs from organ to organ, and that the model may be driven by language for some organs.
6. **For four organs the total credit is close to zero.** It is 0.014 for cerebral cortex, 0.003 for lateral ventricle and 0.000 for choroid plexus and vessel (Table 2). For cerebral cortex and lateral ventricle the organ word credit of +0.303 and +0.283 is cancelled by the other words with −0.340 and −0.360, and the same happens on the left and on the right side (Figure 1, Figure 5, Figure 9). This means that the score of these organs is not limited by one missing part of the prompt, since the four parts together give almost nothing.
7. **The organs differ much more than the subjects.** The total credit has a standard deviation of 0.032 over the subjects, with the subject means between 0.435 and 0.614 (Table 4), while the organs range from 0.000 to 0.878 (Table 2). This means that the balance between the parts is a property of the organ and not of the subject.

## Conclusion

The organ word carries most of the credit for the Dice score (72.9% of the total), the image carries 20.3% and the position word 18.1%, while the other words of the sentence lower the score slightly (−11.4%). The position word and the image each add credit mainly together with the organ word, and the other words get in the way of the organ word. The balance between the parts depends on the organ and not on the subject, and four organs get almost no total credit.
