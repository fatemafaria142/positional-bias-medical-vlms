# RQ10: How Much of the Segmentation Score Is Credited to the Image, the Organ Word and the Position Word of the Prompt?

## What this research question asks

A prompt such as "Segment the left hippocampus in this image." has several parts. The image shows the organ, the organ word says which organ to find, and the position word says on which side. We take each of these parts away in turn and measure how much the Dice score falls. Then we divide the final Dice score fairly between the three parts. This tells us which part of the prompt the model relies on most, and whether the position word really matters.

**Setting.** VoxTell on the Neurite-OASIS dataset, for 16 bilateral organ pairs, using both the left and the right organ, which gives 32 organs per subject and 1,600 subject-organ cases for 50 subjects. The other words of the prompt, "Segment the" and "in this image", are the same in every prompt. They are always present and are not scored.

## Definitions

- **Players.** The three parts that are switched on or off are the image, the organ word and the position word. With three players there are 8 combinations, and each combination is one prompt variant that the model segments.
- **Image off.** The voxels of the organ itself are replaced with Gaussian noise that has the mean and spread of the scan, and the rest of the scan is unchanged.
- **Organ word off.** The organ name is replaced with the generic word "structure", for example "Segment the left structure in this image."
- **Position word off.** The word "left" or "right" is dropped, for example "Segment the hippocampus in this image."
- **Dice score.** The overlap between the predicted mask and the true organ, from 0 (no overlap) to 1 (perfect match). It is always computed against the true organ of the prompted side.
- **Shapley credit.** The share of the Dice score that belongs to one player, calculated exactly over all 8 combinations. It is the average gain in Dice when that player is switched on, over every order in which the players can be added. It is given in Dice units. The three credits add up to the Dice of the full prompt minus the Dice of the prompt with all three players off.
- **Total credit.** The sum of the three Shapley credits. Because the Dice with all three players off is 0 in almost every case, it is practically the Dice of the full prompt.
- **Interaction.** The extra Dice that two players give when they are on together, beyond what each gives alone. A positive value means the two players help each other.

## Tables

**Table 1. Dice score of the eight prompt variants over 1,600 subject-organ cases.** The example prompts use the left hippocampus.

| Image | Example prompt | Dice, mean ± SD | Dice, median | Cases with Dice of 0.5 or more |
|---|---|---|---|---|
| Intact | Segment the left hippocampus in this image. | 0.579 ± 0.352 | 0.755 | 71% |
| Organ region replaced by noise | Segment the left hippocampus in this image. | 0.329 ± 0.300 | 0.284 | 36% |
| Intact | Segment the hippocampus in this image. | 0.377 ± 0.246 | 0.498 | 49% |
| Organ region replaced by noise | Segment the hippocampus in this image. | 0.166 ± 0.190 | 0.066 | 9% |
| Intact | Segment the left structure in this image. | 0.000 ± 0.000 | 0.000 | 0% |
| Organ region replaced by noise | Segment the left structure in this image. | 0.000 ± 0.000 | 0.000 | 0% |
| Intact | Segment the structure in this image. | 0.000 ± 0.000 | 0.000 | 0% |
| Organ region replaced by noise | Segment the structure in this image. | 0.000 ± 0.000 | 0.000 | 0% |

**Table 2. Shapley credit of each player in Dice units over 1,600 subject-organ cases.**

| Player | Credit, mean ± SD | Credit, median | Share of the total credit | Cases with positive credit |
|---|---|---|---|---|
| Image | 0.119 ± 0.128 | 0.078 | 20.5% | 74% |
| Organ word | 0.366 ± 0.234 | 0.419 | 63.2% | 87% |
| Position word | 0.095 ± 0.077 | 0.105 | 16.3% | 80% |
| Total credit | 0.579 ± 0.352 | 0.755 | 100.0% | 83% |

**Table 3. Shapley credit of each player for each organ in Dice units, mean over 50 subjects and over the left and right organ, sorted by total credit.**

| Organ | Image | Organ word | Position word | Total credit |
|---|---|---|---|---|
| Cerebellum white matter | 0.355 | 0.413 | 0.109 | 0.878 |
| Cerebral white matter | 0.376 | 0.402 | 0.092 | 0.870 |
| Putamen | 0.174 | 0.520 | 0.170 | 0.864 |
| Cerebellum cortex | 0.101 | 0.575 | 0.157 | 0.832 |
| Pallidum | 0.000 | 0.682 | 0.138 | 0.819 |
| Thalamus | 0.220 | 0.396 | 0.182 | 0.797 |
| Caudate | 0.106 | 0.544 | 0.133 | 0.783 |
| Hippocampus | 0.040 | 0.613 | 0.120 | 0.772 |
| Accumbens | 0.110 | 0.467 | 0.134 | 0.712 |
| Ventral DC | 0.171 | 0.393 | 0.148 | 0.712 |
| Amygdala | 0.220 | 0.396 | 0.076 | 0.691 |
| Inferior lateral ventricle | 0.029 | 0.423 | 0.068 | 0.520 |
| Cerebral cortex | −0.007 | 0.018 | 0.002 | 0.014 |
| Lateral ventricle | 0.003 | 0.014 | −0.015 | 0.003 |
| Choroid plexus | 0.000 | 0.000 | 0.000 | 0.000 |
| Vessel | 0.000 | 0.000 | 0.000 | 0.000 |
| **Mean of 16 organs** | 0.119 | 0.366 | 0.095 | 0.579 |

**Table 4. Interaction between each pair of players in Dice units over 1,600 subject-organ cases.**

| Pair of players | Interaction, mean ± SD | Interaction, median |
|---|---|---|
| Image and organ word | 0.231 ± 0.242 | 0.161 |
| Image and position word | 0.020 ± 0.085 | 0.000 |
| Organ word and position word | 0.183 ± 0.154 | 0.192 |

## Figures

**Figure 1. Shapley credit of the image, the organ word and the position word shown on the words of the prompt, for the mean of all 32 organs and for five example organs.** Each scored player is drawn as a box and its credit in Dice units is written under it, with the shade of blue showing the size of the credit and a light red box showing a negative credit. The dashed boxes are the fixed words of the prompt, which are always present and are not scored. The bold number at the right end of each row is the total credit. Values for the example organs are means over 50 subjects.

**Figure 2. Predicted masks of the left hippocampus and the left thalamus of one subject for four prompt variants, with the image intact and with the organ region replaced by noise.** The four columns are the full prompt, the prompt without the position word, the prompt without the organ word, and the prompt with neither word. The green area is the predicted mask, the white outline is the true organ on the left side, and the Dice score is written under each panel.

## Results

1. **The organ word receives most of the credit.** It gets 0.366 Dice units, which is 63.2% of the total credit of 0.579, against 0.119 (20.5%) for the image and 0.095 (16.3%) for the position word (Table 2, Figure 1). The organ word has positive credit in 87% of the cases, more than the image (74%) and the position word (80%). This means that the model relies mainly on the name of the organ, and much less on the image or on the side word.
2. **Without the organ word the model produces no mask.** The two prompts that replace the organ with "structure" have a Dice of 0.000 in every case, with the image intact and with the organ region replaced by noise, and with or without the position word (Table 1). Figure 2 shows empty panels for these prompts in both examples. This means that neither the image nor the position word can find the organ on its own, and the organ word is a requirement for any segmentation.
3. **The position word helps only together with the organ word.** The credit of the position word is the smallest (0.095, Table 2), and the interaction between the organ word and the position word is 0.183, while the interaction between the image and the position word is only 0.020 (Table 4). With the image intact, dropping the position word lowers the Dice from 0.579 to 0.377, and with the organ region replaced by noise it lowers it from 0.329 to 0.166 (Table 1). In Figure 2 the prompt without the position word gives a mask on both sides in the hippocampus example (Dice 0.55) and in the thalamus example (Dice 0.23), as well as the left mask. This means that the position word is what places the mask on the correct side, but only once the organ is already named.
4. **The image adds a smaller but real credit, and it works together with the organ word.** With the full prompt, replacing the organ region with noise lowers the mean Dice from 0.579 to 0.329, and the image credit of 0.119 is positive in 74% of the cases (Table 1, Table 2). The interaction between the image and the organ word is 0.231, the largest of the three pairs (Table 4). In the thalamus example the Dice falls from 0.72 to 0.00 when the organ region is replaced by noise (Figure 2). This means that the model needs the real organ in the image to draw a mask, and the organ word is what tells it which organ to look for.
5. **The credit depends on the organ.** The image gets almost as much credit as the organ word for cerebral white matter (0.376 against 0.402) and cerebellum white matter (0.355 against 0.413), but almost none for pallidum (0.000 against 0.682) and hippocampus (0.040 against 0.613) (Table 3). The position word gets the most credit for thalamus (0.182) and putamen (0.170), and the least for inferior lateral ventricle (0.068) and amygdala (0.076) among the organs where the prompt works (Table 3). In Figure 1 the organ word is the darkest box for the hippocampus (0.601) and the putamen (0.506), while for the cerebral white matter the image box (0.374) is almost as dark as the organ box (0.385). This means that how the prompt is used differs from organ to organ.
6. **For four organs no part of the prompt gives a usable mask.** The total credit is 0.014 for cerebral cortex, 0.003 for lateral ventricle and 0.000 for choroid plexus and vessel, so even the full prompt reaches almost no Dice (Table 3). Figure 1 shows the lateral ventricle row with credits close to zero (total 0.001), and its position word has a slightly negative credit (−0.033). This means that the score for these organs is not limited by one missing part of the prompt, since all three players together give almost nothing.
7. **The credit of single cases varies widely.** The standard deviation of the total credit is 0.352, which is as large as its mean of 0.579, and the total credit is positive in 83% of the cases (Table 2). The median of 0.755 is higher than the mean. This means that most prompts work well, while a smaller group, which includes the four organs above, gets a Dice close to zero and lowers the mean.

## Conclusion

The organ word carries most of the credit for the Dice score (63.2% of the total), the image carries 20.5% and the position word 16.3%. Without the organ word no mask is produced at all, so naming the organ is a requirement. The position word and the image each add credit mainly when they are combined with the organ word, as shown by the interactions of 0.183 and 0.231. The position word is what separates the correct side from both sides, and the image provides the organ that the word refers to. The balance between the three parts differs between organs, and four organs get almost no credit for any part.
