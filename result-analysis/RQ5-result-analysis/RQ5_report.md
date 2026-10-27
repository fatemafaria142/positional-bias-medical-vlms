# RQ5: Does the Segmentation of an Organ Depend More on the Image Block That Contains the Organ Than on the Surrounding Blocks?

## What this research question asks

For one organ, such as the thalamus, we draw a box around it and cut the box into 8 blocks. We hide blocks by replacing them with noise, and the Shapley value of a block is how much the Dice score drops when that block is hidden. The question is whether the model depends more on the organ block, which is the block holding most of the organ, than on the 7 surrounding blocks. The answer is given as one number per organ, the Shapley value of the organ block minus the mean Shapley value of the surrounding blocks. A positive number means that the model relies more on the organ itself, and a negative number means that it relies more on the surroundings.

## Research question

For a given organ, does the Dice score of the predicted mask depend more on the image block that contains most of the organ (the organ block) than on the blocks that surround it?

**Hypothesis.** If the model finds an organ by looking at the organ itself, the Shapley value of the organ block is higher than the mean Shapley value of the 7 surrounding blocks. If the model also relies on the surrounding tissue, the difference is small or negative.

**Measure.** For each organ, the difference between the Shapley value of the organ block and the mean Shapley value of the 7 surrounding blocks, in Dice units, and the share of cases where the organ block has the higher Shapley value.

## Procedure

What happens in one test.

1. Take one scan and one organ, for example the thalamus.
2. Draw a box around the thalamus and cut it into 8 blocks.
3. Hide some of the blocks by replacing them with noise, and run the model on the damaged scan.
4. Measure the Dice score between the predicted mask and the true thalamus.
5. Repeat for many different choices of hidden blocks, and compute from them how much each block adds to the Dice score. This is the Shapley value of the block.

## Definitions

- **Dice score.** The overlap between the predicted mask and the true organ, from 0 (no overlap) to 1 (perfect match).
- **Shapley value of a block.** How much the Dice score rises when the block is visible instead of hidden, averaged over the different choices of the other blocks. It is given in Dice units. A large Shapley value means that the model needs the block, and a negative value means that showing the block makes the Dice lower. The values are Monte Carlo estimates from two randomly chosen sets of the other blocks, and not the exact values from all 256 combinations of blocks, so the 8 Shapley values of a case are close to, but do not always add up to, the Dice of the full scan.
- **Organ block.** The block that holds the largest part of the organ.
- **Surrounding blocks.** The other 7 blocks of the box.
- **Layer of blocks.** The 8 blocks are arranged in two layers along the head–foot direction, with 4 blocks in each layer. The upper layer is the half of the box nearer the top of the head and the lower layer is the half nearer the neck. A layer is a part of the input image and not a layer of the neural network, because the experiment only changes the input image and runs the whole model.
- **Slice shown in the figures.** Each figure row shows one axial slice, the slice through the centre of the organ inside that layer, so the two rows are two different slices of the same organ.
- **Share of the organ inside a block.** The number of voxels of the organ in the block divided by the number of voxels of the whole organ.
- **Organs with a Shapley value.** The 12 organs whose mean sum of the 8 Shapley values is above 0.1. The other 4 organs are left out of Table 2, because hiding blocks does not change their Dice.
- **95% interval.** Obtained by resampling the subjects 2,000 times, because the 32 organs of one subject are not independent.

## Tables

**Table 1. Shapley value of the organ block and of the surrounding blocks for each organ, in Dice units.** Each value is the mean over the subjects and over the left and the right organ. A positive difference means that the organ block matters more than the surrounding blocks. The organs are sorted by the sum of the 8 Shapley values.

| Organ | Shapley value of the organ block | Mean Shapley value of the 7 surrounding blocks | Difference, organ block minus surrounding blocks | Cases where the organ block is higher | Sum of the 8 Shapley values |
|---|---|---|---|---|---|
| Pallidum | 0.208 | 0.116 | +0.092 | 73% | 1.018 |
| Cerebellum white matter | 0.242 | 0.094 | +0.148 | 98% | 0.900 |
| Thalamus | 0.145 | 0.108 | +0.037 | 57% | 0.898 |
| Cerebral white matter | 0.122 | 0.107 | +0.015 | 66% | 0.868 |
| Cerebellum cortex | 0.139 | 0.095 | +0.044 | 91% | 0.805 |
| Putamen | 0.100 | 0.088 | +0.012 | 64% | 0.715 |
| Hippocampus | 0.058 | 0.092 | −0.034 | 18% | 0.700 |
| Caudate | 0.076 | 0.087 | −0.011 | 32% | 0.685 |
| Ventral DC | 0.112 | 0.066 | +0.046 | 68% | 0.572 |
| Accumbens | 0.078 | 0.062 | +0.016 | 58% | 0.513 |
| Amygdala | 0.139 | 0.052 | +0.087 | 75% | 0.505 |
| Inferior lateral ventricle | 0.094 | 0.049 | +0.045 | 60% | 0.439 |
| Cerebral cortex | 0.009 | −0.001 | +0.010 | 42% | 0.001 |
| Choroid plexus | 0.000 | 0.000 | +0.000 | 0% | 0.000 |
| Vessel | 0.000 | 0.000 | +0.000 | 0% | 0.000 |
| Lateral ventricle | −0.005 | −0.002 | −0.003 | 49% | −0.019 |
| **Mean of 16 organs** | 0.095 | 0.063 | +0.032 | 53% | 0.538 |
| **Mean of the 12 organs with a Shapley value** | 0.126 | 0.085 | +0.041 | 63% | 0.718 |

**Table 2. Shapley value of a block by the share of the organ inside it, for the 12 organs with a Shapley value.** Each row is a group of blocks with a similar share of the organ.

| Share of the organ inside the block | Number of blocks | Mean Shapley value | Median Shapley value | 95% interval of the mean | Blocks with a negative Shapley value |
|---|---|---|---|---|---|
| No voxel of the organ | 560 | 0.035 | 0.018 | 0.030 to 0.040 | 25% |
| Up to 5% | 1,848 | 0.053 | 0.039 | 0.050 to 0.056 | 16% |
| 5% to 15% | 3,512 | 0.095 | 0.081 | 0.093 to 0.096 | 7% |
| 15% to 30% | 3,385 | 0.113 | 0.105 | 0.110 to 0.116 | 5% |
| More than 30% | 295 | 0.098 | 0.070 | 0.086 to 0.111 | 2% |

## Figures

**Figure 1. How the experiment works, shown on the left thalamus of one example scan.**

- **What it shows.** The test hides blocks, measures the Dice of the predicted mask, and repeats this for many choices of hidden blocks. The Shapley value of a block is how much that block adds to the Dice.
- **What to notice.** The organ block is the block that holds most of the organ, and the other 7 blocks are the surrounding blocks.
- **Why it matters.** Because the Shapley value averages over many combinations of hidden blocks, it gives each block its fair share of the Dice and not only the effect of hiding it alone.

**Figure 2. Shapley value of the organ block compared with the surrounding blocks for each organ.**

- **What it shows.** The organ block is higher for 10 of the 12 organs with a Shapley value, with a mean of 0.126 against 0.085 for the surrounding blocks.
- **What to notice.** The difference is largest for the cerebellum white matter (+0.148, higher in 98% of the cases) and negative for the hippocampus (−0.034, 18%) and the caudate (−0.011, 32%).
- **Why it matters.** The model relies more on the block that holds the organ, but not for every organ, since the hippocampus and the caudate are exceptions.

**Figure 3. Scan, true organ and Shapley values of the 8 blocks around the left cerebellum white matter of one example scan.**

- **What it shows.** The organ block has the highest Shapley value (+0.242), and the 4 upper blocks, which hold almost no organ, have only +0.015 to +0.064.
- **What to notice.** The lower layer holds the larger values, from +0.083 to +0.242.
- **Why it matters.** The support for this organ comes mostly from the lower layer of blocks, so the useful context is not spread evenly around the organ.

**Figure 4. Scan, true organ and Shapley values of the 8 blocks around the left cerebellum cortex of one example scan.**

- **What it shows.** The 8 blocks have Shapley values between +0.054 and +0.182, and all of them are positive.
- **What to notice.** No block lowers the Dice, so every block of the box helps the model a little for this organ.
- **Why it matters.** Every block helps a little here, so the model uses the whole box around this organ and not only the organ block.

**Figure 5. Scan, true organ and Shapley values of the 8 blocks around the left thalamus of one example scan.**

- **What it shows.** The organ block has a Shapley value of +0.290, and two other blocks have +0.181 and +0.230.
- **What to notice.** One block has a negative Shapley value (−0.075), shown in red, which means that showing it lowers the Dice.
- **Why it matters.** A surrounding block can lower the Dice, so not every part of the context helps the model.

**Figure 6. Scan, true organ and Shapley values of the 8 blocks around the left hippocampus of one example scan.**

- **What it shows.** The organ block has a Shapley value of +0.067, which is lower than the block below it (+0.215).
- **What to notice.** This agrees with the negative difference of the hippocampus in Figure 2, since a surrounding block matters more than the organ block.
- **Why it matters.** For the hippocampus a surrounding block matters more than the organ block, which is why this organ has a negative difference.

## Key findings

1. **For most organs the organ block has a higher Shapley value than the surrounding blocks.** Over the 12 organs with a Shapley value, the organ block has a mean of 0.126 against 0.085 for the surrounding blocks, a difference of +0.041, and it has the higher value in 63% of the cases (Table 1, Figure 2). This means that the model relies more on the block that contains the organ, but it also relies on the surroundings.
2. **The difference depends on the organ.** The organ block is higher for 10 of the 12 organs with a Shapley value (Figure 2). It is largest for cerebellum white matter (+0.148, with the organ block higher in 98% of the cases), pallidum (+0.092, 73%) and amygdala (+0.087, 75%), and small for cerebral white matter (+0.015, 66%) and putamen (+0.012, 64%). It is negative for hippocampus (−0.034, 18%) and caudate (−0.011, 32%), which means that for these two organs the surrounding blocks matter more than the organ block (Table 1, Figure 2). The examples show the same (Figures 3 to 6). In the cerebellum white matter example the organ block has the highest Shapley value (+0.242), and the upper blocks, which hold almost no organ, have only +0.015 to +0.064. In the thalamus example the organ block has +0.290, and one block has a negative Shapley value (−0.075). In the hippocampus example the organ block has +0.067, while the block below it has +0.215.
3. **Four organs have a Shapley value of about 0.** Cerebral cortex (0.001), choroid plexus (0.000), vessel (0.000) and lateral ventricle (−0.019) have a sum of the 8 Shapley values of about 0 (Table 1, Figure 2). This means that hiding blocks does not change the Dice of these organs, so the hypothesis cannot be tested for them.
4. **A block with more of the organ has a higher Shapley value, up to a limit.** The mean Shapley value is 0.035 for blocks with no voxel of the organ, 0.053 for up to 5%, 0.095 for 5% to 15% and 0.113 for 15% to 30%, and then it is 0.098 for more than 30% (Table 2). The share of blocks with a negative Shapley value falls from 25% to 2% over the same groups. This means that blocks that hold more of the organ are more useful, but blocks with no organ still have a positive Shapley value on average.
5. **The Shapley values are estimates.** The mean sum of the 8 Shapley values of pallidum is 1.018, which is above the maximum possible Dice of 1 (Table 1). This means that a single value has some noise, and the means over many cases are more reliable than one block of one case.

## Conclusion

On average, the Shapley value of the organ block is higher than that of the surrounding blocks (0.126 against 0.085 over the 12 organs with a Shapley value, higher in 63% of the cases), so the hypothesis is supported overall. It is supported strongly for cerebellum white matter and not supported for hippocampus and caudate, where the surrounding blocks matter more. Four organs have a Shapley value of about 0, so the hypothesis cannot be tested for them.
