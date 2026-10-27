# RQ3: How Does the KL Divergence Between Left and Right Prompt Attention Change Across the Prompt Decoder Layers?

## What this research question asks

When the model reads a prompt such as "Segment the left thalamus in this image.", each of its six decoder layers decides which parts of the image the prompt should look at. We give the model two prompts that differ only in the side word, "left X" and "right X", and compare where each of them looks. The KL divergence is one number per layer that tells how different these two ways of looking are. A value near 0 means the two prompts look at the image in the same way, and a larger value means they look at different parts. This question asks how that number changes from the first layer to the last.

## Hypothesis

The prompt decoder is expected to reduce the difference between the left and the right prompt, so the KL divergence between their attention patterns is expected to be highest in the first layer and to fall in the later layers. A rise near the end would mean that a difference between the prompts returns.

## Definitions

- **Attention distribution.** In each decoder layer, the prompt gives a weight to every part of the image. The weights add up to 1, so they form a distribution that shows where the prompt is looking. For example, a prompt may give most of its weight to the region around the thalamus and little to the rest of the image.
- **Left attention and right attention.** The left attention, written p, is the attention distribution of the "left X" prompt. The right attention, written q, is the attention distribution of the "right X" prompt. Both are taken in the same layer, for the same scan.
- **KL divergence.** A number that compares two distributions. It is 0 when the two are identical and grows as they differ. It is not a share or a percentage, so it is not limited to 1. The values in this study range from 0.012 to 0.079 on average.
- **Left from right.** The KL divergence of the left attention from the right attention, which is the sum over all parts of the image of p × log(p / q). It compares the two patterns and counts mostly the parts of the image where the left prompt gives a large weight.
- **Right from left.** The KL divergence of the right attention from the left attention, which is the sum over all parts of the image of q × log(q / p). It counts mostly the parts of the image where the right prompt gives a large weight.
- **Why both directions are reported.** The two directions are not equal in general. If both prompts look at the same parts, both values are close to 0. If one prompt looks closely at a part where the other gives little weight, the value that counts the first prompt's parts becomes large. Reporting both shows whether the result depends on which prompt is taken as the reference.
- **Layer 1 to Layer 6.** The six layers of the prompt decoder, from first to last. Attention exists only inside these layers, so KL divergence is not defined before Layer 1.

## Table

**Table 1. Attention KL divergence between the left and right prompts at each layer of the prompt decoder, in both directions (mean ± SD and median over subjects and organs).**

| Attention KL divergence | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Layer 6 |
|---|---|---|---|---|---|---|
| Left from right, mean ± SD | 0.079 ± 0.033 | 0.018 ± 0.006 | 0.012 ± 0.003 | 0.013 ± 0.007 | 0.021 ± 0.009 | 0.015 ± 0.025 |
| Right from left, mean ± SD | 0.063 ± 0.027 | 0.018 ± 0.006 | 0.012 ± 0.004 | 0.013 ± 0.007 | 0.023 ± 0.011 | 0.018 ± 0.039 |
| Left from right, median | 0.071 | 0.017 | 0.012 | 0.011 | 0.021 | 0.008 |
| Right from left, median | 0.060 | 0.017 | 0.012 | 0.011 | 0.022 | 0.008 |

*Note.* Each value is taken over subjects and organs. Lower values mean the left and right prompts look at the image in a more similar way.

## Figures

**Figure 1. Attention KL divergence between the left and right prompts across the six decoder layers, in both directions.**

- **What it shows.** The KL divergence is 0.079 (left from right) and 0.063 (right from left) at Layer 1, and it falls to 0.012 at Layer 3.
- **What to notice.** A small rise appears at Layer 5 (0.021 and 0.023), and from Layer 2 on the two directions are within 0.004 of each other.
- **Why it matters.** The side word changes where the model looks mainly in Layer 1, and from Layer 2 on both prompts look at almost the same parts of the image.

**Figure 2. Attention KL divergence in the six decoder layers for the inferior lateral ventricle, lateral ventricle, vessel and choroid plexus.**

- **What it shows.** These four pairs have the highest Layer 1 divergence of the 16 pairs, from 0.165 and 0.138 for the inferior lateral ventricle to 0.102 and 0.069 for the choroid plexus, and it falls to at most 0.047 in Layers 2 to 5. Across Figures 2 to 5, left from right is higher than right from left at Layer 1 in all 16 pairs.
- **What to notice.** Only the inferior lateral ventricle has filled predicted masks. The lateral ventricle, vessel and choroid plexus show "No mask predicted" although their Layer 1 values are high (0.131, 0.117 and 0.102 for left from right), and the choroid plexus rises again at Layer 6 to 0.093 and 0.135.
- **Why it matters.** A high Layer 1 divergence does not guarantee a predicted mask, so the attention difference alone does not decide the output.

**Figure 3. Attention KL divergence in the six decoder layers for the hippocampus, cerebral white matter, cerebellum cortex and cerebellum white matter.**

- **What it shows.** The Layer 1 divergence is between 0.072 and 0.100 for left from right and between 0.061 and 0.076 for right from left, and every value in Layers 2 to 6 is at most 0.031.
- **What to notice.** The hippocampus has the highest Layer 1 value (0.100 and 0.076), and the predicted masks mostly follow the dashed outlines. Layer 6 is at most 0.009 in all four rows, and the two white-matter organs show a small rise in Layer 4 (0.029 and 0.031 for the cerebellum white matter, 0.021 and 0.022 for the cerebral white matter).
- **Why it matters.** The Layer 1 peak is also present for organs whose masks follow the true outline, so it is not tied to failed masks.

**Figure 4. Attention KL divergence in the six decoder layers for the cerebral cortex, amygdala, putamen and thalamus.**

- **What it shows.** The Layer 1 divergence is between 0.058 and 0.067 for left from right and between 0.047 and 0.057 for right from left, and Layer 6 is at most 0.010 in all four rows.
- **What to notice.** For the cerebral cortex almost no filled mask is visible, only the dashed outlines of the true organ, and its divergence falls from 0.067 to 0.007 at Layer 6. The amygdala, putamen and thalamus have masks that lie on the outlines, and the amygdala and the thalamus rise again in Layer 5 (0.019 to 0.021).
- **Why it matters.** The same fall from Layer 1 to Layer 6 appears even where almost no mask is filled, so the shape of the attention curve does not depend on mask quality.

**Figure 5. Attention KL divergence in the six decoder layers for the caudate, pallidum, accumbens and ventral DC.**

- **What it shows.** These four pairs have the lowest Layer 1 divergence, from 0.054 and 0.042 for the caudate to 0.031 and 0.026 for the ventral DC.
- **What to notice.** The masks of the caudate, pallidum and accumbens lie on the dashed outlines, while the masks of the ventral DC cover only part of them. The caudate and the pallidum rise in Layer 5 (0.020 to 0.022), whereas the ventral DC stays at or below 0.009 in Layers 2 to 6.
- **Why it matters.** The early peak is weakest for these organs, so the influence of the side word on attention depends on the organ.

## Key findings

1. **The KL divergence is highest in the first layer.** At Layer 1 it is 0.079 (left from right) and 0.063 (right from left), the highest of all layers, and about four times the value at Layer 2 (0.018) (Figure 1, Table 1). This means that the left and right prompts look at different parts of the image mainly in the first layer. The first layer is therefore where the side word has its strongest effect on the attention.
2. **It drops sharply after the first layer.** It falls to 0.018 at Layer 2 and reaches its lowest value of 0.012 in both directions at Layer 3, which is about six times lower than at Layer 1 (Figure 1, Table 1). This means that from the second layer on, both prompts look at almost the same parts of the image. The side word then has little influence on where the model looks.
3. **A small rise appears at Layer 5.** At Layer 5 the KL divergence increases again to 0.021 (left from right) and 0.023 (right from left), and then falls to 0.015 and 0.018 at Layer 6 (Figure 1, Table 1). This rise is small compared with the value at Layer 1. This means that a small difference between the prompts comes back near the end of the decoder, but it stays far below the difference in the first layer.
4. **Layer 6 is raised by a few large values.** The mean at Layer 6 is 0.015 and 0.018, but the median is only 0.008 in both directions, and the SD (0.025 and 0.039) is larger than the mean (Table 1). The median at Layer 6 is lower than the medians at Layers 2 to 5, which are 0.011 to 0.022. This means that in a typical case the two prompts look at the image almost identically in the last layer, and only a few cases differ strongly. These few cases raise the average.
5. **The two directions agree after the first layer.** At Layer 1 the left-from-right value (0.079) is higher than the right-from-left value (0.063), but from Layer 2 onward the two directions are within 0.004 of each other (Figure 1, Table 1). This means that, apart from the first layer, the result does not depend on which prompt is taken as the reference. The first layer is the only place where the difference between the two prompts is not symmetric.

## Conclusion

The KL divergence between the left and right prompt attention is largest in the first decoder layer (0.079 and 0.063), falls to its lowest at Layer 3 (0.012), rises slightly at Layer 5 (0.021 and 0.023), and is small again at Layer 6 (median 0.008). The two prompts therefore look at the image most differently in the first layer and in a very similar way in the later layers, so the difference between "left" and "right" is mainly expressed early in the prompt decoder.
