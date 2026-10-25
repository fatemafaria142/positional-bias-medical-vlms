# RQ3: How Does the KL Divergence Between Left and Right Prompt Attention Change Across the Prompt Decoder Layers?

## What this research question asks

In each of the six decoder layers, the text prompt looks at the image to decide where to focus. KL divergence measures how different two such focus patterns are. This question asks how the KL divergence between the "left X" prompt and the "right X" prompt changes from the first layer to the last. A value near 0 means the two prompts look at the image in the same way. A larger value means they look at it differently.

**Setting.** VoxTell on the Neurite-OASIS dataset, for 16 bilateral organ pairs. For each organ, the KL divergence between the "left X" and "right X" attention is measured in every one of the six decoder layers.

## Definitions

- **Attention distribution.** In each decoder layer, the prompt gives a weight to every part of the image. The weights add up to 1, so they form a distribution that shows where the prompt is looking.
- **KL divergence.** A measure of how different two distributions are. It is 0 when the two are identical and grows as they differ. Here it compares the attention distribution of the "left X" prompt with that of the "right X" prompt.
- **Left from right and right from left.** Let the left attention be the weights the "left X" prompt gives to the image, and the right attention the weights the "right X" prompt gives. "Left from right" is the KL divergence that takes the left attention as the reference, which asks how much the left attention differs from the right attention. "Right from left" does the opposite. KL divergence is not symmetric, so the two values are not equal, and both are reported.
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

## Figure

**Figure 1. Attention KL divergence between the left and right prompts across the six layers of the prompt decoder, in the direction left from right (green) and right from left (red).** Each point is the mean over subjects and organs, and each mean is written next to its point.

## Results

1. **The KL divergence is highest in the first layer.** At Layer 1 it is 0.079 (left from right) and 0.063 (right from left), the highest of all layers, and about four times the value at Layer 2 (0.018) (Figure 1, Table 1). This means that the left and right prompts look at different parts of the image mainly in the first layer. The first layer is therefore where the side word has its strongest effect on the attention.
2. **It drops sharply after the first layer.** It falls to 0.018 at Layer 2 and reaches its lowest value of 0.012 in both directions at Layer 3, which is about six times lower than at Layer 1 (Figure 1, Table 1). This means that from the second layer on, both prompts look at almost the same parts of the image. The side word then has little influence on where the model looks.
3. **A small rise appears at Layer 5.** At Layer 5 the KL divergence increases again to 0.021 (left from right) and 0.023 (right from left), and then falls to 0.015 and 0.018 at Layer 6 (Figure 1, Table 1). This rise is small compared with the value at Layer 1. This means that a small difference between the prompts comes back near the end of the decoder, but it stays far below the difference in the first layer.
4. **Layer 6 is raised by a few large values.** The mean at Layer 6 is 0.015 and 0.018, but the median is only 0.008 in both directions, and the SD (0.025 and 0.039) is larger than the mean (Table 1). The median at Layer 6 is lower than the medians at Layers 2 to 5, which are 0.011 to 0.022. This means that in a typical case the two prompts look at the image almost identically in the last layer, and only a few cases differ strongly. These few cases raise the average.
5. **The two directions agree after the first layer.** At Layer 1 the left-from-right value (0.079) is higher than the right-from-left value (0.063), but from Layer 2 onward the two directions are within 0.004 of each other (Figure 1, Table 1). This means that, apart from the first layer, the result does not depend on which prompt is taken as the reference. The first layer is the only place where the difference between the two prompts is not symmetric.

## Conclusion

The KL divergence between the left and right prompt attention is largest in the first decoder layer (0.079 and 0.063), falls to its lowest at Layer 3 (0.012), rises slightly at Layer 5 (0.021 and 0.023), and is small again at Layer 6 (median 0.008). The two prompts therefore look at the image most differently in the first layer and in a very similar way in the later layers, so the difference between "left" and "right" is mainly expressed early in the prompt decoder.
