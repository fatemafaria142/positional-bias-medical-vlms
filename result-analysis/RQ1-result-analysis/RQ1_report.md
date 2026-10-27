# RQ1: Do the 32 Text–Image Fusion Channels Distinguish Left and Right Anatomical Structures?

## What this research question asks

VoxTell finds a brain part from a text prompt such as "left hippocampus". We run the same scan twice, once with "left" and once with "right". Then we check whether the model's inside values are different for the two prompts. If they are the same, the model is not using the word "left" or "right" at that point. If they are different, the word is changing what the model does. We check this at every step of the decoder, to see where the difference appears and where it disappears.

## Hypothesis

The left-prompt and right-prompt fusion channels are expected to differ, since the side word is the only change, and the difference is expected to be larger at some decoder stages than at others. Nearly identical channels at a stage would mean weak sensitivity to the side word at that stage, and would not prove that the text has no side information.

## Definition of the decoder stages

VoxTell first encodes the image with a 3D U-Net encoder. The text prompt is embedded separately, so the image features are identical for the left and right prompt. A 6-layer transformer decoder then fuses text and image into a *mask embedding*. Finally, a U-Net convolutional decoder rebuilds the full-resolution mask, and at each of its **5 fusion stages** the mask embedding is projected and injected into the decoder.

- **Stage 0–3:** the first four fusion stages of the U-Net decoder. Each injects a **32-channel map**, which is the quantity compared between the left and right prompt. Stage 0 is the first (coarsest resolution); resolution increases through Stage 3.
- **Finest logit:** the fifth and last stage. It has no 32-channel map; it produces the single full-resolution logit map that is thresholded into the predicted mask.
- These stages belong to the convolutional decoder, not to the 6 layers of the transformer decoder.

### What "32 channels" means

The 32 channels are **inside each stage**, not spread across the stages. Stage 0, Stage 1, Stage 2 and Stage 3 each contain one feature map with 32 channels. The model produces one such map for the "left X" prompt and one for the "right X" prompt. Within a stage, channel *c* of the left map is compared with channel *c* of the right map, which gives 32 values per stage (one per channel). The table reports the average of these 32 values (and, for cosine similarity, also the lowest of them). The Finest logit has a single channel, so its average and its minimum are the same number.

Similarity is computed between the left-prompt and right-prompt maps on the same image. Cosine similarity of 1.0 means identical channels; lower values mean the prompt changed the channels more. (The stage order, with Stage 0 as the coarsest, follows the decoder's processing order and was not separately verified against each stage's tensor size.)

## Table

**Table 1. Left-versus-right similarity of the 32 fusion channels at each U-Net decoder stage, reported as mean ± SD over subjects and organ pairs.**

| Metric | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Finest logit |
|---|---|---|---|---|---|
| Cosine similarity, average over the 32 channels of each stage | 0.695 ± 0.078 | 0.848 ± 0.073 | 0.917 ± 0.045 | 0.979 ± 0.009 | 0.979 ± 0.012 |
| Cosine similarity, lowest of the 32 channels of each stage | 0.160 ± 0.232 | 0.505 ± 0.230 | 0.724 ± 0.182 | 0.820 ± 0.129 | 0.979 ± 0.012 |
| Pearson correlation, average over channels | 0.711 ± 0.068 | 0.671 ± 0.124 | 0.681 ± 0.105 | 0.829 ± 0.061 | 0.652 ± 0.169 |
| Normalized L2 difference, average over channels | 0.780 ± 0.084 | 0.519 ± 0.135 | 0.409 ± 0.116 | 0.234 ± 0.133 | 0.242 ± 0.090 |

*Note.* Stage 0–3 each have 32 channels; the Finest logit has 1 channel, so its average and lowest cosine similarity are identical. Each ± is the standard deviation over subjects and organ pairs. Cosine similarity of 1.0 means the left and right prompts give identical channels; for normalized L2 difference, 0 means identical.

## Figure

**Figure 1. Left-versus-right cosine similarity of the fusion channels for each organ at each U-Net decoder stage.**

- **What it shows.** The similarity rises from Stage 0 to Stage 3 for all 16 organs. At Stage 0 the organs range from 0.50 to 0.77, and at Stage 3 and the finest logit all of them are at 0.950 or above.
- **What to notice.** Choroid plexus (0.50) and vessel (0.54) are the most different at Stage 0, and cerebral white matter keeps its difference up to Stage 2 (0.80).
- **Why it matters.** The fading of the left/right difference is a property of the decoder as a whole, since all 16 organs follow the same pattern from Stage 0 to Stage 3.

## Key findings

1. **The fusion channels separate left from right only at early stages.** The mean cosine similarity between the left-prompt and right-prompt channels rises from 0.695 at Stage 0 to 0.848 at Stage 1, 0.917 at Stage 2 and 0.979 at Stage 3, and stays at 0.979 at the finest logit. The normalized L2 difference falls in the same way, from 0.780 at Stage 0 to 0.242 at the finest logit (Table 1). This means that the word "left" or "right" changes what the model computes most in the first fusion stage, and that this effect fades at every later stage. By Stage 3 the two prompts give almost the same channels, so the later stages mostly carry what is shared by both prompts.
2. **The same pattern holds for every organ.** Every row of Figure 1 gets darker from left to right, so the similarity increases for each of the 16 organs. At Stage 0 the organs range from 0.50 to 0.77, and by Stage 3 and the finest logit all of them are at 0.950 or above (Figure 1). This means that the fading of the left/right difference is a general behavior of the decoder and does not depend on one organ. The trend can therefore be described for the decoder as a whole and not organ by organ.
3. **Some organs keep their left/right difference longer.** At Stage 2, cerebral white matter is the lowest (0.80), followed by cerebellum cortex and cerebral cortex (0.86 each), while most other organs are already at 0.92 or above. At Stage 0, choroid plexus (0.50) and vessel (0.54) are the lowest, and the other organs lie between 0.66 and 0.77 (Figure 1). This means that the prompt has its strongest early influence on these two organs, and that the stage at which the difference disappears depends on the organ. For cerebral white matter the difference lasts the longest, up to Stage 2.
4. **The worst-case channel differs far more than the average.** The lowest cosine among the 32 channels is 0.160 at Stage 0 and 0.505 at Stage 1, much lower than the average values of 0.695 and 0.848, and it rises to 0.724 at Stage 2 and 0.820 at Stage 3 (Table 1). This means that the average hides the strongest effect, because some individual channels are almost unrelated between the two prompts even when the average looks moderate. The left/right word therefore acts very strongly on a few channels in the early stages.
5. **Pearson correlation does not follow the same trend.** It stays between 0.652 and 0.829 at all stages and does not rise steadily, being 0.711 at Stage 0, 0.829 at Stage 3 and 0.652 at the finest logit (Table 1). Cosine similarity and L2 difference, in contrast, change steadily from stage to stage. This means that correlation does not show the convergence of the left and right channels, so cosine similarity and L2 difference are the more useful measures here.

## Conclusion

The 32 fusion channels carry a left/right difference, but only at the early decoder stages. The difference is largest at Stage 0 and shrinks at each later stage, until the left- and right-prompt maps are nearly identical at Stage 3 and the finest logit. This holds for all 16 organs, so the model uses the side word early and then largely stops separating the two prompts in its channels.
