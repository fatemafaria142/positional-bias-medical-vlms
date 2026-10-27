# RQ2: Do Text and Image Representations Encode Different Information About Laterality?

## What this research question asks

VoxTell gets its input from two sources, the text prompt and the image. First, we check whether the words "left X" and "right X" are different from each other as text. Then we check whether that difference is still there at the end of the model, where the mask is produced. The image is the same for both prompts, so it cannot tell the model which side to pick.

## Hypothesis

The text embeddings of "left X" and "right X" are expected to differ, while the image features cannot differ, because the image encoder never receives the prompt. If the decoder output of the two prompts is more similar than their raw text, part of the side difference is lost on the way through the decoder.

## Table

**Table 1. Left-versus-right similarity of the prompts as raw text embeddings and as U-Net decoder outputs for each organ, sorted by text cosine similarity.** The last column is the decoder finest-logit cosine similarity minus the raw text cosine similarity, so a positive value means the two prompts are more alike at the decoder output than as raw text.

| Organ | Raw text cosine similarity | Raw text L2 distance | Decoder Stage 0 cosine similarity | Decoder finest logit cosine similarity | Change (finest logit minus raw text) |
|---|---|---|---|---|---|
| Vessel | 0.759 | 95.4 | 0.540 | 0.960 | +0.202 |
| Thalamus | 0.840 | 78.4 | 0.739 | 0.980 | +0.140 |
| Amygdala | 0.850 | 75.0 | 0.656 | 0.984 | +0.134 |
| Cerebral cortex | 0.851 | 75.4 | 0.772 | 0.979 | +0.129 |
| Choroid plexus | 0.853 | 73.3 | 0.503 | 0.961 | +0.108 |
| Cerebellum cortex | 0.857 | 73.1 | 0.754 | 0.970 | +0.113 |
| Caudate | 0.858 | 72.2 | 0.766 | 0.988 | +0.130 |
| Cerebral white matter | 0.860 | 72.0 | 0.713 | 0.950 | +0.089 |
| Hippocampus | 0.861 | 72.3 | 0.676 | 0.990 | +0.129 |
| Accumbens | 0.862 | 73.1 | 0.667 | 0.986 | +0.124 |
| Pallidum | 0.880 | 67.3 | 0.766 | 0.988 | +0.108 |
| Cerebellum white matter | 0.882 | 65.5 | 0.702 | 0.984 | +0.102 |
| Putamen | 0.885 | 65.7 | 0.770 | 0.988 | +0.103 |
| Lateral ventricle | 0.911 | 57.6 | 0.687 | 0.978 | +0.067 |
| Inferior lateral ventricle | 0.912 | 57.4 | 0.657 | 0.981 | +0.069 |
| Ventral DC | 0.920 | 55.8 | 0.756 | 0.992 | +0.072 |
| **Mean of 16 organs** | 0.865 | 70.6 | 0.695 | 0.979 | +0.114 |

*Note.* Raw text values are identical for every subject, because they depend only on the prompt. Decoder values are means over subjects. Cosine similarity of 1.0 means identical, and an L2 distance of 0 means identical.

## Figure

**Figure 1. Left-versus-right cosine similarity of the raw text and of the finest logit of the U-Net decoder for each organ.**

- **What it shows.** The red circle is above the green circle in every column, so the finest logit is more similar than the raw text for all 16 organs. The mean cosine similarity rises from 0.865 to 0.979, an increase of 0.114.
- **What to notice.** Vessel has the lowest raw text similarity (0.759) and the largest increase (+0.202), lateral ventricle has the smallest increase (+0.067), and cerebral white matter has the lowest finest-logit value (0.950).
- **Why it matters.** The side difference that exists in the text is partly lost by the decoder output, so the decoder narrows the gap between the two prompts.

## Key findings

1. **The raw text of the left and right prompts is different.** The cosine similarity between the "left X" and "right X" text embeddings ranges from 0.759 (vessel) to 0.920 (ventral DC), with a mean of 0.865, and the L2 distance ranges from 55.8 to 95.4, with a mean of 70.6. Vessel is the most different on both measures, while ventral DC, lateral ventricle and inferior lateral ventricle (0.91 to 0.92) are the most similar (Table 1, Figure 1). This means that the text encoder already separates "left" from "right" before VoxTell uses the prompt, so the side information starts in the text.
2. **The image features carry no left/right difference.** Their similarity between the two prompts is exactly 1.00, because the image encoder never receives the prompt, and this was checked directly for every subject. The image features of a scan are therefore the same whichever side is asked for. This means that the image cannot tell the model which side to choose, and the whole left/right instruction comes from the text.
3. **The difference becomes smaller by the final decoder output.** At the finest logit the two prompts are more similar than as raw text for all 16 organs, and the mean cosine similarity rises from 0.865 to 0.979, an increase of 0.114 (Table 1). In Figure 1 the red circle is above the green circle in every column. This means that the decoder reduces the text difference on its way to the output, and only a small difference remains at the end (0.021 below 1.00 on average).
4. **The size of the increase depends on the organ.** It is largest for vessel (+0.202, from 0.759 to 0.960) and thalamus (+0.140), and smallest for lateral ventricle (+0.067) and inferior lateral ventricle (+0.069). Cerebral white matter has the lowest finest-logit similarity (0.950) although its raw text value (0.860) is mid-range (Table 1). This means that how much of the text difference is kept depends on the organ and not only on how different the raw text is. The decoder therefore treats organs differently, in addition to treating the two sides.
5. **The early decoder stage separates the prompts more than the raw text.** The mean cosine similarity at Stage 0 is 0.695, below the raw text value of 0.865, and it is lower than the raw text for all 16 organs, while at the finest logit it is 0.979 (Table 1). This means that the decoder first increases the left/right difference at Stage 0 and then loses it at the later stages. The difference between the two prompts is therefore largest in the early part of the decoder, even larger than in the raw text.

## Conclusion

Text and image representations encode laterality differently. The text embeddings of "left X" and "right X" differ (cosine 0.76 to 0.92), while the image features are identical for both prompts (cosine 1.00), so the side comes from the text alone. The text difference is not kept to the end of the pipeline, since at the finest logit the two prompts are more similar than as raw text for every organ (0.979 against 0.865).

## Definitions

- **Raw text embedding.** The prompt as encoded by the frozen Qwen3 text encoder, before any VoxTell-specific processing. The similarity between the "left X" and "right X" embeddings is measured with cosine similarity (1.0 = identical) and L2 distance (0 = identical; larger = more different). Because the embedding depends only on the prompt, each organ has a single value, the same for every subject.
- **Image features.** The 3D U-Net encoder never receives the text prompt, so the image features for a scan are the same whichever prompt is used. This was checked directly: the image features computed twice for the same scan were identical for every subject, so the left-versus-right image-feature similarity is exactly 1.00.
- **Final decoder output.** The cosine similarity between the left-prompt and right-prompt maps at the finest logit of the U-Net decoder, averaged over subjects. Fusion cosine at Stage 0 (the first decoder stage) is given for comparison. These are the same quantities as in RQ1.
