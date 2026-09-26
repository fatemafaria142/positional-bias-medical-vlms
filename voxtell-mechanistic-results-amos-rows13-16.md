# VoxTell mechanistic experiments, rows 13-16: AMOS22 MRI results

**Data:** 3 subjects (`amos_0508`, `amos_0530`, `amos_0555`), the ones whose rows 14-16 finished; organs = left/right kidney and
left/right adrenal gland, unrotated. Row 13 also ran for the other 7 subjects, but they are left out here so that all four rows
describe the same 3 subjects. Source CSVs: `AMOS-ROW(13 to 16)/*.csv`. Figures: `AMOS-ROW(13 to 16)/figures/`
(made by `scripts/plot_amos_rows13_16.py`).

**Read this first.** Three subjects is a pilot, not a distribution. Every number below is a mean or a count over 3 subjects
(6 organs per organ type in row 13); treat the patterns as leads to confirm on more subjects, not as established results.
All 12 unmodified baseline predictions had the correct side (kidney Dice 0.94-0.97, adrenal gland 0.46-0.74), so any change
below is a change from a working starting point.

---

## Summary of the answers

| Row | Question | Answer (3 subjects) |
|---|---|---|
| 13 | Does the model need image evidence, or does it draw a memorized organ? | **Kidney: needs the image.** Replace the kidney by noise and the mask disappears (5 of 6 empty). **Adrenal gland: partly context.** The mask survives (about 80% of its size, 0 of 6 empty) even when the gland itself is noise. A pure noise image gives no mask for either organ. |
| 14 | Where is the side decided, and does the mask change smoothly? | The side follows the text embedding: it flips between alpha 0.25 and 0.75 (sharp, not gradual), and extrapolating to -0.5 / 1.5 keeps the side. No single decoder layer can flip it (0 of 72 masks); patching layers 1-6 together can (kidney 3 of 3 left to right), layers 4-6 alone cannot (0 of 12). So the side is settled in layers 1-3 together. |
| 15 | Does "left kidney" look at the left kidney inside the model? | **Kidney: yes**, from layer 1. Attention on the named kidney is 2.4x to 8.6x the volume average vs 0.6x to 2.6x on the other-side kidney. **Adrenal gland: barely**; enrichment stays at or below 1 until layer 4, and only layer 6 favors the named gland (2.3 vs 1.7). |
| 16 | At which layer do left and right first diverge? | They already differ at the input (cosine distance 0.24). For the kidney the gap grows a little after attention (to 0.33 by layer 6), for the adrenal gland it shrinks (to 0.17). Left/right is always smaller than the organ-vs-organ distance (0.37 rising to 0.51). |

---

## Row 13 - organ replaced by noise

![row 13](AMOS-ROW%2813%20to%2016%29/figures/fig_row13_noise.png)

The named organ's voxels (or both organs of the pair, or the whole image) are replaced by Gaussian noise with the image's mean and
spread, then the same prompt is run again. Size is the predicted mask size divided by the size of the unmodified prediction.

| Organ | Condition | Median size vs. original | Empty masks | Mask overlap with the noised region | Centroid offset (voxels, median) |
|---|---|---|---|---|---|
| kidney | organ noised | 0.00 | 5 / 6 | 0.00 | n/a for empty |
| kidney | both organs noised | 0.00 | 4 / 6 | 0.00 | |
| kidney | whole image noise | 0.00 | 5 / 6 | 0.00 | |
| adrenal gland | organ noised | 0.80 | 0 / 6 | 0.53 | 2.4 (original: 1.6) |
| adrenal gland | both organs noised | 0.78 | 0 / 6 | 0.53 | 2.2 |
| adrenal gland | whole image noise | 0.00 | 6 / 6 | 0.00 | |

- **Kidney:** with the kidney replaced by noise the model draws nothing. It is not filling in a "usual kidney" from memory; it needs
  the kidney's own pixels. One kidney of the whole-image-noise case gave a mask (size 0.52) but far from any organ (overlap 0.01).
- **Adrenal gland:** the mask stays about where it was (centroid moves about 1 voxel) and keeps about 80% of its size, although the
  gland itself is now noise. With a pure noise image it disappears completely. So it is not a fixed picture drawn from the prompt
  alone, but it also does not depend on the gland's own voxels: the model is using the surrounding anatomy (position relative to the
  kidney, liver and so on). Caution: an adrenal gland is only a few hundred voxels, so the noise leaves almost all of its
  surroundings intact; this test cannot separate "learned position prior" from "reading the neighboring organs".

---

## Row 14 - blending and patching the text representation

### Blend of the left and right embeddings

![row 14 blend](AMOS-ROW%2813%20to%2016%29/figures/fig_row14_blend.png)

| alpha | Kidney: mask covers | Adrenal gland: mask covers |
|---|---|---|
| -0.5 | left (3 of 3) | left (3 of 3) |
| 0.25 | left (3 of 3) | left (3 of 3) |
| 0.5 | mixed: 2 of 3 cover both kidneys, 1 covers neither | 2 of 3 right, 1 neither |
| 0.75 | right (3 of 3) | right (3 of 3) |
| 1.5 | right (3 of 3) | right (3 of 3) |

The text embedding controls the side. The switch is abrupt around alpha 0.5, not a smooth slide, and going beyond either prompt
(-0.5, 1.5) does not produce anything new: the mask stays on the corresponding side.

### Patching one decoder layer, or several

![row 14 patches](AMOS-ROW%2813%20to%2016%29/figures/fig_row14_patches.png)

"Left to right" means: the prompt says left, but the update made by the chosen decoder layer(s) is replaced by the one computed
for the right prompt. A mask counts as switched when it ends up on the other side.

- **Single layer (layers 1-6, both directions, both organs):** 0 of 72 masks switched. No single layer decides the side.
- **Layers 4-6 together:** 0 of 12 switched. The late layers do not decide it.
- **Layers 1-6 together:** kidney left to right 3 of 3, kidney right to left 1 of 3 (the other two ended up on neither kidney),
  adrenal gland left to right 2 of 3, right to left 0 of 3 (two empty masks). Because layers 4-6 alone do nothing, the side
  is set by layers 1-3 acting together (the redundancy means any one of them alone is not enough), and the effect is stronger in
  the left-to-right direction than right-to-left.
- **Control:** patching a whole layer state reproduced the source prompt's mask exactly (voxel agreement 1.0000, minimum 0.999999,
  24 of 24 runs), so the patching code does what it should.

---

## Row 15 - cross-attention of each prompt

![row 15](AMOS-ROW%2813%20to%2016%29/figures/fig_row15_attention.png)

Attention density inside each ground-truth organ divided by the volume average (1 = no preference), averaged over the 3 subjects.

| Layer | Kidney: named | Kidney: other side | Kidney: other organ | Adrenal: named | Adrenal: other side | Adrenal: other organ |
|---|---|---|---|---|---|---|
| 1 | 2.42 | 0.58 | 0.53 | 0.56 | 0.25 | 0.31 |
| 2 | 4.42 | 0.64 | 0.29 | 0.33 | 0.23 | 0.33 |
| 3 | 3.76 | 1.17 | 0.40 | 0.28 | 0.23 | 0.32 |
| 4 | 8.55 | 2.62 | 0.80 | 0.78 | 0.52 | 0.71 |
| 5 | 4.54 | 1.73 | 1.04 | 1.26 | 0.74 | 0.81 |
| 6 | 4.41 | 2.28 | 1.86 | 2.34 | 1.71 | 1.13 |

- **Kidney:** the attention is on the named kidney, and it prefers it to the same organ on the other side by about 2x to 7x (it is highest at layer 2). So at
  the attention step the model already separates left from right, not only "kidney" from "not kidney".
- **Adrenal gland:** almost no preference before layer 4, and only a small one at layer 6. The attention grid is coarse
  (12x12x12 tokens per patch) and the gland is tiny, so weak enrichment is expected, but it means the adrenal side is not
  visibly decided by the attention map.
- The attention centre along the right-left axis was closer to the named organ in 67-83% of cases (kidney) and 50-83% (adrenal),
  but each of those is out of only 6 prompts, so it is not informative on its own.

---

## Row 16 - distance between the left and right representations, layer by layer

![row 16](AMOS-ROW%2813%20to%2016%29/figures/fig_row16_distance.png)

Cosine distance between the "left X" and "right X" query states at the decoder input (layer 0) and after each layer; the reference
is the distance between two different organs ("left kidney" vs "left adrenal gland").

| Layer | Kidney: left vs right | Adrenal: left vs right | Reference: organ vs organ |
|---|---|---|---|
| 0 (input) | 0.244 | 0.245 | 0.369 |
| 1 | 0.284 | 0.246 | 0.377 |
| 2 | 0.303 | 0.238 | 0.411 |
| 3 | 0.311 | 0.229 | 0.449 |
| 4 | 0.314 | 0.218 | 0.480 |
| 5 | 0.325 | 0.191 | 0.503 |
| 6 | 0.328 | 0.170 | 0.509 |

- The left/right difference is present from the start (the text encoder already separates the words), so the decoder is not
  creating it from nothing.
- **Kidney:** the gap widens mostly at layer 1 (0.244 to 0.284) and keeps growing slowly, so the side signal is kept and
  amplified, staying at about 64% of the organ-vs-organ distance.
- **Adrenal gland:** the gap shrinks steadily (down to 0.17, about a third of the organ-vs-organ distance). The side signal is
  partly washed out, which fits its weak attention in row 15. The masks still land on the correct side for all 3 subjects, so
  what is left of the signal is enough.
- Left/right stays smaller than organ-vs-organ at every layer: side is a smaller component of the representation than which organ.
- The "organ patches only" and "all patches" averages are identical: all 6 sliding-window patches of these small AMOS crops
  overlap the organs, so that filter changed nothing.

---

## Caveats

- 3 subjects. Per-subject curves in the figures (faint lines) show the spread; counts such as "2 of 3" can change with one subject.
- "Neither" calls in row 14 (mask on neither organ) are counted as not switched; they show that patching layers 1-6 together
  disrupts the mask as well as, or instead of, moving it.
- Row 13 uses Gaussian noise with the image's overall mean and spread; a different noise level or a smoother fill could give different
  numbers for the adrenal gland.
- Row 16 uses cosine distance only in this write-up; the CSV also has the relative L2 distance.
