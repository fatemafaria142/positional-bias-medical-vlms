# Axis III — Proving Text–Image Alignment: Methodology for RQ1–RQ11

Mechanistic investigation of whether and where anatomical laterality (left vs. right) is
represented, preserved, and causally used inside VoxTell, a free-text-promptable 3D medical
segmentation model. RQ1–RQ5 are the original research questions from
`Axis_III_Proving_Text_Image_Alignment_RQ_Plans.docx`; RQ6–RQ11 are follow-up mechanistic
experiments added during the investigation, reusing already-implemented analysis code
(`scripts/voxtell_mechanistic.py`, `scripts/voxtell_experiments.py`) rather than introducing new
methodology from scratch. All eleven are implemented end to end in
`RQ1_to_RQ11_axis3_experiments_kaggle.ipynb`, which exports every result as CSV (ground-truth
masks, predicted masks, and all intermediate representations needed for visualization and
metric computation — no image files, so results can be downloaded from Kaggle as plain CSVs).

Every RQ below runs independently on **every bilateral (left/right) organ pair** in the
Neurite-OASIS brain-MRI dataset (16 pairs — see the Organ coverage section at the end of this
document), across all 100 staged subjects unless noted otherwise, and is repeated per organ,
per subject.

---

## RQ1: Do the 32 Text–Image Fusion Channels Distinguish Left and Right Anatomical Structures?

**Research question:** When the same medical image is paired with "left `<organ>`" and "right `<organ>`" prompts, do the 32 fusion channels produce distinguishable similarity maps — and does this hold across every bilateral organ pair available, not just kidney?

### Mechanism

VoxTell's checkpoint (`voxtell_v1.1`) instantiates `VoxTellModel(..., num_heads=32, num_maskformer_stages=5, ...)`. The transformer decoder's fused text+image output (`mask_embedding`) is projected through `project_to_decoder_channels` into **32 channels per stage**, then concatenated onto the U-Net decoder's skip features at each of the **5 maskformer fusion stages**. These 32 channels are the object of RQ1 — a U-Net-decoder quantity, not the 6-layer transformer's cross-attention heads (RQ3/RQ4's territory). Only 4 of the 5 maskformer stages produce this literal 32-channel tensor; the 5th (finest, full-resolution) stage contracts the *unreshaped* mask embedding directly into the final per-prompt segmentation logit, with no 32-channel intermediate, and is therefore analyzed as its own single-channel comparison.

### Procedure (repeated independently per organ pair)

1. Select scans containing both sides of the organ pair, with verified ground-truth masks.
2. Run the model on the same image with the prompts "left `<organ>`" and "right `<organ>`" — the only difference between the two forward passes is that one word.
3. Extract the 32 fusion channels at each of the 4 intermediate decoder stages, plus the finest stage's direct logit.
4. Compute channel-wise cosine similarity, Pearson correlation, and normalized L2 difference between the left-prompt and right-prompt maps at each stage.
5. Threshold the finest stage's logit into a predicted mask for each prompt; score against both the named organ's and the mirrored organ's ground truth.

### Output

Per subject × organ pair × stage: `cosine_mean`, `cosine_min`, `corr_mean`, `norm_l2_mean`. Per subject × organ pair × side: ground-truth mask, predicted mask, `dice_named_organ`, `dice_other_side_organ` (the latter is a direct laterality-error signal — high overlap with the *wrong* organ's GT). Plus the exact image patch each prediction ran on.

### Visualization

1. Organ × stage heatmap of mean cosine similarity (low = strong laterality discrimination at that stage).
2. Per-organ bar chart ranking the 4 intermediate stages by discriminative power.
3. Image + GT + prediction overlay for the organ with the weakest vs. strongest discrimination.
4. Scatter of stage-wise cosine similarity vs. `dice_other_side_organ` — tests whether weak fusion-channel discrimination actually predicts a laterality error in the output, rather than assuming the link.

### Metrics

Cosine similarity (mean + min over 32 channels), Pearson correlation, normalized L2 difference (per stage, per organ, per subject); Dice and IoU against both the named and mirrored organ's ground truth.

### Interpretation note

Near-identical maps for a given organ suggest weak prompt sensitivity at that fusion stage for that organ, but do not by themselves prove text contains no laterality information — see RQ2–RQ5 for the representation, causal, and attribution follow-ups.

---

## RQ2: Do Text and Image Representations Encode Different Information About Laterality?

**Research question:** How do text-feature similarity and image-feature similarity differ, and does their relationship explain failures in left/right segmentation?

### Mechanism

Two independent measurements. (a) Raw Qwen3 text-embedding similarity between "left `<organ>`" and "right `<organ>`" prompts, measured **before** any VoxTell-specific projection — the representation RQ1's fusion channels are eventually built from. (b) An empirical verification that image features are prompt-independent: since `model.encoder` never sees the text prompt, encoding the same image patch twice must produce bit-for-bit identical bottleneck features — confirmed directly rather than assumed.

### Procedure

1. Embed the left and right prompts for each organ pair with the frozen Qwen3 text encoder.
2. Compute cosine similarity and L2 distance between the two raw embeddings.
3. Encode the same image patch twice through the image encoder; verify the resulting memory tensor is identical both times.

### Output

Per subject × organ pair: `cosine_similarity`, `l2_distance` (raw text embedding space); one `image_feature_identical_check` boolean per subject.

### Visualization

1. Bar chart of raw-text cosine similarity per organ, sorted ascending.
2. Paired scatter: raw-text cosine similarity (x) vs. RQ1's finest-stage cosine similarity (y) for the same organ — points below the diagonal mean VoxTell's pipeline made the representation *more* discriminative than the raw text alone; points above mean discriminability was lost somewhere between text encoding and fusion (RQ3's question).

### Metrics

Cosine similarity and L2 distance (text, raw embedding space); a boolean sanity check (image-feature prompt-independence), not a scientific metric.

### Interpretation note

The cross-modal (text-image) comparison this RQ also conceptually asks for is the same computation as RQ1's per-stage fusion-channel similarity and is not duplicated here, to avoid a second full pass over the same subjects for the same numbers.

---

## RQ3: Does the Prompt Decoder Preserve or Suppress Laterality Information?

**Research question:** Do left/right text embeddings become more similar or less distinguishable after passing through the prompt decoder?

### Mechanism

Cosine distance and relative L2 distance between the left-prompt and right-prompt query states at every point in the 6-layer transformer decoder's residual stream — layer 0 (immediately after `project_text_embed`, before any decoder layer) through layer 6 (after the final layer and norm). Measured for the organ pair's own left-vs-right comparison (the "target" pair) and, in parallel, for a different-organ, same-side "reference" pair, so laterality-specific distance can be separated from generic organ-to-organ distance. A genuine KL divergence between the left- and right-prompt cross-attention distributions is also computed at each of the 6 layers, since cross-attention weights already sum to 1 over image tokens.

### Procedure

1. Project text embeddings into query space (layer 0) and run them through the 6-layer decoder, recording the query state after each layer.
2. Compute cosine distance and relative L2 distance between left-prompt and right-prompt states at every layer, for both the target pair and the reference pair.
3. Compute KL divergence (both directions) between the two prompts' cross-attention distributions at each layer.

### Output

Per subject × pair (target or reference) × scope (`organ_patches`/`all_patches`) × layer (0–6): `cosine_distance`, `relative_l2_distance`; for the target pair only, `attention_kl_left_from_right`, `attention_kl_right_from_left` at layers 1–6.

### Visualization

1. Line plot of `cosine_distance` across layers 0–6, one line per organ (or mean with range), with the reference pair's line overlaid as a dashed comparison — a falling line toward layer 6 indicates suppression; a rising or flat line indicates preservation or amplification, relative to the reference.
2. Same style of line plot for `relative_l2_distance`.
3. Line plot of attention KL divergence across layers 1–6.

### Metrics

Cosine distance, relative L2 distance, KL divergence (bidirectional).

### Interpretation note

A small embedding difference does not, by itself, prove laterality information is absent — it must be interpreted alongside RQ4's causal intervention and RQ5's attribution results.

---

## RQ4: Which Decoder Stages Causally Influence Laterality Through Activation Patching?

**Research question:** Does replacing left-prompt text features with right-prompt text features at an individual decoder stage change the segmentation output?

### Mechanism

Three conditions per layer (0–5), per organ pair: (a) an unpatched **baseline**; (b) a positive-control **full-state patch**, replacing the entire running state at one layer with the other prompt's state — this reproduces the other prompt's output by construction (a residual-stream identity, not a finding) and only confirms the intervention pipeline is wired correctly; (c) the primary causal test, an **update-only patch**, replacing only that layer's own contribution (`state_out − state_in`) with the other prompt's update at that layer, leaving the running state from earlier layers untouched.

### Procedure

1. Run the left prompt and the right prompt, caching every layer's state and update.
2. For each layer, construct the full-state and update-only patched conditions described above.
3. Decode each condition to a predicted mask; score Dice/coverage against both sides' ground truth and record whether the prediction's laterality call flipped toward the source prompt's side.

### Output

Per subject × organ pair × condition: `variant`, `dst`/`src`, `layer`, `dice_left_organ`, `dice_right_organ`, `cov_left_organ`, `cov_right_organ`, `call`, `moved_to_source_side` (the laterality-flip flag), `agreement_with_source_baseline` (sanity check for the full-state control — should be ≈1.0), and the predicted mask for every single condition, plus ground-truth masks.

### Visualization

1. Bar chart, by layer, of the fraction of cases where `moved_to_source_side == True` under the update-only condition — directly ranks the 6 layers by causal influence over laterality.
2. Grouped bar chart comparing baseline Dice to each layer's update-only Dice for the same organ.
3. Histogram of `agreement_with_source_baseline` for the full-state-control rows only, as a correctness check on the patching mechanism itself.
4. Reconstructed overlay of the predicted mask for the layer with the highest flip rate.

### Metrics

Dice and coverage (both sides); laterality-flip rate; full-state-control agreement rate (mechanism correctness check, not a scientific result).

### Interpretation note

If patching has no effect, the stage may not be influential, the information may be redundant across layers, or another pathway may bypass the intervention — this does not independently prove that text is irrelevant. Report effect sizes from update-only patching as the primary finding; use full-state patching only to confirm the intervention pipeline is wired correctly.

---

## RQ5: How Much Do Text and Image Features Contribute to the Final Segmentation?

**Research question:** What are the relative contributions of text and image representations to correct anatomical segmentation?

### Mechanism

An exact, two-player Shapley decomposition of the final prediction's Dice score. Players: "image informative" (the real image vs. a Gaussian-noise image matched to the real image's mean/standard deviation) and "text informative" (the sided prompt, e.g. "left kidney", vs. a neutral prompt with the side word removed, e.g. "kidney"). With two players there are exactly four coalitions, so the Shapley value is exact, not sampled or estimated.

### Procedure

1. For each organ, generate the four coalition conditions (neither informative, image only, text only, both).
2. Predict a mask under each condition; score Dice against ground truth.
3. Compute the exact Shapley value for each player from the four Dice scores.

### Output

Per subject × organ: `dice_neither`, `dice_image_only`, `dice_text_only`, `dice_both`, `shapley_image`, `shapley_text` (sum to `dice_both − dice_neither` by the Shapley efficiency axiom), `interaction`; the predicted mask for all four coalitions plus ground truth.

### Visualization

1. Grouped or stacked bar chart per organ of `shapley_image` vs. `shapley_text`.
2. Line plot of the layer-wise Dice-drop proxy derived from RQ4's update-only patching results (re-expressed as an approximate per-layer text contribution — explicitly labeled as a proxy, not a true per-layer Shapley value, since a genuine per-layer decomposition would require the same per-layer patching machinery RQ4 already performs).
3. Scatter of `shapley_text` vs. RQ1's `dice_other_side_organ` for the same organ.

### Metrics

Exact two-player Shapley value (image, text), interaction term, the four raw coalition Dice scores.

### Interpretation note

This is a final-output-level decomposition. The layer-wise proxy derived from RQ4 is a Dice difference, not a Shapley value, and should be reported as such.

---

## RQ6: Is There One Shared Left→Right Direction, and Does Adding It Alone Flip the Output?

**Research question (added experiment):** Is laterality represented as one shared, composable direction across every organ (the classic "king − man + woman" word-vector arithmetic), or does each organ get its own, unrelated left/right direction? If one shared direction exists, is it causally sufficient to flip the output on its own?

### Mechanism

Part A (geometry): for every pair of organs, compute `embed("right X") − embed("left X")` in raw Qwen3 embedding space, and measure the cosine similarity between every pair of organs' diff vectors, summarized by a `coherence_ratio` (`‖mean(diff vectors)‖ / mean(‖diff vector‖)`: 1.0 = one shared direction for every organ, 0.0 = unrelated per-organ directions). This step is subject- and image-independent. Part B (causal steering): the single mean diff vector across every organ is added, at several scales, to a **neutral** prompt's raw embedding (no side word at all, and no opposite-side prompt involved anywhere in the test), then run through the full model to see which side the resulting mask lands on.

### Procedure

1. Compute per-organ left→right diff vectors in raw embedding space; measure pairwise cosine similarity and the coherence ratio.
2. Compute the mean diff vector across all organs.
3. For each organ, add the mean diff vector (scaled by 0, 0.5, 1.0, 1.5, 2.0) to a neutral prompt's embedding and predict; score against both sides' ground truth.

### Output

`kind='pair'` rows (pairwise cosine similarity between organs' diff vectors) and one `kind='summary'` row (`mean_cosine_similarity`, `coherence_ratio`); per subject × organ × scale: `dice_vs_left_gt`, `dice_vs_right_gt`, `call`, `moved_to_right_without_right_word`, predicted mask, ground-truth masks.

### Visualization

1. Organ × organ heatmap of the pairwise cosine-similarity matrix.
2. Line plot (dose-response curve): steering scale (x) vs. fraction of cases flipped to the right side without ever naming a side (y).

### Metrics

Cosine similarity between per-organ diff vectors; `coherence_ratio`; Dice against both sides' ground truth; flip rate at each steering scale.

### Interpretation note

A causal flip from pure vector addition, with no opposite-side prompt ever presented, is stronger evidence of a composable, steerable laterality direction than RQ4's matched-pair patching, which always requires a real opposite-side prompt to exist.

---

## RQ7: Which Specific Attention Head Causally Matters?

**Research question (added experiment):** Is a layer's causal influence over laterality (RQ4) spread evenly across all its attention heads, or concentrated in one specific head?

### Mechanism

Zero-ablation of each of the 8 cross-attention heads, one at a time, at the final decoder layer (extendable to all 6 layers at proportionally higher cost), via a `forward_hook` on `multihead_attn` (verified against a real `nn.MultiheadAttention` — a `forward_pre_hook` on `out_proj` was tried first and found to silently never fire, since PyTorch's internal attention implementation calls `F.linear` directly rather than invoking `out_proj` as a module). Every result is preceded by a mandatory no-op validation: an empty-head-list ablation, which must leave the prediction unchanged, confirming the hook mechanism itself is not introducing spurious effects.

### Procedure

1. Run the no-op validation; confirm agreement is 1.0 (or within floating-point rounding).
2. For each organ and each head, zero that head's output, re-predict the full mask, and score against ground truth and against the unablated baseline.

### Output

Per subject × organ × head: `dice_vs_gt`, `dice_vs_baseline`, `call`, plus the ground-truth mask, the unablated baseline mask, and that head's ablated prediction mask; a separate validation-agreement record (must read 1.0).

### Visualization

1. Heatmap, organ × head, of `1 − dice_vs_baseline` — which heads, for which organs, change the prediction at all when zeroed.
2. Bar chart of the fraction of cases where ablating head *h* flips the call to the mirror organ — the head-level analogue of RQ4's layer-ranking chart.

### Metrics

Dice against ground truth and against the unablated baseline; per-head flip rate.

### Interpretation note

Run on a small subject subset by default — this experiment's cost is combinatorial (organs × heads full predictions per subject), unlike the linear-in-organ cost of RQ1–RQ6.

---

## RQ8: Where in the Image Does Gradient/Grad-CAM Saliency Concentrate?

**Research question (added experiment):** When only the prompt's side word changes, where in the image does the model's own gradient concentrate, and is it actually looking at the named organ?

### Mechanism

Vanilla-gradient saliency and Grad-CAM, computed from a single encoder-plus-backward pass per organ per side (far cheaper than RQ9's full occlusion enumeration), summarized by an enrichment ratio (mean saliency inside the organ's ground-truth region divided by mean saliency outside it).

### Procedure

1. For each organ and side, crop one patch around the organ's ground-truth centroid.
2. Compute vanilla-gradient saliency and Grad-CAM with respect to that prompt.
3. Summarize as the in-organ vs. outside-organ enrichment ratio; for a small set of representative organs, also retain the full dense maps.

### Output

Per subject × organ pair × side: `mean_saliency_in_organ`, `mean_saliency_outside_organ`, `saliency_enrichment`, and the equivalent Grad-CAM quantities. For representative organs: the dense saliency and Grad-CAM maps, the crop box they are defined in, and the matching ground-truth mask in that same crop frame.

### Visualization

1. Bar chart of saliency/Grad-CAM enrichment per organ and side.
2. Saliency/Grad-CAM overlay, reconstructed from the dense maps and the matching ground truth, for a representative organ.

### Metrics

Saliency and Grad-CAM enrichment ratio (in-organ vs. outside-organ mean).

### Interpretation note

High enrichment is a sanity check that the model is actually attending to the organ when it segments it correctly — a complementary, spatial counterpart to RQ5's single scalar image-contribution number.

---

## RQ9: Which Image Region Matters Most (a Spatial Shapley Map)?

**Research question (added experiment):** Rather than one global "how much does the image matter" number (RQ5), which specific spatial region of the image does the model rely on?

### Mechanism

A padded box around each organ's ground-truth region is tiled into `2³ = 8` blocks, giving `2⁸ = 256` present/absent coalitions — cheap enough to enumerate exactly, the same affordability argument as RQ5's exact two-player game, scaled up to 8 players. Absent blocks are replaced with Gaussian noise; each coalition's Dice score is converted into an exact Shapley credit per block.

### Procedure

1. Tile a padded box around the organ's ground truth into 8 blocks.
2. Enumerate all 256 present/absent combinations of the 8 blocks, noising the absent ones, and score Dice for each.
3. Convert the 256 coalition scores into one exact Shapley value per block.

### Output

Per subject × organ × block: `block_bbox` (absolute full-volume coordinates), `shapley_value`, plus the organ's ground-truth mask for overlay.

### Visualization

1. Spatial heatmap of block-wise Shapley value overlaid on the image and ground truth.
2. Bar chart comparing the block containing the organ's centroid to the mean of all other blocks.

### Metrics

Exact Shapley value per block (sums to `dice_both − dice_neither` for that organ, the same efficiency axiom as RQ5).

### Interpretation note

Run on a small subject subset by default — at 256 full predictions per organ per subject, this is the single most expensive experiment in the investigation.

---

## RQ10: Which Specific Word Contributes More — Organ Word, Position Word, Other Words, or Image?

**Research question (added experiment):** RQ5 splits credit between "image" and the *entire* text prompt as two lumped players. Within the text side, is a laterality error attributable to the position word ("left"/"right") not being used, the organ word not being used, or neither — and does the sentence's surrounding scaffolding contribute anything at all?

### Mechanism

An exact, four-player Shapley decomposition of the final prediction's Dice score. Players: **image** (real vs. noised over the organ's ground-truth region), **organ word** (e.g. "kidney" vs. a generic "structure"), **position word** (e.g. "left" vs. dropped), and **other words** (the fixed scaffolding "Segment", "the", "in", "this", "image" vs. dropped down to the bare content words, e.g. "Left kidney." instead of "Segment the left kidney in this image."). Four players give `2⁴ = 16` coalitions, still enumerated exactly. Shapley values are a property of the entire player set: this is a genuinely different game from RQ5's two-player one, not the same numbers with a third or fourth added — splitting "text" into sub-players redistributes shared interaction credit across all players.

### Procedure

1. For each organ, construct all 16 present/absent combinations of the four roles (noising the image when absent; substituting "structure" when the organ word is absent; dropping the position word when absent; dropping the sentence scaffolding when "other words" is absent).
2. Predict a mask under each of the 16 conditions; score Dice against ground truth and against the fully-informed baseline.
3. Compute the exact Shapley value for each of the four roles under both scoring functions.

### Output

Per subject × organ × role × metric: `shapley_value` (Dice-score units; the four `f_dice_gt` values sum exactly to `dice_all_informed − dice_none_informed`). Per subject × organ × coalition (16 rows): the four role booleans, `dice_gt`, `dice_self`, the predicted mask, and the ground-truth mask.

### Visualization

1. Grouped bar chart per organ of the four roles' Shapley values.
2. Scatter of the position word's Shapley value against RQ1's `dice_other_side_organ` or RQ4's flip rate for the same organ.
3. Reconstructed overlay of the "image and organ word present, position word absent" coalition's mask against ground truth — literally, what the model predicts when the prompt never states a side.

### Metrics

Exact four-player Shapley value (Dice-score units) under both `f_dice_gt` and `f_dice_self`; the 16 raw coalition Dice scores.

### Interpretation note

Not directly comparable in raw magnitude to RQ5's two-player split — the two experiments answer related but distinct questions, at different resolutions, and should be reported as complementary rather than reconciled into one number. A large Shapley value for "other words" would be a notable finding on its own: it would suggest the model's behavior depends on sentence framing, not purely on keyword presence.

---

## RQ11: Prompt-Injection Bias Probe — When Confused by Injected or Conflicting Text, Does the Model Default Left or Right?

**Research question (added experiment):** RQ1–RQ10 all compare a clean, unambiguous "left X" prompt against a clean "right X" prompt. None of them ask what happens when the prompt gives the model genuinely conflicting or insufficient signal about which side is intended. When that ambiguity is injected deliberately, does the model resolve it toward left or right at random, or systematically toward one side — revealing a laterality bias that clean, unambiguous prompts can never surface?

### Mechanism

Six prompt variants, each injecting conflicting or irrelevant text alongside or instead of a clean "left X"/"right X" instruction: (1) a left-then-right correction ("...left X... Ignore that, segment the right X instead."); (2) the mirror, right-then-left correction; (3) no side word at all plus an irrelevant injected sentence ("...the X... This scan was acquired under standard protocol."); (4)–(5) a more aggressive, system-override-shaped injection in both mention orders; (6) a prompt whose only side word appears inside a secondary, injected clause rather than the primary instruction ("Segment the X... Actually, make it the right one."). VoxTell has no mechanism to "obey instructions" the way a conversational model does — it is a single-shot text-to-mask model with a frozen text encoder, not an agent — so this is not a security exploit in the classical prompt-injection sense. What it probes instead is how the model resolves genuine ambiguity, which is the relevant bias question.

### Procedure

1. For each organ, construct the six injected/conflicting prompt variants.
2. Predict a mask for each variant; score against both sides' ground truth (which side, if any, the output resolves to) and against this same subject's own clean, unambiguous "left X"/"right X" predictions (how closely the injected output resembles a normal prediction for either side).

### Output

Per subject × organ × variant: the injected prompt text, `dice_left_organ`/`dice_right_organ`, `call` (`'left'`/`'right'`/`'both'`/`'empty'`/`'neither'`), `dice_vs_clean_left`/`dice_vs_clean_right`, the predicted mask, and both sides' ground-truth masks.

### Visualization

1. Bar chart, one bar per variant, of the fraction of rows resolving to `'left'` vs. `'right'` vs. other — the headline result.
2. Paired comparison of the two conflict variants (left-then-right vs. right-then-left): if the model always follows whichever side is mentioned first, these two variants' resolution distributions should mirror each other; if it favors one specific side regardless of mention order, they will not.
3. Per-organ breakdown of the same bar chart, cross-referenced against RQ1's `dice_other_side_organ` and RQ4's `moved_to_source_side` for the same organs.

### Metrics

Per-variant resolution rate (fraction landing on `'left'`/`'right'`/other); Dice against each side's ground truth; Dice against this subject's own clean baseline predictions for each side.

### Interpretation note

A model with no laterality bias would show roughly even left/right resolution rates across the conflict variants, resolving toward whichever side is mentioned last (ordinary instruction-following) or first (a recency/anchoring effect, interesting but not itself a left/right bias unless it is asymmetric between the two conflict variants). A genuine laterality bias specifically means the resolution rate favors one literal side more than the other after accounting for mention order — e.g. if the right-then-left variant resolves to "right" more often than the left-then-right variant resolves to "left", that asymmetry is not explained by recency alone.

---

## Organ Coverage (applies to every RQ above)

Every RQ above runs once per bilateral (left/right) organ pair below. Structures with no left/right counterpart (brain midline structures) are out of scope — they carry no laterality signal to test.

### OASIS-1 (brain MRI, FreeSurfer aseg35 labels)

| # | Organ pair | Prompt pair |
|---|---|---|
| 1 | Cerebral white matter | "left cerebral white matter" / "right cerebral white matter" |
| 2 | Cerebral cortex | "left cerebral cortex" / "right cerebral cortex" |
| 3 | Lateral ventricle | "left lateral ventricle" / "right lateral ventricle" |
| 4 | Inferior lateral ventricle | "left inferior lateral ventricle" / "right inferior lateral ventricle" |
| 5 | Cerebellum white matter | "left cerebellum white matter" / "right cerebellum white matter" |
| 6 | Cerebellum cortex | "left cerebellum cortex" / "right cerebellum cortex" |
| 7 | Thalamus | "left thalamus" / "right thalamus" |
| 8 | Caudate | "left caudate" / "right caudate" |
| 9 | Putamen | "left putamen" / "right putamen" |
| 10 | Pallidum | "left pallidum" / "right pallidum" |
| 11 | Hippocampus | "left hippocampus" / "right hippocampus" |
| 12 | Amygdala | "left amygdala" / "right amygdala" |
| 13 | Accumbens | "left accumbens" / "right accumbens" |
| 14 | Ventral DC | "left ventral DC" / "right ventral DC" |
| 15 | Vessel | "left vessel" / "right vessel" |
| 16 | Choroid plexus | "left choroid plexus" / "right choroid plexus" |

*Excluded (no left/right counterpart):* 3rd ventricle, 4th ventricle, brain stem.

### AMOS22 (abdominal CT and MRI) — for future extension beyond the brain

| # | Organ pair | Prompt pair |
|---|---|---|
| 1 | Kidney | "left kidney" / "right kidney" |
| 2 | Adrenal gland | "left adrenal gland" / "right adrenal gland" |

## Aggregation Across Organs (applies to RQ1–RQ11)

Once every organ pair has been run through the procedure above for a given RQ:

- Roll up the per-organ results into one organ × metric matrix, to see whether an effect is organ-specific or consistent across anatomy.
- Flag organs showing the weakest laterality discrimination/strongest laterality error as candidates for closer inspection across the other RQs (e.g. an organ weak in RQ1 should be checked against RQ4's flip rate and RQ10's position-word Shapley value for the same organ).
- When extending to AMOS22, analyze CT and MRI separately before pooling with the OASIS-1 brain results.
