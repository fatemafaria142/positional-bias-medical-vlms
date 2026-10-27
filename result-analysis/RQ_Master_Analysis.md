# Laterality in VoxTell, Consolidated Analysis of RQ1 to RQ4 and RQ5 to RQ7

## 0. How to read this document

This document puts the seven research questions into one format and compares them with the VoxTell paper. Section 1 lists the key findings that must be noticed. Section 2 compares the paper with our results and lists the issues in the paper. Section 3 describes where the model fails and where it looks, and whether it is biased to one side. Section 4 checks every item of the Axis III plan against what was done. Section 5 gives each research question in the same fixed format, which is the specific question, what it does, hypothesis, expected outcome, methodology, table, figure caption, analysis, results analysis, key finding, conclusion and future work.

**Sources.** The Axis III plan document (RQ1 to RQ4), the VoxTell paper (arXiv 2511.11450), and our own analyses of VoxTell on the Neurite-OASIS brain MRI dataset, using 16 bilateral organ pairs. The paper statements below were read from the HTML version of the paper and should be checked against the PDF, with table numbers, before they are cited in a manuscript.

**Status of the seven research questions.**

| Research question | Result analysis | Note |
|---|---|---|
| RQ1 to RQ4 | Done | Plan questions, reports available |
| RQ5 to RQ7 | Done | Follow-up questions, reports available |

## 1. Key findings that must be noticed

1. **The model does not confuse left and right when the prompt is clean.** With clean prompts the Dice with the wrong side is 0.000 ± 0.001, and only about 6% of the rows overlap the wrong organ at all (RQ1).
2. **The real failure on clean prompts is missing organs, and not the wrong side.** The predicted mask is empty in 14.8% of the prompts, and cerebral cortex, lateral ventricle, choroid plexus and vessel get a Dice of about 0 (RQ1, RQ6).
3. **The side word acts early and then fades.** The 32 fusion channels of the left and right prompts have a cosine similarity of 0.695 at Stage 0 and 0.979 at Stage 3 (RQ1). The raw text embeddings differ (cosine 0.865), while the image features are identical for both prompts (1.00), so the side comes from the text alone (RQ2). The similarity at the later stages may be too high, because the finest logit has a cosine of 0.979 but a Pearson correlation of only 0.652, so this part needs a recomputation.
4. **Attention differs between the two prompts mainly in the first decoder layer.** The KL divergence is 0.079 at Layer 1 and falls to 0.012 at Layer 3 (RQ3).
5. **No single decoder layer decides the side.** Replacing the update of any single layer never moves the mask, and only the replacement of all six layers moves 20.9% of the masks (RQ4).
6. **The model is biased, and the bias appears only when the prompt is under pressure.** When all six layers are replaced, 38.8% of the left prompts move to the right side, against 3.4% of the right prompts moving to the left side (RQ4). When a prompt contradicts itself, the mask follows the last named side to the right in 58.6% of the cases, against 36.6% to the left, and "Actually, make it the right one" is followed in 70.8% (RQ7).
7. **The organ word carries most of the credit, and the image carries little for some organs.** The organ word gets 72.9% of the Dice credit, the image 20.3% and the position word 18.1% (RQ6). For pallidum and hippocampus the image credit is 0.009 and 0.041 (RQ6).
8. **The words around the organ name slightly hurt.** The other words of the sentence have a mean credit of −0.066, and −0.340 and −0.360 for cerebral cortex and lateral ventricle (RQ6).
9. **The model looks at the organ block more than at its surroundings, but it also uses the surroundings.** The organ block has a Shapley value of 0.126 against 0.085 for the surrounding blocks, and it is lower than the surroundings for hippocampus and caudate (RQ5).
10. **All results come from one dataset.** Neurite-OASIS brain MRI with 16 organ pairs was analyzed. The kidney laterality plan on AMOS22 and the lesion dataset have not been analyzed, so the findings do not yet cover abdominal organs or lesions.

## 2. Comparison with the VoxTell paper

### 2.1 What the paper claims and reports

The paper presents VoxTell as a free-text promptable 3D segmentation model with multi-stage vision-language fusion in the decoder. A ResEncL encoder has six stages, the text encoder is the frozen Qwen3-Embedding-4B, and a six-layer prompt decoder with eight heads turns the text into guidance tensors with 32 channels for each decoder scale. The model was trained on about 158 datasets with more than 62,000 CT, MRI and PET volumes and 1,087 concepts, with synonyms and rephrased prompts, and with negative prompts that teach it to return an empty mask. In the zero-shot benchmark it reaches a mean Dice of 70.85 against 51.23 for the next model, and the authors state that it is robust to linguistic variation. The paper says that performance on fully unseen regions remains challenging, and it reports a mean Dice of 10.72 on knee MRI, with lateral meniscus 0.0 and patellar cartilage 0.07, and a Dice of 0 on the left and right lacrimal glands in the RADCURE data.

### 2.2 Issues in the paper

1. **Laterality is claimed but not measured.** The paper says that the model can distinguish left from right structures, but it gives no left-versus-right error rate. The only per-side numbers found are modest, with left breast 41.52 and right breast 45.07.
2. **Robustness is claimed for rephrased prompts only.** The paper shows stable Dice across prompt formulations and gives no numbers in the text. It does not test prompts that contradict themselves or that name the side in an added sentence.
3. **Mirroring was removed from training to avoid ambiguity.** The paper states that left-right mirroring was left out for laterality-sensitive structures. It does not test whether this leaves a preference for one side.
4. **Failure analysis is limited.** The failures that are discussed come from the knee dataset, and the paper does not analyze where its own predictions fail inside the organs it was trained on.
5. **Empty and false-positive predictions are not reported.** Negative prompts are used in training, but the paper gives no empty-mask or false-positive rates.
6. **The fusion is not examined.** The 32 guidance channels and the six prompt decoder layers are described, and the paper gives no analysis of which stage carries the side information.

### 2.3 Paper claims compared with our results

| Claim or gap in the paper | What we found | Verdict |
|---|---|---|
| The model distinguishes left from right structures, shown qualitatively | With clean prompts the wrong-side Dice is 0.000 ± 0.001 and about 6% of the rows overlap the wrong organ (RQ1) | Supported for clean prompts on brain MRI |
| The model is robust to linguistic variation | When the prompt contradicts itself, the mask goes to the last named side in only 58.6% (left then right) and 36.6% (right then left) of the cases, and 41.4% and 63.4% of the masks land on neither side (RQ7) | Not supported for conflicting prompts |
| A single instruction prompt gives a consistent mapping | The sentence words have a mean credit of −0.066, and −0.340 and −0.360 for cerebral cortex and lateral ventricle (RQ6) | The template is not neutral, and it should be compared with the exact prompt of the paper |
| Strong performance on laterality-sensitive structures | Left prompts move to the right side in 38.8% of the cases when all layers are replaced, against 3.4% in the other direction (RQ4) | The two sides are not treated symmetrically |
| Negative prompts teach the model to return an empty mask, with no rates reported | The prediction is empty in 14.8% of our prompts, and in 100% of the choroid plexus prompts (RQ1) | Empty masks are common for some organs, and should be reported |
| Unseen or sparsely represented regions remain challenging | Cerebral cortex, lateral ventricle, choroid plexus and vessel get a Dice of about 0, and the cerebral cortex prediction has 476 voxels against 290,171 true voxels (RQ1) | Consistent with the paper, and the cause for these four organs is not yet tested |

## 3. Where the model fails, where it looks, and whether it is biased

### 3.1 Where the model fails

- **Whole organs are missed.** Cerebral cortex, lateral ventricle, choroid plexus and vessel have a Dice of about 0 for the full prompt, and the predicted masks are tiny compared with the true organs (RQ1, RQ6). This is the largest failure of the model, and it is larger than any laterality error.
- **Organs differ in what they need.** For white matter the image credit is almost as high as the organ word credit (0.361 against 0.434 for cerebral white matter), and for pallidum and hippocampus the image adds almost nothing (RQ6).
- **The sentence around the organ name can hurt.** For cerebral cortex and lateral ventricle the other words remove almost all of the credit (RQ6).
- **A few subjects are harder for some organs.** The total credit of caudate ranges from 0.312 to 0.881 across subjects, and of amygdala from 0.311 to 0.802 (RQ6).

### 3.2 Where the model looks

- **In the image.** The Dice depends more on the block that holds the organ than on the blocks around it (0.126 against 0.085 in Shapley value), and for hippocampus and caudate the surrounding blocks matter more (RQ5).
- **In the prompt decoder.** The left and right prompts look at different parts of the image mainly in Layer 1, with a KL divergence of 0.079 and 0.063, and almost identically from Layer 2 on (RQ3).

### 3.3 Is the model biased to one side

- **With clean prompts, no.** The wrong-side Dice is 0.000 (RQ1).
- **With a conflict, yes, to the right.** The last named side wins, and right wins more often (58.6% against 36.6%). A side added only in a later sentence is followed 70.8% of the time for "right" (RQ7).
- **With no side named, slightly to the left.** The mask lands on the left organ in 10.1% and on the right organ in 4.2% of the cases, and 85.6% land on neither (RQ7).
- **Inside the decoder, the sides are not symmetric.** Left prompts are pushed to the right organ in 38.8% of the cases and right prompts to the left in 3.4% (RQ4).
- **The size of the bias depends on the organ.** Putamen, ventral DC and pallidum show almost no right bias, while thalamus, cerebral white matter and accumbens show a very strong one (RQ7).

The right bias in RQ7 and the easy push from left to right in RQ4 point in the same direction. We have not tested that they share one cause.

## 4. Plan-wise verification

The Axis III plan asks for research questions and a dataset plan, and this study keeps RQ1 to RQ4 of them. The table compares each item with what exists.

| Plan item | Status | Evidence or gap |
|---|---|---|
| Dataset plan, OASIS-1 brain MRI | Done | 16 bilateral pairs |
| Dataset plan, AMOS22 kidneys | Not done | No AMOS22 results exist, and the plan starts with AMOS22 for kidney laterality |
| Dataset plan, lesion dataset | Not done | The dataset was never selected |
| RQ1, 32 fusion channels at each decoder stage | Done | Four stages and the finest logit, with cosine, correlation and normalized L2 (RQ1 Table 1) |
| RQ1, side-by-side plots of the 32 channels | Not done | The figure shows the organ by stage mean, and not the 32 channels one by one |
| RQ1, comparison with the ground-truth masks | Done | Dice with the named and the wrong side (RQ1) |
| RQ2, text and image similarity | Done | Text cosine 0.865, image 1.00 by construction, as the plan's interpretation note expected (RQ2) |
| RQ2, text-image similarity maps | Not done | No maps in the analysis |
| RQ3, left and right embeddings before and after the prompt decoder | Partly done | Raw text from RQ2 and the layer values from RQ3, and the analysis of RQ3 reports the KL divergence |
| RQ3, comparison with a different-region control | Partly done | The data holds a reference pair, and the report does not use it |
| RQ3, PCA of the prompt embeddings | Not done | Not in the analysis |
| RQ3, link to segmentation accuracy | Partly done | Figures 2 to 5 of RQ3 show the predicted masks next to the KL values of one subject |
| RQ4, full-state patching as a control and update-only patching as the main test | Done | Both done for six layers, and for Layers 3 to 5 and Layers 0 to 5 together (RQ4) |
| RQ4, same-feature patching control | Not found | Not seen in the analysis |
| RQ4, wrong-side rates and examples | Done | Tables 1 and 2 and Figures 1 to 5 (RQ4) |

## 5. The research questions

### RQ1. Do the 32 text-image fusion channels distinguish left and right?

**Specific research question.** When the same scan is given to the model with the prompts "left X" and "right X", do the 32 fusion channels at each decoder stage differ, and do they differ more at some stages than at others?

**What it does.** The model is run twice on each scan, and only the side word differs. The 32 fusion channels are taken at four decoder stages and at the finest logit, and the two prompts are compared with cosine similarity, Pearson correlation and normalized L2 difference. The predicted masks are scored against the true organ and against the mirrored organ.

**Hypothesis.** The channels of the left and right prompts are more different at some stages than at others. Nearly identical channels would mean weak prompt sensitivity at that stage, and would not prove that the text has no side information.

**Expected outcome.** A list of the stages and the channels that respond most to the side word, and the organs for which the response is weakest.

**Methodology.** Neurite-OASIS brain MRI, 16 bilateral organ pairs, one forward pass per prompt and scan, and the 32 channels read at each stage.

**Table 1. Left-versus-right similarity of the 32 fusion channels at each U-Net decoder stage, reported as mean ± SD over subjects and organ pairs.**

| Metric | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Finest logit |
|---|---|---|---|---|---|
| Cosine similarity, average over the 32 channels of each stage | 0.695 ± 0.078 | 0.848 ± 0.073 | 0.917 ± 0.045 | 0.979 ± 0.009 | 0.979 ± 0.012 |
| Cosine similarity, lowest of the 32 channels of each stage | 0.160 ± 0.232 | 0.505 ± 0.230 | 0.724 ± 0.182 | 0.820 ± 0.129 | 0.979 ± 0.012 |
| Pearson correlation, average over channels | 0.711 ± 0.068 | 0.671 ± 0.124 | 0.681 ± 0.105 | 0.829 ± 0.061 | 0.652 ± 0.169 |
| Normalized L2 difference, average over channels | 0.780 ± 0.084 | 0.519 ± 0.135 | 0.409 ± 0.116 | 0.234 ± 0.133 | 0.242 ± 0.090 |

**Figure 1. Left-versus-right cosine similarity of the fusion channels for each organ at each U-Net decoder stage.**

**Analysis.** The comparison is made for each organ and for each stage, and the mean over the 32 channels is reported together with the lowest channel, so that a strong effect on a few channels is not hidden by the mean.

**Results analysis.** The cosine similarity of the two prompts rises from 0.695 at Stage 0 to 0.848, 0.917 and 0.979 at Stages 1 to 3 and stays at 0.979 at the finest logit, and the normalized L2 difference falls from 0.780 to 0.242. The pattern holds for all 16 organs, at Stage 0 they range from 0.50 to 0.77 and by Stage 3 all are at 0.95 or above. Cerebral white matter keeps its difference longest (0.80 at Stage 2), and choroid plexus (0.50) and vessel (0.54) are the lowest at Stage 0. The lowest of the 32 channels is 0.160 at Stage 0, far below the mean. The cosine similarity may be too high at the later stages because a value shared by the whole volume is not removed, which the lower Pearson values suggest, and the fix is to recompute the similarity with the mean of each map removed. The Dice with the named side is 0.572 ± 0.349 and the Dice with the wrong side is 0.000 ± 0.001, the prediction is empty in 14.8% of the rows, and 71.0% of the predictions have a Dice of 0.5 or more.

> **Key finding.** The side word changes the fusion channels mostly at the first stage, and the later stages look nearly identical in cosine similarity, although this may be caused by a shared background value, since the Pearson correlation at the finest logit is only 0.652. The model almost never segments the wrong side, and its main failure on clean prompts is an empty or very small mask for four organs.

**Conclusion.** The fusion channels carry a left and right difference only at the early stages, for all 16 organs.

**Future work and methodology.** Repeat the analysis on AMOS22 kidneys, as the plan asks. Add the side-by-side plots of single channels for strong and weak cases. Identify the few channels with the lowest cosine and test them with channel ablation. Relate the channel difference of a case to its Dice.

### RQ2. Do text and image representations encode different information about laterality?

**Specific research question.** Is the left and right difference present in the raw text embedding, in the image features, and in the decoder output, and how does the difference change between them?

**What it does.** The raw Qwen3 text embeddings of "left X" and "right X" are compared with cosine similarity and L2 distance. The image features of the same scan are compared between the two prompts. The decoder output at Stage 0 and at the finest logit is compared for the two prompts.

**Hypothesis.** If the text embedding differs and the image features do not, the side must come from the text. If the decoder output is more similar than the raw text, the decoder removes part of the difference.

**Expected outcome.** A decision on whether the laterality information is missing from the text, weakly aligned during fusion, or lost later in the decoder.

**Methodology.** The same 16 organ pairs as RQ1. The image encoder never receives the prompt, so the image features are identical by construction, and this was checked for every subject.

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

**Figure 1. Left-versus-right cosine similarity of the raw text and of the finest logit of the U-Net decoder for each organ.**

**Analysis.** The comparison is made at three points, which are the raw text, Stage 0 and the finest logit, and for each organ separately.

**Results analysis.** The cosine similarity of the raw text is 0.865 on average, from 0.759 for vessel to 0.920 for ventral DC, and the L2 distance averages 70.6. The image features have a similarity of exactly 1.00. At Stage 0 the mean similarity is 0.695, lower than the raw text for all 16 organs, and at the finest logit it is 0.979, higher than the raw text for all 16 organs, an increase of 0.114. The increase is largest for vessel (+0.202) and thalamus (+0.140), and smallest for lateral ventricle (+0.067).

> **Key finding.** The side information is in the text and only in the text. The decoder first increases the difference at Stage 0 and then removes most of it before the output.

**Conclusion.** Text and image encode laterality differently, since the text differs and the image is the same for both prompts.

**Future work and methodology.** Add the text-image similarity maps of the plan for left and right prompts and compare them with the true organ. Test whether a weaker text difference for an organ goes with a lower Dice. Repeat with other text encoders, since the paper reports that its choice of encoder saturates below a Dice of 63.

### RQ3. Does the prompt decoder preserve or suppress laterality information?

**Specific research question.** How different are the attention patterns of the "left X" and "right X" prompts in each of the six prompt decoder layers?

**What it does.** In each of the six layers, the KL divergence between the attention distribution of the left prompt and that of the right prompt is computed in both directions. One further figure set shows the organ image, the true organs, the predicted masks and the KL values of one subject for every organ pair.

**Hypothesis.** The KL divergence is largest in the first layer, where the side word has its strongest effect, and falls in later layers. A small rise near the end would mean that a difference between the prompts comes back.

**Expected outcome.** The layer or layers where the two prompts look at the image in the most different way.

**Methodology.** Neurite-OASIS, 16 bilateral pairs. The left attention is p and the right attention is q. "Left from right" is the sum of p times log(p over q) and "right from left" is the same with p and q swapped, so each direction counts mostly the parts where its own prompt looks.

**Table 1. Attention KL divergence between the left and right prompts at each layer of the prompt decoder, in both directions (mean ± SD and median over subjects and organs).**

| Attention KL divergence | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Layer 6 |
|---|---|---|---|---|---|---|
| Left from right, mean ± SD | 0.079 ± 0.033 | 0.018 ± 0.006 | 0.012 ± 0.003 | 0.013 ± 0.007 | 0.021 ± 0.009 | 0.015 ± 0.025 |
| Right from left, mean ± SD | 0.063 ± 0.027 | 0.018 ± 0.006 | 0.012 ± 0.004 | 0.013 ± 0.007 | 0.023 ± 0.011 | 0.018 ± 0.039 |
| Left from right, median | 0.071 | 0.017 | 0.012 | 0.011 | 0.021 | 0.008 |
| Right from left, median | 0.060 | 0.017 | 0.012 | 0.011 | 0.022 | 0.008 |

**Figure 1. Attention KL divergence between the left and right prompts across the six decoder layers, in both directions.**

**Figure 2. Attention KL divergence in the six decoder layers for the inferior lateral ventricle, lateral ventricle, vessel and choroid plexus.**

**Analysis.** The mean, standard deviation and median over subjects and organs are given for each layer. The organ figures show the same values for one subject, with the scan, the true organs and the predicted masks next to them.

**Results analysis.** The KL divergence is 0.079 (left from right) and 0.063 (right from left) at Layer 1, falls to 0.018 at Layer 2 and reaches 0.012 at Layer 3, rises to 0.021 and 0.023 at Layer 5, and falls to 0.015 and 0.018 at Layer 6. The median at Layer 6 is only 0.008, so a few large values raise its mean. From Layer 2 on the two directions are within 0.004 of each other. In the single subject, left from right is higher than right from left at Layer 1 for all 16 pairs, and the choroid plexus rises again at Layer 6 to 0.093 and 0.135. Pairs with no predicted mask have a Layer 1 divergence at least as high as pairs with a good mask.

> **Key finding.** The left and right prompts look at different parts of the image mainly in Layer 1, and from Layer 2 they look at almost the same parts, whether or not the predicted mask is good.

**Conclusion.** The attention of the two prompts differs mainly in the first layer, and the side word has little influence on where the model looks in later layers.

**Future work and methodology.** Add the cosine and L2 distance before and after the decoder, the PCA of the prompt embeddings, and the different-organ control pair that the plan asks for. Relate the KL divergence of a case to its Dice over all subjects. Test Layer 1 by patching, to see whether it is the layer that carries the side.

### RQ4. Which decoder stages causally influence laterality?

**Specific research question.** Does replacing the update of one decoder layer, with the update from the opposite-side prompt, move the predicted mask to the other side?

**What it does.** For each layer, the update of the left prompt is replaced with that of the right prompt (and the reverse). The same is done for Layers 3 to 5 together, for all six layers together, and for the whole state as a positive control.

**Hypothesis.** One or a few layers carry the side, so that replacing them moves the mask. If no single layer moves the mask, the side is spread over the layers or held in a path that the patch does not touch.

**Expected outcome.** A ranking of the layers by their causal effect on the side of the mask, with wrong-side rates and Dice changes.

**Methodology.** Neurite-OASIS, 16 pairs. Only prompts for which the unpatched model already lands on the prompted side are used. Update-only patching is the main test, and full-state patching only confirms that the method works, since it must reproduce the other prompt's output.

**Table 1. Effect of replacing the update of each prompt decoder layer, of several layers together, and of the whole state, over the prompts where the unpatched prediction is on the prompted side.**

| Layers whose update is replaced | Mask moves to the other side | Mask stays on the prompted side | Mask on both sides, on neither side or empty | Dice of the prompted side | Dice of the other side |
|---|---|---|---|---|---|
| None (unpatched) | 0.0% | 100.0% | 0.0% | 0.780 ± 0.114 | 0.000 ± 0.001 |
| Layer 0 only | 0.0% | 99.5% | 0.5% | 0.771 ± 0.115 | 0.002 ± 0.006 |
| Layer 1 only | 0.0% | 99.7% | 0.3% | 0.780 ± 0.113 | 0.001 ± 0.005 |
| Layer 2 only | 0.0% | 99.7% | 0.3% | 0.778 ± 0.113 | 0.001 ± 0.005 |
| Layer 3 only | 0.0% | 99.7% | 0.3% | 0.777 ± 0.115 | 0.001 ± 0.005 |
| Layer 4 only | 0.0% | 99.7% | 0.3% | 0.779 ± 0.115 | 0.001 ± 0.004 |
| Layer 5 only | 0.0% | 99.7% | 0.3% | 0.779 ± 0.114 | 0.001 ± 0.004 |
| Layers 3–5 together | 0.0% | 89.8% | 10.2% | 0.680 ± 0.181 | 0.012 ± 0.033 |
| Layers 0–5 together | 20.9% | 9.6% | 69.6% | 0.165 ± 0.213 | 0.262 ± 0.275 |
| Whole state at Layer 0 | 98.7% | 0.0% | 1.3% | 0.000 ± 0.001 | 0.774 ± 0.131 |
| Whole state at Layer 5 | 98.7% | 0.0% | 1.3% | 0.000 ± 0.001 | 0.774 ± 0.131 |

**Figure 1. Where the predicted mask lies for the hippocampus of one example scan when the update of some decoder layers is replaced with the update of the other prompt.**

**Analysis.** For each test the share of masks that move to the other side, stay on the prompted side, or land on both sides, on neither side or empty is reported with the Dice of both sides, and the share of the mask on each organ is given for left and for right prompts.

**Results analysis.** Replacing one layer moves no mask (0.0% at all six layers) and at least 99.5% stay on the prompted side. Layers 3 to 5 together lower the Dice from 0.780 to 0.680 and move none (10.2% are no longer on one side). All six layers together move 20.9% of the masks, leave 9.6% on the prompted side and put 69.6% on both sides, on neither or empty. Left prompts move far more easily, with 38.8% moving to the right side against 3.4% of right prompts moving to the left. Replacing the whole state moves 98.7%, which confirms the method.

> **Key finding.** No single layer decides the side, and the side is carried by all layers together. The two sides are not symmetric, since left prompts are pushed to the right organ far more easily than right prompts are pushed to the left.

**Conclusion.** The side is spread over the layers of the prompt decoder and not held by one layer.

**Future work and methodology.** Add the same-feature patching control of the plan. Patch pairs of layers to find the smallest set that moves the mask. Patch attention heads and the text embedding. Test whether the left to right asymmetry also appears on AMOS22 kidneys.

### RQ5. Does the segmentation depend more on the block that contains the organ than on the surrounding blocks?

**Specific research question.** For a given organ, does the Dice depend more on the image block that holds most of the organ than on the 7 surrounding blocks?

**What it does.** A box around the organ is cut into 2 by 2 by 2 blocks, blocks are replaced with noise, and the Shapley value of each block is estimated from the change in Dice.

**Hypothesis.** If the model finds an organ by looking at the organ, the Shapley value of the organ block is higher than the mean of the 7 surrounding blocks. If it also relies on the surrounding tissue, the difference is small or negative.

**Expected outcome.** One difference for each organ, and the share of cases where the organ block is higher.

**Methodology.** Neurite-OASIS. The Shapley values are Monte Carlo estimates, and not exact values from all 256 combinations.

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

**Figure 2. Shapley value of the organ block compared with the surrounding blocks for each organ.**

**Analysis.** The values are averaged over subjects and over the left and right organ, and a second table groups the blocks by the share of the organ inside them.

**Results analysis.** Over the 12 organs with a Shapley value, the organ block has a mean of 0.126 against 0.085 for the surrounding blocks (+0.041), and it is higher in 63% of the cases and for 10 of the 12 organs. The difference is largest for cerebellum white matter (+0.148, 98%), pallidum (+0.092) and amygdala (+0.087), and negative for hippocampus (−0.034, 18%) and caudate (−0.011, 32%). Four organs, cerebral cortex, choroid plexus, vessel and lateral ventricle, have a sum of Shapley values of about 0, so the hypothesis cannot be tested for them. A block with more of the organ has a higher value up to a limit, but blocks with no organ still have a positive value on average (0.035). The estimates are noisy, and the sum of the values for pallidum is 1.018, above the maximum Dice of 1.

> **Key finding.** The model relies more on the organ block, but it also relies on the tissue around it, and for hippocampus and caudate it relies more on the surroundings.

**Conclusion.** The hypothesis is supported overall and is strongly supported for cerebellum white matter, and it is not supported for hippocampus and caudate.

**Future work and methodology.** Use finer blocks than 2 by 2 by 2, and compute exact Shapley values for more organs. Check which surrounding structures the model uses for hippocampus and caudate. Test whether a model trained on more diverse data relies less on the surroundings.

### RQ6. Which word of the prompt carries the credit, the organ word, the position word, the other words or the image?

**Specific research question.** How is the Dice score of a prompt divided between the image, the organ word, the position word and the other words of the sentence?

**What it does.** The four parts are switched on and off in all 16 combinations, the image by replacing the organ region with noise, the organ word by "structure", the position word by dropping it, and the other words by cutting the sentence down to its content words. The exact Shapley value of each part is computed on the Dice score.

**Hypothesis.** The organ word carries most of the credit, the position word carries a smaller credit that helps only once the organ is named, and the other words carry almost none.

**Expected outcome.** The credit of each part for every organ, and the interactions between the parts.

**Methodology.** Neurite-OASIS, 16 pairs. The four credits add up to the Dice of the full prompt.

**Table 1. Shapley credit of each player in Dice units over all subject-organ cases.**

| Player | Credit, mean ± SD | Credit, median | Share of the total credit | Cases with positive credit | Cases with negative credit |
|---|---|---|---|---|---|
| Image | 0.118 ± 0.118 | 0.079 | 20.3% | 84% | 3% |
| Organ word | 0.422 ± 0.198 | 0.456 | 72.9% | 88% | 0% |
| Position word | 0.105 ± 0.065 | 0.112 | 18.1% | 86% | 1% |
| Other words | −0.066 ± 0.114 | −0.024 | −11.4% | 13% | 74% |
| Total credit | 0.579 ± 0.352 | 0.755 | 100.0% | 83% | 0% |

**Table 2. Shapley credit of each player for each organ in Dice units, mean over the subjects and over the left and right organ, sorted by total credit.**

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

**Figure 1. Shapley credit of the image, the organ word, the position word and the other words for each organ.**

**Analysis.** Credits are given over all cases, for each organ, for pairs of parts, and for each subject, and the figures show the credit on the words of the prompt, with the organ image, the true organ and the predicted mask for one subject.

**Results analysis.** The organ word gets 0.422 Dice units (72.9% of the total of 0.579), the image 0.118 (20.3%), the position word 0.105 (18.1%), and the other words −0.066 (−11.4%), negative in 74% of the cases. The image and the organ word help each other (interaction 0.230), and so do the organ word and the position word (0.217), while the organ word and the other words get in each other's way (−0.122). The image credit is 0.361 for cerebral white matter and 0.344 for cerebellum white matter, and 0.009 for pallidum and 0.041 for hippocampus. For cerebral cortex and lateral ventricle the other words have a credit of −0.340 and −0.360, so their total credit is 0.014 and 0.003. Choroid plexus and vessel get 0.000 for every part. The subjects behave alike, with a standard deviation of 0.032 in total credit over subjects, and the organs differ much more than the subjects do, with the largest spread for caudate, amygdala and inferior lateral ventricle.

> **Key finding.** The organ word carries most of the credit and the sentence words slightly hurt. The image matters little for pallidum and hippocampus, so the model may be driven by language for some organs.

**Conclusion.** The organ word carries most of the credit, the image and the position word each add credit together with it, and the balance depends on the organ and not on the subject.

**Future work and methodology.** Split the other words into single words, which needs 128 prompt variants for each organ instead of 16, and shows which word of the sentence hurts. Test other prompt templates, including the exact template of the paper. Test whether the organ word credit is lower for organs with a high image credit.

### RQ7. Which side does the model choose when the prompt is confusing?

**Specific research question.** When a prompt contradicts itself or hides the side in an extra sentence, does the mask go to the left or the right organ, and is the choice biased?

**What it does.** Four prompt types are tested, which are "left then right", "right then left", no side named with a distractor sentence, and a side named only in an added sentence. The side of the predicted mask is recorded.

**Hypothesis.** Without a bias the masks would split about evenly between left and right, whichever side is named last.

**Expected outcome.** The share of masks on the left organ, on the right organ and on neither, for each prompt type and organ.

**Methodology.** Neurite-OASIS, 16 pairs.

**Table 1. Where the predicted mask lands for each prompt type, over all subject-organ cases.** Read each row as one prompt type. The three share columns add up to 100%. A positive lean means the masks go more often to the right organ, and a negative lean means they go more often to the left organ.

| Prompt type | Example prompt (X is the organ) | Mask on the LEFT organ, % of cases (95% interval) | Mask on the RIGHT organ, % of cases (95% interval) | Mask on neither side, % of cases | Lean, right minus left (percentage points) | Leans to |
|---|---|---|---|---|---|---|
| Left named first, then right | Segment the left X in this image. Ignore that, segment the right X instead. | 0.0% (0.0 to 0.0) | 58.6% (56.1 to 61.0) | 41.4% | +58.6 | Right |
| Right named first, then left | Segment the right X in this image. Ignore that, segment the left X instead. | 36.6% (34.5 to 38.6) | 0.0% (0.0 to 0.0) | 63.4% | -36.6 | Left |
| No side named, distractor sentence | Segment the X in this image. This scan was acquired under standard protocol. | 10.1% (8.4 to 11.8) | 4.2% (3.1 to 5.4) | 85.6% | -5.9 | Left |
| Side named only in an added sentence | Segment the X in this image. Actually, make it the right one. | 0.0% (0.0 to 0.0) | 70.8% (69.4 to 72.1) | 29.2% | +70.8 | Right |

**Figure 2. Share of masks on the left organ and on the right organ for each prompt.**

**Analysis.** The shares have 95% intervals from resampling the subjects, and a second table shows the result for each organ.

**Results analysis.** When the last named side is right, 58.6% of the masks land on the right organ, and when it is left, 36.6% land on the left organ, and the intervals do not overlap. The side named first is never kept (0.0%). "Actually, make it the right one" is followed in 70.8% with none on the left. With no side named, 10.1% land on the left organ and 4.2% on the right, and 85.6% land on neither. The right bias is strongest for thalamus, cerebral white matter, accumbens, hippocampus and amygdala, and absent for putamen, ventral DC and pallidum. The left bias with no side named comes mainly from accumbens (86% of the subjects), ventral DC (34%) and putamen (22%). Four organs show no mask on either side.

> **Key finding.** The model is biased to the right when the prompt contradicts itself, and the first named side is always dropped.

**Conclusion.** The prompt is biased to the right under contradiction and slightly to the left when no side is named, and the size of the bias depends on the organ.

**Future work and methodology.** Test more contradiction styles and longer clinical sentences, since the paper claims robustness to clinical language. Test whether the bias is the same on AMOS22 kidneys and in the lesion data. Link the bias to the asymmetry of RQ4 by patching the layers in the conflicting prompts.

## 6. Future work, in order of priority

1. **Repeat the main questions on AMOS22 kidneys**, as the plan asks, to see whether the findings hold outside the brain.
2. **Explain the four organs with a Dice of about 0.** Test the size of the organ, the training coverage of the concept, and the prompt template.
3. **Explain the right bias.** Connect the asymmetry of RQ4 with the bias of RQ7, and test whether the paper's choice to leave left-right mirroring out of the training plays a role.
4. **Compare prompt templates with the exact template of the paper**, since the sentence words are not neutral in RQ6.
5. **Report empty-mask rates and side errors together with Dice**, since the paper reports neither.
