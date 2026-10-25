# RQ4: Which Decoder Stages Causally Influence Laterality Through Activation Patching?

## What this research question asks

VoxTell turns the prompt, for example "Segment the left hippocampus in this image.", into a vector and passes it through six decoder layers, numbered 0 to 5. Each layer adds its own update to this vector. We take the update of one layer for the left prompt and replace it with the update from the right prompt. Then we check whether the predicted mask moves to the other side. We repeat this for each layer alone, for several layers together and for the whole state.

**Setting.** VoxTell on the Neurite-OASIS dataset, for 16 bilateral organ pairs, each with a left and a right prompt. A flip can only be judged when the unpatched model already predicts the prompted side, so only those prompts are used. The image figures show one example subject.

## Definitions

- **Unpatched.** The model's own prediction for a prompt when nothing is replaced. It is not the ground truth, so its Dice is below 1.
- **Ground truth.** The true organ outline from the dataset labels. In the image figures it is drawn as white outlines, solid for the prompted side and dashed for the other side. Dice and the shares below compare every predicted mask with it.
- **Activation patching.** Replacing a piece of the model's internal values for one prompt with the values from the other prompt, and then checking how the output changes. Here the replaced piece is the update that a prompt decoder layer adds to the prompt's query vector.
- **Update of a layer.** The change that one prompt decoder layer makes to the query vector, calculated as the vector after the layer minus the vector before it. Each prompt has its own update at each layer.
- **Layer on and layer off.** A layer is "on" when its update for the prompt is replaced with the update that the same layer makes for the other-side prompt, and "off" when it is left unchanged. Concretely, a left prompt at an "on" layer receives its own vector before the layer plus the right prompt's update. In the main test, one layer is on and the other five are off.
- **Layers together.** The same replacement is made at several layers at once. Layers 3 to 5 are on together in one condition, and Layers 0 to 5 are on together in the other.
- **Whole state.** The entire query vector of the left prompt after one layer is replaced with the right prompt's vector after the same layer. The later layers see the same image, so this should reproduce the right prompt's output, and it only checks that the patching is done correctly.
- **Where the mask ends up.** The predicted mask is assigned to a side according to how much of each side's true organ it covers. It moves to the other side when it lands on the other side's organ, it stays on the prompted side when it lands on the prompted side's organ, and otherwise it is on both sides, on neither side or empty. The share that moves to the other side is called the flip rate.
- **Share of the predicted mask.** The part of the predicted mask that lies on the true left organ, on the true right organ, or outside both organs. These three shares add up to 100% for each mask, and they are averaged over the prompts. Empty masks have no shares and are left out of the averages.
- **Dice of the prompted side and Dice of the other side.** How well the predicted mask overlaps the true organ on the prompted side and on the other side. A value of 1 is a perfect match and 0 means no overlap. The two Dice values are separate scores and do not add up to 1.

## Tables

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

*Note.* Dice values are mean ± SD. When the whole state is replaced, the prediction agrees with the unpatched prediction of the other prompt in every case (agreement 1.000).

**Table 2. Where the predicted mask lies (share on the left organ, on the right organ and outside both organs) and the share of masks that move to the other side, for left prompts and for right prompts.**

| Layers whose update is replaced | Left prompts, mask on the left organ | Left prompts, mask on the right organ | Left prompts, mask outside both organs | Left prompts, mask moves to the other side | Right prompts, mask on the left organ | Right prompts, mask on the right organ | Right prompts, mask outside both organs | Right prompts, mask moves to the other side |
|---|---|---|---|---|---|---|---|---|
| None (unpatched) | 82.2% | 0.0% | 17.7% | 0.0% | 0.0% | 84.9% | 15.1% | 0.0% |
| Layer 0 only | 83.9% | 0.3% | 15.8% | 0.0% | 0.1% | 84.9% | 14.9% | 0.0% |
| Layer 1 only | 83.1% | 0.2% | 16.7% | 0.0% | 0.1% | 84.9% | 15.0% | 0.0% |
| Layer 2 only | 83.1% | 0.2% | 16.7% | 0.0% | 0.1% | 85.2% | 14.7% | 0.0% |
| Layer 3 only | 82.8% | 0.2% | 17.0% | 0.0% | 0.1% | 85.3% | 14.6% | 0.0% |
| Layer 4 only | 82.9% | 0.2% | 17.0% | 0.0% | 0.1% | 84.9% | 15.1% | 0.0% |
| Layer 5 only | 82.6% | 0.2% | 17.2% | 0.0% | 0.1% | 85.0% | 14.9% | 0.0% |
| Layers 3–5 together | 84.2% | 2.2% | 13.6% | 0.0% | 1.8% | 85.9% | 12.3% | 0.0% |
| Layers 0–5 together | 26.5% | 66.6% | 6.9% | 38.8% | 42.2% | 49.6% | 8.2% | 3.4% |
| Whole state at Layer 0 | 0.0% | 84.9% | 15.1% | 100.0% | 81.9% | 0.0% | 18.1% | 97.5% |
| Whole state at Layer 5 | 0.0% | 84.9% | 15.1% | 100.0% | 81.9% | 0.0% | 18.1% | 97.5% |

*Note.* The three shares of each prompt group add up to 100%. Empty masks are left out of the shares.

## Figures

**Figure 1. Where the predicted mask lies for one example, the hippocampus of one subject, when the update of some decoder layers is replaced with the update of the other prompt.** Each row is one test. The gray squares on the left show which layers were replaced, and the other layers were left unchanged. The left panel uses the prompt "Segment the left hippocampus in this image." and the right panel uses "Segment the right hippocampus in this image." Each bar shows how the predicted mask is divided and adds up to 100%, with green for the part on the left organ, red for the part on the right organ, and gray for the part outside both organs. The gray value is written to the right of the bar. The last two rows replace Layers 3 to 5 together and Layers 0 to 5 together. The same subject and organ appear in the first rows of Figure 2 and Figure 3.

**Figure 2. Predicted masks for the left prompt on two organs of one subject, when the update of decoder layers is replaced with the update of the right prompt.** The two rows are the left hippocampus and the left cerebral white matter, shown on one axial MRI slice each, and the prompts were "Segment the left hippocampus in this image." and "Segment the left cerebral white matter in this image." The first column is the ground truth, with the true left organ in green and the true right organ in red. The other columns are predictions, in green where the mask lies on the left side and in red where it lies on the right side. They show the unchanged model, then the update of each of Layers 0 to 5 replaced alone, then Layers 0 to 5 replaced together, and then the whole state replaced. The solid white outline is the true organ on the prompted side, and the dashed white outline is the true organ on the other side. The Dice and the side of the mask are written under each panel, and the side is the one that the mask covers best.

**Figure 3. Predicted masks for the right prompt on two organs of one subject, when the update of decoder layers is replaced with the update of the left prompt.** The two rows are the right hippocampus and the right cerebral white matter, and the prompts were "Segment the right hippocampus in this image." and "Segment the right cerebral white matter in this image." The panels, colors and outlines are the same as in Figure 2.

## Results

1. **One layer on its own never moves the mask to the other side.** The share that moves is 0.0% at all six layers, and at least 99.5% of the masks stay on the prompted side. The Dice of the prompted side is 0.771 to 0.780 against 0.780 unpatched, with the largest drop at Layer 0, and the Dice of the other side stays at 0.002 or below (Table 1). This means that the side is not stored in the update of any single layer, and that replacing one layer changes the accuracy only very slightly.
2. **In the example, a single layer does not change where the mask lies.** For the left prompt 91% to 92% of the mask stays on the left organ, and for the right prompt 90% to 91% stays on the right organ, as when unpatched (Figure 1). The images show the same, with the mask staying on the prompted side for every single layer (Figure 2, Figure 3). This means that replacing one layer is not enough to change the answer, even though the replaced update comes from the prompt of the opposite side.
3. **Layers 3 to 5 together lower the accuracy but still do not move the mask.** The share that moves is 0.0% and 89.8% of the masks stay on the prompted side. The Dice of the prompted side falls from 0.780 to 0.680, and 10.2% of the masks are no longer on a single side (Table 1). In the example, 93% of the mask is still on the prompted organ (Figure 1). This means that the later layers help to build a good mask but do not decide which side it is on.
4. **All six layers together split the mask instead of moving it.** Only 20.9% of the masks move to the other side, 9.6% stay on the prompted side and 69.6% are on both sides, on neither side or empty (Table 1). In the example, the left prompt's mask is 23% on the left organ and 73% on the right organ, and the right prompt's mask is 10% on the left organ and 86% on the right organ (Figure 1). On average, the left prompt has 26.5% of its mask on the left organ and 66.6% on the right organ, and the right prompt has 42.2% on the left organ and 49.6% on the right organ (Table 2). This means that the side is carried by the updates of all layers together, and that removing all of them does not turn the left prompt into a clean right prompt but mostly breaks the mask.
5. **Left prompts move much more easily than right prompts.** With all six layers replaced, 38.8% of the left prompts move to the right side, against only 3.4% of the right prompts moving to the left side (Table 2). The left prompts end mostly on the right organ (66.6% of the mask), while the right prompts stay split, with 49.6% of the mask on the right organ and 42.2% on the left organ (Table 2). This means that the two sides are not treated symmetrically. The model is much easier to push from left to right than from right to left.
6. **Replacing the whole state moves almost every mask, which confirms that the patching works.** The share that moves is 98.7% (100.0% for left prompts and 97.5% for right prompts). For left prompts 84.9% of the mask then lies on the right organ, and for right prompts 81.9% lies on the left organ (Table 1, Table 2). The images show the mask on the other side as well (Figure 2, Figure 3). This means that the other prompt's complete state carries the side, and that the failure to move the mask in the earlier tests is a real result and not a fault of the method.

## Conclusion

No single decoder layer decides the side of the prediction. Replacing the update of any one layer, or of Layers 3 to 5 together, never moves the mask to the other side. The side changes only when all six layers are replaced together, and even then the mask is mostly split between the two sides or broken, with only 20.9% of the masks moving. This happens far more for left prompts (38.8%) than for right prompts (3.4%). Replacing the whole state moves 98.7% of the masks, which confirms that the patching itself is correct. The side is therefore spread across the layers of the prompt decoder and not held by one layer.
