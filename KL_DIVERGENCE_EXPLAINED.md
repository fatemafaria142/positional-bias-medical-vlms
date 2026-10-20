# How KL Divergence Works (Row 18: Attention–Ground-Truth Alignment)

This explains the KL divergence computation behind `row18_attention_kl_*.csv` and the
`row18_explainer.png` / `row18_paper.png` figures (`scripts/voxtell_experiments.py::run_attention_kl`,
`scripts/make_rows17_21_figures.py`).

## What it's measuring

For a given organ (e.g. "left cerebral white matter") and a given decoder layer, KL divergence answers:
**does the model's cross-attention land on the voxels where the organ actually is?**

- **P** = the cross-attention density — the text prompt (e.g. *"Segment the left cerebral white
  matter in this image"*) is embedded and used as the query in the decoder's cross-attention,
  attending over image patch tokens. The result is a spatial weight over image voxels: how much the
  model "looked" at each location, for that layer, given that prompt.
- **Q** = the ground-truth mask — the real, annotated location of the organ, treated as a uniform
  distribution over its voxels (zero everywhere else).

Both P and Q are normalized so they sum to 1, making them proper probability distributions over the
same voxel grid before comparison.

## The steps

**1. Crop to the working region.**
The image is preprocessed and cropped to the bounding box VoxTell actually operates on
(`predictor.preprocess`). All subsequent steps happen inside this crop.

**2. Get the ground-truth distribution, Q.**
```
gt = (label_crop == organ_id)        # boolean mask, this organ's true voxels
gt_norm = gt / gt.sum()              # normalize to a probability distribution
```

**3. Get the attention distribution, P.**
Run the model with the organ's text prompt as the cross-attention query. For each decoder layer `l`:
```
d = attn[l, organ]                   # raw attention density over image voxels, this layer
d_norm = d / d.sum()                 # normalize to a probability distribution
```

**4. Compute the KL divergence between them.**
```
KL(Q || P) = sum_x  Q(x) * ( log(Q(x)) - log(P(x)) )
```
In code:
```python
def kl(p_norm, q_norm):
    return float((p_norm * (torch.log(p_norm.clamp_min(eps)) - torch.log(q_norm.clamp_min(eps)))).sum())
```
(a small `eps` floor avoids `log(0)` where either distribution is exactly zero at a voxel)

- **Low KL (near 0)** → P and Q are nearly identical → the attention is concentrated right where the
  ground-truth structure actually is → **good alignment**.
- **High KL** → P puts weight where Q has none (or misses where Q is concentrated) → the attention is
  scattered away from the true anatomical region → **poor alignment**.
- KL is always ≥ 0, with no fixed upper bound, and is **not symmetric**
  (`KL(Q‖P) ≠ KL(P‖Q)`) — it's a measure of mismatch, not a true distance.

**5. Repeat for every decoder layer.**
Steps 3–4 run once per layer, giving one KL value per layer per organ — this is the per-layer line
you see in the charts (`kl_divergence` column).

**6. (Optional) Isolate the prompt's effect — the generic-prompt-normalized variant.**
To separate "attention driven by the organ/position words in the prompt" from "wherever the model
looks by default regardless of prompt," the same process is repeated with a generic, non-spatial
prompt (*"Segment the anatomy in this image."*), giving a baseline attention density `g_norm`. The
real prompt's density is then divided by this baseline before computing KL:
```python
ratio = d_norm / g_norm.clamp_min(eps)
ratio_norm = ratio / ratio.sum().clamp_min(eps)
kl_divergence_vs_generic_normalized = kl(gt_norm, ratio_norm)
```
This is the `kl_divergence_vs_generic_normalized` column — a prompt-specific attention signal with
the model's default spatial bias divided out. The plain `kl_divergence` column (used in
`row18_explainer.png`) does **not** do this normalization — it's the raw attention-vs-ground-truth
comparison.

**7. Compare Left vs. Right.**
Steps 1–6 run independently for the left and right version of the same organ (two different text
prompts, two different ground-truth masks). The per-layer gap (`Left KL − Right KL`) is the
positional-bias signal: a consistently positive or negative gap means the model's attention aligns
with one side's true anatomy systematically better than the other's, even though both sides contain
the same type of structure.

## Why it isn't just "the overlap"

KL divergence is stricter than something like IoU/Dice overlap: it penalizes attention mass placed
*confidently* in the wrong place much more heavily than attention that's merely spread thin, because
of the `log(Q/P)` term. Two attention maps with the same overlap area can have very different KL
values depending on how concentrated the "wrong" mass is.
