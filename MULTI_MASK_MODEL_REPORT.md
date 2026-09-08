# Model Survey: One Image + One Text → Multiple Masks

## Task requirement

Given a single image and a single free-text prompt that may name **more than one**
target region, produce **separate, distinguishable masks per named target** in one
model call — not one merged blob, and not N separate calls.

This report evaluates every model discussed against that requirement, split into two
groups: medical referring-segmentation models (cloned and inspected directly) and
general-domain models (assessed from public documentation). Each entry states what the
model actually outputs, why it does or doesn't meet the requirement, and the evidence
behind that verdict.

Legend: ✅ meets the requirement natively · ⚠️ meets it only as a two-stage pipeline
(grounder + SAM) · ❌ does not meet it · ❓ insufficient public information.

---

## Part 1 — Medical referring-segmentation models (cloned and inspected directly)

| Model | Output type | Multiple *distinct* targets, one call? | Verdict |
|---|---|---|---|
| MedCLIP-SAMv2 | pixel mask | No — pools whole prompt into one embedding | ❌ single merged mask only |
| CausalCLIPSeg | pixel mask | No — same pooling limitation, plus unrelated bugs | ❌ single merged mask only |
| RecLMIS | pixel mask | No — `n_labels=1`, mask hard-binarized | ❌ single merged mask only |
| LG_SDG | pixel mask | N/A — no text input at inference at all | ❌ not a referring-segmentation model |

### MedCLIP-SAMv2
- **What it does:** zero-shot pipeline — BiomedCLIP saliency → k-means postprocessing
  → SAM box prompt → one mask.
- **Why it fails the requirement:** BiomedCLIP pools the *entire* text prompt into a
  single embedding before producing saliency. A prompt naming 3 organs collapses to
  one saliency map with no per-organ signal to split apart.
- **Evidence:** this was already tried and reverted in this project's own code
  (`scripts/medclip_samv2/run_medclip_samv2.py` docstring) — forcing multiple output
  blobs via `--num-contours>1` + SAM's `--multicontour` was tested and produced only
  corner-noise artifacts, not real per-organ regions. `P_ALL` manifest rows are scored
  against the *union* of all named regions' masks for exactly this reason.
- **Verdict:** ❌ single merged mask only.

### CausalCLIPSeg
- **What it does:** CLIP (RN101, not the advertised ViT-B) backbone → FPN neck with a
  sup/inf masking split → one sigmoid mask.
- **Why it fails the requirement:** architecturally single-target — `forward()`
  returns exactly one `pred` tensor per call, regardless of how many things the input
  text names. There is no per-entity output head.
- **Other issues found while adapting it** (see `models/CausalCLIPSeg/config.py` for
  detail): the released code's hardcoded `Masker(512, 512)` shapes only work with an
  RN101 CLIP backbone (embed_dim 512), not the RN50/ViT-B the spec describes; no
  training script, loss, or pretrained checkpoint ships with the repo; and
  `FPN_AD.forward()` accepts a `state` (text) argument it never actually uses, so the
  causal-masking neck itself is not text-conditioned — only the final `Projector` is.
- **Verdict:** ❌ single merged mask only (and not usable zero-shot at all, on top of that).

### RecLMIS
- **What it does:** CLIP ViT-B/32-based, cross-modal conditioned reconstruction,
  trained with a proper loss/training loop (unlike CausalCLIPSeg).
- **Why it fails the requirement:** `Load_Dataset.py` hard-binarizes every mask
  (`mask[mask <= 0] = 0; mask[mask > 0] = 1`) and every config sets `n_labels = 1`.
  It is built around "one report describes one lesion," not "one text names several
  regions."
- **Verdict:** ❌ single merged mask only. (Better engineered than CausalCLIPSeg —
  has real training code and released checkpoints — but the same output shape limit.)

### LG_SDG
- **What it does:** plain UNet/EfficientUNet trained for single-source domain
  generalization; text embeddings are used only as a training-time contrastive
  regularizer to make the visual encoder more robust across scanners/sites.
- **Why it fails the requirement:** it doesn't take text as an inference input at
  all — `model(data, tau=tau)` is image-only. Output classes are fixed at training
  time by the dataset, not chosen by a runtime prompt.
- **Verdict:** ❌ not a referring-segmentation model — doesn't meet the task's basic
  premise (text input at inference), let alone the multi-mask requirement.

---

## Part 2 — General-domain models (assessed from public documentation, not cloned)

| Model | Output type | Multiple *distinct* targets, one call? | Verdict |
|---|---|---|---|
| **PaliGemma 2 (3B)** | pixel mask (VQ-VAE decoded) | **Yes** — `"segment {a}; {b}; {c}"` → one mask per entity | ✅ confirmed multi-mask, one call |
| CLIPSeg | pixel mask (soft) | No — one prompt, one embedding, one heatmap | ❌ single merged mask only |
| Florence-2 | mask (referring-seg task) / boxes (detection task) | No for masks; yes for boxes only | ❌ segmentation task is single-target |
| Moondream2 | points/boxes, no masks | Multiple instances of *one* category, not masks | ❌ no masks; unclear on distinct-entity naming |
| Molmo2-8B | points, no masks | Multiple instances of *one* category, not masks | ❌ no masks |
| Grounding DINO | boxes, no masks | Yes — multi-phrase → multi-box (native design) | ⚠️ needs +SAM for masks |
| OWLv2 | boxes, no masks | Yes — multi-query → multi-box | ⚠️ needs +SAM for masks |
| Kosmos-2 | boxes, no masks | Yes — grounds multiple phrases to multiple boxes | ⚠️ needs +SAM for masks |
| CogVLM-grounding | boxes, no masks | Yes — multiple referred objects → multiple boxes | ⚠️ needs +SAM for masks |
| Qwen2.5-VL-3B | boxes (structured/JSON), no masks | Yes — multiple named objects → multiple boxes | ⚠️ needs +SAM for masks |
| InternVL2.5 (1B–8B) | boxes, no masks | Partial — REC grounding, box-level | ⚠️ needs +SAM for masks |
| LocateAnything-3B | boxes/points, no masks | Yes — built-in "multi-object detection" | ⚠️ needs +SAM for masks |
| North-Micro-Vision-Instruct | unclear (general VQA/OCR/grounding) | No evidence of mask output at all | ❓ unconfirmed, likely no |

### PaliGemma 2 (3B) — ✅ meets the requirement
Its segmentation task accepts a prompt of the form
`"segment {object description}; {object description}; ..."`. Each named object gets
its own sequence of `<seg000>`–`<seg127>` tokens, decoded by a small VQ-VAE into a
pixel mask — genuinely multiple masks, correctly attributed to their respective
phrases, from one generation call. This is the only model in the whole survey
confirmed to do this natively.
- Caveat: general-domain training (not medical-specific) — mask quality on medical
  images would need to be verified per modality, same as any zero-shot use.

### CLIPSeg — ❌
One text prompt → one CLIP embedding → one soft segmentation heatmap. Same pooling
limitation as BiomedCLIP/MedCLIP-SAMv2.

### Florence-2 — ❌ for masks
`<REFERRING_EXPRESSION_SEGMENTATION>` is one-phrase-in, one-mask-out. Its
`<OPEN_VOCABULARY_DETECTION>` task *can* return multiple boxes for multiple named
objects in one call, but that's boxes, not masks — the segmentation task itself
doesn't support multi-entity naming.

### Moondream2 — ❌ (no masks)
Its "point"/"detect" skills can point at *multiple instances of one described
category* in a single query (e.g., every lesion of a kind), demonstrated in its own
docs. But: (a) it outputs points/boxes, not pixel masks, and (b) this is multi-
*instance* of one category, not confirmed to handle several differently-named targets
attributed separately within one query.

### Molmo2-8B — ❌ (no masks)
Same shape as Moondream2: can point at every instance of a described object in one
query (its counting-by-pointing behavior, documented up to ~40 instances), but again
points only, not masks, and validated mainly for repeated-instance counting of one
category rather than distinct multi-entity naming.

### Grounding DINO / OWLv2 / Kosmos-2 / CogVLM-grounding / Qwen2.5-VL-3B / InternVL2.5 / LocateAnything-3B — ⚠️ boxes/points only
All seven are phrase- or query-grounding models whose core design already handles
multiple named things in one call — that part of the requirement is met. None of them
draw pixels themselves:
- **Grounding DINO**, **OWLv2**, **Kosmos-2**, **CogVLM-grounding**: multi-phrase
  grounding → multiple boxes, natively.
- **Qwen2.5-VL-3B**: structured (JSON-style) output listing multiple boxes for
  multiple named objects.
- **InternVL2.5 (1B–8B)**: referring-expression comprehension, box-level.
- **LocateAnything-3B** (NVIDIA): explicitly built for "multi-object detection" in one
  call, per its own documentation — but its outputs are boxes/points (via Parallel Box
  Decoding), not masks.

Any of these seven, paired with **SAM** (each box/point fed in as a prompt), becomes a
working multi-mask pipeline — the "Grounded-SAM" pattern. This is a *two-stage*
solution, not a single-model one, and the grounder's domain gap (all seven are
trained on natural images, not medical scans) would need separate validation.

### North-Micro-Vision-Instruct — ❓ unconfirmed
Cohere Labs' 2.4B general VQA/OCR/document/grounding model (released 2026-08-12). Its
published capabilities (VQA, captioning, grounding, OCR, charts, documents) mention
"grounding" generically but nothing indicates pixel-mask output or multi-object
segmentation specifically. Not enough public information to confirm either way.

---

## Bottom line

- **If a single merged mask covering everything the text names is acceptable:**
  MedCLIP-SAMv2, CausalCLIPSeg, RecLMIS, and CLIPSeg all already do this (MedCLIP-SAMv2
  is the only one of these usable zero-shot with meaningful quality — the other two
  medical models need training first).
- **If genuinely separate masks per named target, from one call, are required:**
  **PaliGemma 2 (3B)** is the only model surveyed that does this natively. Every other
  path requires a two-stage grounder+SAM pipeline (Grounding DINO or LocateAnything-3B
  are the strongest first-stage candidates given their explicit multi-object design),
  with an added domain-gap risk since none of the grounders are medically trained.
- **LG_SDG** and **North-Micro-Vision-Instruct** don't fit this task's premise at all
  (no runtime text input, and no confirmed mask output, respectively).
