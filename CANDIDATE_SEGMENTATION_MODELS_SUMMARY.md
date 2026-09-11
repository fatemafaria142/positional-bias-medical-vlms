# Candidate Text-Prompted Segmentation Models — Summary & Verdicts

This tracks the models actually surveyed as candidates for this project's
multi-region segmentation work (cardiac MRI / abdominal MRI / dermoscopy
attribute data), scored against three things at once: what modalities each
model actually covers, whether it can be run at all (checkpoint availability,
VRAM budget), and whether it's single- or multi-region per call (see
`MULTI_REGION_MODEL_CAPABILITY_TABLE.md` for the general single- vs.
multi-region distinction this column relies on).

## Summary tables

Split by the `Single- or Multi-Region per Call?` distinction (see
`MULTI_REGION_MODEL_CAPABILITY_TABLE.md` for what that distinction means in
general). Models that aren't segmenters of real images at all (data-generation
pipelines, training frameworks) or whose region-per-call behavior couldn't be
verified are listed separately at the end.

### Multi-region models (one call, multiple masks back)

| Model | Type | Params / VRAM | Modalities Covered | Task Type | Single- or Multi-Region per Call? | Verdict |
|---|---|---|---|---|---|---|
| **BiomedParse** | X-Decoder/SEEM-family foundation model | Not stated (~small, X-Decoder-sized) | 9 modalities incl. MRI, dermoscopy, CT, ultrasound, X-ray, endoscopy, pathology, fundus, OCT | Text -> mask | Multi-region -- takes a list of text prompts in one call (e.g. `['neoplastic cells', 'inflammatory cells']`), returns one mask per prompt | Best match found -- near-exact vocabulary for cardiac MRI + abdominal MRI, weak (whole-lesion only) for dermoscopy attribute data |
| **SAT** | Knowledge-enhanced universal segmenter | SAT-Pro needs 34-62GB VRAM | MR, CT, PET only (3D), 497 classes, 8 body regions | Text -> 3D mask | True multi-region -- label input is a list of class names, `--max_queries` sized to the whole class list, all segmented in a single inference pass | No dermoscopy; 3D-volume-only; VRAM far beyond free Colab T4 |
| **RadGrounder** | PaliGemma-2 (3B) + mask decoder | 3B | CT, MR only (no dermoscopy) | Text -> box or Text -> mask | Multi-region -- detection variant emits one structured token sequence per mentioned finding, with class-wise merging for repeated organs | Weights not downloadable at all; no dermoscopy training |
| **CLIPSeg** | General-purpose CLIP decoder | ~150M | None (general images) | Text -> mask (single call) | Multi-region via batching -- a list of text prompts against the same image returns one mask per prompt in one batched call | Works technically, simplest option, zero medical-domain training |

### Single-region models (one call, one mask back)

| Model | Type | Params / VRAM | Modalities Covered | Task Type | Single- or Multi-Region per Call? | Verdict |
|---|---|---|---|---|---|---|
| **MedCLIPSeg** | CLIP + adapter, per-dataset checkpoints | UniMedCLIP ViT-B/16 based | Breast ultrasound, brain-tumor MRI, chest X-ray/CT, endoscopic ultrasound, skin lesion, GI polyp (named checkpoints only) | Text -> mask (per checkpoint) | Single-region -- one prompt in, one mask out | No cardiac-MRI or abdominal-MRI checkpoint; skin checkpoint targets the wrong task (lesion boundary, not fine-grained attributes) |
| **SegVol** | 3D SAM-style, CLIP text encoder | Not stated | CT only, 3D volumes | Point/box/text -> 3D mask | Single-region -- `cls_idx` selects one category at a time, looped for multiple organs | 3D CT-volume-only; no MRI, no dermoscopy, no 2D input |
| **LViT** | CNN + ViT + BERT fusion | Small (research-scale) | Chest X-ray (COVID), CT (COVID), histopathology nuclei | Text -> mask (fixed report-style text) | Single-region -- binary mask matching the one report/text input | No usable weights (must train from scratch); zero domain overlap; not open-vocabulary |
| **MedCLIP-SAMv2** | Fine-tuned BiomedCLIP (saliency) -> SAM | BiomedCLIP + SAM ViT-H (~2.4GB) | Breast ultrasound, brain MRI, lung X-ray, lung CT | Text -> saliency -> box/point -> SAM mask (3-stage pipeline) | Single-region -- one curated text prompt drives one saliency map drives one SAM prompt/mask | No cardiac/abdominal/dermoscopy coverage; heaviest pipeline (3 sub-repos) |
| **Florence-2** | General-purpose VLM | 0.23B / 0.77B | None (general images, not medical-trained) | Text -> box -> mask (2-step task tokens) | Single-region -- one phrase in, one box->mask out; loop once per region for multi-region rows | Works technically, but zero medical-domain training |
| **MedSAM / SAM-family** | SAM fine-tuned on medical images | Varies (ViT-B/L/H) | Broad (trained on many medical modalities) | Box/point -> mask, not text | Single-region per prompt (one box/point -> one mask); SAM supports batched boxes for multiple masks in one forward pass | Medical-specific and works, but needs a box/point prompt instead of free text -- changes what's being tested |

### Not applicable / unverified

| Model | Type | Params / VRAM | Modalities Covered | Task Type | Single- or Multi-Region per Call? | Verdict |
|---|---|---|---|---|---|---|
| **MedSegFactory** | Dual-stream diffusion (Stable Diffusion-based) | SD-scale | Generates synthetic abdominal-CT / cardiac-MRI / breast-ultrasound / colonoscopy / liver-CT / kidney-CT-style images | Image+mask generation, not segmentation of real input | N/A -- not a segmenter of real images | Wrong task type -- synthesizes fake training pairs, doesn't segment real images |
| **APRIL-MedSeg** | Modular training toolbox (130 archs) | N/A (framework) | Whatever you train it on | Training framework | N/A -- depends entirely on which of its 130 architectures you train | Not a model -- a toolkit to train your own (includes LViT/BiomedParse/CausalCLIPSeg as options) |
| **MRGen** | VAE + diffusion + ControlNet | SD-scale | MRI (abdominal organ, prostate, pancreas, liver) | Synthetic MRI generation, feeds nnU-Net | N/A -- not a segmenter of real images | Wrong task type -- data engine for training nnU-Net, not a segmenter itself |
| **CausalCLIPSeg** | CLIP + causal intervention | Unknown | Unknown | Referring segmentation (per MedCLIPSeg's baseline table) | Unknown -- could not verify | Unverified -- no confirmed public checkpoint found |

*Table 1. 14 candidate text-prompted (or text-adjacent) segmentation models
surveyed for this project's cardiac-MRI, abdominal-MRI, and dermoscopy
multi-region work, grouped by region-per-call capability, with each model's
final feasibility verdict.*

## Detailed notes per model

- **MedCLIPSeg** -- A per-checkpoint model, not one general-purpose one: each
  of its published checkpoints was fine-tuned for a specific modality (breast
  ultrasound, brain-tumor MRI, chest X-ray/CT, endoscopic ultrasound, skin
  lesion, GI polyp), and only outputs one mask per call regardless of the
  prompt. Ruled out because it has no cardiac-MRI or abdominal-MRI checkpoint
  at all, and its one dermoscopy-adjacent checkpoint (skin lesion) solves a
  different task than this project needs (a whole-lesion boundary, not the
  fine-grained attribute regions ISIC Task 2 asks for).

- **RadGrounder** -- Architecturally the most promising of the box/mask
  candidates: it's built on PaliGemma-2 (the same base model already used
  elsewhere in this project) plus a mask decoder, and its detection variant
  can emit one structured token sequence covering every mentioned finding in
  a single call, merging repeated organ names on its own. Ruled out purely on
  access -- there is no public checkpoint to download yet, and it wasn't
  trained on dermoscopy at all, so even gated access wouldn't fully cover this
  project's 3 modalities.

- **BiomedParse** -- The best match found overall. It's the model this
  project already built a runner notebook for
  (`BiomedParse_Multi_Region_Positional_Bias.ipynb`): a single image encoding
  paired with a list of text prompts returns one mask per prompt in that same
  call, and its 9-modality training set includes near-exact vocabulary for
  both cardiac MRI (ACDC-style structures) and abdominal MRI (CHAOS-style
  organs). The one soft spot is dermoscopy: BiomedParse's dermoscopy training
  is whole-lesion segmentation, not the fine-grained attribute patterns
  (pigment network, streaks, etc.) ISIC Task 2 needs -- so results there will
  likely be weaker than the MRI modalities.

- **SegVol** -- A genuinely capable 3D medical segmenter (point, box, or text
  prompts all work), but its whole design is built around 3D CT volumes with
  a `cls_idx`-based category selector -- getting multiple organs out means
  looping the same volume once per category, so it's single-region per call
  by construction. Ruled out because it's CT-only in 3D; this project's data
  is 2D MRI and dermoscopy, neither of which SegVol was built for.

- **LViT** -- A small, research-scale fusion model (CNN + ViT + text encoder)
  matched to a very specific report-style text format, not open-ended
  clinical noun phrases. Ruled out for two independent reasons: no pretrained
  checkpoint was ever released (GitHub code only, so it would need training
  from scratch), and even if trained, its 3 covered modalities (COVID chest
  X-ray/CT, histopathology nuclei) share no overlap with this project's data.

- **MedSegFactory** -- Not a segmentation model at all -- it's a diffusion
  model that *generates* synthetic image+mask training pairs (in a look that
  spans abdominal CT, cardiac MRI, breast ultrasound, colonoscopy, liver/
  kidney CT). Ruled out on task type alone: it doesn't take a real image in
  and return a mask for it, which is what this project needs.

- **APRIL-MedSeg** -- Not a model but a training framework/toolbox bundling
  130 architectures (LViT, BiomedParse, and CausalCLIPSeg among them) that
  you'd train yourself on your own data. Ruled out because there's no single
  pretrained checkpoint to just run -- using it would mean picking one of its
  130 architectures and training it from scratch, which is a different
  (much larger) undertaking than evaluating an existing model zero-shot.

- **MRGen** -- Also not a segmenter of real images: it's a VAE+diffusion+
  ControlNet pipeline that generates *synthetic* MRI images (abdominal organ,
  prostate, pancreas, liver) specifically to feed as training data into
  nnU-Net. Same disqualifying reason as MedSegFactory -- wrong task type for
  this project's need to segment real, already-collected images.

- **MedCLIP-SAMv2** -- Already implemented and run in this repo
  (`scripts/medclip_samv2/`). A 3-stage pipeline (BiomedCLIP saliency map ->
  coarse box/point -> SAM refinement) that is single-region per call: one
  curated text prompt drives one saliency map, which drives one SAM prompt
  and one mask -- documented in this repo's own P_ALL scoring note as unable
  to split a multi-organ prompt into separate per-organ masks. Ruled out for
  further multi-region work both because of that single-region ceiling and
  because none of its 4 covered modalities (breast ultrasound, brain MRI,
  lung X-ray, lung CT) match this project's cardiac/abdominal/dermoscopy set.
  It's also the heaviest pipeline surveyed -- 2 separate checkpoint downloads
  across 3 sub-repos (BiomedCLIP, k-means postprocessing, SAM ViT-H).

- **SAT** -- The only other model besides BiomedParse confirmed as *true*
  multi-region: its label input is a list of class names sized by
  `--max_queries`, and all requested classes come back segmented in one
  inference pass, covering 497 classes across 8 body regions in MR/CT/PET.
  Ruled out purely on resources -- SAT-Pro needs 34-62GB VRAM, far beyond a
  free Colab T4, and it has no dermoscopy coverage at all (3D-volume-only, no
  2D dermoscopy support).

- **CausalCLIPSeg** -- Appears only as a baseline entry in MedCLIPSeg's
  comparison table; no working public repo or checkpoint could be confirmed
  during this survey. Left as unverified rather than guessed at -- flagged
  for follow-up if a working release surfaces later, not ruled out on
  technical grounds.

- **Florence-2** -- A general-purpose (non-medical) VLM that reaches masks
  via a 2-step task-token flow (open-vocabulary detection -> region-to-
  segmentation). Single-region per call natively -- getting N regions means
  looping the same image N times, once per named object. Technically works
  out of the box (public, ungated, plain `transformers`, no license
  click-through), but has zero medical-domain training, so it's a fallback
  option rather than a first choice.

- **CLIPSeg** -- Also general-purpose and non-medical, but reaches
  multi-region output through simple batching: a list of text prompts run
  against the same image in one batched call returns one mask per prompt.
  That's a different (and weaker) kind of "multi-region" than BiomedParse's
  or SAT's shared-encoding, per-prompt-mask design, but it does mean CLIPSeg
  can produce N separate masks without looping calls one at a time. Simplest
  option surveyed to just run (smallest model, no gating), at the cost of no
  medical-domain training at all.

- **MedSAM / SAM-family** -- Medical-domain fine-tunes of SAM, so mask
  quality is solid across a broad range of modalities SAM-family models have
  been fine-tuned on. The catch: none of them take free text as a prompt --
  they need a box or point, so "does this model respond correctly to a text
  description of a region" (the actual question this project's positional-
  bias prompts are testing) can't be asked of them directly. SAM's own
  batched-box support means one forward pass can still produce several masks
  at once if you already have several boxes from somewhere else -- but that
  "somewhere else" is exactly the missing text-to-box step for this project.
