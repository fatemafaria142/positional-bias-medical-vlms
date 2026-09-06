# Candidate Models for Multi-Region Medical Segmentation — Evaluation Notes

This document records every model evaluated as a candidate first model for the
`segmentation-multi-region` study, which spans 3 data types: **cardiac MRI**,
**dermoscopy** (fine-grained skin-lesion attributes), and **abdominal-organ MRI**
— all 2D, text-prompted, "segment {X}" task, 2-4 regions per image. Each model
was checked against the actual model card / GitHub repo, not assumed from
memory. None currently work as an out-of-the-box, text-prompted, pixel-mask
segmenter across all 3 data types — see the Recommendation at the end.

## Our task's requirements (what a candidate must do)

1. Take a **2D image** (not a 3D volume) + a **free-text prompt** naming one region
   (or, for `P_ALL` rows, several).
2. Output a **pixel-level binary mask** for that region (not just a box/point,
   not just a caption/answer).
3. Work **zero-shot** across all 3 data types at once: cardiac MRI, dermoscopy,
   abdominal-organ MRI — ideally without per-data-type fine-tuning or a
   dataset-specific checkpoint.
4. Be loadable/runnable in a single Colab notebook without excessive setup.

## Summary table

| Model | Type | Params / VRAM | Access | Modalities covered | Task type | Verdict |
|---|---|---|---|---|---|---|
| [MedCLIPSeg](#1-medclipseg) | CLIP+adapter, per-dataset checkpoints | UniMedCLIP ViT-B/16 based | Public HF | Breast ultrasound, brain-tumor MRI, chest X-ray/CT, endoscopic ultrasound, skin lesion, GI polyp (named checkpoints only) | Text→mask (per checkpoint) | ❌ No cardiac-MRI/abdominal-MRI checkpoint; skin checkpoint = wrong task (lesion boundary, not fine-grained attributes) |
| [RadGrounder](#2-radgrounder) | PaliGemma-2 (3B) + mask decoder | 3B | **Private/gated**, no public access yet | CT, MR only (no dermoscopy) | Text→box or Text→mask | ❌ Weights not downloadable at all; no dermoscopy training |
| [BiomedParse](#3-biomedparse) | X-Decoder/SEEM-family foundation model | Not stated (~small, X-Decoder-sized) | Gated (license click-through) | 9 modalities incl. MRI, dermoscopy, CT, ultrasound, X-ray, endoscopy, pathology, fundus, OCT | Text→mask | ⚠️ **Best match found** — near-exact vocabulary for cardiac MRI + abdominal MRI, weak (whole-lesion only) for dermoscopy attribute data |
| [SegVol](#4-segvol) | 3D SAM-style, CLIP text encoder | Not stated | Public HF (`trust_remote_code`) | **CT only**, 3D volumes | Point/box/text→3D mask | ❌ 3D CT-volume-only; no MRI, no dermoscopy, no 2D input |
| [LViT](#5-lvit) | CNN+ViT+BERT fusion | Small (research-scale) | GitHub only, **no pretrained checkpoint released** | Chest X-ray (COVID), CT (COVID), histopathology nuclei | Text→mask (fixed report-style text) | ❌ No usable weights (must train from scratch); zero domain overlap; not open-vocabulary |
| [MedSegFactory](#6-medsegfactory) | Dual-stream diffusion (Stable Diffusion-based) | SD-scale | Public HF checkpoint | Generates **synthetic** abdominal-CT/cardiac-MRI/breast-ultrasound/colonoscopy/liver-CT/kidney-CT-**style** images | **Image+mask generation**, not segmentation of real input | ❌ Wrong task type — synthesizes fake training pairs, doesn't segment real images |
| [APRIL-MedSeg](#7-april-medseg) | Modular training toolbox (130 archs) | N/A (framework) | GitHub, no single pretrained checkpoint | Whatever you train it on | Training framework | ❌ Not a model — a toolkit to train your own (includes LViT/BiomedParse/CausalCLIPSeg as options) |
| [MRGen](#8-mrgen) | VAE + diffusion + ControlNet | SD-scale | Public HF checkpoints (per source→target pair) | MRI (abdominal organ, prostate, pancreas, liver) | **Synthetic MRI generation**, feeds nnU-Net | ❌ Wrong task type — data engine for training nnU-Net, not a segmenter itself |
| [MedCLIP-SAMv2](#9-medclip-samv2) | Fine-tuned BiomedCLIP (saliency) → SAM | BiomedCLIP + SAM ViT-H (~2.4GB) | Public GitHub, 2 separate checkpoint downloads | Breast ultrasound, brain MRI, lung X-ray, lung CT | Text→saliency→box/point→SAM mask (3-stage pipeline) | ❌ No cardiac/abdominal/dermoscopy coverage; heaviest pipeline (3 sub-repos) |
| [SAT](#10-sat) | Knowledge-enhanced universal segmenter | SAT-Pro needs **34–62GB VRAM** | Public HF | **MR, CT, PET only** (3D), 497 classes, 8 body regions | Text→3D mask | ❌ No dermoscopy; 3D-volume-only; VRAM far beyond free Colab T4 |
| [CausalCLIPSeg](#11-causalclipseg) | CLIP + causal intervention | Unknown | **Could not verify working repo** | Unknown | Referring segmentation (per MedCLIPSeg's baseline table) | ❓ Unverified — no confirmed public checkpoint found |
| [Florence-2](#12-florence-2-fallback) | General-purpose VLM | 0.23B / 0.77B | Public, ungated, plain `transformers` | None (general images, not medical-trained) | Text→box→mask (2-step task tokens) | ✅ **Works technically**, but zero medical-domain training |
| [CLIPSeg](#13-clipseg-fallback) | General-purpose CLIP decoder | ~150M | Public, ungated, plain `transformers` | None (general images) | Text→mask (single call) | ✅ **Works technically**, simplest option, zero medical-domain training |
| [MedSAM / SAM-family](#14-medsam-box-prompted-alternative) | SAM fine-tuned on medical images | Varies (ViT-B/L/H) | Public | Broad (trained on many medical modalities) | **Box/point→mask**, not text | ⚠️ Medical-specific and works, but needs a box/point prompt instead of free text — changes what's being tested |

## Detailed notes per model

### 1. MedCLIPSeg
- **What it is**: UniMedCLIP ViT-B/16-based adapter checkpoints, one trained set of weights *per named benchmark* (`BUSI`, `BTMRI`, `Covid19`, `EUS`, `ISIC`, `Kvasir`, each with `_10/_25/_50` data-efficiency variants).
- **Why it doesn't fit**: There is no cardiac-MRI or abdominal-organ-MRI checkpoint anywhere in the released set — those two data types are simply not covered. The skin-lesion checkpoint that does exist was trained for single-lesion-boundary segmentation (whole-lesion style), not the 5 fine-grained dermoscopic attributes (pigment network, streaks, milia-like cyst, negative network, globules) our dermoscopy data requires.
- **Integration cost**: Full GitHub repo clone, `INSTALL.md`/`DATASETS.md`/`RUN.md`, not a plug-and-play `transformers.AutoModel`. Total checkpoint set is 33GB.

### 2. RadGrounder
- **What it is**: PaliGemma-2 (3B) fine-tuned on RefRad2D (1.2M CT/MR report pairs), with a detection variant (stock `transformers`) and a segmentation variant (custom `GroundedGemmaForConditionalGeneration`).
- **Why it doesn't fit**: The Hugging Face weights repo is explicitly **private** ("private until the paper is accepted — `hf auth login` with an account that has access"). There is no public download path at all right now. Even disregarding that, training data is **CT and MR only** — no dermoscopy anywhere in RefRad2D, so our skin-lesion data type would be entirely unsupported even with access.
- **Integration cost**: Custom `uv`-managed environment, custom model class, only usable from inside the cloned repo.

### 3. BiomedParse ⭐ (best match found)
- **What it is**: A genuine biomedical foundation model (Nature Methods 2025) for joint segmentation/detection/recognition across 9 imaging modalities, text-prompted, X-Decoder/SEEM-family architecture (Focal-Net/DaViT image encoder + PubMedBERT text encoder).
- **Why it's the best fit**: Its own documented "Data Specification for Deployment" target list is a **near-exact match** for 2 of our 3 data types:
  - Abdominal MRI (our regions: liver, right kidney, left kidney, spleen) ↔ *MRI-abdomen*: liver, right kidney, left kidney, spleen, aorta, pancreas, stomach...
  - Cardiac MRI (our regions: RV, myocardium, LV) ↔ *MRI-heart*: left/right heart ventricle, myocardium, left heart atrium
  - Dermoscopy (our regions: 5 fine-grained lesion attributes) ↔ *Dermoscopy-skin*: only `lesion`, `melanoma` — **not** the fine-grained attribute vocabulary we need. This is the one real weak point.
- **Access**: Gated — requires visiting the HF page, agreeing to share contact info, and using an HF token to download.
- **Integration cost**: Not a plain `transformers.AutoModel` — needs the actual `github.com/microsoft/BiomedParse` repo cloned (`BaseModel`/`build_model`, config YAML, `BIOMED_CLASSES` constants) and `detectron2` as a dependency. Heavier than our previous 4 notebooks (Moondream2/OWLv2/Qwen2.5-VL/Molmo2), but simpler than SAT/SegVol's 3D pipelines.

### 4. SegVol
- **What it is**: A volumetric ("3D medical SAM") segmentation foundation model accepting point, box, and text prompts, trained on 90k unlabeled + 6k labeled **CT** volumes, 200+ anatomical categories.
- **Loading**: Notably simple — `AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True)`, no GitHub clone needed.
- **Why it doesn't fit**: Its entire pipeline (`preprocess_ct_gt`, `zoom_transform`) expects a `.nii.gz` **3D volume** + 3D label map with zoom-in/zoom-out cropping — not a single 2D image. Trained **only on CT** — no MRI, no dermoscopy at all. All 3 of our data types are 2D, and 2 of the 3 aren't even CT (our cardiac and abdominal-organ data are MRI; our skin-lesion data is dermoscopy photography, not a scan at all).

### 5. LViT
- **What it is**: "Language meets Vision Transformer" (TMI 2023) — CNN + ViT + BERT fusion, trained on QaTa-COV19 (chest X-ray + COVID report text), MosMedData+ (CT + text), MoNuSeg (histopathology nuclei).
- **Why it doesn't fit**: (1) **No pretrained checkpoint is released** — only training code + dataset download links; using it means training from scratch first. (2) Its text input is tied to **fixed clinical report-style sentences** bundled with those specific datasets, not an open "segment {X} on the left" interface. (3) **Zero domain overlap** with cardiac MRI, abdominal MRI, or dermoscopy.

### 6. MedSegFactory
- **What it is**: A dual-stream diffusion model (Stable-Diffusion-based) that **generates new synthetic (image, mask) pairs** from a text description — "Text-Guided Generation of Medical Image-Mask Pairs."
- **Why it doesn't fit**: Wrong task type entirely. It doesn't take a real existing image and segment it — it creates a brand-new fake image + mask from scratch given a prompt (interesting detail: its presets even include a cardiac-MRI preset naming right ventricle/myocardium/left ventricle, but that's for generating a *fake* cardiac MRI, not segmenting one of our real slices).

### 7. APRIL-MedSeg
- **What it is**: A massive modular research toolbox (130 architectures, 177 encoders, 45 decoders, 900+ YAML configs) for assembling and training your own segmentation models.
- **Why it doesn't fit**: Not a deployable model at all. Its "text-guided" section literally re-implements LViT, BiomedParse, CausalCLIPSeg, and MedCLIP-SAM as trainable options *inside* it — i.e., it bundles the same already-rejected models plus infrastructure to train them yourself, with no evidence of a ready pretrained checkpoint for any text-guided task.

### 8. MRGen
- **What it is**: "Segmentation Data Engine for Underrepresented MRI Modalities" (ICCV 2025) — a VAE + diffusion + ControlNet pipeline that generates synthetic MRI images/masks (e.g. translating abdominal-MRI T1 scans into synthetic T2 scans with matching masks) to train a downstream **nnU-Net**.
- **Why it doesn't fit**: Same wrong-task-type problem as MedSegFactory — the actual segmentation step in its own pipeline is done by nnU-Net trained on synthetic output, not by MRGen itself.

### 9. MedCLIP-SAMv2
- **What it is**: Fine-tuned BiomedCLIP (with a DHN-NCE loss) generates a saliency map (via M2IB) → converted into visual prompts (boxes/points) → fed into a separately-downloaded raw **SAM** checkpoint (ViT-H/L/B) to produce the final mask.
- **Why it doesn't fit**: Evaluated only on breast tumor ultrasound, brain tumor MRI, lung X-ray, lung CT — no cardiac MRI, no abdominal-organ MRI, no dermoscopy. Brain-tumor-MRI is at least the same *modality* as ACDC/CHAOS but a different anatomical region/pathology entirely.
- **Integration cost**: The heaviest pipeline reviewed — one conda env + 3 vendored sub-repos (`open_clip` fork, `segment-anything`, `nnUNet`) + 2 separate checkpoint downloads + curated (not free-form) `text_prompts.py` per dataset.

### 10. SAT
- **What it is**: "Large-Vocabulary Segmentation for Medical Images with Text Prompts" (npj Digital Medicine 2025) — knowledge-enhanced universal segmenter, 497 classes, trained on 72 public 3D datasets across MR/CT/PET, 8 body regions.
- **Why it doesn't fit**: (1) **MR/CT/PET only** — no dermoscopy/2D-photography modality at all. (2) **3D-volume-native** (NIfTI, `H,W,D` shape with orientation codes) — not 2D PNGs. (3) **SAT-Pro needs ~34–62GB VRAM for inference alone** (SAT-Nano ~24–36GB) — far beyond a free Colab T4 (16GB), independent of every other issue.

### 11. CausalCLIPSeg
- Referenced as a baseline in MedCLIPSeg's own comparison table (a MICCAI-style CLIP-adaptation method for "referring medical image segmentation" via causal language intervention). **Could not verify an actual working GitHub repository** — URL guesses failed (404 / unrelated paper). Not used as a basis for any conclusion; flagged here only so it isn't silently dropped from the record.

### 12. Florence-2 (fallback)
- **What it is**: Microsoft's general-purpose vision foundation model (FLD-5B, natural/everyday images — no medical training data at all), 0.23B (base) / 0.77B (large).
- **Fit**: Genuinely open-vocabulary via task tokens (`<CAPTION_TO_PHRASE_GROUNDING>` then `<REGION_TO_SEGMENTATION>`) — will attempt to ground/segment any phrase on any of our 3 datasets, with no domain-vocabulary mismatch risk, but also **no guarantee it understands what these medical terms actually look like**. Fully public, ungated, plain `transformers`. Already the mask-tier model in the original `position_bias_research_plan.md`.

### 13. CLIPSeg (fallback)
- **What it is**: `CIDAS/clipseg-rd64-refined`, ~150M params, single-call open-vocabulary text→mask segmenter built on CLIP.
- **Fit**: Same general-purpose caveat as Florence-2 (no medical training), but the simplest possible integration — one processor + one model call, no two-step grounding.

### 14. MedSAM / box-prompted alternative
- **What it is**: SAM fine-tuned on medical images across many modalities — genuinely medical-specific and works well, but **takes box/point prompts, not free text**.
- **Fit**: Would require deriving a box from the ground-truth region's approximate location to use as the "prompt" instead of a text description — this changes what the study is actually testing (spatial prompt-following vs. text-comprehension), so it doesn't fit the P0–P4 "segment {X} on the left side" framing at all.

## Recommendation

No model checked is a clean, all-domain fit. In order of preference:

1. **BiomedParse** — best real domain match (near-exact vocabulary for cardiac MRI + abdominal MRI), at the cost of gated access and a heavier (but still single-GPU, single-notebook) integration than our previous 4 models. Dermoscopy results should be reported with the caveat that only whole-lesion vocabulary is in its trained/recommended target list, not the fine-grained attributes.
2. **Florence-2 / CLIPSeg** — trivially easy, fully public, works technically on all 3 data types, but zero medical-domain training (consistent with how every other model in this whole project — Moondream2, OWLv2, Qwen2.5-VL, Molmo2 — has been used: general VLMs, zero-shot, on medical images).
3. **MedSAM (box-prompted)** — medical-specific and reliable, but requires changing the experiment design from text-prompted to box-prompted, which measures a different thing than positional-wording bias.
