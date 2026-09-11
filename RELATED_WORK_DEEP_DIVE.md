# Deep Dive: Papers Directly Aligned With This Repo's Research

For each paper: link, dataset(s), models/baselines it compares against, its own methodology (input/output, how it works), its ablation study or comparison design, and its stated limitations. Papers are grouped by how they connect to this repo: the two 3D segmentation models actually run here, the closest published competitor to Text3DSAM, the paper that structurally resembles this repo's own research question, and the papers that most directly study the same phenomenon (positional/spatial bias in medical VLMs).

---

## 1. Text3DSAM

**Link:** [OpenReview (id: egbzGkOWVf)](https://openreview.net/pdf?id=egbzGkOWVf) — blocked by a Cloudflare browser check for automated fetching; details below are reconstructed from its own Hugging Face checkpoint description (`MagicXin/Text3DSAM`, already used in this repo) plus its characterization as a baseline inside the ESICA paper below. **No independent, fully-read copy of Text3DSAM's own paper was obtainable** — this is the one entry in this list with a genuine access gap; treat the ablation/limitations rows as "not directly verified" rather than absent.

- **Dataset(s) used:** [CVPR-BiomedSegFM](https://huggingface.co/datasets/junma/CVPR-BiomedSegFM) — ~200,000 3D image-mask pairs assembled from public biomedical datasets for the CVPR 2025 text-guided segmentation challenge, spanning CT, MR, PET, ultrasound, and microscopy (CT-dominant).
- **Models/baselines it compares against:** Not directly observed from Text3DSAM's own paper. When it appears as a baseline in *other* papers' tables (ESICA, VoxTell), it is compared against CAT, SAT, ESICA/ESICA-Lite, BioMedParse/BioMedParseV2, SegVol, and TotalSegmentator.
- **Methodology (as far as verified):** SAM-style promptable segmentation adapted to 3D: a whole volume plus a single plain-text class name goes in, one binary mask volume for that class comes out. Text encoder is `nlpie/tiny-clinicalbert` (a small, distilled BERT variant). ~59.30M parameters, 24.55G FLOPs — the smallest and cheapest model in every comparison table it appears in.
- **Ablation study:** Not verified — inaccessible.
- **Limitations (as far as verified):** Not stated in its own words anywhere accessible. Inferred from its performance profile across other papers' tables: consistently the lowest-accuracy method in every comparison it appears in (DSC 0.6091 on its own CVPR-BiomedSegFM benchmark; Dice as low as 0.52-26.6 on VoxTell's zero-shot cross-dataset tests), suggesting the parameter/FLOPs savings come at a real accuracy cost relative to larger competitors.
- **This repo's own result:** AMOS22 MRI mean Dice 0.1025 (liver best ~0.26); Neurite OASIS mean Dice 0.000 (0/350 nonzero) — see `TEXT3DSAM_RESULTS_ANALYSIS.md`.

---

## 2. Medical SAM3

**Link:** [arxiv.org/abs/2601.10880](https://arxiv.org/abs/2601.10880), "Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation" (Jiang, Ding, Song, et al.). Code: [github.com/AIM-Research-Lab/Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3).

- **Dataset(s) used:** 33 datasets, 76,956 images, 263,705 mask annotations, 10 modalities (radiography/X-ray/angiography ~40,160 images, ultrasound ~12,179, endoscopy/fetoscopy ~12,887, plus pathology, fundus/OCT, dermoscopy, microscopy, virtual microscopy, electron microscopy). 85/15 train/val split (~65.4k training images). Internal validation on 10 held-out splits from this pool; external validation on 7 fully unseen datasets.
- **Models/baselines it compares against:** Primarily its own base model, SAM3 (pre- vs. post-fine-tuning) — this is the paper's main comparison axis, not a multi-method leaderboard. Related work mentions MedSAM, Medical SAM Adapter, SAM-Med3D, MedSAM2, nnU-Net, UniSeg/MedUniSeg, and SAT, but **no direct numerical comparison against them is reported.**
- **Methodology:** Full-parameter fine-tuning of SAM3 (all layers updated, no LoRA/adapter) at 1008×1008 resolution. Deliberately 2D-slice-centric even for volumetric data — the authors state this "maximizes applicability across diverse workflows without constraints from inconsistent 3D geometries." Text-only prompts at training and inference (no points/boxes), forcing the model to learn spatial grounding from language alone. Combines text-driven semantic alignment with a set-prediction objective (Hungarian matching for instance discovery + semantic segmentation). Multi-class scenarios: each class is queried independently, then overlaps resolved via pixel-wise maximal confidence — exactly the `--instance`, one-organ-per-call pattern this repo uses.
- **Ablation / comparison design:** Internal validation: mean Dice 77.0% / IoU 67.3% vs. base SAM3's 54.0% / 43.3% (+23.0 pt Dice). External (unseen) validation: mean Dice 73.9% / IoU 64.4% vs. base SAM3's 11.9% / 8.0% (+62.0 pt Dice) — the larger external gain is the paper's headline finding (fine-tuning matters far more out-of-distribution). Notable per-dataset swings: PS-FH-AOP'23 fetal ultrasound 65.7%→91.6% Dice; DRIVE retinal vessels 24.8%→55.8%; GlaS'15 glands 68.9%→88.2%; CVC-ClinicDB and ETIS-LaribPolypDB polyp sets both 0.0%→87.9%/86.1% (base SAM3 fails completely pre-fine-tuning on these).
- **Stated limitations:** (1) Full-parameter fine-tuning at high resolution is computationally demanding; parameter-efficient strategies flagged as future work. (2) The 2D-slice/planar representation "may underutilize native volumetric continuity" — native 3D prompting explicitly recommended as future work. (3) Evaluation limited to atomic concept prompts; synonym-robust and compositional prompts flagged as needed for clinical deployment. (4) Broader multi-center validation and uncertainty estimation needed before deployment.
- **This repo's own result:** Run zero-shot against AMOS22 MRI and Neurite OASIS; predictions in `predictions_medsam3_<dataset>.csv`, figures under `results_medsam3_amos/`, `results_medsam3_oasis/`.

---

## 3. ESICA (the paper that benchmarks Text3DSAM directly)

**Link:** [arxiv.org/abs/2604.24876](https://arxiv.org/abs/2604.24876), "ESICA: A Scalable Framework for Text-Guided 3D Medical Image Segmentation."

- **Dataset(s) used:** CVPR-BiomedSegFM — the same benchmark as Text3DSAM. 35,792 training cases, 2,145 validation cases, across CT (24,786/906), MRI (8,975/977), 3D ultrasound (1,122/110), PET (623/69), microscopy (286/83). Aggregates 68 sub-datasets with class-text prompt pairs. Preprocessing via MONAI: isotropic resampling to 1.5×1.5×1.5mm, 96×96×96-voxel patches.
- **Models/baselines it compares against:** **Text3DSAM** (59.30M params, 24.55G FLOPs), CAT (207.29M, 831.82G), SAT (219.87M, 4.62T), and its own two variants, ESICA-Lite (130.77M, using TinyClinicalBERT) and full ESICA (252.07M, using ClinicalBERT).
- **Methodology:** 3D volume patch-encoded via DCFormerV2; text prompt encoded via BERT; multimodal fusion via a transformer with grouped-query attention and an iterative prompt encoder. Rather than decoding a mask directly, the decoder outputs dense image feature embeddings, and the final mask comes from a dot-product similarity between upscaled image features and the projected text embedding. Two-pass iterative refinement (initial prediction re-encoded as a prompt, fused again with text features for a sharper second pass). Training: two-stage curriculum — 30 epochs positive-only pretraining, then 5 epochs positive-negative fine-tuning.
- **Ablation study (Table 4, training/architecture):**

  | Configuration | DSC | NSD | F1 | DSC-TP |
  |---|---|---|---|---|
  | Full ESICA | 0.6588 | 0.6919 | 0.2162 | 0.5129 |
  | w/o negative fine-tuning | 0.5442 | 0.5472 | 0.1429 | 0.5280 |
  | w/o positive-only pretraining | 0.2123 | 0.2305 | 0.0276 | 0.0943 |
  | w/o similarity alignment mask | 0.5854 | 0.6011 | 0.1480 | 0.4685 |
  | w/o DCFormerV2 | 0.6396 | 0.6817 | 0.2012 | 0.4797 |

  Removing either training stage causes severe collapse (especially dropping positive-only pretraining); similarity alignment and DCFormerV2 both contribute measurably.

  **Ablation study (Table 5, refinement passes):**

  | Passes | DSC | NSD | F1 | DSC-TP |
  |---|---|---|---|---|
  | 1-pass | 0.4185 | 0.3728 | 0.0334 | 0.2434 |
  | 2-pass (default) | 0.6588 | 0.6919 | 0.2162 | 0.5129 |
  | 3-pass | 0.6328 | 0.6643 | 0.2104 | 0.4890 |

  Two passes is optimal; a third pass degrades performance (error accumulation).

- **Stated limitations:** (1) Instance-level detection (F1) lags voxel-level segmentation quality — identifying object *presence* is harder than delineating boundaries, attributed to limited negative sampling during fine-tuning. (2) Current cross-modal fusion may not scale to complex/compositional language prompts. (3) The decomposed convolution + transformer design may have an accuracy ceiling without more substantial architectural changes.

---

## 4. VoxTell (CVPR 2026 — closest published competitor on Text3DSAM's own benchmark)

**Link:** [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/papers/Rokuss_VoxTell_Free-Text_Promptable_Universal_3D_Medical_Image_Segmentation_CVPR_2026_paper.pdf), "VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation" (Rokuss et al.).

- **Dataset(s) used:** Training — 158 public datasets, 62K+ volumetric scans (~4TB), 1,087 anatomical/pathological concepts, across CT, multi-sequence MRI, and PET; vocabulary built via LLM-assisted semantic harmonization to 1,087 unified concepts + 9,682 rewritten label variants. Evaluation — abdominal organs (CT, zero-shot), lung & airway (CT), multiple sclerosis (MRI), brain metastases (MRI, OOD), adrenal tumors (CT, rare pathology), ReXGroundingCT (instance-level), and an in-house radiotherapy cohort (203 patients, real clinical report text as prompts).
- **Models/baselines it compares against (Table 1, zero-shot Dice):**

  | Model | Abdominal Organs | Lung & Airway | Liver Lesions | Mean Dice |
  |---|---|---|---|---|
  | TotalSegmentator | 67.64 | 85.23 | — | — |
  | BioMedParse | 9.12 | 0.00 | 41.25 | 12.91 |
  | **Text3DSAM** | 26.56 | 65.36 | 0.52 | 15.97 |
  | SegVol | 52.50 | 88.67 | 58.35 | 30.35 |
  | BioMedParseV2 | 51.78 | 62.59 | 70.37 | 30.59 |
  | SAT | 68.79 | 87.98 | 96.05 | 51.23 |
  | **VoxTell** | **72.94** | **89.65** | **97.59** | **70.85** |

  On the real-clinical-report radiotherapy cohort: VoxTell 50.2 Dice vs. SAT 0.0, BioMedParseV2 1.2, SegVol 8.1. On ReXGroundingCT: VoxTell 28.2 Dice / 67.8 HIT5% vs. SAT 13.1 Dice / 49.8 HIT5%.

- **Methodology:** UNet-style encoder-decoder with **multi-stage** vision-language fusion (not just a single late-fusion step). Text encoded with a frozen Qwen3-Embedding-4B. A transformer-based prompt decoder takes the text query and bottleneck image features to produce multi-scale "text-guidance" tensors, which are fused into every decoder stage via channel-wise dot products (image features modulated by text guidance at each resolution), plus deep supervision (loss computed at every scale, not just the final output).
- **Ablation study (Table 2, fusion strategy):**

  | Configuration | Fusion Stages | Deep Supervision | Dice |
  |---|---|---|---|
  | Mask2Former (baseline) | 1 | ✗ | 51.68 |
  | MaskFormer (SAT-style) | 1 | ✗ | 55.11 |
  | Multi-stage (3 stages) | 3 | ✗ | 60.16 |
  | Multi-stage (5 stages) | 5 | ✗ | 61.54 |
  | + Deep supervision | 5 | ✓ | 62.55 |
  | + Batch scaling (128) | 5 | ✓ | 69.43 |

  Also ablated: text encoder choice (Qwen3-Embedding-4B selected after comparison, detailed in their Appendix D, not extracted here), and prompt robustness (VoxTell stays stable across synonyms/rephrasings/spelling errors where competitors show high variability).

- **Stated limitations:** "While VoxTell represents a significant step toward broader generalization... performance on fully out-of-distribution (OOD) cases remains challenging." Explicit failure case: the Stanford Knee dataset, where the model "struggles" because the knee region is sparsely represented in training with few segmentation targets — "the model's text-guidance cannot extrapolate to entirely unfamiliar spatial or visual patterns." General finding: models "fail to correctly identify or segment truly unseen classes and perform best on concepts closely related to known categories." Future work: few-shot adaptation with minimal labeled examples.
- **Relevance to this repo:** Trained/evaluated on the same CVPR-BiomedSegFM-style benchmark family as Text3DSAM, and directly reports Text3DSAM as a baseline with real numbers (mean Dice 15.97 vs. VoxTell's 70.85) — the most apples-to-apples comparison point available for Text3DSAM found anywhere in this search.

---

## 5. MedREAL (ECCV 2026 — structurally closest to this repo's own research shape)

**Link:** [media.eventhosts.cc/.../12494.pdf](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/12494.pdf), "From Reasoning to Pixels: Grounded Medical Multimodal LLMs for VQA and Segmentation" (Gu, Pei, Mao, Wang, Ren, Yao). Code: [github.com/NUST-Machine-Intelligence-Laboratory/MedREAL](https://github.com/NUST-Machine-Intelligence-Laboratory/MedREAL).

- **Dataset(s) used:** Built its own benchmark, **MedRAVS-13K** (Medical Reasoning, Answering, and Visual Segmentation) — 13,824 expertly curated image-text-mask triplets across four modalities/datasets: BUSI (breast ultrasound), COVID-QU-Ex (chest X-ray), ISIC-2018 (dermoscopy), Kvasir-SEG (endoscopy). Each sample pairs a `[SEG]`-augmented clinical QA pair with a pixel-level mask; original train/val splits from the source datasets preserved.
- **Models/baselines it compares against (Table 2, gIoU/cIoU across all 4 datasets, overall column):**

  | Method | Category | Overall gIoU | Overall cIoU |
  |---|---|---|---|
  | OVSeg | Referring-based | 13.48 | 18.75 |
  | SAM3 | Referring-based | 5.43 | 1.39 |
  | **MedSAM3** (Liu et al. paper) | Referring-based | 70.99 | 78.28 |
  | SAM3-Agent | Agent-based | 16.21 | 4.20 |
  | MedSAM3-Agent | Agent-based | 55.71 | 74.33 |
  | LISA | Reasoning-based | 55.70 | 56.05 |
  | **MedREAL (ours)** | Reasoning-based | **68.49** | **70.47** |

  Notably, referring-based **MedSAM3 actually outperforms MedREAL on overall cIoU** (78.28 vs. 70.47) and wins outright on ISIC-2018 and Kvasir-SEG individually — MedREAL's advantage is in overall gIoU consistency and its added QA-reasoning capability, not a clean sweep. On BUSI specifically (noisy ultrasound), MedREAL's gIoU advantage over LISA is largest (69.15% vs. 41.76%).
- **Methodology:** A unified MLLM framework using Qwen3-VL-2B-Instruct as the MLLM and SAM as the segmentor. Introduces **Seg Anchored Reasoning Pooling (SARP)** to distill task-relevant semantic evidence directly from `[SEG]` tokens in the MLLM's hidden states (rather than using the whole global hidden state), and a **Reasoning-to-Visual (R2V)** fusion mechanism to inject that distilled evidence into the segmentation pipeline for mask decoding. Input: an image + a clinical question; output: both a text answer *and* a segmentation mask that must be consistent with that answer (not two independent predictions).
- **Ablation study (Table 4, isolating Box Guidance / SARP / R2V):**

  | Box Guide | SARP | R2V | gIoU | cIoU |
  |---|---|---|---|---|
  | ✗ | ✗ | ✗ | 11.35 | 19.69 |
  | ✓ | ✗ | ✗ | 45.02 | 52.84 |
  | ✗ | ✓ | ✓ | 36.24 | 37.74 |
  | ✓ | ✗ | ✓ | 23.22 | 47.12 |
  | ✓ | ✓ | ✗ | 45.50 | 46.96 |
  | ✓ | ✓ | ✓ | **68.49** | **70.47** |

  Key finding: removing all three components collapses performance to near-chance (11.35% gIoU); R2V *without* SARP actually hurts (23.22%, worse than Box Guidance alone) because "forcefully injecting unrefined, global linguistic representations directly into the visual pipeline introduces severe semantic noise" — the components are complementary, not independently additive.
- **Stated limitations:** "Currently, MedREAL operates exclusively on 2D imaging modalities and incurs computational overhead due to the autoregressive nature of the MLLM backbone." Future work: integrate 3D visual encoders into the R2V module to support volumetric data (CT/MRI), and knowledge distillation for faster inference.
- **Relevance to this repo:** Independently cites **both** "MedSAM3" papers as separate references (confirming the naming-collision finding from earlier in this conversation), and its whole framing — jointly evaluating a model's answer *and* whether its segmentation mask is spatially consistent with that answer — is structurally close to this repo's own RQ2/RQ3 (does the model's reported location match what it says, and does it match ground truth).

---

## 6. PRS-Med (CVPR Workshop 2026, CV4Clinic — closest match to "positional bias" specifically)

**Link:** [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026W/CV4Clinic2026/papers/Trinh_PRS-Med_Position_Reasoning_Segmentation_in_Medical_Imaging_CVPRW_2026_paper.pdf), "PRS-Med: Position Reasoning Segmentation in Medical Imaging" (Trinh et al.). Code: [github.com/huyquoctrinh/PRS-Med](https://github.com/huyquoctrinh/PRS-Med).

- **Dataset(s) used:** Built its own benchmark, **PosMed** — 116,000 expert-validated question-answer pairs, 38,731 images, across six modalities: lung X-ray (48.0%), lung CT (33.0%), brain MRI (7.9%), plus polyp endoscopy, skin, and breast ultrasound. 84.3% anatomy-centric, 15.7% tumor-centric. Source datasets: BUSI, two brain MRI sources, LungCT, lung X-ray sets, Kvasir-SEG/ClinicDB/CVC300/ColonDB/ETIS (polyp), ISIC (skin). Original train/test splits preserved; endoscopy generalization tested by holding out CVC-300, ETIS, and ColonDB entirely from training.
- **Models/baselines it compares against (Table 1, segmentation mDice):**

  | Method | Breast US | Brain MRI | Lung CT |
  |---|---|---|---|
  | G-Dino + SAM-Med2D | 0.515 | 0.667 | 0.540 |
  | BiomedParse | 0.783 | 0.294 | 0.516 |
  | LISA-7B | 0.299 | 0.478 | 0.478 |
  | LISA-13B | 0.705 | 0.439 | 0.656 |
  | **PRS-Med** | **0.817** | **0.803** | **0.968** |

  Position-reasoning comparison (Table 2, ROUGE/accuracy on breast): LLaVA-Med (0.33 / 75.22%), Med-MoE (0.61 / 81.63%), **PRS-Med (0.64 / 92.92%)**.
- **Methodology:** Combines LLaVA-Med (the language backbone, adapted via LoRA rank 16) with TinySAM (a lightweight TinyViT-based vision encoder producing W/16×H/16 embeddings) and a custom Prompt Mask Decoder. The decoder has a fusion module (cross-attention between image features and MLLM embeddings) and a mask-prediction module (transposed-convolution upsampling to 1024×1024). Loss combines segmentation loss (BCE + Dice) and text loss (categorical cross-entropy). Input: image + question; output: both a segmentation mask *and* a spatial-reasoning text answer.
- **Ablation study:** Vision-encoder choice (Table 3) — full TinySAM: mDice 0.884/IoU 0.834; frozen TinySAM: 0.737/0.662; full SAM-Med: 0.891/0.838 but at 292.6M params vs. TinySAM's 31.49M (near-identical accuracy for ~9x fewer parameters). LoRA target modules (Table 4) — applying LoRA to q,k,v,o projections gave the best result (mDice 0.879); restricting to q,v only dropped to 0.573. MLLM backbone (Table 5) — LLaVA-Med (medically pretrained) beat generic LLaVA-1.5/1.6 by a wide margin (0.879 vs. 0.709/0.744), attributed to domain pretraining. TinySAM initialization — without pretrained weights, mDice dropped from 0.884 to 0.674.
- **Stated limitations:** Boundary precision remains a weakness — the authors write they "observe that the boundary problems are still the limitations of PRS-Med, and we are planning to improve in the future." One reported exception where PRS-Med wins on ROUGE but loses accuracy to a competitor (HuoGPT) on Lung CT-Scan data, suggesting room for modality-specific tuning. Future work: extend beyond localization to estimate clinically meaningful attributes (lesion size, inter-structure distances).
- **Relevance to this repo:** The single closest CVPR-family match to "does a medical VLM correctly reason about *where* something is" — but as a trained model+dataset contribution, not a zero-shot bias probe of existing models the way this repo's study is designed.

---

## 7. "Your other Left!" — Vision-Language Models Fail to Identify Relative Positions in Medical Images (MICCAI 2025 — closest match to this repo's exact research question)

**Link:** [arxiv.org/abs/2508.00549](https://arxiv.org/abs/2508.00549) (Published at MICCAI 2025).

- **Dataset(s) used:** Built its own benchmark, **MIRP** (Medical Image Relative Positioning), from CT volumes in **BTCV** (Beyond the Cranial Vault) and **AMOS** (Abdominal Multi-Organ Segmentation) — **the same AMOS family this repo already uses for AMOS22 MRI**, though this paper uses AMOS's CT slices, not the MRI arm. Anatomical structures extracted via TotalSegmentator (117 structures: organs, bones, muscles, vessels) on axial PNG slices. Structure pairs selected only if they appear once, are ≥50px apart, and are at least twice the marker size. Ground truth is derived from the center-of-mass coordinates of the segmented structures — methodologically very close to this repo's own centroid-based ground truth derivation.
- **Models/baselines it compares against:** GPT-4o-2024-08-06 (closed-source, via API), Pixtral-12B-2409 (Mistral), Llama3.2-11B-Vision-Instruct (Meta), JanusPro-7B (DeepSeek-AI) — four general-purpose VLMs, no medical-specific model included.
- **Methodology:** Binary yes/no relative-position questions about two anatomical structures in a CT slice (e.g. "Is the left kidney below the stomach?"), evaluated under three conditions:
  - **RQ1 (plain image):** No markers, structure names only in the text. Every model scored ~50% — chance level.
  - **RQ2 (marked image, names given):** Structures also marked with numbers/letters/colored dots, question mentions both markers and names (e.g. "Is the left kidney (1) below the stomach (2)?"). GPT-4o and Pixtral improved slightly; JanusPro and Llama3.2 showed little to no improvement.
  - **RQ3 (marked image, names removed):** Question refers *only* to the markers, no anatomical names at all (e.g. "Is the number 1 below the number 2?"), forcing the model to rely purely on image content. GPT-4o jumped to >85% accuracy, Pixtral >75% — evidence that when anatomical names are present, these two models lean on learned anatomical priors rather than the actual image; JanusPro and Llama3.2 stayed comparatively weak even here.
  - **Rotation/flip control:** CT slices randomly rotated/flipped to separate "correct relative to the displayed image" from "correct relative to standard anatomy." GPT-4o scored 0.506 (image-view, chance) vs. 0.757 (anatomical-consistency) — strong evidence its answers track memorized anatomy, not the pixels actually shown. Pixtral showed a smaller version of the same pattern (0.508 vs. 0.574); JanusPro and Llama3.2 stayed near chance on both readings.
- **Ablation study:** A domain-free control — white images with randomly placed markers only (no anatomy at all), 100 images per marker type, same yes/no relative-position questions (e.g. "Is the number 1 above the number 2?"). Pixtral improved on dot markers; the others performed similarly to RQ3. JanusPro and especially Llama3.2 still struggled even in this simplified, anatomy-free setting — evidence their failure is a **general** spatial-reasoning weakness, not specific to medical imagery.
- **Stated limitations (from the paper's own Conclusion):** Evaluation restricted to CT images only — MRI/PET generalization assumed but not tested. Restricted to 2D slices, not full 3D volumes, "since current VLMs operate on 2D images," though the authors explicitly flag volumetric evaluation as more clinically applicable future work. Only general-purpose VLMs evaluated; medically fine-tuned VLMs left for future work.
- **Relevance to this repo:** This is the closest existing paper to this repo's actual research question found in the entire search — same style of ground-truth derivation (segmentation-mask centroid → relative position), a directly overlapping dataset family (AMOS), and the same core finding this repo's own P0-P4 experiment is designed to test (does the model's answer track the image, or does it default to a learned prior regardless of what's actually shown). It is MICCAI, not CVPR/ECCV, so it sits outside the venue scope asked for earlier, but it's the single most citation-worthy related-work item found across this whole search.

---

## 8. Localization Lens for Improving Medical Vision-Language Models (MICCAI 2025 — partially inaccessible)

**Link:** [papers.miccai.org/miccai-2025/paper/2723_paper.pdf](https://papers.miccai.org/miccai-2025/paper/2723_paper.pdf) / [Springer](https://link.springer.com/chapter/10.1007/978-3-032-05114-1_33). **Access note:** the MICCAI open-access PDF is behind a bot-check redirect loop, and the Springer version is paywalled behind institutional login — neither could be fully read. What's below is abstract-level only, gathered via search snippets, not a full-text read.

- **Dataset(s) used:** Not accessible.
- **Models/baselines it compares against:** Not accessible.
- **Methodology (abstract-level only):** Proposes "localization lens" — expert-validated representations that inject richer anatomical/positional context into a Med-VLM pipeline, at three levels: data, architecture, and alignment. Integrates pixel shuffle within the architecture to filter/refine these representations while preserving anatomical continuity, and adds a decoupled contrastive loss (DCL) alongside the standard training loss to align the localization-lens representations with textual features.
- **Ablation study:** Not accessible.
- **Stated limitations:** Not accessible.
- **Relevance to this repo:** Directly on-topic (improving a medical VLM's spatial/anatomical understanding), but I could not verify enough of it to build a real methodology/ablation/limitations picture. Flagging this gap explicitly rather than guessing at ablation numbers or limitations that weren't actually read.
