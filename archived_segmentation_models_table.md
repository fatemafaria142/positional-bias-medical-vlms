# Table 1 — Text-Prompted Medical Segmentation Models (Built by Other Researchers)

This is the model table that was removed from the position-bias research plan (it lists published segmentation architectures from other researchers' papers, not usable as your own model pool for that study) — kept here as a standalone reference in case you need to cite, compare against, or re-use it later.

## Medical segmentation models (13)

| # | Model | Year | Venue | Base architecture | Text mechanism | Params | Code | License | CPU-only feasible? | Kaggle / Colab Pro |
|---|-------|------|-------|--------------------|-----------------|--------|------|---------|----------------------|----------------------|
| 1 | LViT | 2022/23 | IEEE TMI | ViT + text encoder, cross-modal fusion | Free-text clinical annotation guides pixel attention | Not stated | [repo](https://github.com/HUANGLIZI/LViT) | Not stated | ✅ Yes — small hybrid CNN+ViT, no LLM | ✅ Both — fits free Kaggle T4 easily |
| 2 | MedCLIP-SAMv2 | 2024 | MedIA | BiomedCLIP + SAM | Text → visual prompt (M2IB) → SAM mask | ~196M (BiomedCLIP) | [repo](https://github.com/HealthX-Lab/MedCLIP-SAMv2) | Not stated | ⚠️ Slow — includes SAM ViT-H (~630M); 8–16GB RAM | ✅ Both — SAM ViT-H fits 16GB Kaggle T4 |
| 3 | SimTxtSeg | 2024 | MICCAI | Text-to-visual-cue converter + SAM | Simple text cue → pseudo visual prompt for SAM | Not stated | [repo](https://github.com/xyx1024/SimTxtSeg) | Apache-2.0 | ⚠️ Slow — includes SAM | ✅ Both |
| 4 | CausalCLIPSeg | 2024 | MICCAI | CLIP + causal intervention module | Referring text prompt with confounder removal | Not stated | [repo](https://github.com/WUTCM-Lab/CausalCLIPSeg) | Not stated | ✅ Yes — CLIP-scale, no SAM/LLM | ✅ Both |
| 5 | RecLMIS | 2024 | IEEE TMI | Bidirectional visual-text reconstruction | Text and image mutually reconstruct each other for alignment | Not stated | [repo](https://github.com/ShawnHuang497/RecLMIS) | Not stated | ✅ Yes — CLIP-scale | ✅ Both |
| 6 | VLSM-Adapter | 2024 | MICCAI | Adapter on CLIPSeg/CRIS backbone | Free-text prompt through frozen VLSM + adapter | ~3M (adapter only) | [repo](https://github.com/naamiinepal/vlsm-adapter) | Not stated | ✅ Yes — smallest model in this list | ✅ Both |
| 7 | TuneVLSeg | 2024 | ACCV | Prompt-tuning wrapper on CLIPSeg/CRIS | Learnable + free-text prompt tuning | Not stated | [repo](https://github.com/naamiinepal/tunevlseg) | Not stated | ✅ Yes — CLIP-scale | ✅ Both |
| 8 | BiomedParse | 2024 | Nature Methods | MLLM-style joint seg/detect/recognize, 9 modalities | Open-vocabulary clinical noun phrase → mask | Not stated | [repo](https://github.com/microsoft/BiomedParse) | Not stated (MSR license) | ⚠️ Slow — SAM-scale (~640M) backbone; 8–16GB RAM | ✅ Both — SAM-scale fits 16GB Kaggle T4 |
| 9 | VLSM-Ensemble | 2025 | MIDL | Ensemble of CLIPSeg/CRIS/BiomedCLIPSeg | Free-text prompt, ensembled mask output | Not stated | [repo](https://github.com/juliadietlmeier/VLSM-Ensemble) | Not stated | ✅ Yes but slower — runs multiple CLIP-scale models per image | ✅ Both |
| 10 | BiPVL-Seg | 2025 | arXiv | Bidirectional progressive vision-language fusion | Text-class embeddings aligned at global-local level | Not stated | [repo](https://github.com/rafiibnsultan/bipvl-seg) | Not stated | ✅ Yes — CNN+text fusion, no LLM | ✅ Both |
| 11 | MedSAM3 (Joey-S-Liu) | 2025 | arXiv | SAM 3 fine-tuned + agentic MLLM refinement | Open-vocabulary concept text ("Promptable Concept Segmentation") | Not stated | [repo](https://github.com/Joey-S-Liu/MedSAM3) | Not stated | ❌ Not practical — MLLM agent loop, needs GPU | ⚠️ Colab Pro (A100) likely needed |
| 12 | UniBiomed | 2025/26 | Nature Communications | MLLM + SAM, 10 modalities, 27M triplets | Diagnostic text query grounds segmentation + report | Not stated | [repo](https://github.com/Luffy03/UniBiomed) | Not stated | ❌ Not practical — MLLM-scale (multi-billion params), needs GPU | ⚠️ Colab Pro (A100) likely needed |
| 13 | Medical-SAM3 (AIM-Research-Lab) | 2026 | arXiv | SAM 3 fully fine-tuned on 33 datasets, 10 modalities | Text prompt drives universal 2D+3D segmentation | Not stated | [repo](https://github.com/AIM-Research-Lab/Medical-SAM3) | Not stated | ⚠️ Slow — SAM 3-scale backbone, no LLM | ✅ Both — fits within 16GB Kaggle T4 |

## General-purpose segmentation models (2)

| # | Model | Year | Venue | Base architecture | Text mechanism | Code | License | CPU-only feasible? | Kaggle / Colab Pro |
|---|-------|------|-------|--------------------|-----------------|------|---------|----------------------|----------------------|
| 14 | CLIPSeg | 2022 | CVPR | CLIP + conv decoder | Free text or image prompt → mask | [timojl/clipseg](https://github.com/timojl/clipseg) | MIT | ✅ Yes | ✅ Both |
| 15 | CRIS | 2022 | CVPR | CLIP + text-to-pixel contrastive decoder | Referring expression → mask | [DerrickWang005/CRIS.pytorch](https://github.com/DerrickWang005/CRIS.pytorch) | MIT | ✅ Yes | ✅ Both |

**Kaggle / Colab Pro legend:** Kaggle free tier = single/dual T4 GPU, 16GB VRAM each, ~30 GPU-hrs/week quota. Colab Pro = priority access to V100/A100, more VRAM (up to 40GB on A100), longer sessions.

**Two need Colab Pro, not Kaggle free tier alone:** MedSAM3 (Joey-S-Liu) and UniBiomed — both MLLM-scale, marginal on Kaggle's single 16GB T4 without 4-bit quantization.

## Note

These are all pre-built, published mask-producing segmentation architectures, not general-purpose VLMs. They're a different model class from the 15 models in your active position-bias plan (`position_bias_research_plan.md`), which are off-the-shelf grounding/pointing VLMs you run yourself on the P0 to P4 prompts. This table is kept for reference/comparison only and isn't part of the active research plan.
