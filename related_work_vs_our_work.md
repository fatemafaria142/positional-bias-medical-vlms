# Related work vs. our work (medical domain)

Title of our work: *Disentangling the Relative Contributions of Language and Region to Positional Hallucinations in VLMs*

**How to read this table.** Key: ✅ yes · ◐ partly · ❌ no · ? not confirmed.
Entries for other papers come from search summaries and the repo's `RELATED_WORK_*` documents, **not from reading the full papers**. Every `?` must be checked in the paper before you cite or claim novelty. Our row comes from the repo branches.

## Table 1: Task and setting

| Work | Venue | Task | Output | Data | Models studied |
|---|---|---|---|---|---|
| **Our work** | (this study) | Text-prompted localization and segmentation; positional prompts | Box, point or mask (2D); mask (3D) | 2D: LGG, Kvasir, ISIC. 3D: AMOS22 MRI, Neurite-OASIS | 11 2D VLMs; VoxTell, Text3DSAM, MedSAM3 |
| [Your other Left!](https://arxiv.org/html/2508.00549) | MICCAI 2025 | Relative position between two structures | Text answer | 2D / CT slices (MIRP benchmark) | GPT-4o, Llama3.2, Pixtral, JanusPro |
| [HEAL-MedVQA](https://arxiv.org/html/2505.00744) | IJCAI 2025 | Grounded medical VQA | Text answer | 2D, 67K VQA pairs with doctor-annotated masks | Medical LMMs (?) |
| [MedVIGIL](https://arxiv.org/html/2605.07919v2) | arXiv | Medical VLMs under broken visual evidence | Text answer | ? | Medical VLMs (?) |
| [Diagnostic Robustness under Visual and Textual Perturbations](https://arxiv.org/pdf/2608.04885) | arXiv | Diagnosis under perturbations | Text answer | ? | VLMs (?) |
| [How Do Medical MLLMs Fail?](https://arxiv.org/abs/2603.14323) | ICLR 2026 (per ML Anthology listing) | Visual grounding failure study on Med-VQA; VGRefine refines attention at inference time | Text answer | VGMED evaluation set; 6 Med-VQA benchmarks | 8 medical MLLMs |
| [MedSIGHT](https://arxiv.org/html/2606.06760v1) | arXiv | Grounded visual comprehension | Text + regions (?) | ? | Medical large VLMs |
| [PRS-Med](https://openaccess.thecvf.com/content/CVPR2026W/CV4Clinic2026/papers/Trinh_PRS-Med_Position_Reasoning_Segmentation_in_Medical_Imaging_CVPRW_2026_paper.pdf) | CVPRW 2026 | Position reasoning segmentation | Mask + text | 2D, PosMed: 116K QA pairs, 6 modalities | Own model vs. baselines |
| [MedREAL](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/12494.pdf) | ECCV 2026 | Grounded VQA + segmentation | Answer + mask | 2D, MedRAVS-13K | Own model vs. MedSAM3, LISA and others |
| [VoxTell](https://arxiv.org/abs/2511.11450) | CVPR 2026 | Free-text 3D segmentation | 3D mask | 3D, 158 datasets | Own model vs. SAT, BiomedParse and others |
| [Spatial-aware Symmetric Alignment](https://arxiv.org/pdf/2512.22981) | arXiv | Text-guided segmentation with directional guidance | Mask | ? | Own method |
| [Medal S](https://arxiv.org/html/2511.13001) | arXiv | Spatio-textual prompted segmentation | Mask | ? | Own model |
| [Medical SAM3](https://arxiv.org/pdf/2601.10880) | arXiv | Universal prompt-driven segmentation | Mask | Many | Own model |
| [MedGround](https://arxiv.org/pdf/2601.06847) | arXiv | Verified grounding data for medical VLMs | ? | Own data | Medical VLMs (?) |

## Table 2: What is varied and what is analysed

| Work | Language perturbed | Image perturbed | Language and image varied separately | Looks inside the model | Attribution (Shapley/SHAP/LIME) | Positional / laterality focus |
|---|---|---|---|---|---|---|
| **Our work** | ✅ word ablation, embedding swap and blend, positional wording | ◐ rotation (done), organ noise (pilot), mirror (partial) | ✅ separately, but not yet on one scale | ✅ layers, attention, fused features | ✅ | ✅ |
| Your other Left! | ? | ◐ visual markers on structures | ❌ | ❌ | ❌ | ✅ |
| HEAL-MedVQA | ✅ Textual Perturbation Test | ✅ Visual Perturbation Test | ✅ | ? | ❌ | ◐ (localization focus) |
| MedVIGIL | ✅ false-premise, wording perturbations, knowledge-only rewrites | ✅ ROI-masked / ROI-isolated (abstract; a left/right flip probe appeared only in a search summary) | ✅ | ❌ (external behavioural evaluation) | ❌ | ◐ (laterality only in some probes) |
| Diagnostic Robustness | ✅ label-position swap in prompts (presentation order, not anatomy) | ✅ slice reordering / reversal (brain MRI) | ✅ | ❌ | ❌ | ❌ (studies presentation-order stability, not anatomical position) |
| How Do Medical MLLMs Fail? | ? | ? | ? | ? | ? | ◐ (grounding) |
| MedSIGHT | ❌ | ❌ | ❌ | ? | ❌ | ◐ |
| PRS-Med | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| MedREAL | ❌ | ❌ | ❌ | ❌ | ❌ | ◐ |
| VoxTell | ✅ paraphrase, synonym, typo robustness | ❌ | ❌ | ❌ | ❌ | ❌ |
| Spatial-aware Symmetric Alignment | ◐ (spatial words in text) | ❌ | ❌ | ❌ | ❌ | ✅ (left/right guidance) |
| Medal S | ◐ | ❌ | ❌ | ❌ | ❌ | ◐ |
| Medical SAM3 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MedGround | ❌ | ❌ | ❌ | ❌ | ❌ | ◐ |

## Table 3: Findings and how they relate to ours

| Work | Key finding (from summaries) | Overlap with ours | What our work adds |
|---|---|---|---|
| Your other Left! | VLMs are at chance on relative position; they rely on prior anatomical knowledge over the image | Same language-prior-over-image conclusion | Segmentation, 3D volumes, controlled interventions on both sides |
| HEAL-MedVQA | Text and image shortcut tests expose localization and hallucination problems | Same two-sided perturbation idea | Segmentation, internal attribution, a rotation-induced opposite-side result |
| MedVIGIL | Models are weak when visual evidence is broken; image probes include flips and ROI masks | Same image-side probes (flip, ROI mask) | Prompt-side attribution, 3D segmentation |
| Diagnostic Robustness | Perturbing image and text degrades diagnosis | Two-sided design | Positional focus and segmentation |
| How Do Medical MLLMs Fail? | Grounding is weaker on medical than natural images | Failure-analysis spirit | Interventions and internal analysis |
| PRS-Med | A model and a benchmark for position reasoning segmentation | Position + segmentation | Diagnosis of why positional errors occur |
| MedREAL | Grounded VQA and segmentation improves with its method | Medical grounding | Analysis instead of a new method |
| VoxTell | Strong 3D segmentation, robust to paraphrase | It is our main model | Tests left/right grounding, which its paper does not |
| Spatial-aware Symmetric Alignment | Text like "left lower lung" can segment both sides | Same both-sides failure as our position-word removal | Attribution of why |
| Medal S / Medical SAM3 / MedGround | Models or data for prompted segmentation and grounding | Context | Analysis of failure modes |

## Table 4: Our claims vs. prior work

| Claim in our title | Already shown by others? | Our evidence | Still needed |
|---|---|---|---|
| Language contributes to positional errors | Partly (Your other Left!, HEAL-MedVQA) | Rows 1–11; 2D VLM prompt sweep | Nothing major |
| Region (image layout) contributes | Partly (MedVIGIL flip and ROI masks, HEAL-MedVQA) | Rotation sweep; row 13 noise (3 subjects) | More region experiments (removal, organ swap, translation) |
| Positional hallucination in text-prompted segmentation | Not found in summaries | Rotation: masks land on the opposite-side organ at 150-180 degrees | Write up the rotation analysis; add a removal experiment |
| Disentangling the two | Two-sided perturbation exists in HEAL-MedVQA and Diagnostic Robustness | Separate interventions in one study | One shared design |
| Relative contributions on one scale | Not found | Not yet | A common Dice-drop comparison |
| 3D segmentation setting with internal attribution | Not found | Shapley, SHAP, LIME, embedding swap, layer patching | Nothing major |

## Table 5: Papers matching our task type (text-prompted localization / segmentation with position)

Kept: papers whose task is localization or segmentation from a text prompt, and that deal with position, spatial words, or are models we run. VQA-only papers are excluded (Your other Left!, HEAL-MedVQA, MedVIGIL, Diagnostic Robustness, How Do Medical MLLMs Fail?).

| Paper | Task | 2D / 3D | Prompt | Position handling | Match with ours | Difference from ours |
|---|---|---|---|---|---|---|
| **Our work** | Localization (2D VLMs) + segmentation (3D) | Both | Positional text prompt | Tested: does the output follow the position word? | n/a | n/a |
| PRS-Med | Position reasoning segmentation | 2D, 6 modalities | Text query about an anatomical zone | Central: segment within a named zone | Position + segmentation | A model and a benchmark; does not diagnose failures |
| Spatial-aware Symmetric Alignment | Text-guided lesion segmentation | Not stated | Clinical text with location | Builds guidance masks from spatial words | Same both-sides failure ("left lower lung") | A fix; we explain the failure |
| Medal S | Text + spatial prompted segmentation | 3D | Text and spatial prompts | Spatial prompts | Text-prompted 3D segmentation | A model; no analysis of position errors |
| VoxTell | Free-text segmentation | 3D | Free text | Not tested | Our main 3D model | We test left/right grounding, which its paper does not |
| Medical SAM3 | Prompt-driven segmentation | 2D + 3D | Text, box or point | Not tested | A model we run | Same |

Use this file as a checklist: fill in each `?` by reading the paper, then keep only the claims that survive.
