# MICCAI 2024-2025 Papers: Medical Image Segmentation, Localization, and Benchmarking

Companion to `CVPR_ECCV_2024_2026_SEGMENTATION_VLM_PAPERS.md`. MICCAI turned out to be where most of this repo's closest-matching research actually lives (text/VLM-guided segmentation, positional/spatial reasoning benchmarks), so it gets its own file rather than being folded into the CVPR/ECCV one. Scope: MICCAI 2024 and MICCAI 2025 papers on medical segmentation, spatial/positional localization, and — the specific focus of this pass — papers that **benchmark multiple existing models against each other**, not just propose one new method.

## MICCAI 2025

| Paper | Link | Benchmarking? | Why it's relevant |
|---|---|---|---|
| "Your other Left!" Vision-Language Models Fail to Identify Relative Positions in Medical Images | [arxiv.org/abs/2508.00549](https://arxiv.org/abs/2508.00549) | **Yes — benchmarks 4 existing VLMs**: GPT-4o, Llama3.2, Pixtral, JanusPro, all evaluated on a new relative-position dataset (MIRP) | **The closest existing match to this repo's core research question anywhere.** Shows VLMs perform at or near chance level identifying the relative position of two anatomical structures in a CT slice, and tests visual markers as a fix. Directly parallel to this repo's own positional-bias study, just on general-purpose VLMs rather than segmentation-specific ones |
| RadSAM: Segmenting 3D radiological images with a 2D promptable model | [papers.miccai.org](https://papers.miccai.org/miccai-2025/paper/4118_paper.pdf) · [arXiv](https://arxiv.org/abs/2504.20837) | **Yes — direct model-vs-model benchmark**: nnU-Net, SAM, MedSAM, SAM-Med3D | **Strong match.** Primary evaluation dataset is **AMOS** — the same dataset this repo uses for its MRI arm — with secondary evaluation on TotalSegmentator and MSD for out-of-domain generalization. A 2D-promptable model doing iterative slice-by-slice 3D reconstruction, evaluated with a benchmarking methodology directly comparable to this repo's own AMOS22 evaluation |
| Localization Lens for Improving Medical Vision-Language Models | [papers.miccai.org](https://papers.miccai.org/miccai-2025/paper/2723_paper.pdf) | Partial — proposes a new method, but includes a localization-question analysis showing spatial-reasoning gains lift Med-VQA accuracy up to 6.2 points | Proposes richer anatomical/positional representations to improve a medical VLM's spatial understanding — same research angle as this repo (does the model know *where* something is), approached by building a fix rather than auditing existing models |
| Medical Large Vision Language Models with Multi-Image Visual Ability (Med-MIM) | [papers.miccai.org](https://papers.miccai.org/miccai-2025/0545-Paper0784.html) | New benchmark dataset (83.2K samples) + new model, not a pure model-vs-model comparison | Weaker match — this is about multi-image comparison/temporal tracking across scans, not single-image spatial/positional localization within one scan |
| MoE-SAM: Enhancing SAM for Medical Image Segmentation with Mixture-of-Experts | [papers.miccai.org](https://papers.miccai.org/miccai-2025/0577-Paper1526.html) | Unconfirmed — page was behind a verification wall, could not extract the baseline list or confirm benchmarking depth | Likely a new-method paper (SAM + MoE routing); flagged as unverified rather than detailed |
| TGSAM-2: Text-Guided Medical Image Segmentation using SAM 2 | [papers.miccai.org](https://papers.miccai.org/miccai-2025/paper/0846_paper.pdf) | New method (adds text-prompt support to SAM2), not primarily a benchmark | Weak match for the benchmarking ask specifically, though topically on-theme (text-guided segmentation) |

## MICCAI 2024

| Paper | Link | Benchmarking? | Why it's relevant |
|---|---|---|---|
| ABP (Asymmetric Bilateral Prompting) | (venue confirmed MICCAI 2024; link not re-verified in this pass) | Not confirmed as multi-model benchmark - proposes its own prompting method | Text-guided medical segmentation approach, found in earlier CVPR/ECCV search as the closest MICCAI 2024 match to this repo's research line |
| Common Vision-Language Attention for Text-Guided Medical Image Segmentation of Pneumonia | (venue confirmed MICCAI 2024; link not re-verified in this pass) | Not confirmed as multi-model benchmark - proposes its own attention mechanism | Text-guided segmentation via vision-language attention, same relevance as above |
| SimTxtSeg: Weakly-Supervised Medical Image Segmentation with Simple Text Cues | [papers.miccai.org](https://papers.miccai.org/miccai-2024/699-Paper0802.html) | New method (text-cue pseudo-labeling), not a comparison-focused benchmark paper | Weak match - no multi-model benchmarking angle found, but on-theme (text-guided segmentation) |

No MICCAI 2024 paper doing genuine multi-model benchmarking of promptable/text-guided segmentation or VLM spatial-localization was found beyond ABP and the Common Vision-Language Attention paper, and neither of those is itself a benchmarking paper (both propose their own new method rather than comparing existing ones).

## Bottom line for "which MICCAI papers actually benchmark models, for this type of research"

Two papers clear that bar cleanly:
- **RadSAM (MICCAI 2025)** - a genuine multi-model benchmark (nnU-Net vs. SAM vs. MedSAM vs. SAM-Med3D) on **AMOS**, the same dataset this repo evaluates Text3DSAM and Medical SAM3 on. Worth treating as a benchmarking-methodology reference even though it doesn't include Text3DSAM/Medical SAM3 themselves.
- **"Your other Left!" (MICCAI 2025)** - benchmarks 4 general-purpose VLMs (GPT-4o, Llama3.2, Pixtral, JanusPro) specifically on relative-position identification in medical images. This is the closest paper found anywhere, at any venue, to this repo's actual research question, and it is fundamentally a benchmarking study, not a new-model paper.

Everything else found either proposes one new method (RadSAM's contemporaries, ABP, SimTxtSeg, TGSAM-2, MoE-SAM) or builds a new benchmark dataset without deeply comparing existing models against each other (Med-MIM).

## Venue-unconfirmed (excluded)

- **K-Prism**: A Knowledge-Guided and Prompt Integrated Universal Medical Image Segmentation Model ([arxiv.org/abs/2509.25594](https://arxiv.org/abs/2509.25594)) - posted on OpenReview, no MICCAI acceptance confirmed. Benchmarks against 18 public datasets - would be a strong match if the venue is later confirmed as MICCAI, but excluded here for now.
- Assorted narrative/survey reviews on foundation-model medical segmentation (a PMC review on language-guided segmentation, a QIMS narrative review) - journal publications, not MICCAI proceedings papers.

If any of the "not confirmed" entries above turn out to be MICCAI papers after all, or if you want ABP's and the Common Vision-Language Attention paper's links independently re-verified, say so and I'll chase them down directly.
