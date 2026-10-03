# Positional Bias in Medical Vision-Language Models: A Benchmark Study

## Problem

Medical image analysis increasingly relies on vision-language models (VLMs) that accept natural-language prompts to localize or segment anatomical structures — e.g., "segment the *right* kidney," or "locate the *upper* lobe." These prompts routinely encode **spatial/positional language** (left, right, upper, lower, anterior, posterior) alongside the **entity** being described (kidney, tumor, lobe).

For such a model to be clinically trustworthy, its output must genuinely respond to the stated position — moving to the anatomically correct location when "left" becomes "right" — rather than relying on the image content alone, a fixed output bias, or a memorized shortcut that happens to work on common cases but fails silently elsewhere. If a model ignores positional language, or follows it even when it contradicts the true anatomy, this is a safety-relevant failure mode that standard accuracy metrics (e.g., Dice score, IoU) do not expose, since they are typically computed without any stress-testing of positional phrasing.

Despite the growing use of VLMs for medical image grounding and segmentation, there is no systematic, model-agnostic benchmark that isolates and quantifies **positional bias** — the gap between a model's *reported* accuracy and its *actual, verifiable* use of spatial language.

## Gap

Existing evaluation practice in medical VLM / grounding research has several blind spots that this work addresses:

1. **No standardized positional-bias benchmark.** Models are evaluated on aggregate localization/segmentation accuracy, but not on whether their outputs correctly and consistently track positional language across controlled prompt variations.
2. **Limited to 2D, limited model coverage.** Prior grounding benchmarks largely target natural images or 2D medical slices; few studies systematically compare how a broad set of 2D grounding/VLM architectures handle positional language in medical imaging contexts.
3. **No 3D / volumetric evaluation.** Clinically, most meaningful spatial reasoning (left vs. right kidney, superior vs. inferior lobe) is inherently volumetric. 2D slice-based evaluation cannot capture whether a model's spatial grounding survives in full 3D, text-guided segmentation.
4. **No stress-testing under geometric transformation.** It is unknown how robust positional grounding is to transformations that alter the image's spatial layout (e.g., rotation) while leaving semantic content intact — a setting that disentangles true anatomical grounding from superficial correlations.
5. **No mechanistic account of *why* a model's predictions shift (or don't) with positional words.** Even when a model appears to respond "correctly" to positional language at the output level, it is unclear whether this reflects genuine grounding in the image versus a shortcut (e.g., a memorized prior, or sensitivity to prompt wording without real visual evidence).

This work addresses these gaps by constructing a dedicated benchmark suite for positional bias in medical VLMs, spanning 2D and 3D segmentation/grounding, controlled geometric stress tests, and internal-mechanism analysis.

## Approach (Benchmark Study)

This study is framed as a **multi-stage benchmark**, evaluating medical vision-language models for positional bias along four complementary axes. The precise analytical techniques used within each axis are withheld at this stage; the axes themselves are summarized below.

### 1. 2D Model Benchmarking
A broad set of 2D vision-language and grounding models is evaluated on medical imaging datasets using prompts that vary only in their positional component (e.g., "locate the {organ}" vs. "locate the left {organ}" vs. "locate the right {organ}"). This axis establishes a baseline picture of how widely-used 2D grounding architectures behave when positional language is introduced or altered, across multiple imaging modalities and anatomical targets.

### 2. 3D Model Benchmarking
Recognizing that clinically relevant spatial reasoning is inherently volumetric, the benchmark extends to full 3D, text-guided segmentation models evaluated on multi-subject, multi-organ volumetric datasets. Candidate 3D models are compared on standard segmentation quality metrics, and the strongest-performing candidate is carried forward for deeper positional-bias analysis across a wide range of organs/structures and bilateral (left/right) anatomical pairs.

### 3. Rotation-Based Stress Testing
To disentangle genuine anatomical grounding from superficial or memorized shortcuts, the selected 3D model is stress-tested under controlled geometric transformations of the input volumes — including systematic rotation and reflection of the anatomy — while tracking whether the model's predictions move with the true (transformed) anatomy or remain anchored to the original, untransformed orientation. This axis probes robustness and consistency of positional grounding under conditions that break simple positional heuristics.

### 4. Internal Mechanism Analysis
Finally, the study moves beneath output-level behavior to examine the model's internal decision process when resolving positional language — investigating, at a high level, how the model's internal representations and attention relate to the entity word versus the positional word, and whether the model's response to positional language reflects genuine image-grounded reasoning or a non-visual shortcut. This includes high-level analyses in the spirit of word-attribution and image-attention interpretability, without disclosing the specific methodology.

---

*This document intentionally omits implementation-level detail of the analytical methods used in each stage, consistent with the current phase of the research.*
