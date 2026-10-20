# Neurite-OASIS Dataset

## Overview
Neurite-OASIS is a preprocessed, analysis-ready neuroimaging dataset built by the
Neurite lab (led by Adrian Dalca et al., MIT/MGH) on top of the original **OASIS**
(Open Access Series of Imaging Studies) brain MRI collection. It repackages OASIS
T1-weighted brain scans with standardized preprocessing and dense anatomical
segmentations, making it a popular benchmark for image registration, segmentation,
and generative modeling research.

## Source
- **Base cohort:** OASIS-1 (OASIS-3 variants also exist), a public cross-sectional
  collection of T1-weighted structural brain MRI scans of subjects spanning
  healthy aging and Alzheimer's disease.
- **Subjects:** ~414 subjects in the original OASIS-1 release, ages 18–96,
  including both cognitively normal adults and patients with mild-to-moderate
  Alzheimer's disease.
- **Preprocessing by Neurite:** skull-stripping, intensity normalization, affine
  registration to a common atlas space, and resampling to a consistent voxel grid
  (commonly 256×256×256 at 1mm isotropic resolution).

## What's Included
- **T1-weighted MRI volumes** — normalized, skull-stripped brain scans.
- **Segmentation labels** — dense multi-structure brain parcellations (typically
  ~24–35 anatomical labels) generated with FreeSurfer, covering structures such as:
  - Cerebral cortex / white matter (left & right)
  - Cerebellum cortex / white matter (left & right)
  - Lateral ventricles
  - Thalamus, Putamen, Caudate, Hippocampus (left & right)
  - Brainstem, and other subcortical structures
- **2D slice subsets** — many works (including this project) extract a canonical
  2D axial/coronal slice per subject for lighter-weight experimentation.

## Why It's Used in VLM/Segmentation Benchmarks
- **Clean, consistent labels:** FreeSurfer-derived parcellations give reliable
  ground truth for left/right paired structures, which is exactly what's needed
  for studying **positional (left-vs-right) bias** in vision-language or
  segmentation models.
- **Bilateral symmetry:** Because many structures come in left/right pairs
  (hippocampus, thalamus, ventricle, etc.), it's a natural fit for testing whether
  a model's prediction follows the prompt's stated side or defaults to a fixed/
  mirrored answer.
- **Registration-friendly:** Being pre-aligned to a common atlas space, it's also
  a standard benchmark for deformable/affine registration methods (e.g., VoxelMorph,
  SynthMorph — also from the Neurite lab).

## Typical Use in This Project
In this repo's experiments, Neurite-OASIS is one of the core datasets (alongside
AMOS22-MRI, KiTS23, TotalSegmentatorV2, and HaN-Seg CT) used to evaluate VoxTell
and other models on:
- Left/right segmentation accuracy under image rotation
- Attribution of predictions to the "left"/"right" prompt word vs. the organ name
- Attention-grounding (KL-divergence) analysis per anatomical structure
- Activation patching across decoder layers to localize where left/right
  decisions are made

## Access
- Distributed via the Neurite lab's public data releases (commonly hosted
  alongside their registration/segmentation tooling, e.g. through the
  `neurite`/`voxelmorph` GitHub ecosystems).
- Requires standard OASIS data-use agreement acknowledgment for the underlying
  OASIS-1 imaging data.
