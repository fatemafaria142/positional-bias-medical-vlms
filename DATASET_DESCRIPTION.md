# Dataset Description - 3D Segmentation Datasets

Both datasets are used for the **3D text-guided volumetric segmentation** arm of the positional-bias study (`samples/amos22_mri/`, `samples/neurite_oasis/`), scored with Text3DSAM/MedSAM3-style models that take a whole volume + text prompt and predict a whole mask volume. 10 subjects are sampled from each (`scripts/common3d.py: N_SUBJECTS = 10`, `SEED = 42`), and every organ/structure actually annotated per subject gets one plain-text segmentation prompt, no directional (left/right/upper/lower) prompt variants, since the goal here is *cross-organ* drift, not prompt-phrasing bias.

| Field | AMOS22 MRI | Neurite OASIS |
|---|---|---|
| **Full name** | AMOS 2022, Abdominal Multi-Organ Segmentation, MRI arm | Neurite OASIS (OASIS-1, FreeSurfer/SAMSEG-processed release) |
| **Body region / domain** | Abdomen | Brain |
| **Modality** | MRI (multiple scanners/sequences, not intensity-standardized) | MRI (T1-weighted, skull-stripped, bias-corrected, affine-aligned to a common template) |
| **Dimensionality** | 3D volumes (NIfTI) | 3D volumes (NIfTI) |
| **Source / link** | [huggingface.co/datasets/MedOtter/amos22-mri-dataset](https://huggingface.co/datasets/MedOtter/amos22-mri-dataset) (original challenge: [amos22.grand-challenge.org](https://amos22.grand-challenge.org/)) | Local tar `neurite-oasis.v1.0.tar` (6.6 GB, gitignored), from [Neurite OASIS release](https://www.nitrc.org/projects/oasis) / [Andysbrainbook Neurite mirror](https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar) |
| **Total subjects available in source** | 60 (40 train + 20 validation splits) | 414 (full OASIS-1 cross-sectional cohort) |
| **Subjects sampled for this study** | 10 (`amos_0508, amos_0522, amos_0530, amos_0532, amos_0544, amos_0553, amos_0554, amos_0555, amos_0558, amos_0575`) | 10 (`OASIS_OAS1_0014_MR1, OASIS_OAS1_0057_MR1, OASIS_OAS1_0062_MR1, OASIS_OAS1_0076_MR1, OASIS_OAS1_0123_MR1, OASIS_OAS1_0135_MR1, OASIS_OAS1_0151_MR1, OASIS_OAS1_0363_MR1, OASIS_OAS1_0419_MR1, OASIS_OAS1_0421_MR1`) |
| **Organs / structures annotated (total in taxonomy)** | 15-class AMOS label map (spleen, R/L kidney, gallbladder, esophagus, liver, stomach, aorta, IVC, pancreas, R/L adrenal gland, duodenum, urinary bladder, prostate/uterus) | 35-structure FreeSurfer-style segmentation (cerebral white matter, cerebral cortex, lateral/inferior-lateral ventricle, cerebellum WM/cortex, thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens, ventral DC, vessel, choroid plexus, each L/R, plus 3rd/4th ventricle and brain stem) |
| **Organs / structures actually present in the 10 sampled subjects** | 13 (all except prostate/uterus and urinary bladder, which are sex-specific / not annotated in the sampled subjects) | Up to 35 per subject (a few tiny structures like Left/Right-Vessel can be occasionally empty in some subjects) |
| **Manifest rows (subject x organ pairs)** | 130 (`samples/amos22_mri/manifest_amos22_mri.csv`) | 350 (`samples/neurite_oasis/manifest_neurite_oasis.csv`) |
| **Volume shape** | Varies per subject (e.g. 260x80x320 to 512x168x512); different scan planes (axial/coronal/sagittal), thin axis position varies per subject | Fixed 160x192x224 for all 10 subjects (common template space) |
| **Voxel spacing** | Varies per subject (e.g. 1.4x1.5x1.4 mm to 0.82x1.1x0.82 mm) | 1.000x1.000x1.000 mm (isotropic, identical for all 10 sampled subjects; common-template registration) |
| **Ground truth provided** | Per-organ label volume: voxel centroid, voxel bbox, and true anatomical side (per axis, from the NIfTI affine) | Per-structure label volume (`aligned_seg35.nii.gz`); same centroid/bbox/side derivation |
| **Preprocessing applied by this repo** | `thin_axis_first_order()` (axis-order fix, see `TEXT3DSAM_RESULTS_ANALYSIS.md` section 0) before resizing to model input size | None needed; already same-shape, thin axis already first (reordering is a no-op) |
| **License / access** | Hosted redistribution of AMOS22 (CC BY-NC 4.0-style academic/research use; verify current HF repo terms before commercial use) | Neurite OASIS release terms (OASIS Data Use Agreement; academic/research use, attribution required) |
| **Role in this study** | Graded difficulty case: model has partial competence (liver/kidney/aorta/spleen best, adrenal glands worst); used to test whether wrong predictions drift toward a neighboring organ | Out-of-domain/failure-mode case: zero training exposure to brain anatomy or neuroimaging preprocessing style in the segmentation models tested so far; used as a boundary-case contrast point |

## Notes

- Both datasets are sampled and exported by dedicated scripts (`scripts/sample_amos22_mri.py`, `scripts/sample_neurite_oasis.py`) using shared logic in `scripts/common3d.py`, so re-running with a different seed or `N_SUBJECTS` regenerates a fresh, reproducible sample.
- Full raw downloads are *not* committed to the repo. Only the exported `samples/<dataset>/volumes/` (image + label NIfTI pairs for the sampled subjects) and the manifest CSV are kept, so the full 786 MB (AMOS22 MRI) / 6.6 GB (OASIS tar) source data can be deleted after sampling.
- Neither dataset carries native free text. All segmentation prompts here are generated plain-text organ/structure names (e.g. "segment the liver"), not sourced from the datasets themselves.
