# Text3DSAM Dataset Description

Three bilateral-organ 3D datasets are used for positional-bias evaluation with Text3DSAM, sourced from `samples/text3dsam/manifest_text3dsam_rotated.csv`. Each dataset contains 5 subjects, and every subject's volume is rendered at 7 rotation angles (0°, 30°, 60°, 90°, 120°, 150°, 180°) to test left/right positional bias.

| Dataset | Modality | Subjects | Bilateral Organs (paired L/R) | Rows (organ × subject × angle) | Voxel Spacing (mm) | Axis Orientation | Rotation Angles |
|---|---|---|---|---|---|---|---|
| **CT_HaN-Seg** | CT (head & neck) | 5 | 4 pairs: carotid artery, cochlea, parotid gland, submandibular gland | 280 | ~0.56–0.68 × same × 2.0 (in-plane varies per subject, slice thickness fixed) | SAR | 0°, 30°, 60°, 90°, 120°, 150°, 180° |
| **CT_TotalSeg_cardiac** | CT (cardiac/vascular) | 5 | 5 pairs: brachiocephalic vein, common carotid artery, iliac artery, iliac vena, subclavian artery | 168 | 1.500 × 1.500 × 1.500 (isotropic) | SAR | 0°, 30°, 60°, 90°, 120°, 150°, 180° |
| **MR_TotalSeg** | MR (abdomen/pelvis) | 5 | 4 pairs: adrenal gland, hip, iliac artery, kidney | 169 | Variable, ~1.1–1.9 in-plane × 1.5–7.7 slice thickness | ASR | 0°, 30°, 60°, 90°, 120°, 150°, 180° |

## Notes

- All three datasets are bilateral-organ subsets built for left/right positional-bias testing — each organ appears as a left/right pair with a `gt_side_axis0`/`gt_side_axis1`/`gt_side_axis2` label in the manifest.
- Each subject's volume is rendered at 7 rotation angles (multi-angle rotation sweep), which is why row counts ≈ subjects × organs × 7.
- CT_TotalSeg_cardiac (168 rows) and MR_TotalSeg (169 rows) fall slightly short of the full 5 × organs × 7 count (175 for 5 organ pairs), suggesting a few missing organ/angle combinations for specific subjects.
