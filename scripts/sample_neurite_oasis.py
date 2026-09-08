"""
Sample 10 subjects from Neurite OASIS (brain MRI, 35-structure segmentation)
for positional-bias probing, and export full 3D image+label volumes plus a
text-guided volumetric-segmentation manifest -- for 3D models (Text3DSAM,
MedSAM3, ...) that take a whole volume + text prompt and predict a whole
mask volume. Same convention as scripts/sample_amos22_mri.py.

Usage:
    python scripts/sample_neurite_oasis.py

Reads directly from the already-downloaded local tar
(data/raw/neurite-oasis.v1.0.tar -- 6.6 GB, gitignored, never pushed) and
extracts only the 10 sampled subjects' aligned_norm.nii.gz (image) +
aligned_seg35.nii.gz (label) -- the common-template-space versions, so every
subject shares the same 160x192x224 shape/orientation, which is what makes
cross-subject positional comparison meaningful. Writes:
    samples/neurite_oasis/volumes/<subject>_image.nii.gz
    samples/neurite_oasis/volumes/<subject>_label.nii.gz
    samples/neurite_oasis/manifest_neurite_oasis.csv

Each row is one (subject, structure) pair with a single plain segmentation
prompt -- 10 subjects x up to 35 structures (whichever of the 35 are
actually present with nonzero voxels in a given subject's automated
segmentation; small structures like Left-Vessel/Right-Vessel can
occasionally be empty). See scripts/common3d.py for the shared logic.
"""

import random
import shutil
import tarfile
from pathlib import Path

from common3d import N_SUBJECTS, SEED, export_subject, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "samples" / "neurite_oasis"
RAW_TAR = REPO_ROOT / "data" / "raw" / "neurite-oasis.v1.0.tar"
EXTRACT_DIR = REPO_ROOT / "data" / "raw" / "neurite-oasis-extracted"
DATASET_NAME = "neurite_oasis"

# seg35_labels.txt (label id -> structure name), extracted verbatim from the
# dataset's own tar. Names lowercased/hyphen-to-space for natural prompts.
STRUCTURE_MAP = {
    1: "left cerebral white matter",
    2: "left cerebral cortex",
    3: "left lateral ventricle",
    4: "left inferior lateral ventricle",
    5: "left cerebellum white matter",
    6: "left cerebellum cortex",
    7: "left thalamus",
    8: "left caudate",
    9: "left putamen",
    10: "left pallidum",
    11: "3rd ventricle",
    12: "4th ventricle",
    13: "brain stem",
    14: "left hippocampus",
    15: "left amygdala",
    16: "left accumbens",
    17: "left ventral DC",
    18: "left vessel",
    19: "left choroid plexus",
    20: "right cerebral white matter",
    21: "right cerebral cortex",
    22: "right lateral ventricle",
    23: "right inferior lateral ventricle",
    24: "right cerebellum white matter",
    25: "right cerebellum cortex",
    26: "right thalamus",
    27: "right caudate",
    28: "right putamen",
    29: "right pallidum",
    30: "right hippocampus",
    31: "right amygdala",
    32: "right accumbens",
    33: "right ventral DC",
    34: "right vessel",
    35: "right choroid plexus",
}


def list_all_subjects(tar: tarfile.TarFile) -> list:
    member = tar.getmember("subjects.txt")
    text = tar.extractfile(member).read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_subject_volumes(tar: tarfile.TarFile, subject_id: str) -> tuple:
    """Extract just this subject's aligned_norm.nii.gz + aligned_seg35.nii.gz
    from the tar into EXTRACT_DIR, returning their local paths."""
    subj_dir = EXTRACT_DIR / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    img_path = subj_dir / "aligned_norm.nii.gz"
    lbl_path = subj_dir / "aligned_seg35.nii.gz"

    if not img_path.exists():
        member = tar.getmember(f"{subject_id}/aligned_norm.nii.gz")
        with tar.extractfile(member) as src, img_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    if not lbl_path.exists():
        member = tar.getmember(f"{subject_id}/aligned_seg35.nii.gz")
        with tar.extractfile(member) as src, lbl_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    return img_path, lbl_path


if __name__ == "__main__":
    if not RAW_TAR.exists():
        raise SystemExit(f"[error] tar not found: {RAW_TAR}")

    with tarfile.open(RAW_TAR, "r") as tar:
        all_subjects = list_all_subjects(tar)
        print(f"[info] {len(all_subjects)} subjects available in tar")

        rng = random.Random(SEED)
        sampled = sorted(rng.sample(all_subjects, min(N_SUBJECTS, len(all_subjects))))
        print(f"[info] sampled {len(sampled)} subjects (seed={SEED}): {sampled}")

        all_rows = []
        for subject_id in sampled:
            img_path, lbl_path = extract_subject_volumes(tar, subject_id)

            rows = export_subject(
                subject_id=subject_id,
                image_path=str(img_path),
                label_path=str(lbl_path),
                organ_map=STRUCTURE_MAP,
                out_dir=OUT_DIR,
                dataset_name=DATASET_NAME,
                split="all",  # Neurite OASIS has no official train/valid split
            )
            print(f"[ok] {subject_id}: {len(rows)} structure(s) exported"
                  if rows else f"[warn] {subject_id}: no structures found")
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("[error] no rows produced -- check tar contents/label ids")

    write_manifest(all_rows, OUT_DIR, DATASET_NAME)
    print(f"[done] {DATASET_NAME}: samples/neurite_oasis/ is self-contained and safe to "
          f"upload/share; you can delete {EXTRACT_DIR} to reclaim disk space "
          f"(the original {RAW_TAR} is untouched, gitignored, stays local).")
