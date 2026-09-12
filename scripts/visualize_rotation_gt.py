"""
QC plot for the multi-angle rotation sweep (scripts/rotate_multiangle_samples.py):
for one subject + one organ per dataset, show the image slice at every
rotation angle (0/30/60/90/120/150/180 degrees) with its ground truth
overlaid -- so you can eyeball that (a) the organ visibly moves around the
frame as the angle changes, while (b) the reported anatomical side
(gt_side_axis*) stays fixed, exactly as designed.

Two rows, one column per angle, kept deliberately separate:
  - row 1 ("Rotation"): the rotated image alone, no overlay, titled with
    just the angle.
  - row 2 ("Ground truth"): the SAME slice with the actual ground-truth
    segmentation mask filled in green (a true mask, not a bounding box).
Both rows are sliced at the organ's ground-truth centroid along that
volume's own Superior/Inferior axis. The prompt is identical at every angle
by design (rotating the image doesn't change what was asked), so it is
shown once, at the very top of the figure.

One organ is used per dataset (chosen to be a clearly lateral structure so
positional drift is obvious): "right kidney" for amos22_mri, "left
hippocampus" for neurite_oasis. Reads the already-generated per-annotator
manifest (default: faria/), which already has all 7 angles per subject.

Usage:
    python scripts/visualize_rotation_gt.py [person]
        person: faria (default) | nipu | hridoy
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]

PERSON = sys.argv[1] if len(sys.argv) > 1 else "faria"

# col_width/wspace tuned per dataset so portrait-ish brain slices (neurite_oasis)
# don't sit in an over-wide panel with lots of side whitespace between columns.
DATASETS = [
    {"name": "amos22_mri", "organ": "right kidney", "col_width": 4.0, "wspace": 0.05},
    {"name": "neurite_oasis", "organ": "left hippocampus", "col_width": 2.7, "wspace": 0.015},
]

MASK_GREEN = "#22c55e"
MASK_GREEN_DARK = "#15803d"


def plot_dataset(person: str, dataset_name: str, organ: str, col_width: float, wspace: float):
    data_dir = REPO_ROOT / person / "samples" / dataset_name
    manifest_path = data_dir / f"manifest_{dataset_name}.csv"
    manifest = pd.read_csv(manifest_path)

    subject_id = manifest["subject_id"].iloc[0]
    rows = manifest[(manifest["subject_id"] == subject_id) & (manifest["organ"] == organ)]
    rows = rows.set_index("rotation_angle_deg").loc[ANGLES_DEG].reset_index()
    if rows.empty:
        raise SystemExit(f"[{dataset_name}] organ '{organ}' not found for subject {subject_id}")

    n = len(ANGLES_DEG)
    prompt = rows.iloc[0]["prompt_text"]  # identical at every angle by design

    # Height is fixed (independent of col_width) so title/label font sizes
    # take up the same vertical fraction of the figure regardless of how
    # narrow the per-dataset columns are.
    fig, axes = plt.subplots(2, n, figsize=(col_width * n, 9.0))

    for col, (_, row) in enumerate(rows.iterrows()):
        angle = int(row["rotation_angle_deg"])
        label_id = int(row["label_id"])
        image_path = data_dir / row["image_path"]
        label_path = data_dir / row["label_path"]

        img_nii = nib.load(str(image_path))
        image = img_nii.get_fdata().astype(np.float32)
        label = nib.load(str(label_path)).get_fdata().astype(int)
        mask = label == label_id

        # Always slice along the volume's own S/I axis (the axis rotation is
        # applied about) so the sweep is visible -- the smallest-dimension
        # axis used for model input elsewhere isn't necessarily the S/I axis
        # (e.g. neurite_oasis has S/I at axis 1, not the thinnest axis 0).
        axcodes = nib.aff2axcodes(img_nii.affine)
        slice_axis = next(a for a, code in enumerate(axcodes) if code in ("S", "I"))

        centroid = [row["gt_centroid_vox_0"], row["gt_centroid_vox_1"], row["gt_centroid_vox_2"]]
        slice_idx = int(round(centroid[slice_axis]))
        idx = [slice(None)] * 3
        idx[slice_axis] = slice_idx
        img_slice = image[tuple(idx)]
        mask_slice = mask[tuple(idx)]

        tag = "0° (original)" if angle == 0 else f"{angle}°"

        # Row 1: rotation alone -- the plain image, no overlay, titled with
        # just the angle (extra pad = a visible gap between label and image).
        ax_img = axes[0, col]
        ax_img.imshow(img_slice.T, cmap="gray", origin="lower")
        ax_img.set_title(tag, fontsize=20, fontweight="bold", pad=16)
        ax_img.axis("off")

        # Row 2: ground truth alone -- the same slice with the true
        # segmentation mask filled in green (fill + crisp edge contour), not
        # a bounding box.
        ax_gt = axes[1, col]
        ax_gt.imshow(img_slice.T, cmap="gray", origin="lower")
        mask_t = mask_slice.T
        rgba = np.zeros((*mask_t.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(MASK_GREEN)
        rgba[..., 3] = np.where(mask_t, 0.55, 0.0)
        ax_gt.imshow(rgba, origin="lower")
        if mask_t.any():
            ax_gt.contour(mask_t, levels=[0.5], colors=MASK_GREEN_DARK, linewidths=1.8)
        ax_gt.axis("off")

    axes[0, 0].axis("on")
    axes[1, 0].axis("on")
    for a in (axes[0, 0], axes[1, 0]):
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel("Rotation", fontsize=22, fontweight="bold", labelpad=18)
    axes[1, 0].set_ylabel("Ground truth", fontsize=22, fontweight="bold", labelpad=18)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.03, hspace=0.30, wspace=wspace)
    fig.suptitle(prompt, fontsize=26, fontweight="bold", y=0.96)

    out_path = OUT_DIR / f"rotation_gt_{dataset_name}_{subject_id}_{organ.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    for ds in DATASETS:
        plot_dataset(PERSON, ds["name"], ds["organ"], ds["col_width"], ds["wspace"])
