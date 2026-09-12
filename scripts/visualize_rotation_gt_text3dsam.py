"""
QC plot for the Text3DSAM multi-angle rotation sweep
(scripts/rotate_multiangle_text3dsam.py): for one subject + one organ per
dataset, show the image slice at every rotation angle (0/30/60/90/120/150/180
degrees) with its ground truth overlaid -- so you can eyeball that (a) the
organ visibly moves around the frame as the angle changes, while (b) the
reported anatomical side (gt_side_axis*) stays fixed, exactly as designed.

Same layout as scripts/visualize_rotation_gt.py in the voxtell3d-rotation
branch:
  - row 1 ("Rotation"): the rotated image alone, no overlay, titled with
    just the angle.
  - row 2 ("Ground truth"): the SAME slice with the actual ground-truth
    segmentation mask filled in green (a true mask, not a bounding box).
Both rows are sliced at the organ's ground-truth centroid along that
volume's own Superior/Inferior axis. The prompt is identical at every angle
by design, so it is shown once, at the top of the figure.

One organ is used per dataset (the "right"-side organ of each bilateral
pair, chosen so positional drift is obvious): right kidney (MR_TotalSeg,
CT_TotalSeg_organs), right hip (CT_TotalSeg_muscles), right cochlea
(CT_HaN-Seg), right iliac artery (CT_TotalSeg_cardiac).

Usage:
    python scripts/visualize_rotation_gt_text3dsam.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "samples" / "text3dsam"
OUT_DIR = DATA_ROOT / "visual_check"
OUT_DIR.mkdir(exist_ok=True)

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]

DATASETS = [
    {"name": "MR_TotalSeg", "organ": "right kidney", "col_width": 3.2, "wspace": 0.05},
    {"name": "CT_HaN-Seg", "organ": "right cochlea", "col_width": 3.2, "wspace": 0.05},
    {"name": "CT_TotalSeg_cardiac", "organ": "right iliac artery", "col_width": 3.2, "wspace": 0.05},
]

MASK_GREEN = "#22c55e"
MASK_GREEN_DARK = "#15803d"


def plot_dataset(dataset_name: str, organ: str, col_width: float, wspace: float):
    ds_dir = DATA_ROOT / dataset_name
    manifest_path = ds_dir / f"manifest_{dataset_name}_rotated.csv"
    manifest = pd.read_csv(manifest_path)

    subject_id = manifest["subject_id"].iloc[0]
    rows = manifest[(manifest["subject_id"] == subject_id) & (manifest["organ"] == organ)]
    rows = rows.set_index("rotation_angle_deg").loc[ANGLES_DEG].reset_index()
    if rows.empty:
        raise SystemExit(f"[{dataset_name}] organ '{organ}' not found for subject {subject_id}")

    n = len(ANGLES_DEG)
    prompt = rows.iloc[0]["prompt_text"]

    fig, axes = plt.subplots(2, n, figsize=(col_width * n, 9.0))

    for col, (_, row) in enumerate(rows.iterrows()):
        angle = int(row["rotation_angle_deg"])
        label_id = int(row["label_id"])
        image_path = ds_dir / row["image_path"]
        label_path = ds_dir / row["label_path"]

        img_nii = nib.load(str(image_path))
        image = img_nii.get_fdata().astype(np.float32)
        label = nib.load(str(label_path)).get_fdata().astype(int)
        mask = label == label_id

        axcodes = nib.aff2axcodes(img_nii.affine)
        slice_axis = next(a for a, code in enumerate(axcodes) if code in ("S", "I"))

        centroid = [row["gt_centroid_vox_0"], row["gt_centroid_vox_1"], row["gt_centroid_vox_2"]]
        slice_idx = int(round(centroid[slice_axis]))
        idx = [slice(None)] * 3
        idx[slice_axis] = slice_idx
        img_slice = image[tuple(idx)]
        mask_slice = mask[tuple(idx)]

        tag = "0° (original)" if angle == 0 else f"{angle}°"

        ax_img = axes[0, col]
        ax_img.imshow(img_slice.T, cmap="gray", origin="lower")
        ax_img.set_title(tag, fontsize=18, fontweight="bold", pad=14)
        ax_img.axis("off")

        ax_gt = axes[1, col]
        ax_gt.imshow(img_slice.T, cmap="gray", origin="lower")
        mask_t = mask_slice.T
        rgba = np.zeros((*mask_t.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(MASK_GREEN)
        rgba[..., 3] = np.where(mask_t, 0.55, 0.0)
        ax_gt.imshow(rgba, origin="lower")
        if mask_t.any():
            ax_gt.contour(mask_t, levels=[0.5], colors=MASK_GREEN_DARK, linewidths=1.6)
        ax_gt.axis("off")

    axes[0, 0].axis("on")
    axes[1, 0].axis("on")
    for a in (axes[0, 0], axes[1, 0]):
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel("Rotation", fontsize=20, fontweight="bold", labelpad=16)
    axes[1, 0].set_ylabel("Ground truth", fontsize=20, fontweight="bold", labelpad=16)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.03, hspace=0.30, wspace=wspace)
    fig.suptitle(f"{dataset_name}: {prompt}", fontsize=22, fontweight="bold", y=0.96)

    out_path = OUT_DIR / f"rotation_gt_{dataset_name}_{subject_id}_{organ.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    for ds in DATASETS:
        plot_dataset(ds["name"], ds["organ"], ds["col_width"], ds["wspace"])
