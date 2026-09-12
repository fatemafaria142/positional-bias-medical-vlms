"""
Text3DSAM version of scripts/visualize_rotation_gt_pred.py (the VoxTell
rotation+prediction visual check) -- SAME 3-row x 7-angle-column layout,
SAME colors, SAME Dice badge, SAME legend, SAME output naming convention.
Ported rather than reused as-is because two things differ from VoxTell:

  1. One combined predictions_text3dsam_<person>.csv covers all 3 datasets
     (VoxTell has one CSV per dataset), so it's filtered by the `dataset`
     column instead of loaded per-dataset.
  2. That single CSV is self-sufficient -- every row already carries the
     ground-truth fields (gt_centroid_vox_*, image_path, label_path,
     prompt_text, label_id, axcodes) alongside the prediction fields
     (pred_mask_rle_b64, dice), so there's no separate per-dataset manifest
     to join against.
  3. No RAS-reorientation/resampling step: unlike VoxTell (whose predictor
     works in a canonically-reoriented, transposed grid different from the
     manifest's raw image array), Text3DSAM's image, ground-truth mask, and
     predicted mask are all already in the SAME native array space (the
     inference notebook never reorients), so the predicted mask can be
     sliced directly with the same slice_axis/slice_idx as the other two
     rows -- no nibabel.processing.resample_from_to needed.

Row 1 ("Rotation"): the rotated image alone, no overlay.
Row 2 ("Ground truth"): the same slice with the true mask filled green.
Row 3 ("Prediction"): the same slice with Text3DSAM's predicted mask filled
    red, titled with a Dice badge (orange = Dice > 0, blue = Dice == 0).

Also saves the raw predicted masks (one .nii.gz per angle, same affine as
the source image, since no reorientation is involved here) for inspection.

Plots EVERY (subject, organ) pair present in the predictions CSV for each
dataset -- not just one hand-picked case.

Output: visual_check_<person>/<dataset_name>/
    rotation_gt_pred_<dataset_name>_<subject_id>_<organ>.png
    pred_masks/<subject_id>_rot<angle>_<organ>.nii.gz

Usage:
    python scripts/visualize_rotation_gt_pred_text3dsam.py [person]
        person: faria (default)
"""

import base64
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]

PERSON = sys.argv[1] if len(sys.argv) > 1 else "faria"
PREDICTIONS_CSV = REPO_ROOT / f"predictions_text3dsam_{PERSON}.csv"

DATASETS = [
    {"name": "MR_TotalSeg", "col_width": 3.4, "wspace": 0.03},
    {"name": "CT_HaN-Seg", "col_width": 3.4, "wspace": 0.03},
    {"name": "CT_TotalSeg_cardiac", "col_width": 3.4, "wspace": 0.03},
]

MASK_GREEN = "#22c55e"
MASK_GREEN_DARK = "#15803d"
MASK_RED = "#ef4444"
MASK_RED_DARK = "#b91c1c"
DICE_NONZERO_COLOR = "#FF7145"  # Dice > 0 (some real overlap)
DICE_ZERO_COLOR = "#1E74FF"     # Dice == 0 (total miss)


def decode_mask_rle(encoded: str, shape) -> np.ndarray:
    """Inverse of the Kaggle notebook's encode_mask_rle(); shape = volume_shape_ras split on 'x'."""
    runs = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    idx = 0
    value = False
    for run_len in runs:
        if value and run_len:
            flat[idx:idx + run_len] = True
        idx += int(run_len)
        value = not value
    return flat.reshape(shape, order="C")


def overlay_mask(ax, mask_t, color, color_dark, aspect="equal"):
    rgba = np.zeros((*mask_t.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.where(mask_t, 0.55, 0.0)
    ax.imshow(rgba, origin="lower", aspect=aspect)
    if mask_t.any():
        ax.contour(mask_t, levels=[0.5], colors=color_dark, linewidths=1.8)


def plot_organ(dataset_name: str, subject_id: str, organ: str, col_width: float, wspace: float,
               df: pd.DataFrame, out_dir: Path, mask_dir: Path, volume_cache: dict):
    rows = df[(df["dataset"] == dataset_name) & (df["subject_id"] == subject_id) & (df["organ"] == organ)]
    rows = rows.set_index("rotation_angle_deg").reindex(ANGLES_DEG).dropna(how="all").reset_index()
    if rows.empty or set(ANGLES_DEG) - set(rows["rotation_angle_deg"].astype(int)):
        print(f"[skip] [{dataset_name}] {subject_id}/{organ}: missing angle(s)")
        return

    n = len(ANGLES_DEG)
    prompt = rows.iloc[0]["prompt_text"]  # identical at every angle by design

    # Height is fixed (independent of col_width) so title/label font sizes
    # take up the same vertical fraction of the figure regardless of how
    # narrow the per-dataset columns are.
    fig, axes = plt.subplots(3, n, figsize=(col_width * n, 13.0))

    for col, (_, row) in enumerate(rows.iterrows()):
        angle = int(row["rotation_angle_deg"])
        label_id = int(row["label_id"])
        image_path = REPO_ROOT / row["image_path"]
        label_path = REPO_ROOT / row["label_path"]
        shape = tuple(int(s) for s in row["volume_shape_ras"].split("x"))

        # Cache raw image/label arrays by path -- the same (subject, angle)
        # volume is shared across every organ, so without this every organ
        # in a subject re-decompresses/reloads the same .nii.gz from scratch.
        cache_key = str(image_path)
        if cache_key not in volume_cache:
            img_nii = nib.load(str(image_path))
            image = np.asarray(img_nii.dataobj).astype(np.float32)
            axcodes = tuple(row["axcodes"])
            slice_axis = next(a for a, code in enumerate(axcodes) if code in ("S", "I"))
            volume_cache[cache_key] = (img_nii, image, slice_axis)
        img_nii, image, slice_axis = volume_cache[cache_key]

        gt_mask = decode_mask_rle(row["gt_mask_rle_b64"], shape)
        pred_mask = decode_mask_rle(row["pred_mask_rle_b64"], shape)

        centroid = [row["gt_centroid_vox_0"], row["gt_centroid_vox_1"], row["gt_centroid_vox_2"]]
        slice_idx = int(round(centroid[slice_axis]))
        idx = [slice(None)] * 3
        idx[slice_axis] = slice_idx
        img_slice = image[tuple(idx)]
        gt_slice = gt_mask[tuple(idx)]
        pred_slice = pred_mask[tuple(idx)]

        tag = "0° (original)" if angle == 0 else f"{angle}°"

        # Row 1: rotation alone -- the plain image, no overlay.
        ax_img = axes[0, col]
        ax_img.imshow(img_slice.T, cmap="gray", origin="lower")
        ax_img.set_title(tag, fontsize=26, fontweight="bold", pad=18)
        ax_img.axis("off")

        # Row 2: ground truth -- same slice, true mask in green.
        ax_gt = axes[1, col]
        ax_gt.imshow(img_slice.T, cmap="gray", origin="lower")
        overlay_mask(ax_gt, gt_slice.T, MASK_GREEN, MASK_GREEN_DARK)
        ax_gt.axis("off")

        # Row 3: Text3DSAM prediction -- same img_slice as rows 1-2 (no
        # resampling needed: image/GT/prediction all share one native grid).
        ax_pred = axes[2, col]
        ax_pred.imshow(img_slice.T, cmap="gray", origin="lower")
        overlay_mask(ax_pred, pred_slice.T, MASK_RED, MASK_RED_DARK)
        dice = row["dice"]
        # Orange = non-zero Dice (some real overlap), blue = Dice == 0 (total
        # miss) -- deliberately NOT green/red, since those are already the
        # mask colors (green=ground truth, red=prediction).
        dice_color = DICE_NONZERO_COLOR if dice > 0 else DICE_ZERO_COLOR
        ax_pred.set_title(f"Dice = {dice:.3f}", fontsize=20, fontweight="bold",
                           color="white", pad=14,
                           bbox=dict(boxstyle="round,pad=0.35", facecolor=dice_color,
                                     edgecolor="none", alpha=0.95))
        ax_pred.axis("off")

        # Keep the actual predicted mask, not just the plot.
        organ_slug = organ.replace(" ", "_")
        pred_mask_nii = nib.Nifti1Image(pred_mask.astype(np.uint8), img_nii.affine)
        nib.save(pred_mask_nii, str(mask_dir / f"{subject_id}_rot{angle:03d}_{organ_slug}.nii.gz"))

    for r, label in enumerate(("Rotation", "Ground truth", "Prediction")):
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
        axes[r, 0].set_ylabel(label, fontsize=30, fontweight="bold", labelpad=22)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=MASK_GREEN,
                   markeredgecolor=MASK_GREEN_DARK, markersize=22, label="Ground truth mask"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=MASK_RED,
                   markeredgecolor=MASK_RED_DARK, markersize=22, label="Predicted mask"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=DICE_NONZERO_COLOR,
                   markersize=22, label="Dice score > 0 (some overlap)"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=DICE_ZERO_COLOR,
                   markersize=22, label="Dice score = 0 (no overlap)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=17,
               bbox_to_anchor=(0.5, -0.015), frameon=False, columnspacing=1.8, handletextpad=0.6)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.89, bottom=0.05, hspace=0.28, wspace=wspace)
    fig.suptitle(prompt, fontsize=28, fontweight="bold", y=0.97)

    out_path = out_dir / f"rotation_gt_pred_{dataset_name}_{subject_id}_{organ.replace(' ', '_')}.png"
    # bbox_inches="tight" expands the saved canvas to include the legend
    # below the axes -- without it, the legend (placed via bbox_to_anchor at
    # a negative y, i.e. below the subplot area) gets clipped off in the PNG.
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def plot_dataset(dataset_name: str, col_width: float, wspace: float, df: pd.DataFrame):
    ds_df = df[df["dataset"] == dataset_name]

    out_dir = REPO_ROOT / f"visual_check_{PERSON}" / dataset_name
    mask_dir = out_dir / "pred_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    pairs = ds_df[["subject_id", "organ"]].drop_duplicates()

    volume_cache = {}  # path -> decoded arrays, shared across organs of the same subject/angle
    print(f"[{dataset_name}] plotting {len(pairs)} (subject, organ) pairs...")
    for i, (subject_id, organ) in enumerate(pairs.itertuples(index=False), 1):
        print(f"[{dataset_name}] ({i}/{len(pairs)}) {subject_id} / {organ}", flush=True)
        plot_organ(dataset_name, subject_id, organ, col_width, wspace, df, out_dir, mask_dir, volume_cache)

    print(f"[done] {dataset_name}: wrote plots -> {out_dir}, predicted masks -> {mask_dir}")


if __name__ == "__main__":
    if not PREDICTIONS_CSV.exists():
        raise SystemExit(f"Not found: {PREDICTIONS_CSV}")
    df = pd.read_csv(PREDICTIONS_CSV)
    for ds in DATASETS:
        plot_dataset(ds["name"], ds["col_width"], ds["wspace"], df)
