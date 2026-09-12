"""
Extends scripts/visualize_rotation_gt.py with a third row: the VoxTell
PREDICTED mask (from predictions_voxtell_<dataset>_rotation_<person>.csv,
produced by the Kaggle notebooks voxtell-*-rotation-multiangle-kaggle.ipynb),
alongside the same "Rotation" and "Ground truth" rows as before -- same
subject/organ/angles, same layout and styling, one column per angle
(0/30/60/90/120/150/180 degrees).

Row 1 ("Rotation"): the rotated image alone, no overlay.
Row 2 ("Ground truth"): the same slice with the true segmentation mask
    filled in green, from the per-annotator manifest (as before).
Row 3 ("Prediction"): the SAME image slice as rows 1-2, with VoxTell's
    PREDICTED mask filled in red. The predicted mask is decoded from the
    prediction CSV's pred_mask_rle_b64, which lives in that CSV's own
    RAS-reoriented + transposed voxel grid (see decode_mask_rle() /
    canonical_affine() below) -- a DIFFERENT array orientation than the
    manifest's raw image array used for rows 1-2. Reusing rows 1-2's own
    image_slice as-is under that mask (or independently reslicing the
    RAS-oriented image for row 3) would show the mask on a differently
    rotated/flipped background than rows 1-2. Instead the predicted mask is
    resampled (nibabel.processing.resample_from_to, nearest-neighbor) from
    its own RAS-reoriented grid onto the manifest image's raw voxel grid,
    so it can be sliced with the exact same slice_axis/slice_idx and drawn
    over the exact same img_slice as rows 1-2 -- guaranteeing identical
    orientation, framing, and zoom across all three rows.

Also saves the raw predicted masks (one .nii.gz per angle, in the same RAS
voxel grid/affine as the loaded image) so the actual predicted mask itself
-- not just the plot -- is kept for inspection.

Plots EVERY (subject, organ) pair present in both the manifest and the
prediction CSV for each dataset (all subjects, all organs) -- not just one
hand-picked organ.

Output: visual_check_<person>/<dataset_name>/
    rotation_gt_pred_<dataset_name>_<subject_id>_<organ>.png
    pred_masks/<subject_id>_rot<angle>_<organ>.nii.gz

Usage:
    python scripts/visualize_rotation_gt_pred.py [person]
        person: faria (default) | nipu | hridoy
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
import nibabel.processing
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]

PERSON = sys.argv[1] if len(sys.argv) > 1 else "faria"

# col_width/wspace tuned per dataset so portrait-ish brain slices (neurite_oasis)
# don't sit in an over-wide panel with lots of side whitespace between columns.
DATASETS = [
    {"name": "amos22_mri", "col_width": 4.0, "wspace": 0.05,
     "pred_csv": REPO_ROOT / "predictions_voxtell_amos22_mri_rotation_faria.csv"},
    {"name": "neurite_oasis", "col_width": 2.7, "wspace": 0.015,
     "pred_csv": REPO_ROOT / "predictions_voxtell_neurite_oasis_rotation_faria.csv"},
]

MASK_GREEN = "#22c55e"
MASK_GREEN_DARK = "#15803d"
MASK_RED = "#ef4444"
MASK_RED_DARK = "#b91c1c"
DICE_NONZERO_COLOR = "#FF7145"  # Dice > 0 (some real overlap)
DICE_ZERO_COLOR = "#1E74FF"     # Dice == 0 (total miss)


def decode_mask_rle(encoded: str, shape) -> np.ndarray:
    """Inverse of the notebooks' encode_mask_rle(); shape = volume_shape_ras split on 'x'."""
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


def canonical_affine(path: Path):
    """The affine of the RAS-reoriented (nib.as_closest_canonical) version of
    a volume, without decoding its pixel data -- this is the affine that
    pairs with volume_shape_ras / the decoded masks (see module docstring),
    cheap to compute since it only needs the header, not get_fdata()."""
    return nib.as_closest_canonical(nib.load(str(path))).affine


def overlay_mask(ax, mask_t, color, color_dark, aspect="equal"):
    rgba = np.zeros((*mask_t.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.where(mask_t, 0.55, 0.0)
    ax.imshow(rgba, origin="lower", aspect=aspect)
    if mask_t.any():
        ax.contour(mask_t, levels=[0.5], colors=color_dark, linewidths=1.8)


def plot_organ(person: str, dataset_name: str, subject_id: str, organ: str,
               col_width: float, wspace: float, manifest: pd.DataFrame, pred_df: pd.DataFrame,
               data_dir: Path, out_dir: Path, mask_dir: Path, volume_cache: dict):
    rows = manifest[(manifest["subject_id"] == subject_id) & (manifest["organ"] == organ)]
    rows = rows.set_index("rotation_angle_deg").reindex(ANGLES_DEG).dropna(how="all").reset_index()
    if rows.empty or set(ANGLES_DEG) - set(rows["rotation_angle_deg"].astype(int)):
        print(f"[skip] [{dataset_name}] {subject_id}/{organ}: missing angle(s) in manifest")
        return

    pred_rows = pred_df[(pred_df["subject_id"] == subject_id) & (pred_df["organ"] == organ)]
    pred_rows = pred_rows.set_index("rotation_angle_deg").reindex(ANGLES_DEG).dropna(how="all").reset_index()
    if pred_rows.empty or set(ANGLES_DEG) - set(pred_rows["rotation_angle_deg"].astype(int)):
        print(f"[skip] [{dataset_name}] {subject_id}/{organ}: missing angle(s) in predictions")
        return

    n = len(ANGLES_DEG)
    prompt = rows.iloc[0]["prompt_text"]  # identical at every angle by design

    # Height is fixed (independent of col_width) so title/label font sizes
    # take up the same vertical fraction of the figure regardless of how
    # narrow the per-dataset columns are.
    fig, axes = plt.subplots(3, n, figsize=(col_width * n, 13.0))

    for col, ((_, row), (_, pred_row)) in enumerate(zip(rows.iterrows(), pred_rows.iterrows())):
        angle = int(row["rotation_angle_deg"])
        label_id = int(row["label_id"])
        image_path = data_dir / row["image_path"]
        label_path = data_dir / row["label_path"]

        # Cache raw image/label arrays by path -- the same (subject, angle)
        # volume is shared across every organ, so without this every organ
        # in a subject re-decompresses/reloads the same .nii.gz from scratch.
        cache_key = str(image_path)
        if cache_key not in volume_cache:
            img_nii = nib.load(str(image_path))
            image = img_nii.get_fdata().astype(np.float32)
            axcodes = nib.aff2axcodes(img_nii.affine)
            # Always slice along the volume's own S/I axis (the axis rotation
            # is applied about) so the sweep is visible -- the smallest-dimension
            # axis used for model input elsewhere isn't necessarily the S/I axis
            # (e.g. neurite_oasis has S/I at axis 1, not the thinnest axis 0).
            slice_axis = next(a for a, code in enumerate(axcodes) if code in ("S", "I"))
            label = nib.load(str(label_path)).get_fdata().astype(int)
            volume_cache[cache_key] = (img_nii, image, label, slice_axis)
        img_nii, image, label, slice_axis = volume_cache[cache_key]
        mask = label == label_id

        centroid = [row["gt_centroid_vox_0"], row["gt_centroid_vox_1"], row["gt_centroid_vox_2"]]
        slice_idx = int(round(centroid[slice_axis]))
        idx = [slice(None)] * 3
        idx[slice_axis] = slice_idx
        img_slice = image[tuple(idx)]
        mask_slice = mask[tuple(idx)]

        tag = "0° (original)" if angle == 0 else f"{angle}°"

        # Row 1: rotation alone -- the plain image, no overlay.
        ax_img = axes[0, col]
        ax_img.imshow(img_slice.T, cmap="gray", origin="lower")
        ax_img.set_title(tag, fontsize=26, fontweight="bold", pad=18)
        ax_img.axis("off")

        # Row 2: ground truth -- same slice, true mask in green.
        ax_gt = axes[1, col]
        ax_gt.imshow(img_slice.T, cmap="gray", origin="lower")
        overlay_mask(ax_gt, mask_slice.T, MASK_GREEN, MASK_GREEN_DARK)
        ax_gt.axis("off")

        # Row 3: VoxTell prediction. pred_mask_full is decoded in the
        # predictions CSV's own RAS-reoriented + transposed (Z, Y, X) grid --
        # a different array orientation than `image`/`mask` above (raw,
        # un-reoriented). Resample it (nearest-neighbor) onto the raw image's
        # own grid so it can be drawn on the SAME img_slice used by rows 1-2,
        # guaranteeing identical orientation/framing across all three rows.
        pred_shape = tuple(int(s) for s in pred_row["volume_shape_ras"].split("x"))
        pred_mask_full = decode_mask_rle(pred_row["pred_mask_rle_b64"], pred_shape)

        affine_cache_key = ("canonical_affine", pred_row["image_path"])
        if affine_cache_key not in volume_cache:
            volume_cache[affine_cache_key] = canonical_affine(REPO_ROOT / pred_row["image_path"])
        pred_affine = volume_cache[affine_cache_key]

        # Undo the (Z, Y, X) transpose to get back to canonical (X, Y, Z)
        # order (matching pred_affine) before wrapping it as a NIfTI image.
        pred_mask_canon_img = nib.Nifti1Image(
            pred_mask_full.transpose(2, 1, 0).astype(np.uint8), pred_affine)
        pred_mask_raw = nibabel.processing.resample_from_to(
            pred_mask_canon_img, img_nii, order=0).get_fdata().astype(bool)
        pred_mask_slice = pred_mask_raw[tuple(idx)]

        ax_pred = axes[2, col]
        ax_pred.imshow(img_slice.T, cmap="gray", origin="lower")
        overlay_mask(ax_pred, pred_mask_slice.T, MASK_RED, MASK_RED_DARK)
        dice = pred_row["dice"]
        # Orange = non-zero Dice (some real overlap), blue = Dice == 0 (total
        # miss) -- deliberately NOT green/red, since those are already the
        # mask colors (green=ground truth, red=prediction) and reusing them
        # here for the score itself would be confusing.
        dice_color = DICE_NONZERO_COLOR if dice > 0 else DICE_ZERO_COLOR
        ax_pred.set_title(f"Dice = {dice:.3f}", fontsize=20, fontweight="bold",
                           color="white", pad=14,
                           bbox=dict(boxstyle="round,pad=0.35", facecolor=dice_color,
                                     edgecolor="none", alpha=0.95))
        ax_pred.axis("off")

        # Keep the actual predicted mask, not just the plot.
        organ_slug = organ.replace(" ", "_")
        nib.save(pred_mask_canon_img, str(mask_dir / f"{subject_id}_rot{angle:03d}_{organ_slug}.nii.gz"))

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


def plot_dataset(person: str, dataset_name: str, col_width: float, wspace: float, pred_csv: Path):
    data_dir = REPO_ROOT / person / "samples" / dataset_name
    manifest_path = data_dir / f"manifest_{dataset_name}.csv"
    manifest = pd.read_csv(manifest_path)
    pred_df = pd.read_csv(pred_csv)

    out_dir = REPO_ROOT / f"visual_check_{person}" / dataset_name
    mask_dir = out_dir / "pred_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    pairs = (pred_df[["subject_id", "organ"]]
             .drop_duplicates()
             .merge(manifest[["subject_id", "organ"]].drop_duplicates(), on=["subject_id", "organ"]))

    volume_cache = {}  # path -> decoded arrays, shared across organs of the same subject/angle
    print(f"[{dataset_name}] plotting {len(pairs)} (subject, organ) pairs...")
    for i, (subject_id, organ) in enumerate(pairs.itertuples(index=False), 1):
        print(f"[{dataset_name}] ({i}/{len(pairs)}) {subject_id} / {organ}", flush=True)
        plot_organ(person, dataset_name, subject_id, organ, col_width, wspace,
                   manifest, pred_df, data_dir, out_dir, mask_dir, volume_cache)

    print(f"[done] {dataset_name}: wrote plots -> {out_dir}, predicted masks -> {mask_dir}")


if __name__ == "__main__":
    for ds in DATASETS:
        plot_dataset(PERSON, ds["name"], ds["col_width"], ds["wspace"], ds["pred_csv"])
