"""
Text3DSAM visual check -- one 3-panel (Original / Ground Truth / Prediction)
figure per organ, for one representative subject per dataset, saved under
each dataset's own samples/text3dsam/<dataset>/visual_check/ folder --
mirroring this repo's existing per-dataset visual_check convention
(samples/isic/visual_check/, etc.) and the exact panel format the VoxTell
Kaggle notebook's save_panel_figure() produces: original image, ground
truth mask filled green, predicted mask filled red, Dice score in the
title.

The representative subject per dataset is whichever one has the MOST
organs actually annotated at rotation_angle_deg == 0 (the un-rotated
baseline) -- the most complete case available, so the check covers as many
organs as possible in one go:
    CT_HaN-Seg:            case_01              (8 organs)
    CT_TotalSeg_cardiac:   s0011                (10 organs)
    MR_TotalSeg:           s0019                (8 organs)

CT_TotalSeg_cardiac has 5 subjects total, unlike the other two datasets
(one representative case each is all there is). Its other 4 subjects
(s0000, s0034, s0066, s0069) are additionally checked here, one
subfolder per subject under samples/text3dsam/CT_TotalSeg_cardiac/visual_check/,
so every case in that dataset gets a look, not just the most-complete one.

Usage:
    python scripts/visual_check_text3dsam.py
"""

import base64
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "samples" / "text3dsam"
PREDICTIONS_CSV = REPO_ROOT / "predictions_text3dsam_faria.csv"

GT_COLOR = (0.13, 0.70, 0.30)
PRED_COLOR = (0.85, 0.15, 0.15)

DATASETS = ["MR_TotalSeg", "CT_HaN-Seg", "CT_TotalSeg_cardiac"]


def decode_mask_rle(encoded: str, shape) -> np.ndarray:
    """Inverse of the Kaggle notebook's encode_mask_rle(): a zlib-compressed,
    base64-encoded run-length encoding of a boolean mask, C-order flattened."""
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


def struct_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "-")


def window(img_slice):
    lo, hi = np.percentile(img_slice, [1, 99])
    if hi <= lo:
        return img_slice
    return np.clip((img_slice - lo) / (hi - lo), 0, 1)


def most_complete_subject(df: pd.DataFrame, dataset_name: str) -> str:
    baseline = df[(df["dataset"] == dataset_name) & (df["rotation_angle_deg"] == 0)]
    counts = baseline.groupby("subject_id").size().sort_values(ascending=False)
    return counts.index[0]


def save_panel_figure(out_path, prompt_text, dice, img_slice, gt_slice, pred_slice):
    h, w = img_slice.shape
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 9.6 * h / w / 3 + 0.9),
                              gridspec_kw={"wspace": 0.02})
    for ax, label in zip(axes, ("Original", "Ground Truth", "Prediction")):
        ax.imshow(img_slice, cmap="gray", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(label, fontsize=11, labelpad=6)

    gt_overlay = np.zeros((*gt_slice.shape, 4))
    gt_overlay[gt_slice] = (*GT_COLOR, 0.5)
    axes[1].imshow(gt_overlay, origin="lower")

    pred_overlay = np.zeros((*pred_slice.shape, 4))
    pred_overlay[pred_slice] = (*PRED_COLOR, 0.5)
    axes[2].imshow(pred_overlay, origin="lower")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=GT_COLOR, markersize=12, label="Ground Truth"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=PRED_COLOR, markersize=12, label="Prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f'"{prompt_text}"', fontsize=13, fontweight="bold", y=1.06)
    ax_title_color = PRED_COLOR if dice < 0.3 else "#1a1a1a"
    axes[1].set_title(f"Dice = {dice:.3f}", fontsize=12, fontweight="bold", color=ax_title_color, pad=8)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.16, wspace=0.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


def visual_check_subject(rows: pd.DataFrame, out_dir: Path, dataset_name: str, subject_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name}] subject {subject_id}: {len(rows)} organ(s)")
    for _, row in rows.iterrows():
        image_path = REPO_ROOT / row["image_path"]
        shape = tuple(int(s) for s in row["volume_shape_ras"].split("x"))

        img = np.asarray(nib.load(str(image_path)).dataobj).astype(np.float32)
        gt_mask = decode_mask_rle(row["gt_mask_rle_b64"], shape)
        pred_mask = decode_mask_rle(row["pred_mask_rle_b64"], shape)

        axcodes = tuple(row["axcodes"])
        slice_axis = next((a for a, code in enumerate(axcodes) if code in ("S", "I")), int(np.argmin(shape)))
        gt_coords = np.nonzero(gt_mask)
        slice_idx = int(round(gt_coords[slice_axis].mean())) if len(gt_coords[0]) else shape[slice_axis] // 2

        def take_slice(vol, ax=slice_axis, idx=slice_idx):
            sl = [slice(None)] * 3
            sl[ax] = idx
            return vol[tuple(sl)].T

        out_path = out_dir / f"{struct_slug(row['organ'])}.png"
        save_panel_figure(out_path, row["prompt_text"], row["dice"],
                           window(take_slice(img)), take_slice(gt_mask), take_slice(pred_mask))
        print(f"    {row['organ']}: dice={row['dice']:.3f} -> {out_path.relative_to(REPO_ROOT)}")


def visual_check_dataset(df: pd.DataFrame, dataset_name: str):
    subject_id = most_complete_subject(df, dataset_name)
    rows = df[(df["dataset"] == dataset_name) & (df["subject_id"] == subject_id)
              & (df["rotation_angle_deg"] == 0)]
    out_dir = DATA_ROOT / dataset_name / "visual_check"
    visual_check_subject(rows, out_dir, dataset_name, subject_id)


def visual_check_remaining_subjects(df: pd.DataFrame, dataset_name: str):
    """Check every other subject in dataset_name beyond the one representative
    case visual_check_dataset() already covers, one subfolder per subject."""
    representative = most_complete_subject(df, dataset_name)
    baseline = df[(df["dataset"] == dataset_name) & (df["rotation_angle_deg"] == 0)]
    other_subjects = sorted(s for s in baseline["subject_id"].unique() if s != representative)

    for subject_id in other_subjects:
        rows = baseline[baseline["subject_id"] == subject_id]
        out_dir = DATA_ROOT / dataset_name / "visual_check" / subject_id
        visual_check_subject(rows, out_dir, dataset_name, subject_id)


if __name__ == "__main__":
    df = pd.read_csv(PREDICTIONS_CSV)
    for ds in DATASETS:
        visual_check_dataset(df, ds)
    visual_check_remaining_subjects(df, "CT_TotalSeg_cardiac")
