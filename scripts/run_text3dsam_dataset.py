"""
Generic Text3DSAM runner for any samples/<dataset>/ manifest built with
common3d.py (currently: amos22_mri, neurite_oasis). For every (subject,
structure) prompt row:
  1. Runs the pretrained Text3DSAM checkpoint on the real image volume +
     real prompt text, thresholds the predicted probability map, and
     computes Dice against the real ground-truth mask.
  2. Records predicted centroid/bbox/anatomical-side (via common3d's
     mask_ground_truth_3d, same method used for the ground truth) into
     samples/<dataset>/predictions_text3dsam.csv.
  3. Saves a publication-style 3-panel (Original / Ground Truth /
     Prediction) figure with the prompt + Dice in the title and a color
     legend, under results_text3d_<tag>/<subject>/<structure>.png.

Both outputs come from the same forward pass, so nothing is run twice.

Usage (from repo root, using the Text3DSAM CPU venv):
    Text3DSAM/.venv/Scripts/python.exe scripts/run_text3dsam_dataset.py amos22_mri
    Text3DSAM/.venv/Scripts/python.exe scripts/run_text3dsam_dataset.py neurite_oasis
    Text3DSAM/.venv/Scripts/python.exe scripts/run_text3dsam_dataset.py neurite_oasis OASIS_OAS1_0014_MR1  # one subject
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monai.transforms as mtf
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT3DSAM_ROOT = REPO_ROOT / "Text3DSAM"
sys.path.insert(0, str(TEXT3DSAM_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.modeling import Med3DSeg  # noqa: E402
from transformers import BertTokenizer  # noqa: E402
from common3d import (  # noqa: E402
    axcodes_for,
    inverse_order,
    mask_ground_truth_3d,
    thin_axis_first_order,
)

MODEL_PATH = TEXT3DSAM_ROOT / "checkpoints" / "Text3DSAM"
THRESHOLD = 0.5

# short tag used for the results_text3d_<tag>/ output folder name
DATASET_TAGS = {
    "amos22_mri": "amos",
    "neurite_oasis": "oasis",
}

GT_COLOR = (0.13, 0.70, 0.30)     # green
PRED_COLOR = (0.85, 0.15, 0.15)   # red

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "figure.titlesize": 13,
    "legend.fontsize": 10,
    "savefig.facecolor": "white",
})


def struct_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "-")


def window(img_slice):
    lo, hi = np.percentile(img_slice, [1, 99])
    if hi <= lo:
        return img_slice
    return np.clip((img_slice - lo) / (hi - lo), 0, 1)


def load_manifest_by_subject(manifest_path):
    subjects = {}
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            subjects.setdefault(row["subject_id"], []).append(row)
    return subjects


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def save_panel_figure(out_path, prompt_text, dice, img_slice, gt_slice, pred_slice):
    h, w = img_slice.shape
    fig, axes = plt.subplots(
        1, 3, figsize=(9.6, 9.6 * h / w / 3 + 0.9),
        gridspec_kw={"wspace": 0.02},
    )
    ax0, ax1, ax2 = axes

    for ax, label in zip(axes, ("Original", "Ground Truth", "Prediction")):
        ax.imshow(img_slice, cmap="gray", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(label, fontsize=11, labelpad=6)

    gt_overlay = np.zeros((*gt_slice.shape, 4))
    gt_overlay[gt_slice] = (*GT_COLOR, 0.5)
    ax1.imshow(gt_overlay, origin="lower")

    pred_overlay = np.zeros((*pred_slice.shape, 4))
    pred_overlay[pred_slice] = (*PRED_COLOR, 0.5)
    ax2.imshow(pred_overlay, origin="lower")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=GT_COLOR, markersize=12, label="Ground Truth"),
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=PRED_COLOR, markersize=12, label="Prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(f"“{prompt_text}”", fontsize=13, fontweight="bold", y=1.06)
    ax1.set_title(f"Dice = {dice:.3f}", fontsize=12, fontweight="bold",
                  color=PRED_COLOR if dice < 0.3 else "#1a1a1a", pad=8)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.16, wspace=0.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATASET_TAGS.keys()))
    parser.add_argument("subjects", nargs="*", help="optional subject_id filter")
    args = parser.parse_args()

    tag = DATASET_TAGS[args.dataset]
    data_dir = REPO_ROOT / "samples" / args.dataset
    manifest_path = data_dir / f"manifest_{args.dataset}.csv"
    pred_csv_path = data_dir / "predictions_text3dsam.csv"
    viz_dir = REPO_ROOT / f"results_text3d_{tag}"
    viz_dir.mkdir(exist_ok=True)

    print(f"[info] loading model from {MODEL_PATH}")
    model = Med3DSeg.from_pretrained(str(MODEL_PATH))
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
    model.eval()

    pre_transforms = mtf.Compose(
        [
            mtf.EnsureChannelFirst(channel_dim="no_channel"),
            mtf.Resize(spatial_size=model.config.image_size, mode="trilinear"),
            mtf.ToTensor(dtype=torch.float32),
        ]
    )

    subjects = load_manifest_by_subject(manifest_path)
    if args.subjects:
        subjects = {k: v for k, v in subjects.items() if k in args.subjects}
    print(f"[info] {len(subjects)} subjects, "
          f"{sum(len(v) for v in subjects.values())} prompts total "
          f"-> predictions: {pred_csv_path}, figures: {viz_dir}")

    out_rows = []
    t_start = time.time()

    for si, (subject_id, rows) in enumerate(subjects.items(), 1):
        image_path = data_dir / rows[0]["image_path"]
        label_path = data_dir / rows[0]["label_path"]

        image_np = nib.load(str(image_path)).get_fdata().astype(np.float32)
        label_vol = nib.load(str(label_path)).get_fdata().astype(int)
        axcodes = axcodes_for(image_path)
        origin_shape = image_np.shape
        slice_axis = int(np.argmin(origin_shape))

        # Reorder axes so the physically thinnest axis lands at position 0,
        # matching the model's fixed (128, 256, 256) resize target (see
        # common3d.thin_axis_first_order for why). Model output is
        # transposed back before scoring/plotting so everything downstream
        # still lives in the original array space that the manifest's
        # ground-truth columns were computed in.
        #
        # NOTE: intensity normalization (percentile-clip to [0,1]) was also
        # tried here and made results dramatically worse (ablated liver
        # case: dice 0.21 -> 0.00, predicted probabilities never exceeded
        # ~0.33) -- the checkpoint apparently expects roughly raw-scale
        # intensities, not [0,1]-normalized ones. Left un-normalized.
        model_axis_order = thin_axis_first_order(origin_shape)
        model_axis_restore = inverse_order(model_axis_order)
        image_for_model = np.transpose(image_np, model_axis_order)
        model_input_shape = image_for_model.shape

        preprocessed = pre_transforms(image_for_model).unsqueeze(0)
        subject_dir = viz_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{si}/{len(subjects)}] {subject_id} shape={origin_shape} "
              f"prompts={len(rows)}")

        for row in rows:
            structure = row["organ"]
            label_id = int(row["label_id"])
            prompt_text = row["prompt_text"]

            prompt_tokens = tokenizer(
                prompt_text, padding="max_length", truncation=True,
                max_length=512, return_tensors="pt",
            )

            t0 = time.time()
            with torch.no_grad():
                logits = model.generate(
                    image=preprocessed,
                    input_ids=prompt_tokens["input_ids"],
                    attention_mask=prompt_tokens["attention_mask"],
                )
                logits = F.interpolate(logits, size=model_input_shape, mode="trilinear")
                pred_prob = torch.sigmoid(logits).squeeze(0).squeeze(0).numpy()
                pred_prob = np.transpose(pred_prob, model_axis_restore)
            elapsed = time.time() - t0

            pred_mask = pred_prob > THRESHOLD
            gt_mask = label_vol == label_id
            dice = dice_score(pred_mask, gt_mask)
            n_voxels = int(pred_mask.sum())

            gt = mask_ground_truth_3d(pred_mask, axcodes) if n_voxels > 0 else None

            out_row = {
                "subject_id": subject_id,
                "split": row["split"],
                "organ": structure,
                "label_id": label_id,
                "prompt_text": prompt_text,
                "volume_shape": row["volume_shape"],
                "n_pred_voxels": n_voxels,
                "dice": round(dice, 4),
                "inference_seconds": round(elapsed, 2),
                "pred_centroid_vox_0": None, "pred_centroid_vox_1": None, "pred_centroid_vox_2": None,
                "pred_bbox_vox_0min": None, "pred_bbox_vox_1min": None, "pred_bbox_vox_2min": None,
                "pred_bbox_vox_0max": None, "pred_bbox_vox_1max": None, "pred_bbox_vox_2max": None,
                "pred_side_axis0": None, "pred_side_axis1": None, "pred_side_axis2": None,
                "gt_centroid_vox_0": row["gt_centroid_vox_0"],
                "gt_centroid_vox_1": row["gt_centroid_vox_1"],
                "gt_centroid_vox_2": row["gt_centroid_vox_2"],
                "gt_side_axis0": row["gt_side_axis0"],
                "gt_side_axis1": row["gt_side_axis1"],
                "gt_side_axis2": row["gt_side_axis2"],
            }
            if gt is not None:
                (cx, cy, cz), bbox, (s0, s1, s2) = gt
                out_row.update({
                    "pred_centroid_vox_0": cx, "pred_centroid_vox_1": cy, "pred_centroid_vox_2": cz,
                    "pred_bbox_vox_0min": bbox[0], "pred_bbox_vox_1min": bbox[1], "pred_bbox_vox_2min": bbox[2],
                    "pred_bbox_vox_0max": bbox[3], "pred_bbox_vox_1max": bbox[4], "pred_bbox_vox_2max": bbox[5],
                    "pred_side_axis0": s0, "pred_side_axis1": s1, "pred_side_axis2": s2,
                })
            out_rows.append(out_row)
            print(f"    {structure:32s} dice={dice:.3f} voxels={n_voxels:8d} ({elapsed:.1f}s)")

            gt_coords = np.nonzero(gt_mask)
            slice_idx = int(round(gt_coords[slice_axis].mean())) if len(gt_coords[0]) else origin_shape[slice_axis] // 2

            def take_slice(vol):
                idx = [slice(None)] * 3
                idx[slice_axis] = slice_idx
                return vol[tuple(idx)].T

            save_panel_figure(
                subject_dir / f"{struct_slug(structure)}.png",
                prompt_text, dice,
                window(take_slice(image_np)), take_slice(gt_mask), take_slice(pred_mask),
            )

    with pred_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[done] wrote {len(out_rows)} predictions -> {pred_csv_path}")
    print(f"[done] wrote {len(out_rows)} figures -> {viz_dir}")
    print(f"[done] total time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
