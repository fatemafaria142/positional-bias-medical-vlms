"""
Per-subject, per-organ visualization for Text3DSAM on AMOS22 MRI: for every
individual (subject, organ) prompt, one image with three panels side by
side -- Original image / Ground Truth mask / Predicted mask -- with the
Dice score in the title.

Produces one PNG per (subject, organ) pair (samples/amos22_mri manifest)
under results_text3d_amos/<subject>/<organ>.png. Reruns the model (no
masks were persisted by the original batch run) -- ~13 forward passes per
subject, ~2s each on CPU.

Usage:
    Text3DSAM/.venv/Scripts/python.exe scripts/visualize_text3dsam_grid.py [subject_id ...]
    # with no args, runs all subjects in the manifest
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monai.transforms as mtf
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT3DSAM_ROOT = REPO_ROOT / "Text3DSAM"
sys.path.insert(0, str(TEXT3DSAM_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.modeling import Med3DSeg  # noqa: E402
from transformers import BertTokenizer  # noqa: E402
from common3d import inverse_order, thin_axis_first_order  # noqa: E402

MODEL_PATH = TEXT3DSAM_ROOT / "checkpoints" / "Text3DSAM"
DATA_DIR = REPO_ROOT / "samples" / "amos22_mri"
MANIFEST_PATH = DATA_DIR / "manifest_amos22_mri.csv"
OUT_DIR = REPO_ROOT / "results_text3d_amos"
OUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.5

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


def organ_slug(organ: str) -> str:
    return organ.strip().lower().replace(" ", "_")


def window(img_slice):
    lo, hi = np.percentile(img_slice, [1, 99])
    if hi <= lo:
        return img_slice
    return np.clip((img_slice - lo) / (hi - lo), 0, 1)


def render_subject(subject_id, rows, model, tokenizer):
    image_path = DATA_DIR / rows.iloc[0]["image_path"]
    label_path = DATA_DIR / rows.iloc[0]["label_path"]
    image_np = nib.load(str(image_path)).get_fdata().astype(np.float32)
    label_vol = nib.load(str(label_path)).get_fdata().astype(int)
    origin_shape = image_np.shape
    slice_axis = int(np.argmin(origin_shape))

    pre_transforms = mtf.Compose(
        [
            mtf.EnsureChannelFirst(channel_dim="no_channel"),
            mtf.Resize(spatial_size=model.config.image_size, mode="trilinear"),
            mtf.ToTensor(dtype=torch.float32),
        ]
    )
    model_axis_order = thin_axis_first_order(origin_shape)
    model_axis_restore = inverse_order(model_axis_order)
    image_for_model = np.transpose(image_np, model_axis_order)
    model_input_shape = image_for_model.shape
    preprocessed = pre_transforms(image_for_model).unsqueeze(0)

    subject_dir = OUT_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{subject_id}] shape={origin_shape}")
    out_paths = []
    for i, (_, row) in enumerate(rows.iterrows()):
        organ = row["organ"]
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
        pred_mask = pred_prob > THRESHOLD
        gt_mask = label_vol == label_id

        inter = np.logical_and(pred_mask, gt_mask).sum()
        denom = pred_mask.sum() + gt_mask.sum()
        dice = 1.0 if denom == 0 else 2.0 * inter / denom
        print(f"    {organ:22s} dice={dice:.3f} ({time.time()-t0:.1f}s)")

        gt_coords = np.nonzero(gt_mask)
        if len(gt_coords[0]):
            slice_idx = int(round(gt_coords[slice_axis].mean()))
        else:
            slice_idx = origin_shape[slice_axis] // 2

        def take_slice(vol):
            idx = [slice(None)] * 3
            idx[slice_axis] = slice_idx
            return vol[tuple(idx)].T

        img_slice = window(take_slice(image_np))
        gt_slice = take_slice(gt_mask)
        pred_slice = take_slice(pred_mask)

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
        fig.legend(
            handles=legend_handles, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.02), frameon=False,
        )

        fig.suptitle(
            f"“{prompt_text}”",
            fontsize=13, fontweight="bold", y=1.06,
        )
        ax1.set_title(f"Dice = {dice:.3f}", fontsize=12, fontweight="bold",
                       color=PRED_COLOR if dice < 0.3 else "#1a1a1a", pad=8)

        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.16, wspace=0.02)
        out_path = subject_dir / f"{organ_slug(organ)}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        out_paths.append(out_path)

    print(f"[done] wrote {len(out_paths)} organ images -> {subject_dir}")
    return out_paths


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    requested = sys.argv[1:]
    subject_ids = requested if requested else list(manifest["subject_id"].unique())

    print(f"[info] loading model from {MODEL_PATH}")
    model = Med3DSeg.from_pretrained(str(MODEL_PATH))
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
    model.eval()

    all_paths = []
    for subject_id in subject_ids:
        rows = manifest[manifest["subject_id"] == subject_id].reset_index(drop=True)
        if rows.empty:
            print(f"[warn] subject {subject_id} not found, skipping")
            continue
        all_paths.extend(render_subject(subject_id, rows, model, tokenizer))

    print(f"[all done] wrote {len(all_paths)} organ images under {OUT_DIR}")


if __name__ == "__main__":
    main()
