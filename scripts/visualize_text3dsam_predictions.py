"""
Visualize Text3DSAM predictions on AMOS22 MRI:
  1. A per-organ slice grid for one subject: image slice at the organ's
     ground-truth centroid, with the ground-truth mask contour (green) and
     the model's predicted mask contour (red) overlaid, titled with that
     organ's Dice score -- makes positional drift visible at a glance.
  2. A bar chart of mean Dice per organ, averaged across all 10 subjects
     (from the already-computed samples/amos22_mri/predictions_text3dsam.csv).

Reruns the model only for the chosen subject (13 organs, ~30s on CPU) since
predicted mask volumes weren't persisted by the batch run.

Usage:
    Text3DSAM/.venv/Scripts/python.exe scripts/visualize_text3dsam_predictions.py [subject_id]
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
PRED_CSV = DATA_DIR / "predictions_text3dsam.csv"
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "amos_0522"
THRESHOLD = 0.5


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    rows = manifest[manifest["subject_id"] == SUBJECT].reset_index(drop=True)
    if rows.empty:
        raise SystemExit(f"subject {SUBJECT} not found in manifest")

    image_path = DATA_DIR / rows.loc[0, "image_path"]
    label_path = DATA_DIR / rows.loc[0, "label_path"]
    image_np = nib.load(str(image_path)).get_fdata().astype(np.float32)
    label_vol = nib.load(str(label_path)).get_fdata().astype(int)
    origin_shape = image_np.shape
    print(f"[info] {SUBJECT} shape={origin_shape}")

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
    model_axis_order = thin_axis_first_order(origin_shape)
    model_axis_restore = inverse_order(model_axis_order)
    image_for_model = np.transpose(image_np, model_axis_order)
    model_input_shape = image_for_model.shape
    preprocessed = pre_transforms(image_for_model).unsqueeze(0)

    n = len(rows)
    ncols = 5
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    # through-plane axis = the smallest volume dimension
    slice_axis = int(np.argmin(origin_shape))

    for i, row in rows.iterrows():
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
        print(f"  {organ:22s} dice={dice:.3f} ({time.time()-t0:.1f}s)")

        gt_coords = np.nonzero(gt_mask)
        slice_idx = int(round(gt_coords[slice_axis].mean())) if len(gt_coords[0]) else origin_shape[slice_axis] // 2

        def take_slice(vol):
            idx = [slice(None)] * 3
            idx[slice_axis] = slice_idx
            return vol[tuple(idx)]

        img_slice = take_slice(image_np)
        gt_slice = take_slice(gt_mask)
        pred_slice = take_slice(pred_mask)

        ax = axes[i]
        ax.imshow(img_slice.T, cmap="gray", origin="lower")
        if gt_slice.any():
            ax.contour(gt_slice.T, levels=[0.5], colors="lime", linewidths=1.5)
        if pred_slice.any():
            ax.contour(pred_slice.T, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(f"{organ}\ndice={dice:.2f}", fontsize=10)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Text3DSAM on {SUBJECT}  (green=ground truth, red=prediction)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"text3dsam_overlays_{SUBJECT}.png"
    fig.savefig(out_path, dpi=130)
    print(f"[done] wrote {out_path}")

    # Dice-per-organ bar chart across all 10 subjects
    pred_df = pd.read_csv(PRED_CSV)
    summary = pred_df.groupby("organ")["dice"].mean().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    bars = ax2.bar(summary.index, summary.values, color="#4C72B0")
    ax2.set_ylabel("Mean Dice (10 subjects)")
    ax2.set_title("Text3DSAM mean Dice per organ — AMOS22 MRI")
    ax2.set_ylim(0, max(0.05, summary.values.max() * 1.15))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    for b, v in zip(bars, summary.values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    out_path2 = OUT_DIR / "text3dsam_dice_per_organ.png"
    fig2.savefig(out_path2, dpi=130)
    print(f"[done] wrote {out_path2}")


if __name__ == "__main__":
    main()
