"""
For the FIRST subject of a dataset and each of its organs: three real scan images side by side, on the same image slice (the slice through the
centre of the true organ):

  1. the original image
  2. the ground truth (the true organ, outlined by experts) drawn on the image
  3. VoxTell's mask (segmentation) drawn on the image together with the ground-truth outline, and their Dice overlap

VoxTell's mask is the unmodified prediction for the request "Segment the <organ> in this image." (baseline_predictions_*.csv from the
groundedness notebooks). The mask has to be decoded and resampled onto the scan exactly as in plot_groundedness_results.py.

Output: local_plots_groundedness/word_highlight/<AMOS|OASIS>/<organ>/rows13_16/subject_<N>/image_original_ground_truth_voxtell_mask.png
Usage:  python scripts/make_organ_image_figures.py [OASIS] [AMOS]        (default: OASIS)
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_row14_figures as m  # noqa: E402  (dataset configuration, subject numbers, fonts)
import plot_groundedness_results as pg  # noqa: E402  (decode_mask_rle, canonical_affine)

GT_C, GT_DARK = "#1baf7a", "#0b6b49"          # ground truth: green
PRED_C, PRED_DARK = "#e34948", "#9c1f1e"      # VoxTell's mask: red
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9993"


def overlay(ax, mask, color, alpha=0.55):
    rgba = np.zeros((*mask.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    ax.imshow(rgba, origin="lower")


def figure(name, sid, num, organ, row):
    image_path = m.REPO / row["image_path"]
    label_path = m.REPO / row["label_path"]
    img_nii = nib.load(str(image_path))
    image = img_nii.get_fdata().astype(np.float32)
    axcodes = nib.aff2axcodes(img_nii.affine)
    slice_axis = next((a for a, code in enumerate(axcodes) if code in ("S", "I")), int(np.argmin(image.shape)))
    label = nib.load(str(label_path)).get_fdata().astype(int)
    gt = label == int(row["label_id"])
    coords = np.nonzero(gt)
    idx = [slice(None)] * 3
    idx[slice_axis] = int(round(coords[slice_axis].mean())) if len(coords[0]) else image.shape[slice_axis] // 2
    idx = tuple(idx)
    shape = tuple(int(s) for s in row["volume_shape_ras"].split("x"))
    pred_full = pg.decode_mask_rle(row["pred_mask_rle_b64"], shape)
    pred_img = nib.Nifti1Image(pred_full.transpose(2, 1, 0).astype(np.uint8), pg.canonical_affine(image_path))
    pred = nib.processing.resample_from_to(pred_img, img_nii, order=0).get_fdata().astype(bool)

    img2, gt2, pred2 = image[idx].T, gt[idx].T, pred[idx].T
    body = np.nonzero(img2 > 0.04 * img2.max())                        # crop the black margin around the head / body
    pad_px = 6
    r0, r1 = max(body[0].min() - pad_px, 0), min(body[0].max() + pad_px + 1, img2.shape[0])
    c0, c1 = max(body[1].min() - pad_px, 0), min(body[1].max() + pad_px + 1, img2.shape[1])
    img2, gt2, pred2 = img2[r0:r1, c0:c1], gt2[r0:r1, c0:c1], pred2[r0:r1, c0:c1]
    rem = [a for a in range(3) if a != slice_axis]                   # after .T: horizontal = rem[0], vertical = rem[1]
    h_code = axcodes[rem[0]]
    left_label, right_label = ("R", "L") if h_code == "L" else ("L", "R")   # letter = side of the PATIENT, shown at the image edge
    dice = float(row["dice_gt"])

    pw, gap, margin = 4.1, 0.14, 0.15                                  # panel width, gap between panels, side margin (inches)
    fig_w = 3 * pw + 2 * gap + 2 * margin
    ph = pw * img2.shape[0] / img2.shape[1]                            # panel height follows the shape of the slice
    top_h, bot_h = 1.05, 0.5
    fig_h = top_h + ph + bot_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=170)
    fig.patch.set_facecolor("#fcfcfb")
    Y = lambda inch_from_top: 1 - inch_from_top / fig_h
    fig.text(0.5, Y(0.1), m.TITLE, ha="center", va="top", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.5, Y(0.52), f"Input given to VoxTell: \"{row['prompt_text']}\"", ha="center", va="top", fontsize=13, color=INK2)
    titles = ["Original image", "Ground truth", "VoxTell's mask and the ground truth"]
    x_of = lambda i: (margin + i * (pw + gap)) / fig_w
    axes = [fig.add_axes([x_of(i), bot_h / fig_h, pw / fig_w, ph / fig_h]) for i in range(3)]
    for ax, t in zip(axes, titles):
        ax.imshow(img2, cmap="gray", origin="lower")
        ax.set_title(t, fontsize=12.5, fontweight="bold", color=INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.02, 0.5, left_label, transform=ax.transAxes, color="white", fontsize=11, fontweight="bold", va="center")
        ax.text(0.98, 0.5, right_label, transform=ax.transAxes, color="white", fontsize=11, fontweight="bold", va="center", ha="right")
    overlay(axes[1], gt2, GT_C)
    if gt2.any():
        axes[1].contour(gt2, levels=[0.5], colors=GT_DARK, linewidths=1.6)
    overlay(axes[2], pred2, PRED_C)
    if pred2.any():
        axes[2].contour(pred2, levels=[0.5], colors=PRED_DARK, linewidths=1.6)
    if gt2.any():
        axes[2].contour(gt2, levels=[0.5], colors=GT_C, linewidths=2.2, linestyles="--")
    handles = [Patch(facecolor=GT_C, alpha=0.7, label="ground truth: the true organ (outlined by experts)"),
               Patch(facecolor=PRED_C, alpha=0.7, label="VoxTell's mask: the area VoxTell marks as the organ"),
               Line2D([0], [0], color=GT_C, lw=2.2, linestyle="--", label="ground-truth outline")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    out = m.out_dir(organ, num) / "image_original_ground_truth_voxtell_mask.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main(names):
    for name in names:
        m.configure(name)
        sid = m.SUBJECTS[0]                                          # the first subject of the row 13-16 run
        base = pd.concat([pd.read_csv(f) for f in {"AMOS": [m.REPO / "Voxtell-Amos-Groundness" / "baseline_predictions_amos22_mri.csv"],
                                                    "OASIS": m.CFG["numbering"]}[name]], ignore_index=True)
        base = base[base.subject_id == sid].set_index("organ")
        for organ in m.ORGANS:
            print("wrote", figure(name, sid, m.NUMBERS[sid], organ, base.loc[organ]).relative_to(m.REPO), flush=True)


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or ["OASIS"])
