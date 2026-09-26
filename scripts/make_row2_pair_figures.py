"""
Row 2 for ONE example subject and BOTH organs of a left/right pair, on the scan itself, with the prompts that
were used. Written into the folders of both organs of the pair:

    local_plots_groundedness/word_highlight/<AMOS|OASIS>/<left|right>_<organ>/row02_masks_as_far_apart_as_the_organs.png

The subject is the best-scoring one of each dataset (see DEFAULT_SUBJECT); its identifier is not printed.

Usage:
    python scripts/make_row2_pair_figures.py                   # default example subjects
    python scripts/make_row2_pair_figures.py amos amos_0508    # another subject
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_example_subject_figures import (DEFAULT_SUBJECT, TRUE_C, Subject, crop_box, slices)  # noqa
from make_plan_question_figures import DARK, DATASETS, INK, INK2, MUTED, SURFACE, load  # noqa

ROOT = Path(__file__).resolve().parent.parent / "local_plots_groundedness" / "word_highlight"


def centroid(m):
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean()) if len(xs) else None


def figure(sub, core, out_paths):
    cfg = sub.cfg
    left, right = f"left {core}", f"right {core}"
    idx, img2, box = slices(sub, [left, right])
    tl, tr = sub.true_mask(left)[idx].T, sub.true_mask(right)[idx].T
    pl, pr = sub.predicted_mask(left)[idx].T, sub.predicted_mask(right)[idx].T
    y0, y1, x0, x1 = box

    r2 = load(cfg, "row2_embedding_output_regression")
    r2 = r2[(r2.subject_id == sub.sid) & (r2.organ_pair == f"{left}/{right}")].iloc[0]
    true_gap = abs(sub.bp.loc[left].gt_centroid_vox_2 - sub.bp.loc[right].gt_centroid_vox_2)
    mask_gap = float(r2.output_centroid_dist_vox)
    prompt_l, prompt_r = sub.bp.loc[left].prompt_text, sub.bp.loc[right].prompt_text
    dice_l, dice_r = float(sub.bp.loc[left].dice_gt), float(sub.bp.loc[right].dice_gt)

    fig = plt.figure(figsize=(12.5, 8.4))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(cfg["title"], x=0.5, ha="center", fontsize=19, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, "Does asking for the right instead of the left move the mask as far as the real organs are apart?",
             ha="center", va="top", fontsize=12.5, color=INK2)
    ax = fig.add_axes([0.04, 0.34, 0.92, 0.52])
    ax.imshow(img2[y0:y1, x0:x1], cmap="gray", origin="lower")
    for m, col in [(pl, cfg["org"]), (pr, cfg["pos"])]:
        m = m[y0:y1, x0:x1]
        rgba = np.zeros((*m.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(col)
        rgba[..., 3] = np.where(m, 0.6, 0.0)
        ax.imshow(rgba, origin="lower")
    for m in (tl, tr):
        m = m[y0:y1, x0:x1]
        if m.any():
            ax.contour(m, levels=[0.5], colors=TRUE_C, linewidths=1.8)
    # lines between the centres: the real organs (green) and the two masks (dark)
    def draw_gap(a, b, color, label, dy):
        ca, cb = centroid(a[y0:y1, x0:x1]), centroid(b[y0:y1, x0:x1])
        if ca is None or cb is None:
            return
        ax.plot([ca[0], cb[0]], [ca[1] + dy, cb[1] + dy], color=color, linewidth=2.2, linestyle=(0, (4, 2.5)),
                marker="o", markersize=6, markeredgecolor="white", zorder=6)
        ax.text((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2 + dy + 0.05 * (y1 - y0), label, color=color, ha="center",
                va="bottom", fontsize=11.5, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    span = (y1 - y0)
    draw_gap(tl, tr, TRUE_C, f"real organs: {true_gap:.0f} voxels apart", -0.12 * span)
    draw_gap(pl, pr, DARK, f"the two masks: {mask_gap:.0f} voxels apart", 0.12 * span)
    ax.axis("off")

    # information strip
    info = fig.add_axes([0.04, 0.09, 0.92, 0.2])
    info.axis("off")
    info.set_xlim(0, 1)
    info.set_ylim(0, 1)
    info.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", linewidth=0,
                                           facecolor="#f1f3f8"))
    info.text(0.02, 0.86, "Prompts that were used", fontsize=11.5, fontweight="bold", color=INK, va="center")
    for k, (col, txt, dice) in enumerate([(cfg["org"], prompt_l, dice_l), (cfg["pos"], prompt_r, dice_r)]):
        yy = 0.62 - 0.22 * k
        info.add_patch(mpatches.FancyBboxPatch((0.02, yy - 0.06), 0.03, 0.12, boxstyle="round,pad=0,rounding_size=0.01",
                                               linewidth=0, facecolor=col))
        info.text(0.06, yy, f"“{txt}”", fontsize=11.5, color=INK, va="center")
        info.text(0.62, yy, f"overlap with the true organ (Dice): {dice:.2f}", fontsize=11, color=INK2, va="center")
    info.text(0.02, 0.14, f"Difference between the two prompts (text): {r2.text_embed_cos_dist:.3f}      "
              f"Real organs apart: {true_gap:.0f} voxels      Masks apart: {mask_gap:.0f} voxels",
              fontsize=11, color=INK2, va="center")

    handles = [mpatches.Patch(facecolor=SURFACE, edgecolor=TRUE_C, linewidth=1.8), mpatches.Patch(color=cfg["org"]),
               mpatches.Patch(color=cfg["pos"])]
    fig.legend(handles, ["True organ (outline)", f"Mask for the prompt “{left}”",
                         f"Mask for the prompt “{right}”"], loc="lower center", bbox_to_anchor=(0.5, 0.01),
               frameon=False, ncol=3, fontsize=11)
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160)
    plt.close(fig)


def main(only=None, sid=None):
    for folder, cfg in DATASETS.items():
        if only and folder.lower() != only:
            continue
        sub = Subject(cfg, sid or DEFAULT_SUBJECT[folder])
        for core in [c for c in cfg["pair_order"] if c not in cfg["not_segmented"]]:
            outs = [ROOT / folder / f"{side}_{core.replace(' ', '_')}" / "row02_masks_as_far_apart_as_the_organs.png"
                    for side in ("left", "right")]
            figure(sub, core, outs)
            print("wrote row 2 for", folder, core)


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
