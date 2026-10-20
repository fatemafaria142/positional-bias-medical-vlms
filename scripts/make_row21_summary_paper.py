"""
Paper-ready version of row 21's "image vs. organ-word vs. position-word" cross-organ summary
(the chart previously saved as local_plots_mechanistic/rows17_21/row21_summary_all_organs.png).

Changes from that internal-use chart:
  - Uses this project's own OASIS word-attribution palette (scripts/plot_share_of_importance.py's
    PALETTES["oasis"] = position word #731cff / organ word #00a7ad), instead of the generic
    dataviz blue/orange/aqua -- so this figure's colors match every other word-attribution figure
    already made for this dataset (row21 word-highlight panels, share-of-importance bars).
  - "Image" gets its own neutral dark-slate color, visually distinct from the two word colors
    (since it isn't a "word" and shouldn't compete visually with the real finding, which is
    organ-word vs. position-word).
  - Title no longer references the internal plan-row numbering ("Row 21") -- a reader of the
    paper has no reason to know what "row 21" means. Replaced with a declarative finding-style
    title + a subtitle that states exactly what was measured.
  - White background, clean spines, 300 dpi -- matches this repo's other print/paper figures
    (e.g. figure_row18_paper in scripts/make_rows17_21_figures.py).

Usage:
    python scripts/make_row21_summary_paper.py
    python scripts/make_row21_summary_paper.py --csv PATH --out PATH
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# -- OASIS word-attribution palette (same hexes as scripts/plot_share_of_importance.py's
# PALETTES["oasis"] = (position_word, organ_word)) -- kept identical so this figure's colors read
# as the SAME visual language as the rest of this dataset's word-attribution figures.
POSITION_C = "#731cff"   # position word (left/right) -- OASIS palette
ORGAN_C = "#00a7ad"      # organ word -- OASIS palette
IMAGE_C = "#52514e"      # image -- neutral dark slate, deliberately not a "word" color

INK, MUTED, GRID = "#0b0b0b", "#6b6a64", "#dddddd"


def _organ_sort_key(o):
    parts = o.split(None, 1)
    return (parts[1], parts[0]) if len(parts) == 2 else (o,)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "Result-13-to-24" / "row21_joint_shapley_neurite_oasis.csv")
    ap.add_argument("--out", type=Path,
                     default=REPO_ROOT / "local_plots_mechanistic" / "rows17_21" / "row21_image_organword_positionword_paper.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.metric == "f_dice_gt"]
    organs = sorted(df.organ.unique(), key=_organ_sort_key)
    roles = ["image", "organ_word", "position_word"]
    role_color = {"image": IMAGE_C, "organ_word": ORGAN_C, "position_word": POSITION_C}
    role_label = {"image": "Image", "organ_word": "Organ word", "position_word": "Position word (left/right)"}

    vals = {r: [float(df[(df.organ == o) & (df.role == r)].shapley_value.iloc[0])
                if ((df.organ == o) & (df.role == r)).any() else 0.0 for o in organs] for r in roles}

    fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(organs))
    width = 0.27
    for i, r in enumerate(roles):
        ax.bar(x + (i - 1) * width, vals[r], width=width, color=role_color[r], label=role_label[r],
               zorder=3, edgecolor="white", linewidth=0.4)

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([o.title() for o in organs], rotation=45, ha="right", fontsize=10, color=INK)
    ax.set_ylabel("Contribution to Dice score\n(Shapley value, Dice units)", fontsize=11.5, color=INK)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.975,
              "Shapley Attribution of Segmentation Accuracy to Image, Organ Word, and Position Word",
              ha="center", fontsize=16, fontweight="bold", color=INK)

    fig.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
               loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False, fontsize=10.5,
               handlelength=1.6, columnspacing=1.6)

    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
