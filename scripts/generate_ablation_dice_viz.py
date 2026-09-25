"""Raw ablation Dice score chart -- the data BEHIND the word-attribution chart.

generate_word_highlight_viz.py shows derived Shapley/SHAP/LIME attribution values.
This script shows the actual measured Dice scores for the 4 raw prompt variants
those attribution values are computed from, so the numbers there are traceable
back to real model predictions.

Run:
    python scripts/generate_ablation_dice_viz.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("voxtell_amos22_mri_groundedness_investigation_results")
OUT_PATH = RESULTS_DIR / "visualizations" / "ablation_dice_scores.png"

CMAP = LinearSegmentedColormap.from_list(
    "importance", ["#FFF3E0", "#FDB863", "#E2492A", "#7A0403"]
)

ORGANS = ["left kidney", "right kidney", "left adrenal gland", "right adrenal gland"]
COMBO_ORDER = ["organ,position", "organ", "position", "nan"]
COMBO_LABEL = {
    "organ,position": "Full prompt",
    "organ": "Organ only",
    "position": "Position only",
    "nan": "Neither word",
}


def load_data():
    am = pd.read_csv(RESULTS_DIR / "ablation_masks_amos22_mri.csv")
    out = {}
    for organ in ORGANS:
        sub = am[am.organ == organ]
        g = (
            sub.groupby(sub["roles_present"].fillna("nan"))
            .agg(dice=("f_dice_gt", "mean"), prompt=("prompt_text", "first"))
        )
        out[organ] = g
    return out


def render():
    data = load_data()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6), dpi=200)
    fig.patch.set_facecolor("white")

    fig.text(0.045, 0.975, "Prompt Ablation Study — Raw Dice Scores",
              fontsize=19, fontweight="bold", color="#111318", ha="left", va="top")
    fig.text(0.045, 0.945,
             "The actual measured segmentation Dice (vs. ground truth) behind each word's "
             "attribution value — averaged across 10 subjects per bar.",
             fontsize=10.5, color="#4A4E57", ha="left", va="top")
    fig.text(0.045, 0.918,
              "Darker bar = higher Dice score. Position-only prompts use a generic "
              "placeholder word (“structure”) in place of the dropped organ name.",
              fontsize=9.5, color="#4A4E57", ha="left", va="top")

    for ax, organ in zip(axes.flat, ORGANS):
        g = data[organ]
        vals = [g.loc[c, "dice"] if c in g.index else 0.0 for c in COMBO_ORDER]
        prompts = [g.loc[c, "prompt"] if c in g.index else "" for c in COMBO_ORDER]
        labels = [COMBO_LABEL[c] for c in COMBO_ORDER]
        colors = [CMAP(0.15 + 0.85 * v) for v in vals]

        bars = ax.bar(range(4), vals, color=colors, width=0.62, zorder=3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.6, 3.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=9.8, color="#2C3038")
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel("Dice vs. ground truth", fontsize=9.6, color="#4A4E57")
        ax.set_title(organ.title(), fontsize=13.5, fontweight="bold", color="#111318", pad=10)
        ax.grid(axis="y", color="#E4E6EA", lw=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#C9CCD1")

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111318")

        caption_lines = [f"{lbl}: “{p}”" for lbl, p in zip(labels, prompts)]
        for j, line in enumerate(caption_lines):
            ax.text(0.0, -0.22 - 0.095 * j, line, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.2, style="italic", color="#8A8F98",
                    clip_on=False)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.06, hspace=1.15, wspace=0.28)

    fig.savefig(OUT_PATH, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    render()
