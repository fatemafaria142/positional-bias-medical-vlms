"""
Publication-quality per-organ Dice bar charts for VoxTell on AMOS22 MRI and
Neurite OASIS, built from predictions_voxtell_amos22_mri.csv and
predictions_voxtell_neurite_oasis.csv.

Design:
  - Vertical bars, grouped by anatomical category (fixed category order),
    sorted by mean Dice descending WITHIN each category -- category answers
    "what kind of structure is this," bar height answers "how well did
    VoxTell do," so both questions are visible in one read.
  - Fixed categorical color per category (validated palette, colorblind-safe
    adjacent pairs), never reused across categories, never re-cycled to
    encode magnitude -- magnitude lives in the bar height + direct label
    only, per the "color follows the entity, not its rank" rule.
  - Neurite OASIS (35 structures) is split into one chart PER category
    (white matter / cortex / subcortical gray / ventricles-CSF /
    brainstem-vascular) instead of one crowded 35-column chart -- each
    chart gets to run at a readable text size. AMOS22 MRI (13 organs)
    stays as a single chart across its 4 categories.
  - Direct value labels and axis/tick labels in near-black ink for
    readability; only the muted chart chrome (gridlines, ref-line ticks,
    error whiskers) stays gray.
  - Complete-failure organs (mean Dice == 0.000 across all subjects) are
    marked with a small status-red "0.000" tag rotated up the axis instead
    of an invisible sliver bar, so the failure reads immediately.

Usage:
    python scripts/plot_voxtell_dice_charts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# -- palette: validated 5-slot categorical set (references/palette.md, fixed order) --
CAT_COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#c98500",  # darker step than the default yellow for label/line contrast
    "violet": "#4a3aa7",
}
STATUS_CRITICAL = "#d03b3b"
INK_LABEL = "#000000"     # value labels, axis labels, tick labels -- near-black, readable
INK_MUTED = "#8a8880"     # chart chrome only: gridlines, ref-line ticks, error whiskers
GRIDLINE = "#e5e4dd"
BASELINE = "#b9b8b0"
SURFACE = "#fcfcfb"
PAGE = "#f6f5f2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK_LABEL,
    "figure.facecolor": PAGE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE,
})

# -- category definitions, in fixed display/legend order --
AMOS_CATEGORIES = [
    ("Solid organs", CAT_COLORS["blue"],
     ["liver", "right kidney", "left kidney", "spleen", "pancreas"]),
    ("Vascular", CAT_COLORS["orange"],
     ["aorta", "inferior vena cava"]),
    ("GI tract", CAT_COLORS["aqua"],
     ["stomach", "duodenum", "esophagus", "gallbladder"]),
    ("Endocrine", CAT_COLORS["yellow"],
     ["left adrenal gland", "right adrenal gland"]),
]

OASIS_CATEGORIES = [
    ("White matter", CAT_COLORS["blue"],
     ["right cerebral white matter", "left cerebral white matter",
      "right cerebellum white matter", "left cerebellum white matter"]),
    ("Cortex", CAT_COLORS["orange"],
     ["right cerebral cortex", "left cerebral cortex",
      "left cerebellum cortex", "right cerebellum cortex"]),
    ("Subcortical gray matter", CAT_COLORS["aqua"],
     ["right putamen", "left putamen", "right thalamus", "left thalamus",
      "right caudate", "left caudate", "right pallidum", "left pallidum",
      "right hippocampus", "left hippocampus", "right amygdala", "left amygdala",
      "right accumbens", "left accumbens", "right ventral DC", "left ventral DC"]),
    ("Ventricles / CSF", CAT_COLORS["yellow"],
     ["left lateral ventricle", "right lateral ventricle",
      "right inferior lateral ventricle", "left inferior lateral ventricle",
      "3rd ventricle", "4th ventricle",
      "right choroid plexus", "left choroid plexus"]),
    ("Brainstem / vascular", CAT_COLORS["violet"],
     ["brain stem", "left vessel", "right vessel"]),
]


def slugify(label: str) -> str:
    return label.lower().replace(" / ", "_").replace(" ", "_")


def organ_label(name: str) -> str:
    return name[:1].upper() + name[1:]


def build_cols(df: pd.DataFrame, categories):
    """One column per organ, in category order, sorted by Dice desc within
    each category; returns cols plus (label, color, start, end) spans for
    the background bands / headers."""
    g = df.groupby("organ")["dice"].agg(mean="mean", std="std").fillna(0.0)

    cols = []
    spans = []
    cursor = 0
    for cat_label, color, organs in categories:
        present = [o for o in organs if o in g.index]
        sub = g.loc[present].sort_values("mean", ascending=False)
        for organ, r in sub.iterrows():
            cols.append({"organ": organ, "mean": r["mean"], "std": r["std"], "color": color})
        spans.append((cat_label, color, cursor, cursor + len(sub)))
        cursor += len(sub)
    return cols, spans


def render_chart(df: pd.DataFrame, categories, title: str, subtitle: str,
                  out_path: Path, col_w: float, show_category_headers: bool):
    n_subjects = df["subject_id"].nunique()
    cols, spans = build_cols(df, categories)
    n = len(cols)

    fig_w = max(6.5, col_w * n + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, 7.0))

    x = np.arange(n)
    bar_w = 0.62

    for lbl, color, start, end in spans:
        ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.06, zorder=0)
        if start > 0:
            ax.axvline(start - 0.5, color="white", linewidth=2.6, zorder=1)
        if show_category_headers:
            mid = (start + end - 1) / 2
            ax.text(mid, 1.10, lbl, ha="center", va="bottom", fontsize=13,
                     fontweight="bold", color=color, zorder=3, clip_on=False)

    for xi, col in zip(x, cols):
        m, s, color = col["mean"], col["std"], col["color"]
        is_fail = m == 0.0

        if is_fail:
            ax.text(xi, 0.015, "0.000", rotation=90, va="bottom", ha="center",
                     fontsize=10.5, fontweight="bold", color=STATUS_CRITICAL, zorder=4)
            continue

        w = bar_w / 2
        ax.add_patch(Rectangle((xi - w, 0), bar_w, m, facecolor=color,
                                edgecolor="none", zorder=2))
        if s > 0:
            ax.plot([xi, xi], [max(m - s, 0), m + s], color=INK_MUTED, linewidth=1.4,
                     alpha=0.8, zorder=3, solid_capstyle="round")
            ax.plot([xi - 0.08, xi + 0.08], [m + s, m + s], color=INK_MUTED,
                     linewidth=1.4, alpha=0.8, zorder=3)
        ax.text(xi, m + s + 0.02, f"{m:.3f}", rotation=90, va="bottom", ha="center",
                 fontsize=10.5, fontweight="bold", color=INK_LABEL, zorder=4)

    labels = [organ_label(c["organ"]) for c in cols]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, rotation=55, ha="right",
                        rotation_mode="anchor", color=INK_LABEL)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.7, n - 0.3)

    for ref in (0.5, 0.7):
        ax.axhline(ref, color=BASELINE, linewidth=1.1, linestyle=(0, (3, 3)), zorder=1)
        ax.text(n - 0.35, ref, f"{ref:.1f}", ha="left", va="center", fontsize=10,
                 color=INK_MUTED, clip_on=False)

    ax.set_ylabel(f"Mean Dice ± std across {n_subjects} subjects", fontsize=12.5,
                   color=INK_LABEL)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=3, color=BASELINE, labelsize=11, colors=INK_LABEL)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(0.02, 0.975, title, fontsize=18, fontweight="bold", ha="left", va="top",
              color=INK_LABEL)
    fig.text(0.02, 0.925, subtitle, fontsize=12, color=INK_MUTED, ha="left", va="top")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def main():
    amos_df = pd.read_csv(REPO_ROOT / "predictions_voxtell_amos22_mri.csv")
    render_chart(
        amos_df, AMOS_CATEGORIES,
        title="VoxTell per-organ Dice — AMOS22 MRI",
        subtitle="Grouped by anatomical category · sorted by mean Dice within each group",
        out_path=REPO_ROOT / "voxtell_dice_amos.png",
        col_w=0.75, show_category_headers=True,
    )

    oasis_df = pd.read_csv(REPO_ROOT / "predictions_voxtell_neurite_oasis.csv")
    for cat_label, color, organs in OASIS_CATEGORIES:
        render_chart(
            oasis_df, [(cat_label, color, organs)],
            title=f"VoxTell per-structure Dice — Neurite OASIS",
            subtitle=f"{cat_label} · sorted by mean Dice",
            out_path=REPO_ROOT / f"voxtell_dice_oasis_{slugify(cat_label)}.png",
            col_w=0.85, show_category_headers=False,
        )


if __name__ == "__main__":
    main()
