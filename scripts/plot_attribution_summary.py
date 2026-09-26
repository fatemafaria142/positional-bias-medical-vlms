"""
Summary figures for the word-attribution results (exact Shapley, SHAP, LIME) of the
groundedness investigation. Reads the CSVs downloaded from the Kaggle notebooks and needs
only numpy / pandas / matplotlib.

For each dataset it writes three figures to local_plots_groundedness/summary/:
  1. <ds>_method_share.png   -- how much of the total importance goes to the position word vs.
                                the organ word, per method (Shapley / SHAP / LIME), per organ.
                                Shares (0-100%) put the three methods on one scale; LIME's raw
                                weights are ~100x smaller than Shapley's.
  2. <ds>_shapley_by_metric.png -- exact Shapley (= SHAP here) for the four target metrics,
                                one panel per metric (different units, so separate y-axes).
  3. <ds>_position_vs_organ.png -- every (subject, organ) instance: |Shapley| of the position
                                word (x) against the organ word (y), f_dice_gt. Points above the
                                diagonal mean the organ word matters more.

Usage:
    python scripts/plot_attribution_summary.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "local_plots_groundedness" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()

# validated default palette, categorical slots 1-2 (light surface) + text tokens
SER_A, SER_B = "#2a78d6", "#eb6834"   # scatter: two organ types
W_POSITION, W_ORGAN = "#f400d4", "#5683ff"   # the two words: magenta = left/right word, blue = organ name
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

DATASETS = {
    "AMOS22 MRI": ("amos", [("Voxtell-Amos-Groundness", "amos22_mri")]),
    "Neurite-OASIS": ("oasis", [("Voxtell-OASIS-Batch1-Groundness", "neurite_oasis"),
                                ("Voxtell-OASIS-Batch2-Groundness", "neurite_oasis_batch2")]),
}
METRICS = [("f_dice_gt", "Dice vs. ground truth"), ("f_dice_self", "Dice vs. own baseline"),
           ("f_centroid_shift", "Left-right centroid shift (voxels)"), ("f_volume", "Volume change (voxels)")]
NOT_SEGMENTED = {"cerebral cortex", "lateral ventricle", "choroid plexus"}  # OASIS: baseline Dice <= 0.02


def load(dirs, name):
    frames = [pd.read_csv(REPO / d / f"{name}_{s}.csv") for d, s in dirs]
    df = pd.concat(frames, ignore_index=True)
    df["core"] = df.organ.str.split(n=1).str[1]
    return df


def label_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                    fontsize=8, color=INK2, xytext=(0, 2), textcoords="offset points")


def top_dice_subjects(dirs, n):
    bp = load(dirs, "baseline_predictions")
    return list(bp.groupby("subject_id").dice_gt.mean().sort_values(ascending=False).head(n).index)


EXAMPLE_ORGAN = {"amos": "left kidney", "oasis": "left cerebral white matter"}


def method_share(title, key, dirs, subjects=None, suffix=""):
    """One panel per organ, left and right kept separate (rows = left / right, columns = organ type).
    Plain-language labels; the prompts used are written under the title."""
    wa, sh, li = load(dirs, "word_attribution"), load(dirs, "shap_attribution"), load(dirs, "lime_attribution")
    if subjects is not None:
        wa, sh, li = (df[df.subject_id.isin(subjects)] for df in (wa, sh, li))
    cores = sorted(wa.core.unique())
    if key == "oasis":  # organs the model segments first
        cores = [c for c in cores if c not in NOT_SEGMENTED] + [c for c in cores if c in NOT_SEGMENTED]

    def share(df, col, metric=None):
        d = df if metric is None else df[df.metric == metric]
        m = d.assign(a=d[col].abs()).groupby(["organ", "role"])["a"].mean().unstack()
        return m["position"] / (m["position"] + m["organ"]) * 100

    methods = [("Exact Shapley", share(wa, "shapley_value", "f_dice_gt")),
               ("SHAP", share(sh, "shap_value", "f_dice_gt")),
               ("LIME", share(li, "lime_weight"))]
    sides = ["left", "right"]
    fig, axes = plt.subplots(2, len(cores), figsize=(2.5 * len(cores) + 1.2, 7.0), sharey=True, squeeze=False)
    for r, side in enumerate(sides):
        for c, core in enumerate(cores):
            ax = axes[r][c]
            organ = f"{side} {core}"
            for i, (name, ser) in enumerate(methods):
                p = ser.get(organ, np.nan)
                if np.isnan(p):
                    continue
                ax.bar(i, 100 - p, width=0.62, color=W_ORGAN, edgecolor=SURFACE, linewidth=2)
                ax.bar(i, p, bottom=100 - p, width=0.62, color=W_POSITION, edgecolor=SURFACE, linewidth=2)
                if p >= 8:  # white text on violet, dark text on the lighter teal
                    ax.text(i, 100 - p / 2, f"{p:.0f}%", ha="center", va="center", fontsize=10.5, color="white")
                if 100 - p >= 8:
                    ax.text(i, (100 - p) / 2, f"{100 - p:.0f}%", ha="center", va="center", fontsize=10.5, color="white")
            ax.set_xticks(range(len(methods)), [m[0].replace("Exact ", "Exact\n") for m in methods], fontsize=10)
            label = organ.capitalize() + ("\n(not segmented)" if core in NOT_SEGMENTED else "")
            ax.set_title(label, fontsize=11.5, color=INK if core not in NOT_SEGMENTED else MUTED)
            ax.set_ylim(0, 100)
            ax.tick_params(axis="y", labelsize=9.5)
            ax.grid(axis="x", visible=False)
        axes[r][0].set_ylabel("Share of importance (%)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=W_ORGAN), plt.Rectangle((0, 0), 1, 1, color=W_POSITION)]
    fig.suptitle(title, x=0.5, ha="center", fontsize=18, fontweight="bold", y=0.985)
    question = "Which one matters more for the segmentation, the organ name or the left/right side?"
    wrapped = "\n".join(textwrap.wrap(question, width=int(fig.get_figwidth() * 8.5)))
    fig.text(0.5, 0.945, wrapped, ha="center", va="top", fontsize=12, color=INK2, linespacing=1.3)
    fig.legend(handles, ["Organ name", "Left/right word"], loc="lower center", bbox_to_anchor=(0.5, 0.012),
               frameon=False, ncol=2, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    fig.savefig(OUT / f"{key}_method_share{suffix}.png", dpi=170)
    plt.close(fig)


METRIC_TITLES = {
    "f_dice_gt": "Accuracy (Dice vs. ground truth)",
    "f_dice_self": "Change from its own answer (Dice)",
    "f_centroid_shift": "Left-right shift of the mask (voxels)",
    "f_volume": "Change in mask size (voxels)",
}


def word_attribution(title, key, dirs, subjects=None, suffix=""):
    """Exact Shapley value of each word (organ name, left/right word) for every organ, left and right
    kept separate, one panel per target metric."""
    wa = load(dirs, "word_attribution")
    if subjects is not None:
        wa = wa[wa.subject_id.isin(subjects)]
    if key == "oasis":  # only structures the model actually segments carry meaningful values
        wa = wa[~wa.core.isin(NOT_SEGMENTED)]
    organs = sorted(wa.organ.unique(), key=lambda o: (o.split(None, 1)[1], o.split()[0] != "left"))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), squeeze=False)
    w = 0.38
    x = np.arange(len(organs))
    for ax, metric in zip(axes.ravel(), METRIC_TITLES):
        d = wa[wa.metric == metric]
        m = d.assign(a=d.shapley_value.abs()).groupby(["organ", "role"])["a"].mean().unstack().reindex(organs)
        b1 = ax.bar(x - w / 2, m["organ"], w, color=W_ORGAN, edgecolor=SURFACE, linewidth=2)
        b2 = ax.bar(x + w / 2, m["position"], w, color=W_POSITION, edgecolor=SURFACE, linewidth=2)
        fmt = "{:.2f}" if metric in ("f_dice_gt", "f_dice_self") else "{:,.0f}"
        for bars in (b1, b2):
            for b in bars:
                ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                            va="bottom", fontsize=9.5, color=INK, xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(x, ["\n".join(textwrap.wrap(o.capitalize(), 11)) for o in organs], fontsize=10)
        ax.set_title(METRIC_TITLES[metric], fontsize=11.5)
        ax.set_ylabel("Mean absolute Shapley value", fontsize=10)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=W_ORGAN), plt.Rectangle((0, 0), 1, 1, color=W_POSITION)]
    fig.suptitle(title, x=0.5, ha="center", fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, "How much does each word change the segmentation?", ha="center", va="top",
             fontsize=12, color=INK2)
    fig.legend(handles, ["Organ name", "Left/right word"], loc="lower center", bbox_to_anchor=(0.5, 0.008),
               frameon=False, ncol=2, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUT / f"{key}_word_attribution{suffix}.png", dpi=170)
    plt.close(fig)


def position_vs_organ(title, key, dirs):
    wa = load(dirs, "word_attribution")
    d = wa[wa.metric == "f_dice_gt"].pivot_table(index=["subject_id", "organ", "core"], columns="role",
                                                 values="shapley_value")
    d = d.rename(columns={"position": "pos_w", "organ": "org_w"}).reset_index()
    d["pos"], d["org"] = d.pos_w.abs(), d.org_w.abs()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = max(d.pos.max(), d.org.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(lim * 0.62, lim * 0.56, "equal importance", ha="left", va="top", fontsize=8, color=MUTED, rotation=45)
    if key == "amos":
        groups = [(c, d[d.core == c], col) for c, col in zip(sorted(d.core.unique()), (SER_A, SER_B))]
    else:
        working = d[~d.core.isin(NOT_SEGMENTED)]
        groups = [(c, working[working.core == c], col) for c, col in zip(sorted(working.core.unique()), (SER_A, SER_B))]
        groups.append(("not segmented (3 structures)", d[d.core.isin(NOT_SEGMENTED)], MUTED))
    for name, g, col in groups:
        ax.scatter(g.pos, g.org, s=34, color=col, edgecolor=SURFACE, linewidth=1.2, label=f"{name} (n={len(g)})", zorder=3)
    seg = d if key == "amos" else d[~d.core.isin(NOT_SEGMENTED)]
    above = (seg.org > seg.pos).mean() * 100
    scope = "" if key == "amos" else " (segmented only)"
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("|Shapley| of the position word")
    ax.set_ylabel("|Shapley| of the organ word")
    ax.set_title(f"{title} (Dice vs. ground truth)\norgan word above diagonal in {above:.0f}%{scope}",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"{key}_position_vs_organ.png", dpi=170)
    plt.close(fig)


for title, (key, dirs) in DATASETS.items():
    top5 = top_dice_subjects(dirs, 5)
    method_share(title, key, dirs)
    print(title, "top-5 subjects by mean Dice:", top5)
    method_share(title, key, dirs, subjects=top5, suffix="_top5")
    word_attribution(title, key, dirs)
    word_attribution(title, key, dirs, subjects=top5, suffix="_top5")
    position_vs_organ(title, key, dirs)
    print("wrote figures for", title)
print("output:", OUT)
