"""
One figure per question of the investigation plan (rows 1-11), for AMOS and OASIS.

Output: local_plots_groundedness/plan_questions/{AMOS,OASIS}/rowNN_<topic>.png

  row 1   matched vs. mismatched overlap with the true organ (image-text alignment)
  row 2   text-embedding distance vs. output distance
  row 3   does the mask land on the named side (left prompt -> left, right prompt -> right)
  row 4   difference between the left-prompt and right-prompt raw outputs
  row 5-7 importance of each word: exact Shapley, LIME, SHAP
  row 8   accuracy (Dice vs. ground truth) when a word is removed
  row 9   change from the model's own answer when a word is removed
  row 10  left-right shift of the mask when the position word is removed
  row 11  mask size when a word is removed

OASIS: rows 1-4 show all five structure pairs (the ones the model does not segment are marked); rows 5-11 show
only the structures the model segments, because the others have Dice ~ 0 even with the full prompt.

Usage:  python scripts/make_plan_question_figures.py
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "local_plots_groundedness" / "plan_questions"

FONT = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#9a9993", "#e6e5e1", "#fcfcfb"
DARK, LIGHT_GREY = "#2f3037", "#b9b8b2"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

DATASETS = {
    "AMOS": dict(title="AMOS22 MRI", pos="#f400d4", org="#5683ff",
                 dirs=[("Voxtell-Amos-Groundness", "amos22_mri")],
                 pair_order=["kidney", "adrenal gland"], not_segmented=[],
                 organs=["left kidney", "right kidney", "left adrenal gland", "right adrenal gland"]),
    "OASIS": dict(title="Neurite-OASIS", pos="#731cff", org="#00a7ad",
                  dirs=[("Voxtell-OASIS-Batch1-Groundness", "neurite_oasis"),
                        ("Voxtell-OASIS-Batch2-Groundness", "neurite_oasis_batch2")],
                  pair_order=["cerebral white matter", "cerebellum cortex", "cerebral cortex", "lateral ventricle",
                              "choroid plexus"],
                  not_segmented=["cerebral cortex", "lateral ventricle", "choroid plexus"],
                  organs=["left cerebral white matter", "right cerebral white matter", "left cerebellum cortex",
                          "right cerebellum cortex"]),
}
VERSIONS = [("none", "Full prompt"), ("position", "Without position"), ("organ", "Without organ"),
            ("organ,position", "Without position and organ")]


def load(cfg, name):
    return pd.concat([pd.read_csv(REPO / d / f"{name}_{s}.csv") for d, s in cfg["dirs"]], ignore_index=True)


def core_of(x):
    return x.split("/")[0].split(None, 1)[1] if "/" in x else x.split(None, 1)[1]


def wrap(name, n=12):
    return "\n".join(textwrap.wrap(name, n))


# ------------------------------------------------------------------------------------------------
def frame(fig, cfg, question, handles=None, labels=None, ncol=None):
    fig.suptitle(cfg["title"], x=0.5, ha="center", fontsize=19, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, question, ha="center", va="top", fontsize=12.5, color=INK2)
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), frameon=False,
                   ncol=ncol or len(handles), fontsize=11)


def patch(color, edge=None):
    return mpatches.Patch(facecolor=color, edgecolor=edge or color)


def grouped_bars(cfg, cats, series, question, ylabel, fmt, out, ref=None, ref_label=None, ylim=None,
                 grey_cats=(), wrap_n=12, marks=None, marks_label=None, log_from=None, inside_labels=False,
                 width=None, label_fs=9.5, show_labels=True, xlabel=None):
    """series: [(label, [values per category], colour)]; NaN values are skipped.
    marks: [y per category] drawn as a dashed dark bar on each category (e.g. the midline);
    log_from: draw on a log axis with the bars starting at this value (used for ratios, 1 = no change)."""
    from matplotlib.lines import Line2D
    n = len(series)
    width = width or max(8.6, 3.0 + len(cats) * (0.5 + 0.48 * n))
    fig, ax = plt.subplots(figsize=(width, 6.0))
    bw = 0.8 / n
    x = np.arange(len(cats))
    lo, hi = 0.0, 0.0
    allv = []
    if log_from:
        ax.set_yscale("log")
    for k, (label, vals, col) in enumerate(series):
        xs = x - 0.4 + bw * (k + 0.5)
        vals = np.array(vals, dtype=float)
        if log_from:
            bars = ax.bar(xs, np.nan_to_num(vals, nan=log_from) - log_from, bw * 0.92, bottom=log_from, color=col,
                          edgecolor=SURFACE, linewidth=1.5)
        else:
            bars = ax.bar(xs, np.nan_to_num(vals), bw * 0.92, color=col, edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            if np.isnan(v) or not show_labels:
                continue
            up = v >= (log_from or 0)
            if inside_labels:
                ax.text(b.get_x() + b.get_width() / 2, v / 2, fmt(v), ha="center", va="center", fontsize=11,
                        fontweight="bold", color="white")
                continue
            ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom" if up else "top",
                        fontsize=label_fs, color=INK, xytext=(0, 3 if up else -3), textcoords="offset points")
        allv += [v for v in vals if not np.isnan(v)]
        lo, hi = min(lo, np.nanmin(vals)), max(hi, np.nanmax(vals))
    extra = []
    if marks is not None:
        for xi, y in zip(x, marks):
            ax.hlines(y, xi - 0.42, xi + 0.42, color=DARK, linewidth=1.8, linestyle=(0, (3, 2)), zorder=4)
            lo, hi = min(lo, y), max(hi, y)
        extra.append((Line2D([0], [0], color=DARK, linewidth=1.8, linestyle=(0, (3, 2))), marks_label))
    if log_from:
        ax.axhline(log_from, color=DARK, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        extra.append((Line2D([0], [0], color=DARK, linewidth=1.1, linestyle=(0, (4, 3))), ref_label))
        ax.set_ylim(min(allv) / 2.2, max(allv) * 2.2)
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(["×0.01", "×0.1", "×1", "×10", "×100"])
        ax.minorticks_off()
    else:
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=2)
        span = (hi - lo) or 1.0
        ax.set_ylim(ylim if ylim else (lo - 0.12 * span if lo < 0 else 0, hi + 0.14 * span))
    ax.set_xticks(x, [wrap(c, wrap_n) for c in cats], fontsize=10.5)
    for tick, c in zip(ax.get_xticklabels(), cats):
        if c in grey_cats:
            tick.set_color(MUTED)
    ax.set_ylabel(ylabel, fontsize=10.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", visible=False)
    handles = [patch(c) for _, _, c in series] + [h for h, _ in extra]
    labels = [s for s, _, _ in series] + [l for _, l in extra]
    frame(fig, cfg, question, handles, labels)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def order_pairs(cfg, cores):
    return [c for c in cfg["pair_order"] if c in set(cores)]


def pair_label(cfg, c):
    return c + ("\n(not segmented)" if c in cfg["not_segmented"] else "")


# ------------------------------------------------------------------------------------------------
def row1(cfg, out):
    d = load(cfg, "row1_image_text_alignment")
    d["core"] = d.organ_pair.map(core_of)
    g = d.groupby("core")[["iou_matched", "iou_mismatched"]].mean()
    cats = order_pairs(cfg, g.index)
    grouped_bars(cfg, [pair_label(cfg, c) for c in cats],
                 [("Prompt for this side (correct)", [g.loc[c, "iou_matched"] for c in cats], cfg["org"]),
                  ("Prompt for the other side (mismatched)", [g.loc[c, "iou_mismatched"] for c in cats], LIGHT_GREY)],
                 "Does each prompt point to the right region of the image?",
                 "Overlap with the true organ (IoU)", lambda v: f"{v:.2f}", out, ylim=(0, 1.1),
                 grey_cats=[pair_label(cfg, c) for c in cfg["not_segmented"]])


def row2(cfg, out):
    d = load(cfg, "row2_embedding_output_regression").dropna(subset=["output_centroid_dist_vox"])
    d["core"] = d.organ_pair.map(core_of)
    cats = order_pairs(cfg, d.core.unique())
    cols = [cfg["org"], cfg["pos"], DARK, MUTED, LIGHT_GREY]
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    for c, col in zip(cats, cols):
        s = d[d.core == c]
        ax.scatter(s.text_embed_cos_dist, s.output_centroid_dist_vox, s=70, color=col, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3, label=c)
    if d.text_embed_cos_dist.nunique() > 1:
        m, b = np.polyfit(d.text_embed_cos_dist, d.output_centroid_dist_vox, 1)
        r2 = np.corrcoef(d.text_embed_cos_dist, d.output_centroid_dist_vox)[0, 1] ** 2
        xs = np.linspace(d.text_embed_cos_dist.min(), d.text_embed_cos_dist.max(), 20)
        ax.plot(xs, m * xs + b, color=DARK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(0.5, 0.97, f"Fitted line: slope {m:,.0f}, R² {r2:.2f}", transform=ax.transAxes, fontsize=10.5,
                color=INK2, va="top", ha="center")
    ax.set_xlabel("Difference between the left and right prompt (text embedding distance)", fontsize=10.5)
    ax.set_ylabel("Distance between the two masks (voxels)", fontsize=10.5)
    ax.tick_params(labelsize=10)
    frame(fig, cfg, "Does a bigger difference between the prompts give a bigger difference in the output?",
          [patch(c) for c in cols[:len(cats)]], [pair_label(cfg, c).replace("\n", " ") for c in cats], ncol=min(len(cats), 3))
    fig.tight_layout(rect=(0, 0.12 if len(cats) > 3 else 0.07, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def row3(cfg, out):
    d = load(cfg, "row3_causal_embedding_swap")
    d["core"] = d.organ_pair.map(core_of)
    g = d.groupby("core")[["left_prompt_predicts_left_side", "right_prompt_predicts_right_side"]].mean() * 100
    cats = order_pairs(cfg, g.index)
    grouped_bars(cfg, [pair_label(cfg, c) for c in cats],
                 [("Left prompt gives the left side", [g.loc[c, "left_prompt_predicts_left_side"] for c in cats], cfg["org"]),
                  ("Right prompt gives the right side", [g.loc[c, "right_prompt_predicts_right_side"] for c in cats], DARK)],
                 "Does the mask move to the side that the prompt names?",
                 "Correct side (% of subjects)", lambda v: f"{v:.0f}%", out, ylim=(0, 118),
                 grey_cats=[pair_label(cfg, c) for c in cfg["not_segmented"]])


def row4(cfg, out):
    d = load(cfg, "row4_fused_feature_diff")
    d["core"] = d.organ_pair.map(core_of)
    g = d.groupby("core")[["logit_diff_mean_abs", "frac_voxels_class_flip"]].mean()
    cats = order_pairs(cfg, g.index)
    fig, axes = plt.subplots(1, 2, figsize=(max(10.5, 2.6 * len(cats) + 4), 6.0))
    for ax, col, label, f in [(axes[0], "logit_diff_mean_abs", "Average difference in the raw output", lambda v: f"{v:.1f}"),
                              (axes[1], "frac_voxels_class_flip", "Voxels that differ between the two masks (%)",
                               lambda v: f"{v:.2f}%")]:
        vals = g[col] * (100 if col == "frac_voxels_class_flip" else 1)
        xs = np.arange(len(cats))
        bars = ax.bar(xs, [vals[c] for c in cats], 0.6, color=cfg["org"], edgecolor=SURFACE, linewidth=1.5)
        for b, c in zip(bars, cats):
            ax.annotate(f(vals[c]), (b.get_x() + b.get_width() / 2, vals[c]), ha="center", va="bottom", fontsize=9.5,
                        xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(xs, [wrap(pair_label(cfg, c)) for c in cats], fontsize=10)
        for tick, c in zip(ax.get_xticklabels(), cats):
            if c in cfg["not_segmented"]:
                tick.set_color(MUTED)
        ax.set_ylabel(label, fontsize=10.5)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    frame(fig, cfg, "Do the left prompt and the right prompt give different outputs inside the model?")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def word_importance(cfg, out, method):
    if method == "shapley":
        d = load(cfg, "word_attribution")
        d = d[d.metric == "f_dice_gt"].rename(columns={"shapley_value": "v"})
        q, yl = "How much does each word matter (exact Shapley)?", "Contribution to the Dice score"
    elif method == "shap":
        d = load(cfg, "shap_attribution")
        d = d[d.metric == "f_dice_gt"].rename(columns={"shap_value": "v"})
        q, yl = "How much does each word matter (SHAP)?", "Contribution to the Dice score"
    else:
        d = load(cfg, "lime_attribution").rename(columns={"lime_weight": "v"})
        q, yl = "How much does each word matter (LIME)?", "LIME weight (its own scale)"
    g = d.groupby(["organ", "role"]).v.mean().unstack()
    organs = cfg["organs"]
    grouped_bars(cfg, [o.capitalize() for o in organs],
                 [("Organ name", [g.loc[o, "organ"] for o in organs], cfg["org"]),
                  ("Left/right word", [g.loc[o, "position"] for o in organs], cfg["pos"])],
                 q, yl, lambda v: f"{v:.3g}", out, wrap_n=11)


def ablation(cfg):
    a = pd.read_csv  # noqa (kept for readability)
    d = pd.concat([pd.read_csv(REPO / dd / f"ablation_masks_{s}.csv",
                               usecols=["subject_id", "organ", "roles_dropped", "n_pred_voxels", "f_dice_gt",
                                        "f_dice_self", "f_centroid_shift"]) for dd, s in cfg["dirs"]], ignore_index=True)
    d["roles_dropped"] = d.roles_dropped.fillna("none")
    return d[d.organ.isin(cfg["organs"])]


def version_colors(cfg):
    return {"none": DARK, "position": cfg["pos"], "organ": cfg["org"], "organ,position": LIGHT_GREY}


def row8(cfg, out):
    d, organs, col = ablation(cfg), cfg["organs"], version_colors(cfg)
    g = d.groupby(["organ", "roles_dropped"]).f_dice_gt.mean().unstack()
    grouped_bars(cfg, [o.capitalize() for o in organs],
                 [(name, [g.loc[o, key] for o in organs], col[key]) for key, name in VERSIONS],
                 "Does removing a word reduce the accuracy?", "Dice vs. ground truth", lambda v: f"{v:.2f}", out,
                 ylim=(0, 1.15), wrap_n=11)


def row9(cfg, out):
    d, organs, col = ablation(cfg), cfg["organs"], version_colors(cfg)
    g = d.groupby(["organ", "roles_dropped"]).f_dice_self.mean().unstack()
    grouped_bars(cfg, [o.capitalize() for o in organs],
                 [(name, [g.loc[o, key] for o in organs], col[key]) for key, name in VERSIONS[1:]],
                 "Does removing a word change the model's own answer?",
                 "Overlap with the model's answer for the full prompt (Dice)", lambda v: f"{v:.2f}", out,
                 ylim=(0, 1.15), wrap_n=11)


def row10(cfg, out):
    d, organs = ablation(cfg), cfg["organs"]
    g = d[d.roles_dropped == "position"].groupby("organ").f_centroid_shift.mean()
    gap = load(cfg, "row2_embedding_output_regression")
    gap["core"] = gap.organ_pair.map(core_of)
    half = gap.groupby("core").output_centroid_dist_vox.mean() / 2   # distance from each organ to the midline
    core = lambda o: o.split(None, 1)[1]
    marks = [(1 if o.startswith("left") else -1) * half[core(o)] for o in organs]
    grouped_bars(cfg, [o.capitalize() for o in organs],
                 [("Mask centre without the position word", [g[o] for o in organs], cfg["pos"])],
                 "Does removing the position word move the mask to the other side?",
                 "Shift of the mask centre (voxels, + = towards the midline)", lambda v: f"{v:+.0f}", out, wrap_n=11,
                 marks=marks, marks_label="Midline", inside_labels=True)


def row11(cfg, out):
    d, organs, col = ablation(cfg), cfg["organs"], version_colors(cfg)
    m = d.groupby(["organ", "roles_dropped"]).n_pred_voxels.mean().unstack()
    ratio = lambda o, k: m.loc[o, k] / m.loc[o, "none"]
    grouped_bars(cfg, [o.capitalize() for o in organs],
                 [(name, [ratio(o, key) for o in organs], col[key]) for key, name in VERSIONS[1:]],
                 "Does removing a word make the mask collapse?", "Mask size compared with the full prompt (log scale)",
                 lambda v: f"×{v:.2f}" if v < 10 else f"×{v:.0f}", out, log_from=1.0,
                 ref_label="Size with the full prompt", wrap_n=11)


# ------------------------------------------------------------------------------------------------
def main():
    for folder, cfg in DATASETS.items():
        o = ROOT / folder
        row1(cfg, o / "row01_image_text_alignment.png")
        row2(cfg, o / "row02_embedding_vs_output_distance.png")
        row3(cfg, o / "row03_named_side_accuracy.png")
        row4(cfg, o / "row04_left_vs_right_output_difference.png")
        word_importance(cfg, o / "row05_exact_shapley.png", "shapley")
        word_importance(cfg, o / "row06_lime.png", "lime")
        word_importance(cfg, o / "row07_shap.png", "shap")
        row8(cfg, o / "row08_accuracy_when_a_word_is_removed.png")
        row9(cfg, o / "row09_change_from_own_answer.png")
        row10(cfg, o / "row10_left_right_shift.png")
        row11(cfg, o / "row11_mask_size.png")
        print("wrote 11 figures to", o)


if __name__ == "__main__":
    main()
