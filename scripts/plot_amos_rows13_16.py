"""
Figures for the AMOS22-MRI rows 13-16 results (VoxTell mechanistic experiments).

Reads the five CSVs in `AMOS-ROW(13 to 16)/`, keeps the subjects listed in SUBJECTS (the three whose rows 14-16 finished),
and writes five combined PNGs to `AMOS-ROW(13 to 16)/figures/`, (all subjects and both organs of a type together;
the per-organ, per-subject figures are made by make_organ_rows13_16_figures.py):

  fig_row13_noise.png        mask size after the organ / both organs / the whole image are replaced by noise
  fig_row14_blend.png        embedding blend: which organ does the mask cover as alpha goes from "left" to "right"
  fig_row14_patches.png      share of masks that switch side after patching decoder-layer updates
  fig_row15_attention.png    cross-attention enrichment on the named organ / other side / other organ, per layer
  fig_row16_distance.png     cosine distance between the left and right query states per layer, with an organ-vs-organ reference

Usage:  python scripts/plot_amos_rows13_16.py            (needs numpy, pandas, matplotlib only)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "AMOS-ROW(13 to 16)"
OUT = DATA / "figures"
SUBJECTS = ["amos_0508", "amos_0530", "amos_0555"]
CORES = ["kidney", "adrenal gland"]
SFX = "_amos22_mri.csv"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"         # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def load(name):
    d = pd.read_csv(DATA / f"{name}{SFX}")
    return d[d.subject_id.isin(SUBJECTS)].copy()


def core_of(organ):
    return organ.split(None, 1)[1]


def head(fig, main, sub):
    """Bold title plus a quieter description underneath (both left-aligned, never clipped)."""
    fig.text(0.012, 0.975, main, ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def foot_legend(fig, ax, ncol=3):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.01))


def fig_row13():
    d = load("row13_organ_noise")
    d["core"] = d.organ.map(core_of)
    conds = ["original", "noise_target", "noise_pair", "noise_image"]
    labels = ["original", "organ\nnoised", "both organs\nnoised", "whole image\nnoise"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, core in zip(axes, CORES):
        s = d[d.core == core]
        for i, c in enumerate(conds):
            v = s[s.condition == c]
            vals = v.size_vs_original.values
            ax.bar(i, np.median(vals), width=0.55, color=BLUE if i == 0 else ORANGE, alpha=0.9, zorder=2)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.14, 0.14, len(vals)), vals, s=22, color=INK, zorder=3,
                       edgecolor=SURF, linewidth=0.8)
            ax.text(i, -0.19, f"empty {int(v['empty'].sum())}/{len(v)}", ha="center", va="top", fontsize=8.5, color=INK2,
                    transform=ax.get_xaxis_transform())
        ax.axhline(1, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels)
        ax.set_title(core)
        ax.set_ylim(0, 1.3)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mask size vs. original")
    head(fig, "Row 13 - the organ is replaced by noise",
         "3 subjects; bar = median, dots = each organ; dashed line = the unmodified prediction")
    fig.subplots_adjust(bottom=0.22, top=0.78, wspace=0.08)
    fig.savefig(OUT / "fig_row13_noise.png", dpi=170)
    plt.close(fig)


def fig_row14_blend():
    d = load("row14_embedding_intervention")
    b = d[d.variant == "blend"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, core in zip(axes, CORES):
        s = b[b.pair == core]
        for col, color, name in (("cov_left_organ", BLUE, "mask covers the LEFT organ"),
                                 ("cov_right_organ", ORANGE, "mask covers the RIGHT organ")):
            g = s.groupby("alpha")[col].mean()
            ax.plot(g.index, g.values, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=name, zorder=3)
            for _, sub in s.groupby("subject_id"):
                ax.plot(sub.alpha, sub[col], color=color, linewidth=0.8, alpha=0.3, zorder=2)
        ax.axvline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_title(core)
        ax.set_xlabel('alpha  (0 = "left", 1 = "right")')
        ax.set_xticks([-0.5, 0.25, 0.5, 0.75, 1.5])
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("fraction of that organ covered")
    head(fig, "Row 14 - blending the left and right text embeddings",
         "mean of 3 subjects (bold), each subject faint; -0.5 and 1.5 extrapolate beyond the two prompts")
    foot_legend(fig, axes[0], ncol=2)
    fig.subplots_adjust(top=0.78, bottom=0.27, wspace=0.08)
    fig.savefig(OUT / "fig_row14_blend.png", dpi=170)
    plt.close(fig)


def fig_row14_patches():
    d = load("row14_embedding_intervention")
    rows = [("single_layer_update", k, f"update of layer {k + 1} only") for k in range(6)] + \
           [("updates_from_layer", 0, "updates of layers 1-6"), ("updates_from_layer", 3, "updates of layers 4-6")]
    cols = [(c, dst) for c in CORES for dst in ("left", "right")]
    M = np.full((len(rows), len(cols)), np.nan)
    N = np.zeros_like(M)
    for i, (var, k, _) in enumerate(rows):
        for j, (core, dst) in enumerate(cols):
            s = d[(d.variant == var) & (d.layer == k) & (d.pair == core) & (d.dst == dst)]
            if len(s):
                M[i, j] = s.moved_to_source_side.astype(float).mean()
                N[i, j] = len(s)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{int(round(M[i, j] * N[i, j]))}/{int(N[i, j])}", ha="center", va="center",
                    color="white" if M[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c}\n{'left -> right' if dst == 'left' else 'right -> left'}" for c, dst in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows])
    ax.axhline(5.5, color=SURF, linewidth=4)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    head(fig, "Row 14 - patching the decoder layers",
         "masks that switched to the other side (of 3 subjects); the prompt keeps its words,\nthe patched layers use the other prompt's update")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="share of subjects")
    fig.subplots_adjust(top=0.80, left=0.24, right=0.93, bottom=0.12)
    fig.savefig(OUT / "fig_row14_patches.png", dpi=170)
    plt.close(fig)


def fig_row15():
    a = load("row15_attention_enrichment")
    a["core"] = a.prompt_organ.map(core_of)
    a["kind"] = np.where(a.is_matched, "named organ", np.where(a.is_wrong_side, "same organ, other side", "different organ"))
    order = [("named organ", BLUE), ("same organ, other side", ORANGE), ("different organ", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, core in zip(axes, CORES):
        s = a[a.core == core]
        for kind, color in order:
            g = s[s.kind == kind].groupby("layer").enrichment.mean()
            ax.plot(g.index, g.values, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=kind, zorder=3)
        ax.axhline(1, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_title(core)
        ax.set_xlabel("decoder layer")
        ax.set_xticks(range(1, 7))
    axes[0].set_ylabel("attention inside region / average")
    head(fig, "Row 15 - where the prompt's cross-attention lands",
         "mean of 3 subjects; 1 (dashed) = as much attention as anywhere else in the volume")
    foot_legend(fig, axes[0], ncol=3)
    fig.subplots_adjust(top=0.78, bottom=0.27, wspace=0.08)
    fig.savefig(OUT / "fig_row15_attention.png", dpi=170)
    plt.close(fig)


def fig_row16():
    l = load("row16_layer_distance")
    l = l[l.scope == "all_patches"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, core in zip(axes, CORES):
        s = l[l.pair.str.startswith(core)]
        lr = s[s.pair.str.contains("left vs right")]
        ref = s[s.pair.str.contains("reference")]
        for sub, color, name in ((lr, BLUE, "left vs. right (same organ)"), (ref, ORANGE, "organ vs. organ (reference)")):
            for _, one in sub.groupby("subject_id"):
                ax.plot(one.layer, one.cosine_distance, color=color, linewidth=0.8, alpha=0.3, zorder=2)
            g = sub.groupby("layer").cosine_distance.mean()
            ax.plot(g.index, g.values, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=name, zorder=3)
        ax.set_title(core)
        ax.set_xlabel("decoder layer  (0 = input)")
        ax.set_xticks(range(0, 7))
        ax.set_ylim(0, 0.6)
    axes[0].set_ylabel("cosine distance")
    head(fig, "Row 16 - how different left and right are, layer by layer",
         "cosine distance between the two query states; mean of 3 subjects (bold), each subject faint")
    foot_legend(fig, axes[0], ncol=2)
    fig.subplots_adjust(top=0.78, bottom=0.27, wspace=0.08)
    fig.savefig(OUT / "fig_row16_distance.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_row13(); fig_row14_blend(); fig_row14_patches(); fig_row15(); fig_row16()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
    print("per-organ, per-subject figures: scripts/make_organ_rows13_16_figures.py")
