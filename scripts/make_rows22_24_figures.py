"""
Local (no GPU, no torch, no voxtell) visualization for PLAN ROWS 22-24 (the
model-internals/circuit suite) of voxtell-oasis-mechanistic-rows13-24-kaggle.ipynb /
voxtell-oasis-mechanistic-experiments-kaggle.ipynb.

Run the Kaggle notebook, download its results zip (or just the three CSVs below) from
the Output tab, then point this script at that directory -- unlike
scripts/make_rows17_21_figures.py, none of rows 22-24 need the original scan/GT at all
(no row0_baseline_masks.csv, no .npz maps, no nibabel) -- these are all plain tables.

Rows covered:
  22  linear probe for left/right at every decoder layer   -- is laterality DECODABLE, not just different?
  23  "right - left" embedding geometry across organs        -- one shared direction, or one per organ?
  24  cross-attention head ablation                           -- which SPECIFIC head is responsible?

Output: local_plots_mechanistic/rows22_24/row{22,23,24}_*.png

Usage:
    python scripts/make_rows22_24_figures.py RESULTS_DIR [--suffix neurite_oasis] [--out-dir DIR]

    # e.g. after downloading the Kaggle output of voxtell-oasis-mechanistic-rows13-24-kaggle.ipynb:
    python scripts/make_rows22_24_figures.py Result-13-to-24
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same fixed categorical/diverging/sequential palette as scripts/make_rows17_21_figures.py
# (dataviz skill's validated reference palette -- kept duplicated here, not imported, so this
# script stays self-contained like every other make_*_figures.py in this repo).
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
GRAY_MID = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, GRAY_MID, RED])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", BLUE, "#0d366b"])


def style_bars(ax, title, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def savefig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def rel(p: Path, repo: Path) -> str:
    try:
        return str(p.relative_to(repo))
    except ValueError:
        return str(p)


def read_csv(results_dir: Path, name: str, suffix: str) -> pd.DataFrame:
    path = results_dir / f"{name}_{suffix}.csv"
    if not path.exists():
        print(f"[skip] not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------------------
# row 22 -- linear probing: is left/right linearly decodable at each layer?
# --------------------------------------------------------------------------------------------------
def figure_row22(df: pd.DataFrame, out_dir: Path):
    d = df.sort_values("layer")
    n_pairs = int(d["n_pairs"].iloc[0]) if "n_pairs" in d and len(d) else None
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.plot(d.layer, d.cv_accuracy, "-o", color=BLUE, linewidth=2, markersize=6,
            label="cross-validated (leave-one-pair-out)")
    ax.plot(d.layer, d.train_accuracy, "-o", color=ORANGE, linewidth=2, markersize=6,
            label="train (fit on all pairs)")
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.5, -0.14, "dashed line = 0.5 (chance)", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, ha="center", va="top")
    ax.set_ylim(-0.02, 1.05)
    title = "Row 22 -- is left/right linearly decodable, by decoder layer?"
    if n_pairs is not None:
        title += f"\n(n_pairs={n_pairs} -- small-sample, read as indicative)"
    style_bars(ax, title, "leave-one-pair-out probe accuracy")
    ax.set_xlabel("decoder state (0 = text embedding before cross-attention, 1-6 = after each layer)",
                  fontsize=10, color=INK2)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    return savefig(fig, out_dir / "row22_laterality_probing.png")


# --------------------------------------------------------------------------------------------------
# row 23 -- embedding geometry: one shared "left -> right" direction, or one per organ?
# --------------------------------------------------------------------------------------------------
def figure_row23(df: pd.DataFrame, out_dir: Path):
    pairs = df[df.kind == "pair"]
    summary = df[df.kind == "summary"].iloc[0]
    organs = sorted(set(pairs.organ_a) | set(pairs.organ_b))
    n = len(organs)
    idx = {o: i for i, o in enumerate(organs)}
    mat = np.full((n, n), np.nan)
    for _, r in pairs.iterrows():
        i, j = idx[r.organ_a], idx[r.organ_b]
        mat[i, j] = mat[j, i] = r.cosine_similarity
    np.fill_diagonal(mat, 1.0)

    fig = plt.figure(figsize=(max(7.5, 0.55 * n + 2.5), max(6.5, 0.55 * n + 1.5)), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.28, 0.08, 0.66, 0.66])
    vmin = float(np.nanmin(mat))
    im = ax.imshow(mat, cmap=SEQ_BLUE, vmin=vmin, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(organs, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(organs, fontsize=8.5)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if mat[i, j] > (vmin + 1.0) / 2 else INK)
    ax.set_title("cosine similarity between organs' \"right - left\" embedding-difference vectors",
                 fontsize=11, fontweight="bold", color=INK, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=8.5)

    ax2 = fig.add_axes([0.03, 0.80, 0.94, 0.16])
    ax2.axis("off")
    headline = ("ONE shared \"left -> right\" direction" if summary.coherence_ratio > 0.7 else
                "a DIFFERENT direction per organ" if summary.coherence_ratio < 0.3 else
                "a partially shared direction")
    ax2.text(0.5, 0.75, f"coherence_ratio = {summary.coherence_ratio:.3f}  ->  {headline}",
             ha="center", va="center", fontsize=14, fontweight="bold", color=INK, transform=ax2.transAxes)
    ax2.text(0.5, 0.15, f"mean pairwise cosine similarity = {summary.mean_cosine_similarity:.3f}  "
                         f"across {int(summary.n_organs)} organs  (1.0 = identical direction, 0.0 = unrelated)",
             ha="center", va="center", fontsize=10.5, color=INK2, transform=ax2.transAxes)
    return savefig(fig, out_dir / "row23_embedding_geometry.png")


# --------------------------------------------------------------------------------------------------
# row 24 -- cross-attention head ablation: which specific head?
# --------------------------------------------------------------------------------------------------
def figure_row24_heatmaps(df: pd.DataFrame, out_dir: Path):
    organs = sorted(df.organ.unique(), key=lambda o: (o.split(None, 1)[1], o.split(None, 1)[0]) if " " in o else (o,))
    paths = []
    for layer, dl in df.groupby("layer"):
        heads = sorted(dl["head"].unique())
        mat_base = np.full((len(organs), len(heads)), np.nan)
        mat_gt = np.full((len(organs), len(heads)), np.nan)
        for _, r in dl.iterrows():
            i, j = organs.index(r.organ), heads.index(r["head"])
            mat_base[i, j] = r.dice_vs_baseline
            mat_gt[i, j] = r.dice_vs_gt
        fig = plt.figure(figsize=(max(8, 0.55 * len(heads) * 2 + 3), max(6, 0.3 * len(organs) + 2)), dpi=170)
        fig.patch.set_facecolor(SURFACE)
        # dice_vs_baseline values cluster tightly near 1.0 (most single-head ablations barely move
        # the answer) -- a fixed 0-1 scale washes that out, so this panel's scale follows the data's
        # own range instead, to actually show which heads matter more than others.
        base_vmin = float(np.nanmin(mat_base)) if np.isfinite(mat_base).any() else 0.0
        for k, (mat, cmap, label, vmin, vmax) in enumerate([
            (mat_base, DIVERGING, "Dice vs. the unablated baseline\n(1.0 = ablating this head changed nothing)", base_vmin, 1),
            (mat_gt, SEQ_BLUE, "Dice vs. ground truth\nafter ablating this head", 0, 1),
        ]):
            ax = fig.add_axes([0.30 + k * 0.38, 0.12, 0.33, 0.78])
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(heads))); ax.set_xticklabels(heads, fontsize=9)
            ax.set_xlabel("head", fontsize=10, color=INK2)
            if k == 0:
                ax.set_yticks(range(len(organs))); ax.set_yticklabels(organs, fontsize=8)
            else:
                ax.set_yticks([])
            ax.set_title(label, fontsize=10.5, fontweight="bold", color=INK, pad=8)
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).ax.tick_params(labelsize=8)
        fig.suptitle(f"Row 24 -- per-head ablation, decoder layer {layer}", fontsize=13, fontweight="bold", color=INK, y=1.02)
        paths.append(savefig(fig, out_dir / f"row24_head_ablation_heatmap_layer{layer}.png"))
    return paths


def figure_row24_call_flips(df: pd.DataFrame, out_dir: Path):
    """How many organs flip away from the correct call ('a' = still lands on the named organ) when
    each head is ablated -- the single clearest "which head matters" summary across all organs."""
    flips = (df.call != "a").groupby(df["head"]).sum().reindex(sorted(df["head"].unique()), fill_value=0)
    n_organs = df.organ.nunique()
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.bar(flips.index.astype(str), flips.values, color=BLUE, width=0.6, zorder=3)
    style_bars(ax, "Row 24 -- how many organs' masks change call when this head is ablated",
               f"organs affected (out of {n_organs})")
    ax.set_xlabel("head", fontsize=10.5, color=INK2)
    fig.tight_layout()
    return savefig(fig, out_dir / "row24_call_flips_by_head.png")


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="directory containing row22/23/24 CSVs")
    ap.add_argument("--suffix", default="neurite_oasis", help="dataset suffix in the CSV file names (default: neurite_oasis)")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help="repo root, only used to print relative output paths")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <repo>/local_plots_mechanistic/rows22_24")
    args = ap.parse_args()
    out_dir = args.out_dir or (args.repo / "local_plots_mechanistic" / "rows22_24")

    row22 = read_csv(args.results_dir, "row22_laterality_probing", args.suffix)
    row23 = read_csv(args.results_dir, "row23_embedding_geometry", args.suffix)
    row24 = read_csv(args.results_dir, "row24_attention_head_ablation", args.suffix)

    if not row22.empty:
        print("wrote", rel(figure_row22(row22, out_dir), args.repo))
    if not row23.empty:
        print("wrote", rel(figure_row23(row23, out_dir), args.repo))
    if not row24.empty:
        for p in figure_row24_heatmaps(row24, out_dir):
            print("wrote", rel(p, args.repo))
        print("wrote", rel(figure_row24_call_flips(row24, out_dir), args.repo))

    if row22.empty and row23.empty and row24.empty:
        print("none of row22/23/24's CSVs were found in", args.results_dir)


if __name__ == "__main__":
    main()
