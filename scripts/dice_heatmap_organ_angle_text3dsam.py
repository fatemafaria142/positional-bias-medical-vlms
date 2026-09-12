"""
Organ x rotation-angle Dice heatmaps for Text3DSAM, one per dataset
(MR_TotalSeg, CT_HaN-Seg, CT_TotalSeg_cardiac) -- same design as
dice_heatmap_organ_angle.py (the VoxTell version), adapted to read from the
single combined predictions_text3dsam_<person>.csv instead of one CSV per
dataset per annotator (Text3DSAM only has one annotator, faria, and one
manifest covering all 3 datasets).

Organs are sorted by their own mean Dice (descending, left = best) so the
heatmap reads as a ranked table, not an arbitrary list. Color is a
DIVERGING red<->blue ramp (ColorBrewer RdBu, a perceptually validated
diverging pair) centered on Dice = 0.5 -- the conventional pass/fail pivot
for a segmentation overlap score -- with a light neutral midpoint, the same
convention used for the VoxTell heatmaps and the inference-time table, so
all 3 read consistently. Cell text color is chosen per-cell from the actual
sampled color's luminance, since a diverging map is dark at BOTH ends and
light in the middle.

Layout is horizontal (wider than tall): organs run left-to-right across
columns, rotation angle runs top-to-bottom across a fixed 7 rows.

Usage:
    python scripts/dice_heatmap_organ_angle_text3dsam.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]
PREDICTIONS_CSV = REPO_ROOT / "predictions_text3dsam_faria.csv"

# Diverging red<->blue ramp (ColorBrewer RdBu-11), centered on a light
# neutral midpoint at Dice = 0.5. Low Dice (failing) -> red; high Dice
# (succeeding) -> blue. Identical to the VoxTell heatmap's palette.
DIVERGING_RAMP = [
    "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
    "#f7f7f7",
    "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061",
]
DICE_CMAP = mcolors.LinearSegmentedColormap.from_list("dice_diverging", DIVERGING_RAMP, N=256)


def _text_color_for(rgba) -> str:
    """Pick white/dark cell text from the ACTUAL sampled color's relative
    luminance -- required for a diverging map, which is dark at both ends
    and light in the middle (unlike a single-hue ramp's monotonic light->dark)."""
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else "#0b0b0b"


DATASETS = [
    {"name": "MR_TotalSeg", "label": "MR TotalSeg"},
    {"name": "CT_HaN-Seg", "label": "CT HaN-Seg"},
    {"name": "CT_TotalSeg_cardiac", "label": "CT TotalSeg Cardiac"},
]


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        raise SystemExit(f"Not found: {PREDICTIONS_CSV} -- run the Text3DSAM Kaggle "
                          f"inference notebook first and copy the CSV here.")
    return pd.read_csv(PREDICTIONS_CSV)


def title_case_organ(name: str) -> str:
    return " ".join(w if w.isupper() else w.capitalize() for w in name.split())


def plot_heatmap(df: pd.DataFrame, dataset_name: str, label: str):
    ds_df = df[df["dataset"] == dataset_name]
    if ds_df.empty:
        print(f"[skip] {dataset_name}: no rows in {PREDICTIONS_CSV.name}")
        return

    pivot = (ds_df.groupby(["organ", "rotation_angle_deg"])["dice"].mean()
             .unstack("rotation_angle_deg").reindex(columns=ANGLES_DEG))
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__", ascending=False).drop(columns="__mean__")
    # Transpose: organs run left-to-right along columns, angles top-to-bottom
    # along rows -- a wide horizontal layout.
    pivot = pivot.T

    organs = [title_case_organ(o) for o in pivot.columns]
    data = pivot.values

    n_rows, n_cols = data.shape  # n_rows = len(ANGLES_DEG), n_cols = n_organs
    fig_w = max(6.0, 0.62 * n_cols + 2.2)
    main_h = max(3.2, 0.62 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, main_h))

    norm = mcolors.Normalize(vmin=0, vmax=1)
    mesh = ax.pcolormesh(data, cmap=DICE_CMAP, norm=norm,
                          edgecolors="white", linewidth=2)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(organs, fontsize=11, rotation=45, ha="right", rotation_mode="anchor")
    for t in ax.get_xticklabels():
        t.set_clip_on(False)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([f"{a}°" for a in ANGLES_DEG], fontsize=12.5)
    ax.invert_yaxis()
    ax.set_ylabel("Rotation angle", fontsize=13.5, labelpad=10)
    ax.set_title(f"Text3DSAM Dice by Organ x Rotation Angle\n{label}",
                 fontsize=15, fontweight="bold", pad=14)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = _text_color_for(DICE_CMAP(norm(val)))
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center",
                     fontsize=12.5, color=text_color, fontweight="bold")

    fig.tight_layout()

    # Grow the figure to add a fixed strip at the BOTTOM for the colorbar,
    # then re-anchor ax at the same distance from the TOP edge it already
    # has post-tight_layout -- see dice_heatmap_organ_angle.py (VoxTell
    # version) for the full rationale: a fraction-of-main-axes reservation
    # either overlaps rotated labels or leaves a huge blank gap depending on
    # organ count, so this uses a fixed absolute-inch budget instead, with a
    # dedicated buffer band to absorb tight_layout's own under-measurement
    # of how far the longest rotated label drops.
    cbar_h = 0.9
    label_overflow_buffer = 0.6
    total_added = cbar_h + label_overflow_buffer
    old_h = main_h
    new_h = old_h + total_added
    ax_pos = ax.get_position()  # fractions of old_h
    dist_from_top_y1 = old_h * (1 - ax_pos.y1)
    dist_from_top_y0 = old_h * (1 - ax_pos.y0)
    fig.set_size_inches(fig_w, new_h)
    new_y1 = 1 - dist_from_top_y1 / new_h
    new_y0 = 1 - dist_from_top_y0 / new_h
    ax.set_position([ax_pos.x0, new_y0, ax_pos.width, new_y1 - new_y0])

    inset_w = min(ax_pos.width * 0.45, 3.2 / fig_w)
    bar_bottom_in = 0.5
    bar_height_in = 0.18
    cbar_ax = fig.add_axes([ax_pos.x0 + (ax_pos.width - inset_w) / 2,
                             bar_bottom_in / new_h, inset_w, bar_height_in / new_h])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean Dice (red = fails, blue = succeeds)", fontsize=11, labelpad=6)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=10)

    # Force a redraw at the FINAL (post-repositioning) axes layout before
    # savefig's bbox_inches="tight" measures extents -- otherwise it can use
    # stale tick-label bounding boxes and clip long rotated labels.
    fig.canvas.draw()
    out_path = OUT_DIR / f"dice_heatmap_text3dsam_{dataset_name}.png"
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    df = load_predictions()
    for ds in DATASETS:
        plot_heatmap(df, ds["name"], ds["label"])
