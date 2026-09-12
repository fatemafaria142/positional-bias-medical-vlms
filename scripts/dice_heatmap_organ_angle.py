"""
Organ x rotation-angle Dice heatmaps for VoxTell, one per dataset -- combines
all 3 annotators' prediction CSVs (faria, nipu, hridoy; same experiment
design, different subjects each), averaging Dice per (organ, angle) cell
across every pooled subject/annotator row.

Organs are sorted by their own mean Dice (descending, left = best) so the
heatmap reads as a ranked table, not an arbitrary list. Color is a
DIVERGING red<->blue ramp (the data-viz skill's validated diverging pair)
centered on Dice = 0.5 -- the conventional pass/fail pivot for a
segmentation overlap score -- with a light neutral midpoint, rather than
the skill's default single-hue sequential rule for plain magnitude: this
is a deliberate exception because it's a two-color heatmap by request, and
Dice has a real, meaningful threshold to diverge around (unlike an
arbitrary magnitude with no such pivot). A 2px white gap sits between
every cell (surface gap, not a border stroke); cell text color is chosen
per-cell from the actual sampled color's luminance, since a diverging map
is dark at BOTH ends and light in the middle (the old "light->dark"
single-hue assumption no longer holds).

Layout is horizontal (wider than tall): organs run left-to-right across
many columns, rotation angle runs top-to-bottom across a fixed 7 rows --
appropriate when there are dozens of organs, since a portrait layout would
put them all in one long, hard-to-scan column instead.

Usage:
    python scripts/dice_heatmap_organ_angle.py
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
PERSONS = ["faria", "hridoy", "nipu"]

# Diverging red<->blue ramp (ColorBrewer RdBu-11, a perceptually validated
# diverging pair), centered on a light neutral midpoint at Dice = 0.5.
# Low Dice (failing predictions) -> red; high Dice (good predictions) -> blue.
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
    {"name": "amos22_mri", "label": "AMOS22 MRI"},
    {"name": "neurite_oasis", "label": "Neurite OASIS"},
]


def load_combined(dataset_name: str) -> pd.DataFrame:
    frames = []
    for person in PERSONS:
        path = REPO_ROOT / f"predictions_voxtell_{dataset_name}_rotation_{person}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"No prediction CSVs found for {dataset_name}")
    return pd.concat(frames, ignore_index=True)


def title_case_organ(name: str) -> str:
    return " ".join(w if w.isupper() else w.capitalize() for w in name.split())


def plot_heatmap(dataset_name: str, label: str):
    df = load_combined(dataset_name)
    pivot = (df.groupby(["organ", "rotation_angle_deg"])["dice"].mean()
             .unstack("rotation_angle_deg").reindex(columns=ANGLES_DEG))
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__", ascending=False).drop(columns="__mean__")
    # Transpose: organs (many) run left-to-right along columns, angles (fixed
    # at 7) run top-to-bottom along rows -- a wide horizontal layout, since
    # with dozens of organs a portrait layout puts them in one long unreadable
    # column instead.
    pivot = pivot.T

    organs = [title_case_organ(o) for o in pivot.columns]
    data = pivot.values

    n_rows, n_cols = data.shape  # n_rows = len(ANGLES_DEG), n_cols = n_organs
    # Horizontal sizing: width follows the organ count (many columns), height
    # follows the fixed 7-angle row count -- so the figure comes out wide
    # rather than tall, matching the "many organs" case.
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
    ax.set_title(f"VoxTell Dice by Organ x Rotation Angle\n{label}",
                 fontsize=15, fontweight="bold", pad=14)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    # Cell value labels -- bumped up from the original 8.3/9.3pt so each
    # number reads clearly at a glance instead of squinting.
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
    # then re-anchor ax at the same distance from the TOP edge it already has
    # post-tight_layout -- so the new inches land entirely below everything
    # tight_layout already placed, instead of negotiating a shared layout
    # with a second axes (which produced both an earlier overlap and a huge
    # blank gap on tall figures).
    #
    # tight_layout's own estimate of how far a 45-degree rotated tick label
    # drops below the axes can under-measure for the longest label in the
    # set (confirmed: "Inferior Vena Cava" visibly dipped into the colorbar
    # panel even though shorter labels cleared it fine) -- so a dedicated
    # `label_overflow_buffer` band sits directly under tight_layout's own
    # margin, purely as slack for that under-measurement, with the colorbar
    # itself only in the strip below THAT.
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

    # Colorbar panel occupies the bottom `cbar_h` inches (the
    # `label_overflow_buffer` band sits above this, already accounted for in
    # ax's new position above). Bar itself sits near the top of that budget,
    # leaving room below it for its own tick numbers + "Mean Dice..." label.
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
    # stale tick-label bounding boxes cached from the pre-reposition layout
    # and clip long rotated labels (e.g. "Inferior Vena Cava" got cropped to
    # "or Vena Cava").
    fig.canvas.draw()
    out_path = OUT_DIR / f"dice_heatmap_{dataset_name}.png"
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    for ds in DATASETS:
        plot_heatmap(ds["name"], ds["label"])
