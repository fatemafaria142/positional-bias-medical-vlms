"""
Combines all 3 annotators' VoxTell rotation-sweep prediction CSVs (faria, nipu,
hridoy -- same experiment design, different subjects each) per dataset, then:

  1. Plots mean Dice vs. rotation angle as a grouped bar chart (one bar group
     per angle, one bar per dataset), with error bars (+/- 1 std across all
     pooled organ/subject/annotator rows at that angle).
  2. Writes a TOTAL-inference-time-per-angle table (summed seconds across
     every organ/subject/annotator row at that angle -- an exact total, not
     an average, so it directly answers "which dataset/angle is slower"),
     one column per dataset (just the 2 real datasets -- no combined
     "All datasets" column, which read as a misleading 3rd dataset) plus a
     bottom bolded grand-Total row. Written as a CSV and a rendered PNG
     table, colored with the exact same diverging red<->blue scale as
     dice_heatmap_organ_angle.py's heatmaps (red = slow, blue = fast, by the
     same red-is-bad/blue-is-good convention as that script's Dice coloring)
     -- a color-coded table, not a heatmap grid.

Usage:
    python scripts/summarize_dice_and_inference.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dice_heatmap_organ_angle import DICE_CMAP, _text_color_for  # same palette as the heatmaps

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180]
PERSONS = ["faria", "hridoy", "nipu"]

DATASETS = [
    {"name": "amos22_mri", "label": "AMOS22 MRI", "color": "#2a78d6"},
    {"name": "neurite_oasis", "label": "Neurite OASIS", "color": "#eb6834"},
]

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})


def load_combined(dataset_name: str) -> pd.DataFrame:
    frames = []
    for person in PERSONS:
        path = REPO_ROOT / f"predictions_voxtell_{dataset_name}_rotation_{person}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"No prediction CSVs found for {dataset_name}")
    return pd.concat(frames, ignore_index=True)


def plot_dice_bar_chart(combined: dict):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_groups = len(ANGLES_DEG)
    n_series = len(DATASETS)
    bar_width = 0.7 / n_series
    x = np.arange(n_groups)

    for i, ds in enumerate(DATASETS):
        df = combined[ds["name"]]
        stats = df.groupby("rotation_angle_deg")["dice"].agg(["mean", "std"]).reindex(ANGLES_DEG)
        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(x + offset, stats["mean"], width=bar_width * 0.92,
                       yerr=stats["std"], capsize=3,
                       color=ds["color"], edgecolor="none", label=ds["label"],
                       error_kw={"ecolor": "#52514e", "elinewidth": 1.2, "alpha": 0.8})
        for rect, val in zip(bars, stats["mean"]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.025,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10.5,
                    color="#0b0b0b", fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}°" for a in ANGLES_DEG], fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean Dice score", fontsize=13, labelpad=10)
    ax.set_xlabel("Rotation angle", fontsize=13, labelpad=10)
    ax.set_title("VoxTell: Mean Dice vs. Rotation Angle", fontsize=16, fontweight="bold", pad=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color="#e3e2dc", linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=12, bbox_to_anchor=(1.0, 1.05))

    fig.tight_layout()
    out_path = OUT_DIR / "mean_dice_by_angle.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def build_inference_table(combined: dict):
    """One row per angle = TOTAL (summed, not averaged) inference_seconds
    across every organ/subject/annotator row at that angle, one column per
    real dataset -- an exact total, so summing a column gives that dataset's
    exact total inference time, not an estimate from an average.

    Below the angle rows, three summary rows spell out the scale and the
    total/per-subject cost in BOTH seconds and minutes, precisely to head
    off "wait, how many subjects/organs is this actually?" confusion:
      - "Total (s)"              exact total, seconds
      - "Total (min)"            same total, minutes (easier to read at a
                                  glance for a whole dataset's run)
      - "Mean / subject (min)"   Total (min) / subject count -- "how long
                                  does ONE subject's full sweep take"

    Also returns a `meta` dict (subjects/organs/angles/predictions per
    dataset) used to caption the table with the exact experiment scale.
    """
    cols = {}
    meta = {}
    for ds in DATASETS:
        df = combined[ds["name"]]
        cols[ds["label"]] = df.groupby("rotation_angle_deg")["inference_seconds"].sum().reindex(ANGLES_DEG)
        n_subjects = df["subject_id"].nunique()
        n_organs = df.groupby("subject_id")["organ"].nunique().iloc[0]
        meta[ds["label"]] = {
            "n_subjects": n_subjects,
            "n_organs": n_organs,
            "n_angles": df["rotation_angle_deg"].nunique(),
            "n_predictions": len(df),
        }

    table = pd.DataFrame(cols)
    table.index = [f"{a}°" for a in ANGLES_DEG]
    table.index.name = "Rotation angle"

    total_s = table.sum()
    table.loc["Total (s)"] = total_s
    table.loc["Total (min)"] = total_s / 60
    table.loc["Mean / subject (min)"] = [
        (total_s[label] / 60) / meta[label]["n_subjects"] for label in table.columns
    ]
    return table.round(3), meta


SUMMARY_ROWS = ("Total (s)", "Total (min)", "Mean / subject (min)")


def render_table_png(table: pd.DataFrame, meta: dict):
    """Color-coded TABLE (not a heatmap grid): per-angle cells (in seconds)
    use the exact same diverging red<->blue scale as
    dice_heatmap_organ_angle.py, normalized GLOBALLY across both dataset
    columns (not per-column) so a glance at color alone already shows which
    dataset runs slower overall. Direction is inverted from the Dice
    heatmap's raw value mapping since here LOW time is good: fastest
    angle/dataset -> blue end, slowest -> red end, keeping the same
    red-is-bad/blue-is-good convention.

    The 3 summary rows (SUMMARY_ROWS) are kept neutral (not color-scaled --
    they're on different scales/units than the angle rows and each other, so
    coloring them on the same axis would mislead, not clarify) and visually
    separated with a thicker top border. A caption line under the title
    spells out each dataset's exact scale (subjects x organs x angles =
    predictions) specifically to head off "wait, how many subjects is this?"
    confusion.
    """
    angle_rows = table.drop(index=list(SUMMARY_ROWS))
    vmin, vmax = angle_rows.values.min(), angle_rows.values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    n_rows, n_cols = table.shape
    row_h_in = 0.5
    # Every vertical element below is placed by an explicit ABSOLUTE-INCH
    # budget converted to a figure fraction, and none of them exceed [0, 1] --
    # deliberately avoiding ax.set_title()'s default behavior of floating the
    # title ABOVE y=1 via its `pad` (in points, outside the axes box). That
    # combined with savefig(bbox_inches="tight") auto-expanding the canvas to
    # include the spilled-over title created a mismatch: fractions computed
    # against the pre-expansion fig_h no longer matched the actual (taller,
    # tight-cropped) saved image, which is what caused the large, seemingly
    # inexplicable gap between the title and the caption/table below it.
    top_margin_in = 0.15
    title_h_in = 0.62
    gap1_in = 0.18
    caption_h_in = 0.32 * len(meta)  # one caption line per dataset
    gap2_in = 0.15
    table_h_in = row_h_in * (n_rows + 1)  # +1 for the header row
    bottom_margin_in = 0.1
    fig_h = (top_margin_in + title_h_in + gap1_in + caption_h_in
             + gap2_in + table_h_in + bottom_margin_in)
    fig, ax = plt.subplots(figsize=(3.1 + 2.1 * n_cols, fig_h))
    ax.axis("off")
    # Pin the axes to the full figure canvas (matplotlib's default subplot
    # margins otherwise leave the axes box smaller than the figure, so the
    # bbox fractions below -- computed relative to the FULL figure height --
    # would land in the wrong place within a shrunk axes).
    ax.set_position([0, 0, 1, 1])

    col_labels = [table.index.name] + list(table.columns)

    def fmt_row(idx, row):
        unit_is_seconds = idx not in SUMMARY_ROWS or idx == "Total (s)"
        suffix = "s" if unit_is_seconds else " min"
        return [idx] + [f"{v:,.2f}{suffix}" for v in row]

    cell_text = [fmt_row(idx, row) for idx, row in zip(table.index, table.values)]

    # Explicit bbox (fractions of the axes) instead of loc="center" -- with
    # loc="center" the table keeps its own natural size and floats centered
    # in whatever the axes' box happens to be, leaving a visible gap between
    # the title/caption and the table whenever the axes is taller than the
    # table itself. bbox=[...] stretches the table to fill a region anchored
    # at `bottom_margin_in` above the figure bottom, so no dead space appears
    # either above or below it.
    table_bottom = bottom_margin_in / fig_h
    table_frac = table_h_in / fig_h
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                   bbox=[0, table_bottom, 1, table_frac])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)

    header_bg = "#0b0b0b"
    summary_bg = "#e1e0d9"
    row_index = {i + 1: idx for i, idx in enumerate(table.index)}  # +1: row 0 is the header

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2.5)  # white surface gap between cells, same convention as the heatmaps

        if row == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue

        if row_index[row] in SUMMARY_ROWS:
            cell.set_facecolor(summary_bg)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#0b0b0b")
            if row_index[row] == SUMMARY_ROWS[0]:  # "Total (s)": first summary row, separate from angle rows
                cell.set_linewidth(2.5)
                cell.visible_edges = "BTLR"
                cell.set_edgecolor("#0b0b0b")
            continue

        if col == 0:
            cell.set_facecolor("#fcfcfb")
            cell.get_text().set_fontweight("medium")
            cell.get_text().set_color("#0b0b0b")
        else:
            val = float(cell.get_text().get_text().replace(",", "").rstrip("s"))
            # Inverted so the FASTEST (lowest-time) cell lands on the blue
            # end and the SLOWEST on the red end -- opposite numeric
            # direction from the Dice heatmap (there, high value = good),
            # but the same red-is-bad/blue-is-good color convention.
            color = DICE_CMAP(1 - norm(val))
            cell.set_facecolor(color)
            cell.get_text().set_color(_text_color_for(color))
            cell.get_text().set_fontweight("medium")

    # Title and caption as fig.text (va="top", anchored within [0, 1]) rather
    # than ax.set_title() -- see the note above the fig_h budget for why.
    title_top = 1 - top_margin_in / fig_h
    fig.text(0.5, title_top, "VoxTell: Total Inference Time by Rotation Angle\n(red = slower, blue = faster)",
              ha="center", va="top", fontsize=15.5, fontweight="bold", color="#0b0b0b")

    # Caption: exact experiment scale per dataset, so the reader never has to
    # guess how many subjects/organs went into these numbers.
    caption_top = 1 - (top_margin_in + title_h_in + gap1_in) / fig_h
    caption_lines = [
        f"{label}: {m['n_subjects']} subjects x {m['n_organs']} organs x {m['n_angles']} angles "
        f"= {m['n_predictions']:,} predictions"
        for label, m in meta.items()
    ]
    fig.text(0.5, caption_top, "\n".join(caption_lines),
              ha="center", va="top", fontsize=11, color="#52514e")

    out_path = OUT_DIR / "mean_inference_time_by_angle.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    combined = {ds["name"]: load_combined(ds["name"]) for ds in DATASETS}

    plot_dice_bar_chart(combined)

    table, meta = build_inference_table(combined)
    csv_path = OUT_DIR / "mean_inference_time_by_angle.csv"
    table.to_csv(csv_path)
    print(f"[done] wrote {csv_path}")
    render_table_png(table, meta)

    print()
    print(table)
    print()
    for label, m in meta.items():
        print(f"{label}: {m['n_subjects']} subjects x {m['n_organs']} organs x "
              f"{m['n_angles']} angles = {m['n_predictions']:,} predictions")
