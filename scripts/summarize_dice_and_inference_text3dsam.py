"""
Text3DSAM dataset-wise summary -- same design as
summarize_dice_and_inference.py (the VoxTell version), adapted to read the
single combined predictions_text3dsam_<person>.csv (Text3DSAM only has one
annotator, faria, and one manifest covering all 3 datasets) instead of one
CSV per dataset per annotator:

  1. Plots mean Dice vs. rotation angle as a grouped bar chart (one bar
     group per angle, one bar per dataset), with error bars (+/- 1 std
     across all pooled organ/subject rows at that angle) -- the
     "dataset-wise visualization."
  2. Writes a TOTAL-inference-time-per-angle table (summed seconds, not
     averaged -- an exact total) with a bottom bolded grand-Total row, a
     Total (min) row, and a Mean / subject (min) row, colored with the same
     diverging red<->blue scale as the heatmaps (red = slower, blue =
     faster), plus a caption spelling out each dataset's exact
     subject/organ/angle/prediction count.

Usage:
    python scripts/summarize_dice_and_inference_text3dsam.py
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

DATASETS = [
    {"name": "MR_TotalSeg", "label": "MR TotalSeg", "color": "#2a78d6"},
    {"name": "CT_HaN-Seg", "label": "CT HaN-Seg", "color": "#eb6834"},
    {"name": "CT_TotalSeg_cardiac", "label": "CT TotalSeg Cardiac", "color": "#1baf7a"},
]

# Same diverging palette as dice_heatmap_organ_angle_text3dsam.py, so the
# heatmaps and this table read consistently.
DIVERGING_RAMP = [
    "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
    "#f7f7f7",
    "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061",
]
DICE_CMAP = mcolors.LinearSegmentedColormap.from_list("dice_diverging", DIVERGING_RAMP, N=256)


def _text_color_for(rgba) -> str:
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else "#0b0b0b"


plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        raise SystemExit(f"Not found: {PREDICTIONS_CSV} -- run the Text3DSAM Kaggle "
                          f"inference notebook first and copy the CSV here.")
    return pd.read_csv(PREDICTIONS_CSV)


def plot_dice_bar_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_groups = len(ANGLES_DEG)
    n_series = len(DATASETS)
    bar_width = 0.7 / n_series
    x = np.arange(n_groups)

    for i, ds in enumerate(DATASETS):
        ds_df = df[df["dataset"] == ds["name"]]
        stats = ds_df.groupby("rotation_angle_deg")["dice"].agg(["mean", "std"]).reindex(ANGLES_DEG)
        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(x + offset, stats["mean"], width=bar_width * 0.92,
                       yerr=stats["std"], capsize=3,
                       color=ds["color"], edgecolor="none", label=ds["label"],
                       error_kw={"ecolor": "#52514e", "elinewidth": 1.2, "alpha": 0.8})
        for rect, val in zip(bars, stats["mean"]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9.5,
                    color="#0b0b0b", fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}°" for a in ANGLES_DEG], fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean Dice score", fontsize=13, labelpad=10)
    ax.set_xlabel("Rotation angle", fontsize=13, labelpad=10)
    ax.set_title("Text3DSAM: Mean Dice vs. Rotation Angle", fontsize=16, fontweight="bold", pad=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color="#e3e2dc", linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=12, bbox_to_anchor=(1.0, 1.05))

    fig.tight_layout()
    out_path = OUT_DIR / "mean_dice_by_angle_text3dsam.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


SUMMARY_ROWS = ("Total (s)", "Total (min)", "Mean / subject (min)")


def format_scale_caption(label: str, m: dict) -> str:
    organ_part = (f"{m['n_organs_distinct']} organs" if m["uniform_organs"]
                  else f"{m['n_organs_distinct']} distinct organs (varies per subject)")
    return (f"{label}: {m['n_subjects']} subjects, {organ_part}, {m['n_angles']} angles, "
            f"{m['n_predictions']:,} predictions total")


def build_inference_table(df: pd.DataFrame):
    """Mirrors summarize_dice_and_inference.py's build_inference_table --
    see there for the full rationale (exact totals, not means; summary rows
    in both seconds and minutes; per-dataset scale metadata for the
    caption)."""
    cols = {}
    meta = {}
    for ds in DATASETS:
        ds_df = df[df["dataset"] == ds["name"]]
        cols[ds["label"]] = ds_df.groupby("rotation_angle_deg")["inference_seconds"].sum().reindex(ANGLES_DEG)
        n_subjects = ds_df["subject_id"].nunique()
        # Unlike VoxTell (every subject has the exact same organ set), which
        # Text3DSAM organ pairs are annotated varies per subject here (e.g.
        # CT_TotalSeg_cardiac ranges 2-10 organs/subject) -- so report the
        # dataset-wide distinct organ count and whether it's uniform, rather
        # than one subject's count implying a (false) subjects x organs x
        # angles = predictions identity.
        organs_per_subject = ds_df.groupby("subject_id")["organ"].nunique()
        n_organs_distinct = ds_df["organ"].nunique()
        uniform_organs = organs_per_subject.nunique() == 1
        meta[ds["label"]] = {
            "n_subjects": n_subjects,
            "n_organs_distinct": n_organs_distinct,
            "uniform_organs": uniform_organs,
            "n_angles": ds_df["rotation_angle_deg"].nunique(),
            "n_predictions": len(ds_df),
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


def render_table_png(table: pd.DataFrame, meta: dict):
    """Mirrors summarize_dice_and_inference.py's render_table_png -- see
    there for the full rationale behind the absolute-inch vertical budget
    (fig.text-based title/caption placement instead of ax.set_title(), to
    avoid a title-pad-vs-bbox_inches="tight" layout bug) and the inverted
    diverging-color direction (low time = good = blue end)."""
    angle_rows = table.drop(index=list(SUMMARY_ROWS))
    vmin, vmax = angle_rows.values.min(), angle_rows.values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    n_rows, n_cols = table.shape
    row_h_in = 0.5
    top_margin_in = 0.15
    title_h_in = 0.62
    gap1_in = 0.18
    caption_h_in = 0.32 * len(meta)
    gap2_in = 0.15
    table_h_in = row_h_in * (n_rows + 1)
    bottom_margin_in = 0.1
    fig_h = (top_margin_in + title_h_in + gap1_in + caption_h_in
             + gap2_in + table_h_in + bottom_margin_in)
    fig, ax = plt.subplots(figsize=(3.3 + 2.1 * n_cols, fig_h))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    col_labels = [table.index.name] + list(table.columns)

    def fmt_row(idx, row):
        unit_is_seconds = idx not in SUMMARY_ROWS or idx == "Total (s)"
        suffix = "s" if unit_is_seconds else " min"
        return [idx] + [f"{v:,.2f}{suffix}" for v in row]

    cell_text = [fmt_row(idx, row) for idx, row in zip(table.index, table.values)]

    table_bottom = bottom_margin_in / fig_h
    table_frac = table_h_in / fig_h
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                   bbox=[0, table_bottom, 1, table_frac])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12.5)

    header_bg = "#0b0b0b"
    summary_bg = "#e1e0d9"
    row_index = {i + 1: idx for i, idx in enumerate(table.index)}

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2.5)

        if row == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue

        if row_index[row] in SUMMARY_ROWS:
            cell.set_facecolor(summary_bg)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#0b0b0b")
            if row_index[row] == SUMMARY_ROWS[0]:
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
            color = DICE_CMAP(1 - norm(val))
            cell.set_facecolor(color)
            cell.get_text().set_color(_text_color_for(color))
            cell.get_text().set_fontweight("medium")

    title_top = 1 - top_margin_in / fig_h
    fig.text(0.5, title_top, "Text3DSAM: Total Inference Time by Rotation Angle\n(red = slower, blue = faster)",
              ha="center", va="top", fontsize=15, fontweight="bold", color="#0b0b0b")

    caption_top = 1 - (top_margin_in + title_h_in + gap1_in) / fig_h
    caption_lines = [format_scale_caption(label, m) for label, m in meta.items()]
    fig.text(0.5, caption_top, "\n".join(caption_lines),
              ha="center", va="top", fontsize=10.5, color="#52514e")

    out_path = OUT_DIR / "mean_inference_time_by_angle_text3dsam.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    df = load_predictions()

    plot_dice_bar_chart(df)

    table, meta = build_inference_table(df)
    csv_path = OUT_DIR / "mean_inference_time_by_angle_text3dsam.csv"
    table.to_csv(csv_path)
    print(f"[done] wrote {csv_path}")
    render_table_png(table, meta)

    print()
    print(table)
    print()
    for label, m in meta.items():
        print(format_scale_caption(label, m))
