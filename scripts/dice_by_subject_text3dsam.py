"""
Per-subject Dice for Text3DSAM -- one panel per dataset (MR_TotalSeg,
CT_HaN-Seg, CT_TotalSeg_cardiac), one bar per subject in that dataset, mean
Dice across all that subject's organs and rotation angles, with error bars
(+/- 1 std across those organ/angle rows). Subject IDs are shown with their
common dataset-name prefix stripped (e.g. "CT_HaN-Seg_case_01" -> "case_01")
since it's redundant with the panel title.

This is the "which SUBJECT is the model doing well/badly on" view -- the
organ x angle heatmaps and the angle-only bar chart both average over
subjects, so no per-subject breakdown was visible anywhere else.

Usage:
    python scripts/dice_by_subject_text3dsam.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

PREDICTIONS_CSV = REPO_ROOT / "predictions_text3dsam_faria.csv"

DATASETS = [
    # id_pattern pulls out just the distinguishing suffix from each dataset's
    # own subject_id naming -- generic prefix-stripping doesn't work here
    # since MR_TotalSeg's ids have an extra "_mr_" modality token the other
    # two don't ("MR_totalseg_mr_s0019" vs "CT_HaN-Seg_case_01"), so a
    # regex written for one dataset silently fails (or over-strips) on the
    # others.
    {"name": "MR_TotalSeg", "label": "MR TotalSeg", "color": "#2a78d6",
     "id_pattern": r"(s\d+)$"},
    {"name": "CT_HaN-Seg", "label": "CT HaN-Seg", "color": "#eb6834",
     "id_pattern": r"(case_\d+)$"},
    {"name": "CT_TotalSeg_cardiac", "label": "CT TotalSeg Cardiac", "color": "#1baf7a",
     "id_pattern": r"(s\d+)$"},
]

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})


def short_subject_label(subject_id: str, id_pattern: str) -> str:
    m = re.search(id_pattern, subject_id)
    return m.group(1) if m else subject_id


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        raise SystemExit(f"Not found: {PREDICTIONS_CSV}")
    return pd.read_csv(PREDICTIONS_CSV)


def plot_dice_by_subject(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15.5, 5.5))

    for ax, ds in zip(axes, DATASETS):
        ds_df = df[df["dataset"] == ds["name"]]
        stats = ds_df.groupby("subject_id")["dice"].agg(["mean", "std", "count"]).sort_index()
        labels = [short_subject_label(sid, ds["id_pattern"]) for sid in stats.index]

        bars = ax.bar(labels, stats["mean"], yerr=stats["std"], capsize=4,
                       color=ds["color"], edgecolor="none", width=0.62,
                       error_kw={"ecolor": "#52514e", "elinewidth": 1.2, "alpha": 0.85})
        for rect, val, n in zip(bars, stats["mean"], stats["count"]):
            # Value and sample count on one line -- a separate below-axis
            # "n=" label collided with the rotated x-tick labels underneath.
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                    f"{val:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9.5,
                    color="#0b0b0b", fontweight="medium", linespacing=1.4)

        ax.set_ylim(0, 1.16)
        ax.set_title(ds["label"], fontsize=13.5, fontweight="bold", pad=10)
        ax.set_xlabel("Subject", fontsize=11, labelpad=22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.yaxis.grid(True, color="#e3e2dc", linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="x", labelrotation=30)
        for t in ax.get_xticklabels():
            t.set_ha("right")

    axes[0].set_ylabel("Mean Dice score\n(across all organs x angles for that subject)",
                        fontsize=11.5, labelpad=10)

    fig.suptitle("Text3DSAM: Mean Dice by Subject", fontsize=16.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = OUT_DIR / "dice_by_subject_text3dsam.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    df = load_predictions()
    plot_dice_by_subject(df)
