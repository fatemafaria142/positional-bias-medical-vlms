"""Word-highlight charts for the raw prompt-ablation study, using the EXACT same
design as generate_word_highlight_viz.py (title, prompt-as-subtitle, bold row
label, chip style, legend wording) -- just one row per file (the combo) instead
of three (Exact Shapley / SHAP / LIME).

Organized as:
voxtell_amos22_mri_groundedness_investigation_results/amos/
    full_prompt/       left_kidney.png  right_kidney.png  left_adrenal_gland.png  right_adrenal_gland.png
    organ_only/        (same 4 organ files)
    position_only/     (same 4 organ files)
    neither_word/      (same 4 organ files)

Each file shows the REAL prompt sent to the model for that combo. The content
word(s) actually present are colored directly by that combo's measured Dice
score (already 0-1, so no per-row rescaling needed).

Run:
    python scripts/generate_word_highlight_ablation.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("voxtell_amos22_mri_groundedness_investigation_results")
AMOS_DIR = RESULTS_DIR / "amos"

CMAP = LinearSegmentedColormap.from_list(
    "importance", ["#FFF3E0", "#FDB863", "#E2492A", "#7A0403"]
)
STRUCTURAL_COLOR = "#E9EAEC"
STRUCTURAL_TEXT = "#4A4E57"
ROW_LABEL_COLOR = "#1B1E23"
DPI = 200

ORGANS = ["left kidney", "right kidney", "left adrenal gland", "right adrenal gland"]

COMBO_FOLDER = {
    "organ,position": "full_prompt",
    "organ": "organ_only",
    "position": "position_only",
    "nan": "neither_word",
}
COMBO_ROW_LABEL = {
    "organ,position": "Full Prompt",
    "organ": "Organ Only",
    "position": "Position Only",
    "nan": "Neither Word",
}


def organ_slug(organ: str) -> str:
    return organ.replace(" ", "_")


def load_data() -> dict[str, pd.DataFrame]:
    am = pd.read_csv(RESULTS_DIR / "ablation_masks_amos22_mri.csv")
    out = {}
    for organ in ORGANS:
        sub = am[am.organ == organ]
        out[organ] = (
            sub.groupby(sub["roles_present"].fillna("nan"))
            .agg(dice=("f_dice_gt", "mean"), prompt=("prompt_text", "first"))
        )
    return out


def build_tokens(organ: str, combo: str):
    """(text, kind) tokens for the REAL prompt sent to the model for this combo.
    kind in {structural, scored}. Matches the real strings in
    ablation_masks_amos22_mri.csv (dropping "position" omits the word; dropping
    "organ" substitutes the generic placeholder "structure")."""
    position_word, organ_word = organ.split(" ", 1)
    if combo == "organ,position":
        return [("Segment", "structural"), ("the", "structural"),
                (position_word, "scored"), (organ_word, "scored"),
                ("in", "structural"), ("this", "structural"), ("image.", "structural")]
    if combo == "organ":
        return [("Segment", "structural"), ("the", "structural"),
                (organ_word, "scored"),
                ("in", "structural"), ("this", "structural"), ("image.", "structural")]
    if combo == "position":
        return [("Segment", "structural"), ("the", "structural"),
                (position_word, "scored"), ("structure", "structural"),
                ("in", "structural"), ("this", "structural"), ("image.", "structural")]
    return [("Segment", "structural"), ("the", "structural"), ("structure", "structural"),
            ("in", "structural"), ("this", "structural"), ("image.", "structural")]


def label_for(text, kind, dice):
    if kind != "scored":
        return text
    shown = "0.000" if dice == 0 else f"{dice:.3g}"
    return f"{text}  ({shown})"


def text_width_in(fig, renderer, text_obj) -> float:
    bbox = text_obj.get_window_extent(renderer=renderer)
    return bbox.width / fig.dpi


CHIP_PAD_IN = 0.13
CHIP_GAP_IN = 0.08
LEFT_MARGIN_IN = 0.42
LABEL_COL_IN = 1.55
RIGHT_MARGIN_IN = 0.42
ROW_PITCH_IN = 0.48
CHIP_H_IN = 0.30
HEADER_H_IN = 0.92
LEGEND_H_IN = 1.4


def measure_row(tokens, dice) -> list[float]:
    scratch = plt.figure(figsize=(20, 4), dpi=DPI)
    ax = scratch.add_subplot(111)
    widths = []
    for text, kind in tokens:
        label = label_for(text, kind, dice)
        t = ax.text(0, 0, label, fontsize=12.5,
                    fontweight=("bold" if kind != "structural" else "normal"))
        scratch.canvas.draw()
        renderer = scratch.canvas.get_renderer()
        widths.append(text_width_in(scratch, renderer, t) + 2 * CHIP_PAD_IN)
    plt.close(scratch)
    return widths


def render_one(organ: str, combo: str, dice: float, prompt: str, out_path: Path):
    tokens = build_tokens(organ, combo)
    widths = measure_row(tokens, dice)
    content_w_in = sum(widths) + CHIP_GAP_IN * (len(widths) - 1)

    fig_w = LEFT_MARGIN_IN + LABEL_COL_IN + content_w_in + RIGHT_MARGIN_IN
    fig_w = max(9.0, min(fig_w, 15.0))
    n_rows = 1
    fig_h = HEADER_H_IN + ROW_PITCH_IN * n_rows + LEGEND_H_IN

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # ---------------- Title + prompt ----------------
    top_y = fig_h - 0.22
    ax.text(LEFT_MARGIN_IN, top_y, "Word-Level Attribution Comparison",
            fontsize=19, fontweight="bold", color="#111318",
            ha="left", va="top", family="DejaVu Sans")

    ax.text(LEFT_MARGIN_IN, top_y - 0.32, f"“{prompt}”",
            fontsize=12.5, style="italic", color="#2C3038",
            ha="left", va="top", family="DejaVu Sans")

    divider_y = top_y - 0.58
    ax.plot([LEFT_MARGIN_IN, fig_w - RIGHT_MARGIN_IN], [divider_y, divider_y],
            color="#DDDFE3", lw=1.1)

    # ---------------- Row ----------------
    chip_start_x = LEFT_MARGIN_IN + LABEL_COL_IN
    row_top = divider_y - 0.10
    row_center_y = row_top - ROW_PITCH_IN / 2

    ax.text(LEFT_MARGIN_IN, row_center_y, COMBO_ROW_LABEL[combo],
            fontsize=12.5, fontweight="bold", color=ROW_LABEL_COLOR,
            ha="left", va="center", family="DejaVu Sans")

    x_cursor = chip_start_x
    for (text, kind), chip_w in zip(tokens, widths):
        label = label_for(text, kind, dice)

        if kind == "structural":
            face, textcolor = STRUCTURAL_COLOR, STRUCTURAL_TEXT
        else:
            face = CMAP(0.15 + 0.85 * dice)
            textcolor = "white" if dice > 0.35 else "#3A1400"

        ax.add_patch(mpatches.FancyBboxPatch(
            (x_cursor, row_center_y - CHIP_H_IN / 2), chip_w, CHIP_H_IN,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=0, facecolor=face, zorder=3,
        ))
        ax.text(x_cursor + CHIP_PAD_IN, row_center_y, label,
                fontsize=12.5, fontweight=("bold" if kind != "structural" else "normal"),
                color=textcolor, ha="left", va="center", family="DejaVu Sans", zorder=5)

        x_cursor += chip_w + CHIP_GAP_IN

    # ---------------- Legend ----------------
    legend_top = row_top - ROW_PITCH_IN * n_rows - 0.12

    cbar_w_in = min(3.2, fig_w * 0.30)
    cbar_h_in = 0.15
    cbar_ax = fig.add_axes([
        LEFT_MARGIN_IN / fig_w, (legend_top - cbar_h_in) / fig_h,
        cbar_w_in / fig_w, cbar_h_in / fig_h,
    ])
    norm_obj = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=CMAP, norm=norm_obj, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=9, length=0, pad=3)

    heading_y = legend_top - cbar_h_in - 0.20
    ax.text(LEFT_MARGIN_IN, heading_y, "Contribution to Dice Score:",
            fontsize=9.8, fontweight="bold", color="#111318", ha="left", va="top",
            family="DejaVu Sans")

    line1_y = heading_y - 0.22
    ax.text(LEFT_MARGIN_IN, line1_y, "Lighter color = smaller contribution",
            fontsize=9.8, color="#4A4E57", ha="left", va="top", family="DejaVu Sans")

    line2_y = line1_y - 0.20
    ax.text(LEFT_MARGIN_IN, line2_y, "Darker color = larger contribution",
            fontsize=9.8, color="#4A4E57", ha="left", va="top", family="DejaVu Sans")

    swatch_y = line2_y - 0.28
    ax.add_patch(mpatches.FancyBboxPatch(
        (LEFT_MARGIN_IN, swatch_y - 0.08), 0.34, 0.16,
        boxstyle="round,pad=0,rounding_size=0.04",
        linewidth=0, facecolor=STRUCTURAL_COLOR, zorder=3,
    ))
    ax.text(LEFT_MARGIN_IN + 0.44, swatch_y,
            "Grammar word (article/preposition) — no attribution score",
            fontsize=9.8, color="#4A4E57", ha="left", va="center", family="DejaVu Sans")

    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({fig_w:.2f}in x {fig_h:.2f}in)")


def main():
    data = load_data()
    for combo, folder_name in COMBO_FOLDER.items():
        folder = AMOS_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for organ in ORGANS:
            g = data[organ]
            dice = float(g.loc[combo, "dice"])
            prompt = g.loc[combo, "prompt"]
            out_path = folder / f"{organ_slug(organ)}.png"
            render_one(organ, combo, dice, prompt, out_path)


if __name__ == "__main__":
    main()
