"""Advanced word-attribution comparison visualization for VoxTell groundedness study.

Rebuilds the "word highlight" figures (Exact Shapley / SHAP / LIME) as compact,
content-sized panels:
  - figure width/height are sized to the actual content (no fixed-size whitespace)
  - dynamically measured chip widths (no overlap / no cropping on multi-word spans)
  - title + the prompt shown directly (no "Prompt:" label)
  - color-meaning caption and a plain-language legend live at the bottom

Run:
    python scripts/generate_word_highlight_viz.py
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
OUT_DIR = RESULTS_DIR / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP = LinearSegmentedColormap.from_list(
    "importance", ["#FFF3E0", "#FDB863", "#E2492A", "#7A0403"]
)
STRUCTURAL_COLOR = "#E9EAEC"
STRUCTURAL_TEXT = "#4A4E57"
METHOD_LABEL_COLOR = "#1B1E23"
DPI = 200

# ----------------------------------------------------------------------------
# Load the real per-subject attribution values and average across subjects.
# Every value here is an actual computed number -- including LIME's 0.000 for
# multi-word organ phrases (verified in lime_attribution_amos22_mri.csv: LIME
# assigns exactly zero weight to "adrenal gland" for every subject, it is not
# missing/unstable data), so nothing needs to be shown as "n/a".
# ----------------------------------------------------------------------------
_wa = pd.read_csv(RESULTS_DIR / "word_attribution_amos22_mri.csv")
_shap = pd.read_csv(RESULTS_DIR / "shap_attribution_amos22_mri.csv")
_lime = pd.read_csv(RESULTS_DIR / "lime_attribution_amos22_mri.csv")


def _organ_role_words(organ: str) -> dict[str, str]:
    parts = organ.split(" ", 1)
    return {"position": parts[0], "organ": parts[1]}


def _load_methods(organ: str) -> dict[str, dict[str, float]]:
    exact = _wa[(_wa.organ == organ) & (_wa.metric == "f_dice_gt")].groupby("role").shapley_value.mean()
    shap = _shap[(_shap.organ == organ) & (_shap.metric == "f_dice_gt")].groupby("role").shap_value.mean()
    lime = _lime[_lime.organ == organ].groupby("role").lime_weight.mean()
    return {
        "Exact Shapley": {"position": float(exact["position"]), "organ": float(exact["organ"])},
        "SHAP": {"position": float(shap["position"]), "organ": float(shap["organ"])},
        "LIME": {"position": float(lime["position"]), "organ": float(lime["organ"])},
    }


CASES = {
    "left_kidney": {
        "words": ["Segment", "the", "left", "kidney", "in", "this", "image."],
        "spans": _organ_role_words("left kidney"),
        "methods": _load_methods("left kidney"),
    },
    "right_kidney": {
        "words": ["Segment", "the", "right", "kidney", "in", "this", "image."],
        "spans": _organ_role_words("right kidney"),
        "methods": _load_methods("right kidney"),
    },
    "left_adrenal_gland": {
        "words": ["Segment", "the", "left", "adrenal gland", "in", "this", "image."],
        "spans": _organ_role_words("left adrenal gland"),
        "methods": _load_methods("left adrenal gland"),
    },
    "right_adrenal_gland": {
        "words": ["Segment", "the", "right", "adrenal gland", "in", "this", "image."],
        "spans": _organ_role_words("right adrenal gland"),
        "methods": _load_methods("right adrenal gland"),
    },
}


def build_row_tokens(case: dict, method_values: dict[str, float | None]):
    """(text, kind, value) per word; kind in {structural, scored}."""
    role_of_word = {v: k for k, v in case["spans"].items()}
    tokens = []
    for w in case["words"]:
        role = role_of_word.get(w)
        if role is None:
            tokens.append((w, "structural", None))
        else:
            tokens.append((w, "scored", method_values[role]))
    return tokens


def normalize_row(tokens):
    vals = [v for _, k, v in tokens if k == "scored"]
    if not vals:
        return [(t, k, v, None) for t, k, v in tokens]
    vmax = max(vals)
    vmin = min(vals) if len(vals) > 1 else 0.0
    span = (vmax - vmin) or 1.0
    return [
        (t, k, v, (v - vmin) / span if k == "scored" else None)
        for t, k, v in tokens
    ]


def label_for(text, kind, val):
    if kind != "scored":
        return text
    shown = "0.000" if val == 0 else f"{val:.3g}"
    return f"{text}  ({shown})"


def text_width_in(fig, renderer, text_obj) -> float:
    bbox = text_obj.get_window_extent(renderer=renderer)
    return bbox.width / fig.dpi


def measure_case(case) -> dict:
    """Dry-run render on a scratch figure to get each row's exact chip widths
    (in inches), independent of the final figure size."""
    scratch = plt.figure(figsize=(20, 10), dpi=DPI)
    ax = scratch.add_subplot(111)
    renderer = scratch.canvas.get_renderer()

    row_widths_in = {}
    for method, values in case["methods"].items():
        tokens = normalize_row(build_row_tokens(case, values))
        widths = []
        for text, kind, val, norm in tokens:
            label = label_for(text, kind, val)
            t = ax.text(0, 0, label, fontsize=12.5,
                        fontweight=("bold" if kind != "structural" else "normal"))
            scratch.canvas.draw()
            renderer = scratch.canvas.get_renderer()
            widths.append(text_width_in(scratch, renderer, t) + 2 * CHIP_PAD_IN)
            t.remove()
        row_widths_in[method] = widths
    plt.close(scratch)
    return row_widths_in


CHIP_PAD_IN = 0.13
CHIP_GAP_IN = 0.08
LEFT_MARGIN_IN = 0.42
LABEL_COL_IN = 1.55
RIGHT_MARGIN_IN = 0.42
ROW_PITCH_IN = 0.48
CHIP_H_IN = 0.30
HEADER_H_IN = 0.92
LEGEND_H_IN = 1.4


def render_case(case_key: str, case: dict, out_path: Path):
    methods = list(case["methods"].keys())
    n_rows = len(methods)

    row_widths_in = measure_case(case)
    content_w_in = max(sum(w) + CHIP_GAP_IN * (len(w) - 1) for w in row_widths_in.values())

    fig_w = LEFT_MARGIN_IN + LABEL_COL_IN + content_w_in + RIGHT_MARGIN_IN
    fig_w = max(9.0, min(fig_w, 15.0))
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

    prompt_text = "Segment the {} {} in this image.".format(
        case["spans"]["position"], case["spans"]["organ"]
    )
    ax.text(LEFT_MARGIN_IN, top_y - 0.32, f"“{prompt_text}”",
            fontsize=12.5, style="italic", color="#2C3038",
            ha="left", va="top", family="DejaVu Sans")

    divider_y = top_y - 0.58
    ax.plot([LEFT_MARGIN_IN, fig_w - RIGHT_MARGIN_IN], [divider_y, divider_y],
            color="#DDDFE3", lw=1.1)

    # ---------------- Rows ----------------
    chip_start_x = LEFT_MARGIN_IN + LABEL_COL_IN
    row_top = divider_y - 0.10

    for i, method in enumerate(methods):
        row_center_y = row_top - ROW_PITCH_IN * i - ROW_PITCH_IN / 2

        ax.text(LEFT_MARGIN_IN, row_center_y, method,
                fontsize=12.5, fontweight="bold", color=METHOD_LABEL_COLOR,
                ha="left", va="center", family="DejaVu Sans")

        tokens = normalize_row(build_row_tokens(case, case["methods"][method]))
        x_cursor = chip_start_x
        for (text, kind, val, norm), chip_w in zip(tokens, row_widths_in[method]):
            label = label_for(text, kind, val)

            if kind == "structural":
                face, textcolor = STRUCTURAL_COLOR, STRUCTURAL_TEXT
            else:
                face = CMAP(0.15 + 0.85 * norm)
                textcolor = "white" if norm > 0.35 else "#3A1400"

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
    for key, case in CASES.items():
        render_case(key, case, OUT_DIR / f"word_highlight_{key}.png")


if __name__ == "__main__":
    main()
