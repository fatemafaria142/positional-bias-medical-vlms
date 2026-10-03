"""
Row-21 "word highlight" figure, in the exact same chip style as
scripts/plot_word_highlight.py's word_attribution.png (rows 5-11) -- but row 21
has a THIRD role plot_word_highlight.py never had: the IMAGE itself, not just
two words. The sentence is shown exactly as before (position word and organ
word highlighted in the prompt, grey for the rest), with one extra chip in
front of it for the image's own contribution, on the same color scale and the
same (0 to 1, in Dice-score units) score convention.

Only ONE method exists for row 21 (exact Shapley over the 3-role coalition --
no SHAP/LIME was computed for the image role), so this draws two rows instead
of three: f_dice_gt (vs. ground truth) and f_dice_self (vs. the model's own
full-prompt answer) -- the same two metrics row 21's bar charts already use.

Run the Kaggle notebook, download row21_joint_shapley_<suffix>.csv, then:
    python scripts/make_row21_word_highlight.py RESULTS_DIR [--suffix neurite_oasis] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

FONT = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
FAMILY = "DejaVu Sans"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    FAMILY = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["font.family"] = FAMILY

# Same categorical colors used by every other rows-17-24 figure in this repo (make_rows17_21_figures.py's
# BLUE/ORANGE/AQUA): image = blue, organ word = orange, position word = aqua -- NOT plot_word_highlight.py's
# magenta/blue pair, which only ever had two roles (position, organ) and no image role to also color.
IMG_C, ORG_C, POS_C = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_TEXT = "#ebebe8", "#5a5954"
INK, INK2 = "#0b0b0b", "#52514e"
FS = 12.5

METRIC_LABEL = {"f_dice_gt": "vs. Ground Truth (f_dice_gt)", "f_dice_self": "vs. Full-Prompt Prediction (f_dice_self)"}


def method_values(df: pd.DataFrame, organ: str) -> dict:
    sub = df[df.organ == organ].set_index(["metric", "role"]).shapley_value
    out = {}
    for metric in ("f_dice_gt", "f_dice_self"):
        if metric not in sub.index.get_level_values(0):
            continue
        out[METRIC_LABEL[metric]] = {
            "image": float(sub.loc[(metric, "image")]),
            "position": float(sub.loc[(metric, "position_word")]),
            "organ": float(sub.loc[(metric, "organ_word")]),
        }
    return out


def tokens_for(organ: str, vals: dict):
    """[Image] chip first, then the sentence -- (text, value, role) per chip; role in
    {"image", "position", "organ", None}. None = grey, no score (word was always kept fixed)."""
    pos, core = organ.split(None, 1)
    out = [("[ Image ]", float(vals["image"]), "image")]
    for w in ["Segment", "the", pos, core, "in", "this", "image."]:
        if w == pos:
            out.append((w, float(vals["position"]), "position"))
        elif w == core:
            out.append((w, float(vals["organ"]), "organ"))
        else:
            out.append((w, None, None))
    return out


def label(text, val):
    if val is None:
        return text
    return f"{text}  ({'0.000' if val == 0 else f'{val:.3g}'})"


def render(organ: str, methods: dict, out_path: Path):
    pad, gap = 0.16, 0.09
    scratch = plt.figure(figsize=(22, 5), dpi=200)
    sax = scratch.add_subplot(111)
    rows, widths = {}, {}
    label_w_in = 0.0
    for m, vals in methods.items():
        toks = tokens_for(organ, vals)
        rows[m] = toks
        ws = []
        for text, val, role in toks:
            t = sax.text(0, 0, label(text, val), fontsize=FS, fontweight="bold" if val is not None else "normal")
            scratch.canvas.draw()
            ws.append(t.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi + 2 * pad)
            t.remove()
        widths[m] = ws
        lt = sax.text(0, 0, m, fontsize=FS, fontweight="bold")
        scratch.canvas.draw()
        label_w_in = max(label_w_in, lt.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi)
        lt.remove()
    plt.close(scratch)

    left, label_col, right = 0.3, label_w_in + 0.35, 0.3
    content = max(sum(w) + gap * (len(w) - 1) for w in widths.values())
    fig_w = max(9.5, left + label_col + content + right)
    row_pitch, chip_h = 0.66, 0.4
    fig_h = 1.65 + row_pitch * len(rows) + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.12
    ax.text(fig_w / 2, top, "Neurite-OASIS", fontsize=19, fontweight="bold", color=INK, ha="center", va="top")
    ax.text(fig_w / 2, top - 0.44, "Which input matters more for the segmentation -- the image, or the words?",
            fontsize=12.5, color=INK2, ha="center", va="top")
    pos, core = organ.split(None, 1)
    ax.text(fig_w / 2, top - 0.84, f"“Segment the {pos} {core} in this image.”", fontsize=13, style="italic",
            color=INK, ha="center", va="top")

    row_top = top - 1.45
    for i, (m, toks) in enumerate(rows.items()):
        cy = row_top - row_pitch * i - row_pitch / 2
        ax.text(left, cy, m, fontsize=FS, fontweight="bold", color=INK, ha="left", va="center")
        x = left + label_col
        for (text, val, role), w in zip(toks, widths[m]):
            if role is None:
                face, tc = GREY, GREY_TEXT
            else:
                face, tc = {"image": IMG_C, "position": POS_C, "organ": ORG_C}[role], "white"
            ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h,
                                                 boxstyle="round,pad=0,rounding_size=0.06", linewidth=0,
                                                 facecolor=face, zorder=3))
            ax.text(x + pad, cy, label(text, val), fontsize=FS, fontweight="bold" if val is not None else "normal",
                    color=tc, ha="left", va="center", zorder=5)
            x += w + gap

    # legend
    ly = row_top - row_pitch * len(rows) - 0.3
    lx = left
    for face, txt in [(IMG_C, "Image"), (POS_C, "Position word"), (ORG_C, "Organ name"),
                       (GREY, "Word kept fixed, so it has no score")]:
        ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), 0.44, 0.18, boxstyle="round,pad=0,rounding_size=0.04",
                                             linewidth=0, facecolor=face))
        ax.text(lx + 0.56, ly, txt, fontsize=10.5, color=INK2, ha="left", va="center")
        lx += 0.56 + 0.088 * len(txt) + 0.4

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="directory containing row21_joint_shapley_<suffix>.csv")
    ap.add_argument("--suffix", default="neurite_oasis")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="default: <repo>/local_plots_mechanistic/rows17_21/word_highlight")
    args = ap.parse_args()
    out_dir = args.out_dir or (REPO / "local_plots_mechanistic" / "rows17_21" / "word_highlight")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = args.results_dir / f"row21_joint_shapley_{args.suffix}.csv"
    if not path.exists():
        print(f"not found: {path}")
        return
    df = pd.read_csv(path)

    for organ in sorted(df.organ.unique()):
        methods = method_values(df, organ)
        if not methods:
            continue
        out = out_dir / f"row21_{organ.replace(' ', '_')}.png"
        render(organ, methods, out)
        print("wrote", out.relative_to(REPO) if REPO in out.parents else out)


if __name__ == "__main__":
    main()
