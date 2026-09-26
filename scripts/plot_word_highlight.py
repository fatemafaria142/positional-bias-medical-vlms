"""
Word-highlight figures for ONE prompt: shows how much each word of the prompt matters, according to
exact Shapley, SHAP and LIME. Reads the CSVs downloaded from the Kaggle notebooks (needs only
numpy / pandas / matplotlib).

Only two words were ever tested in the notebooks: the left/right word and the organ name. The other words
of the template ("Segment the ... in this image.") were kept fixed in every prompt, so they are drawn in
grey with no score -- nothing is invented for them.

Each chip shows the word and its mean Shapley / SHAP / LIME value for Dice vs. ground truth (mean over the
subjects). Colour: within a row, darker = larger contribution.

Output: local_plots_groundedness/word_highlight/<dataset>_<organ>.png

Usage:
    python scripts/plot_word_highlight.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "local_plots_groundedness" / "word_highlight"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
FAMILY = "DejaVu Sans"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    FAMILY = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["font.family"] = FAMILY

CMAP = LinearSegmentedColormap.from_list("importance", ["#e6edff", "#9db8ff", "#5683ff", "#1f3fb0"])
POS_C, ORG_C = "#f400d4", "#5683ff"   # position word = magenta, organ name = blue
GREY, GREY_TEXT = "#ebebe8", "#5a5954"
INK, INK2 = "#0b0b0b", "#52514e"
FS = 12.5

DATASETS = {
    "AMOS22 MRI": ("amos", [("Voxtell-Amos-Groundness", "amos22_mri")],
                   ["left kidney", "right kidney", "left adrenal gland", "right adrenal gland"]),
    "Neurite-OASIS": ("oasis", [("Voxtell-OASIS-Batch1-Groundness", "neurite_oasis"),
                                ("Voxtell-OASIS-Batch2-Groundness", "neurite_oasis_batch2")],
                      ["left cerebral white matter", "right cerebral white matter",
                       "left cerebellum cortex", "right cerebellum cortex"]),
}


def load(dirs, name):
    return pd.concat([pd.read_csv(REPO / d / f"{name}_{s}.csv") for d, s in dirs], ignore_index=True)


def method_values(wa, sh, li, organ):
    ex = wa[(wa.organ == organ) & (wa.metric == "f_dice_gt")].groupby("role").shapley_value.mean()
    sp = sh[(sh.organ == organ) & (sh.metric == "f_dice_gt")].groupby("role").shap_value.mean()
    lm = li[li.organ == organ].groupby("role").lime_weight.mean()
    return {"Exact Shapley": ex, "SHAP": sp, "LIME": lm}


def tokens_for(organ, vals):
    """(text, value or None). The two tested words carry a value; everything else is grey."""
    pos, core = organ.split(None, 1)
    words = ["Segment", "the", pos, core, "in", "this", "image."]
    out = []
    for w in words:
        if w == pos:
            out.append((w, float(vals["position"])))
        elif w == core:
            out.append((w, float(vals["organ"])))
        else:
            out.append((w, None))
    return out


def label(text, val):
    if val is None:
        return text
    return f"{text}  ({'0.000' if val == 0 else f'{val:.3g}'})"


def render(title, key, organ, methods, out_path):
    pad, gap = 0.16, 0.09
    scratch = plt.figure(figsize=(20, 5), dpi=200)
    sax = scratch.add_subplot(111)
    rows, widths = {}, {}
    for m, vals in methods.items():
        toks = tokens_for(organ, vals)
        rows[m] = toks
        ws = []
        for text, val in toks:
            t = sax.text(0, 0, label(text, val), fontsize=FS, fontweight="bold" if val is not None else "normal")
            scratch.canvas.draw()
            ws.append(t.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi + 2 * pad)
            t.remove()
        widths[m] = ws
    plt.close(scratch)

    left, label_col, right = 0.3, 1.4, 0.3
    content = max(sum(w) + gap * (len(w) - 1) for w in widths.values())
    fig_w = max(8.0, left + label_col + content + right)
    row_pitch, chip_h = 0.66, 0.4
    fig_h = 1.65 + row_pitch * len(rows) + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.12
    ax.text(fig_w / 2, top, title, fontsize=19, fontweight="bold", color=INK, ha="center", va="top")
    ax.text(fig_w / 2, top - 0.44, "Which word matters more for the segmentation?", fontsize=12.5, color=INK2,
            ha="center", va="top")
    pos, core = organ.split(None, 1)
    ax.text(fig_w / 2, top - 0.84, f"“Segment the {pos} {core} in this image.”", fontsize=13, style="italic",
            color=INK, ha="center", va="top")

    row_top = top - 1.45
    for i, (m, toks) in enumerate(rows.items()):
        cy = row_top - row_pitch * i - row_pitch / 2
        ax.text(left, cy, m, fontsize=FS, fontweight="bold", color=INK, ha="left", va="center")
        x = left + label_col
        pos_word, org_word = organ.split(None, 1)
        for (text, val), w in zip(toks, widths[m]):
            if val is None:
                face, tc = GREY, GREY_TEXT
            else:
                face, tc = (POS_C if text == pos_word else ORG_C), "white"
            ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h,
                                                 boxstyle="round,pad=0,rounding_size=0.06", linewidth=0,
                                                 facecolor=face, zorder=3))
            ax.text(x + pad, cy, label(text, val), fontsize=FS, fontweight="bold" if val is not None else "normal",
                    color=tc, ha="left", va="center", zorder=5)
            x += w + gap

    # legend
    ly = row_top - row_pitch * len(rows) - 0.3
    lx = left
    for face, txt in [(POS_C, "Position word"), (ORG_C, "Organ name"), (GREY, "Word kept fixed, so it has no score")]:
        ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), 0.44, 0.18, boxstyle="round,pad=0,rounding_size=0.04",
                                             linewidth=0, facecolor=face))
        ax.text(lx + 0.56, ly, txt, fontsize=10.5, color=INK2, ha="left", va="center")
        lx += 0.56 + 0.088 * len(txt) + 0.4

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


for title, (key, dirs, organs) in DATASETS.items():
    wa, sh, li = load(dirs, "word_attribution"), load(dirs, "shap_attribution"), load(dirs, "lime_attribution")
    for organ in organs:
        out = OUT / f"{key}_{organ.replace(' ', '_')}.png"
        render(title, key, organ, method_values(wa, sh, li, organ), out)
        print("wrote", out.name)
