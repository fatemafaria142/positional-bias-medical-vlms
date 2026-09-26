"""
Word importance for ONE prompt, by method (Exact Shapley, SHAP, LIME), in the same style as
plot_prompt_breakdown.py.

Each row shows the prompt as word chips. The two words that were tested carry their importance value
(mean over the subjects, contribution to the Dice score). Words that were kept fixed are grey and have no score.

Usage:
    python scripts/plot_prompt_attribution.py                   # AMOS, left adrenal gland
    python scripts/plot_prompt_attribution.py oasis "left cerebral white matter"
Output: local_plots_groundedness/word_highlight/<dataset>_<organ>_attribution.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "local_plots_groundedness" / "word_highlight"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()

POS_C, ORG_C = "#f400d4", "#5683ff"
GREY, GREY_TEXT = "#ebebe8", "#5a5954"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9993"
SURFACE = "#fcfcfb"
FS = 12.5

DATASETS = {
    "amos": ("AMOS22 MRI", [("Voxtell-Amos-Groundness", "amos22_mri")]),
    "oasis": ("Neurite-OASIS", [("Voxtell-OASIS-Batch1-Groundness", "neurite_oasis"),
                                ("Voxtell-OASIS-Batch2-Groundness", "neurite_oasis_batch2")]),
}


PALETTES = {"amos": ("#f400d4", "#5683ff"), "oasis": ("#731cff", "#00a7ad")}   # (position word, organ name)


def _rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def _lum(h):
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_on(h):
    """White text on dark colours, dark text on light ones."""
    return "white" if _lum(h) < 0.2 else "#0b0b0b"


def mix(h, other, t):
    """t = share of colour h (1 = h, 0 = other)."""
    a, b = _rgb(h), _rgb(other)
    return "#%02x%02x%02x" % tuple(round(b[i] + (a[i] - b[i]) * t) for i in range(3))



def shade(hex_color, strength):
    """Blend a colour with the page colour: strength 1 = full colour, lower = lighter."""
    c = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    bg = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(bg[i] + (c[i] - bg[i]) * strength) for i in range(3))


def load(dirs, name):
    return pd.concat([pd.read_csv(REPO / d / f"{name}_{s}.csv") for d, s in dirs], ignore_index=True)


def main(key="amos", organ="left adrenal gland", out=None):
    global POS_C, ORG_C
    POS_C, ORG_C = PALETTES.get(key, PALETTES["amos"])
    title, dirs = DATASETS[key]
    wa, sh, li = load(dirs, "word_attribution"), load(dirs, "shap_attribution"), load(dirs, "lime_attribution")
    ex = wa[(wa.organ == organ) & (wa.metric == "f_dice_gt")].groupby("role").shapley_value.mean()
    sp = sh[(sh.organ == organ) & (sh.metric == "f_dice_gt")].groupby("role").shap_value.mean()
    lm = li[li.organ == organ].groupby("role").lime_weight.mean()
    methods = [("Exact Shapley", ex), ("SHAP", sp), ("LIME", lm)]

    pos, core = organ.split(None, 1)
    words = [("Segment", "fixed"), ("the", "fixed"), (pos, "position"), (core, "organ"),
             ("in", "fixed"), ("this", "fixed"), ("image.", "fixed")]

    def fmt(v):
        return f"{v:.3g}"

    pad, gap = 0.16, 0.09
    scratch = plt.figure(figsize=(20, 5), dpi=200)
    sax = scratch.add_subplot(111)
    widths = []
    for _, vals in methods:
        ws = []
        for text, kind in words:
            label = text if kind == "fixed" else f"{text}  ({fmt(vals['position' if kind == 'position' else 'organ'])})"
            t = sax.text(0, 0, label, fontsize=FS, fontweight="normal" if kind == "fixed" else "bold")
            scratch.canvas.draw()
            ws.append(t.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi + 2 * pad)
            t.remove()
        widths.append(ws)
    plt.close(scratch)

    left, label_col, right = 0.35, 1.9, 0.35
    chips_w = max(sum(w) + gap * (len(w) - 1) for w in widths)
    fig_w = left + label_col + chips_w + right
    row_pitch, chip_h = 0.66, 0.4
    fig_h = 4.85
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.12
    ax.text(fig_w / 2, top, title, fontsize=20, fontweight="bold", color=INK, ha="center", va="top")
    ax.text(fig_w / 2, top - 0.44, "Which word matters more for the segmentation?", fontsize=13, color=INK2,
            ha="center", va="top")
    ax.text(fig_w / 2, top - 0.84, f"“Segment the {pos} {core} in this image.”", fontsize=12, style="italic",
            color=MUTED, ha="center", va="top")

    chips_x0 = left + label_col
    y = top - 1.2
    for i, ((mname, vals), ws) in enumerate(zip(methods, widths)):
        cy = y - row_pitch * i - row_pitch / 2
        ax.text(left, cy, mname, fontsize=FS, fontweight="bold", color=INK, ha="left", va="center")
        x = chips_x0
        for (text, kind), w in zip(words, ws):
            box = dict(boxstyle="round,pad=0,rounding_size=0.06")
            if kind == "fixed":
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=0, facecolor=GREY,
                                                     zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, color=GREY_TEXT, ha="left", va="center", zorder=5)
            else:
                role = "position" if kind == "position" else "organ"
                lo, hi = sorted([float(vals["position"]), float(vals["organ"])])
                rel = (float(vals[role]) - lo) / ((hi - lo) or 1.0)   # 0 = the smaller word, 1 = the larger word
                strength = 0.35 + 0.65 * rel
                face = shade(POS_C if kind == "position" else ORG_C, strength)
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=0, facecolor=face,
                                                     zorder=3, **box))
                ax.text(x + pad, cy, f"{text}  ({fmt(vals[role])})", fontsize=FS, fontweight="bold",
                        color=text_on(face), ha="left", va="center", zorder=5)
            x += w + gap

    ly = y - row_pitch * len(methods) - 0.3
    lx = left
    for face, txt in [(POS_C, "Position word"), (ORG_C, "Organ name"), (GREY, "Word kept fixed, so it has no score")]:
        ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), 0.44, 0.18, boxstyle="round,pad=0,rounding_size=0.04",
                                             linewidth=0, facecolor=face))
        ax.text(lx + 0.56, ly, txt, fontsize=10.5, color=INK2, ha="left", va="center")
        lx += 0.56 + 0.088 * len(txt) + 0.4

    # second legend row: the icon "( )" = the number next to a word
    ly2 = ly - 0.45
    ax.add_patch(mpatches.FancyBboxPatch((left, ly2 - 0.09), 0.44, 0.18, boxstyle="round,pad=0,rounding_size=0.04",
                                         linewidth=1.2, edgecolor=INK2, facecolor=SURFACE))
    ax.text(left + 0.22, ly2, "( )", fontsize=9.5, color=INK2, ha="center", va="center")
    ax.text(left + 0.56, ly2, "Contribution of the word to the Dice score (0 to 1)", fontsize=10.5, color=INK2,
            ha="left", va="center")

    # third legend row: shade = size of the contribution (smaller = lighter, larger = darker)
    ly3 = ly2 - 0.45
    n_steps, step_w = 8, 0.11
    for c, col in enumerate([POS_C, ORG_C]):
        for k in range(n_steps):
            ax.add_patch(mpatches.Rectangle((left + c * (n_steps * step_w + 0.12) + k * step_w, ly3 - 0.09), step_w,
                                            0.18, linewidth=0, facecolor=shade(col, 0.35 + 0.65 * k / (n_steps - 1))))
    gx = left + 2 * (n_steps * step_w + 0.12) + 0.1
    ax.text(gx, ly3, "Lighter shade = smaller contribution     Darker shade = larger contribution", fontsize=10.5,
            color=INK2, ha="left", va="center")

    out = Path(out) if out else OUT / f"{key}_{organ.replace(' ', '_')}_attribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
