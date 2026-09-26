"""
Share of importance for ONE organ: how the importance of the two tested words splits between the organ name and
the position word, for Exact Shapley, SHAP and LIME (each method's own numbers turned into percentages so the three
can be compared on one scale).

Usage:
    python scripts/plot_share_of_importance.py                   # AMOS, left adrenal gland
    python scripts/plot_share_of_importance.py oasis "left cerebral white matter"
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

FONT = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()

POS_C, ORG_C = "#f400d4", "#5683ff"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#9a9993", "#fcfcfb"

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

    fig_w, fig_h = 9.6, 4.75
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

    left, label_w, bar_w = 0.4, 1.75, 6.9
    bar_x0 = left + label_w
    pitch, bh = 0.68, 0.42
    y0 = top - 1.6
    for i, (name, vals) in enumerate(methods):
        cy = y0 - pitch * i
        p, o = abs(float(vals["position"])), abs(float(vals["organ"]))
        share_p = p / (p + o) * 100 if (p + o) else 50.0
        share_o = 100 - share_p
        wo, wp = bar_w * share_o / 100, bar_w * share_p / 100
        ax.text(left, cy, name, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="center")
        ax.add_patch(mpatches.Rectangle((bar_x0, cy - bh / 2), wo, bh, facecolor=ORG_C, linewidth=0, zorder=3))
        ax.add_patch(mpatches.Rectangle((bar_x0 + wo, cy - bh / 2), wp, bh, facecolor=POS_C, linewidth=0, zorder=3))
        ax.add_patch(mpatches.Rectangle((bar_x0 + wo - 0.015, cy - bh / 2), 0.03, bh, facecolor=SURFACE,
                                        linewidth=0, zorder=4))
        ax.text(bar_x0 + wo / 2, cy, f"{share_o:.0f}%", fontsize=12.5, fontweight="bold", color=text_on(ORG_C),
                ha="center", va="center", zorder=5)
        ax.text(bar_x0 + wo + wp / 2, cy, f"{share_p:.0f}%", fontsize=12.5, fontweight="bold", color=text_on(POS_C),
                ha="center", va="center", zorder=5)

    ly = y0 - pitch * len(methods) - 0.05
    lx = left
    for face, txt in [(ORG_C, "Organ name"), (POS_C, "Position word")]:
        ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), 0.44, 0.18, boxstyle="round,pad=0,rounding_size=0.04",
                                             linewidth=0, facecolor=face))
        ax.text(lx + 0.56, ly, txt, fontsize=10.5, color=INK2, ha="left", va="center")
        lx += 0.56 + 0.088 * len(txt) + 0.4
    ax.text(left, ly - 0.42, "Share of importance: the part of the total importance that belongs to each word "
            "(the two words add up to 100%)", fontsize=10.5, color=INK2, ha="left", va="center")

    out = Path(out) if out else OUT / f"{key}_{organ.replace(' ', '_')}_share_of_importance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
