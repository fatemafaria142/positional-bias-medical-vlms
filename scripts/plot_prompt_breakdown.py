"""
One figure that shows the four prompts given for one organ and how well each segmented.

The four versions: full prompt (position + organ), without position, without organ, without position and organ.
Removed words are drawn as struck-out ghost chips so the reader can see exactly what is missing. When the organ
name is removed it is replaced by "structure near the <nearby organ>"; the nearby organ differs per subject, so the
figure keeps only the subjects whose replacement was the chosen nearby organ (default: pancreas for the left
adrenal gland in AMOS).

Usage:
    python scripts/plot_prompt_breakdown.py                                  # AMOS, left adrenal gland, pancreas
    python scripts/plot_prompt_breakdown.py amos "right kidney" "<organ>"    # <dataset> "<organ>" "<nearby organ>"
Output: local_plots_groundedness/word_highlight/<dataset>_<organ>_all_prompts.png
"""

import re
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

POS_C, ORG_C = "#f400d4", "#5683ff"          # position word = magenta, organ name = blue
GREY, GREY_TEXT = "#ebebe8", "#5a5954"
INK, INK2, MUTED, BAR_BG = "#0b0b0b", "#52514e", "#9a9993", "#dbe4ff"   # bar track: pale blue, not the grey of fixed words
SURFACE = "#fcfcfb"
FS = 12.5

DATASETS = {
    "amos": ("AMOS22 MRI", [("Voxtell-Amos-Groundness", "amos22_mri")]),
    "oasis": ("Neurite-OASIS", [("Voxtell-OASIS-Batch1-Groundness", "neurite_oasis"),
                                ("Voxtell-OASIS-Batch2-Groundness", "neurite_oasis_batch2")]),
}
VERSIONS = [("none", "Full prompt", "position + organ"), ("position", "Without position", ""),
            ("organ", "Without organ", ""), ("organ,position", "Without position and organ", "")]


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



def chips_for(organ, cond, landmark):
    """[(text, kind)]; kind: fixed / position / organ / ghost_position / ghost_organ / replacement."""
    pos, core = organ.split(None, 1)
    drop = cond.split(",")
    out = [("Segment", "fixed"), ("the", "fixed")]
    out.append((pos, "ghost_position" if "position" in drop else "position"))
    if "organ" in drop:
        out.append((core, "ghost_organ"))
        out.append((f"structure near the {landmark}", "replacement"))
    else:
        out.append((core, "organ"))
    return out + [("in", "fixed"), ("this", "fixed"), ("image.", "fixed")]


def main(key="amos", organ="left adrenal gland", landmark="pancreas", out=None):
    global POS_C, ORG_C, BAR_BG
    POS_C, ORG_C = PALETTES.get(key, PALETTES["amos"])
    BAR_BG = mix(ORG_C, SURFACE, 0.22)   # bar track: pale tint of the organ colour, not the grey of fixed words
    title, dirs = DATASETS[key]
    a = pd.concat([pd.read_csv(REPO / d / f"ablation_masks_{s}.csv",
                               usecols=["subject_id", "organ", "roles_dropped", "prompt_text", "f_dice_gt"])
                   for d, s in dirs], ignore_index=True)
    a["roles_dropped"] = a.roles_dropped.fillna("none")
    a = a[a.organ == organ]
    if landmark in (None, "", "auto"):   # default: the replacement organ that most subjects received
        names = a[a.roles_dropped == "organ"].prompt_text.str.extract(r"near the (.+?) in this image")[0]
        landmark = names.value_counts().index[0]
    # keep only the subjects whose organ-removed prompt named the chosen nearby organ
    keep = a[(a.roles_dropped == "organ") & a.prompt_text.str.contains(f"near the {landmark} in", regex=False)].subject_id
    a = a[a.subject_id.isin(set(keep))]
    n_subj = a.subject_id.nunique()
    dice = a.groupby("roles_dropped").f_dice_gt.mean()

    rows = [(cond, name, chips_for(organ, cond, landmark), float(dice[cond])) for cond, name, _ in VERSIONS]

    pad, gap = 0.16, 0.09
    scratch = plt.figure(figsize=(20, 5), dpi=200)
    sax = scratch.add_subplot(111)
    widths = []
    for _, _, chips, _ in rows:
        ws = []
        for text, kind in chips:
            t = sax.text(0, 0, text, fontsize=FS, fontweight="normal" if kind == "fixed" else "bold")
            scratch.canvas.draw()
            ws.append(t.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi + 2 * pad)
            t.remove()
        widths.append(ws)
    plt.close(scratch)

    left, label_col, bar_w, right = 0.35, 3.55, 2.3, 0.35
    chips_w = max(sum(w) + gap * (len(w) - 1) for w in widths)
    fig_w = left + label_col + chips_w + 0.55 + bar_w + 0.7 + right
    row_pitch, chip_h = 0.66, 0.4
    fig_h = 1.65 + row_pitch * len(rows) + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.12
    ax.text(fig_w / 2, top, title, fontsize=20, fontweight="bold", color=INK, ha="center", va="top")
    ax.text(fig_w / 2, top - 0.44, "What happens to the segmentation when a word is removed?", fontsize=13,
            color=INK2, ha="center", va="top")
    ax.text(fig_w / 2, top - 0.84,
            f"{organ.capitalize()}  |  replacement organ: {landmark}",
            fontsize=11.5, color=MUTED, ha="center", va="top")

    chips_x0 = left + label_col
    bar_x0 = chips_x0 + chips_w + 0.55
    y = top - 1.45
    ax.text(bar_x0, y + 0.08, "Dice vs. ground truth", fontsize=10.5, color=INK2, ha="left", va="bottom")
    for i, ((cond, name, chips, d), ws) in enumerate(zip(rows, widths)):
        cy = y - row_pitch * i - row_pitch / 2
        # row number + label
        ax.add_patch(mpatches.Circle((left + 0.16, cy), 0.16, facecolor=INK, linewidth=0, zorder=3))
        ax.text(left + 0.16, cy, str(i + 1), fontsize=11, fontweight="bold", color="white", ha="center",
                va="center", zorder=4)
        ax.text(left + 0.5, cy, name, fontsize=FS, fontweight="bold", color=INK, ha="left", va="center")
        x = chips_x0
        for (text, kind), w in zip(chips, ws):
            box = dict(boxstyle="round,pad=0,rounding_size=0.06")
            if kind == "fixed":
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=0, facecolor=GREY,
                                                     zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, color=GREY_TEXT, ha="left", va="center", zorder=5)
            elif kind in ("position", "organ"):
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=0,
                                                     facecolor=POS_C if kind == "position" else ORG_C, zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, fontweight="bold",
                        color=text_on(POS_C if kind == "position" else ORG_C), ha="left", va="center", zorder=5)
            elif kind in ("ghost_position", "ghost_organ"):
                col = POS_C if kind == "ghost_position" else ORG_C
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=1.3, edgecolor=col,
                                                     facecolor=SURFACE, linestyle=(0, (3, 2)), zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, fontweight="bold", color=col, alpha=0.55, ha="left",
                        va="center", zorder=5)
                ax.plot([x + pad * 0.7, x + w - pad * 0.7], [cy, cy], color=col, lw=1.6, alpha=0.75, zorder=6)
            else:  # replacement phrase
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=1.5, edgecolor=ORG_C,
                                                     facecolor=mix(ORG_C, SURFACE, 0.15), zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, fontweight="bold", color=mix(ORG_C, "#000000", 0.6), ha="left", va="center",
                        zorder=5)
            x += w + gap
        # Dice bar
        ax.add_patch(mpatches.FancyBboxPatch((bar_x0, cy - 0.1), bar_w, 0.2, linewidth=0, facecolor=BAR_BG, zorder=2,
                                             boxstyle="round,pad=0,rounding_size=0.08"))
        if d > 0.005:
            ax.add_patch(mpatches.FancyBboxPatch((bar_x0, cy - 0.1), max(bar_w * d, 0.12), 0.2, linewidth=0,
                                                 facecolor=ORG_C, zorder=3, boxstyle="round,pad=0,rounding_size=0.08"))
        ax.text(bar_x0 + bar_w + 0.14, cy, f"{d:.2f}", fontsize=FS + 1, fontweight="bold", color=INK, ha="left",
                va="center")

    # legend (bottom)
    ly = y - row_pitch * len(rows) - 0.3
    lx = left
    items = [("position", "Position word"), ("organ", "Organ name"), ("ghost", "Word removed"),
             ("replacement", "Replacement for the organ name"), ("fixed", "Word kept fixed")]
    for kind, txt in items:
        w = 0.44
        box = dict(boxstyle="round,pad=0,rounding_size=0.04")
        if kind == "position":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=0, facecolor=POS_C, **box))
        elif kind == "organ":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=0, facecolor=ORG_C, **box))
        elif kind == "ghost":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=1.2, edgecolor=MUTED,
                                                 facecolor=SURFACE, linestyle=(0, (3, 2)), **box))
            ax.plot([lx + 0.05, lx + w - 0.05], [ly, ly], color=MUTED, lw=1.4)
        elif kind == "replacement":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=1.2, edgecolor=ORG_C,
                                                 facecolor=mix(ORG_C, SURFACE, 0.15), **box))
        else:
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=0, facecolor=GREY, **box))
        ax.text(lx + w + 0.12, ly, txt, fontsize=10.5, color=INK2, ha="left", va="center")
        lx += w + 0.12 + 0.088 * len(txt) + 0.4

    out = Path(out) if out else OUT / f"{key}_{organ.replace(' ', '_')}_all_prompts.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out, "-", n_subj, "subjects:", sorted(set(a.subject_id)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
