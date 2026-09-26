"""
Row 14 for AMOS and OASIS: for every subject that was run and every organ, two SEPARATE figures drawn as a diagram of what is changed
inside VoxTell:

  text code (embedding)  ->  6 layers (decoder layers)  ->  result

  A  row14_A_mixed_request.png
       Nothing is swapped inside the model. Before the model runs, the text code is replaced by a mix of the numbers of the
       "left" and the "right" request (for example 75% left + 25% right); the six layers then run normally.
  B  row14_B_<side>_to_<other>_inside_the_model.png
       The input is ONE request, e.g. "Segment the left kidney in this image.", and its text code is left untouched. Inside the
       model, the chosen layers do not use their own result but what the OTHER side's request would have done in that layer on
       the same scan.

Blue = the "left" request, pink = the "right" request (AMOS palette; the OASIS palette is used for OASIS). Each figure is sized to its
content (no spare white space). This module also holds the drawing helpers used by make_row13/15/16_figures.py.

Output: local_plots_groundedness/word_highlight/<AMOS|OASIS>/<organ>/rows13_16/subject_<N>/   (N = subject number 1..10 as in the
        row01-row11 figures)
Input:  <AMOS|OASIS>-ROW(13 to 16)/row14_embedding_intervention_*.csv
Usage:  python scripts/make_row14_figures.py [AMOS] [OASIS]        (default: both)
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_prompt_breakdown as pb  # noqa: E402  (fonts, palette, prompt chips)

REPO = Path(__file__).resolve().parent.parent
DATASETS = {
    "AMOS": dict(title="AMOS22 MRI", dir=REPO / "AMOS-ROW(13 to 16)", sfx="_amos22_mri.csv", vol="amos22_mri", palette=("#f400d4", "#5683ff"),
                 numbering=[REPO / "Voxtell-Amos-Groundness" / "baseline_predictions_amos22_mri.csv"], w_emb=2.7,
                 organs=["left kidney", "right kidney", "left adrenal gland", "right adrenal gland"]),
    "OASIS": dict(title="Neurite-OASIS", dir=REPO / "OASIS-ROW-(13 to 16)", sfx="_neurite_oasis.csv", vol="neurite_oasis", palette=("#731cff", "#00a7ad"),
                  numbering=[REPO / "Voxtell-OASIS-Batch1-Groundness" / "baseline_predictions_neurite_oasis.csv",
                             REPO / "Voxtell-OASIS-Batch2-Groundness" / "baseline_predictions_neurite_oasis_batch2.csv"], w_emb=3.5,
                  organs=["left cerebral white matter", "right cerebral white matter", "left cerebellum cortex", "right cerebellum cortex"]),
}
PLURAL = {"kidney": "kidneys", "adrenal gland": "adrenal glands"}

INK, INK2, MUTED, SURFACE, GREY, GREY_TEXT = pb.INK, pb.INK2, pb.MUTED, pb.SURFACE, pb.GREY, pb.GREY_TEXT
BAR = "#2f3037"
FS = 11.5
SQ, W_BAR = 0.42, 1.8
X_LABEL, X_EMB = 0.3, 3.5
PITCH = 0.6
PAD = 0.3
W = 15.0                                            # canvas width; every figure sets it from its own content (see fit_width)
CFG = {}


def configure(name):
    """Select the dataset: palette, folders, subjects (from the row 14 CSV) and column layout."""
    global CFG, TITLE, ROOT, DIR, SFX, ORGANS, SUBJECTS, NUMBERS, LEFT_C, RIGHT_C, TRACK
    global W_EMB, X_LAY, X_BAR, X_PCT, X_ON
    CFG = DATASETS[name]
    TITLE, DIR, SFX, ORGANS = CFG["title"], CFG["dir"], CFG["sfx"], CFG["organs"]
    ROOT = REPO / "local_plots_groundedness" / "word_highlight" / name
    pb.POS_C, pb.ORG_C = CFG["palette"]
    LEFT_C, RIGHT_C = pb.ORG_C, pb.POS_C             # blue-ish = "left" request, pink/violet = "right" request
    TRACK = pb.mix(pb.ORG_C, SURFACE, 0.22)
    W_EMB = CFG["w_emb"]
    X_LAY = X_EMB + W_EMB + 0.65
    X_BAR = X_LAY + 6 * (SQ + 0.08) + 0.4
    X_PCT = X_BAR + W_BAR + 0.15
    X_ON = X_PCT + 0.65
    SUBJECTS = sorted(read("row14_embedding_intervention").subject_id.unique())
    allsubs = sorted(pd.concat([pd.read_csv(f, usecols=["subject_id"]) for f in CFG["numbering"]]).subject_id.unique())
    NUMBERS = {s: allsubs.index(s) + 1 for s in SUBJECTS}


def read(name):
    return pd.read_csv(DIR / f"{name}{SFX}")


def blend_color(alpha):
    return pb.mix(RIGHT_C, LEFT_C, min(max(alpha, 0.0), 1.0))          # share of the "right" request


def text_width(s, bold=False, size=FS):
    scratch = plt.figure(figsize=(20, 5), dpi=200)
    sax = scratch.add_subplot(111)
    t = sax.text(0, 0, s, fontsize=size, fontweight="bold" if bold else "normal")
    scratch.canvas.draw()
    w = t.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi
    plt.close(scratch)
    return w


def chip_widths(chips, pad=0.16):
    return [text_width(t, k != "fixed") + 2 * pad for t, k in chips]


def fit_width(min_w, *needed):
    """Set the canvas width to the widest thing that has to fit (never below min_w)."""
    global W
    W = max([min_w, *needed])
    return W


def wrap(s, size=FS):
    return textwrap.wrap(s, int((W - 2 * PAD) / (0.0077 * size)))


class Canvas:
    def __init__(self, height):
        self.h = height
        self.fig = plt.figure(figsize=(W, height), dpi=170)
        self.fig.patch.set_facecolor(SURFACE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    def text(self, x, y, s, size=FS, weight="normal", color=INK, ha="left", va="center"):
        self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)

    def rect(self, x, y, w, h, color, rounding=0.06, z=3):
        self.ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, linewidth=0, facecolor=color, zorder=z,
                                                  boxstyle=f"round,pad=0,rounding_size={rounding}"))

    def chips(self, x, cy, chips, pad=0.16, gap=0.09, h=0.4):
        for (t, k), w in zip(chips, chip_widths(chips)):
            if k == "fixed":
                self.rect(x, cy - h / 2, w, h, GREY)
                self.text(x + pad, cy, t, FS, color=GREY_TEXT)
            else:
                col = pb.POS_C if k == "position" else pb.ORG_C
                self.rect(x, cy - h / 2, w, h, col)
                self.text(x + pad, cy, t, FS, "bold", pb.text_on(col))
            x += w + gap


def chips_width(chips, gap=0.09):
    return sum(chip_widths(chips)) + gap * (len(chips) - 1)


def result_of(r, cov, core):
    both = f"BOTH {PLURAL[core]}" if core in PLURAL else "BOTH sides"
    side = {"left": f"LEFT {core}", "right": f"RIGHT {core}", "both": both, "neither": "neither organ", "empty": "no mask"}[r["call"]]
    return float(r[cov]), side


def draw_rows(cv, y, rows, mode, organ, own_c):
    cv.text(X_EMB, y, "Text code", 10.5, color=INK2, va="bottom")
    cv.text(X_LAY, y, "The six layers of the model", 10.5, color=INK2, va="bottom")
    cv.text(X_BAR, y, "Share of the true organ covered", 10.5, color=INK2, va="bottom")
    cv.text(X_ON, y, "Where the mask lands", 10.5, color=INK2, va="bottom")
    y -= 0.08
    for i, r in enumerate(rows):
        cy = y - PITCH * i - PITCH / 2
        cv.text(X_LABEL, cy, r["label"], FS, "bold")
        if mode == "blend":
            share_r = min(max(r["alpha"], 0.0), 1.0)
            if share_r < 1:
                cv.rect(X_EMB, cy - 0.19, W_EMB, 0.38, LEFT_C)
            if share_r > 0:
                cv.rect(X_EMB + W_EMB * (1 - share_r), cy - 0.19, W_EMB * share_r, 0.38, RIGHT_C)
            cv.ax.text(X_EMB + W_EMB / 2, cy, r["emb_txt"], fontsize=10.5, fontweight="bold", color="white", ha="center", va="center",
                       zorder=6, bbox=dict(boxstyle="round,pad=0.25", facecolor="#0b0b0bcc", edgecolor="none"))
        else:
            cv.rect(X_EMB, cy - 0.19, W_EMB, 0.38, own_c)
            cv.text(X_EMB + W_EMB / 2, cy, f"\"{organ}\" request", 10.5, "bold", pb.text_on(own_c), ha="center")
        cv.ax.annotate("", xy=(X_LAY - 0.1, cy), xytext=(X_EMB + W_EMB + 0.1, cy), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
        for j, col in enumerate(r["layers"]):
            x = X_LAY + j * (SQ + 0.08)
            cv.rect(x, cy - SQ / 2, SQ, SQ, col, 0.05)
            cv.text(x + SQ / 2, cy, str(j + 1), 10.5, "bold", pb.text_on(col), ha="center")
        v, side = r["res"]
        cv.rect(X_BAR, cy - 0.1, W_BAR, 0.2, TRACK, 0.08, 2)
        if v > 0.005:
            cv.rect(X_BAR, cy - 0.1, max(W_BAR * v, 0.08), 0.2, BAR, 0.08)
        cv.text(X_PCT, cy, f"{v * 100:.0f}%", FS + 1, "bold")
        cv.text(X_ON, cy, side, FS, "bold" if v > 0.3 else "normal", INK if v > 0.3 else "#b3261e")
    return y - PITCH * len(rows)


def legend(cv, y, items):
    x = PAD
    for col, txt in items:
        if col == "mix":
            cv.rect(x, y - 0.11, 0.21, 0.22, LEFT_C, 0.04)
            cv.rect(x + 0.21, y - 0.11, 0.21, 0.22, RIGHT_C, 0.04)
        else:
            cv.rect(x, y - 0.11, 0.42, 0.22, col, 0.04)
        cv.text(x + 0.55, y, txt, 11, color=INK2)
        x += 0.55 + text_width(txt, size=11) + 0.45
    return y - 0.11


def render(build, out):
    """Two passes: measure how tall the content is, then draw it on a canvas of exactly that height."""
    probe = Canvas(40)
    y_end = build(probe)
    used = (probe.h - 0.12) - y_end
    cv = Canvas(used + 0.12 + PAD)
    build(cv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv.fig.savefig(out, facecolor=cv.fig.get_facecolor())
    plt.close(cv.fig)
    plt.close(probe.fig)
    return out


def context(organ, d, sid):
    pos, core = organ.split(None, 1)
    opos = "right" if pos == "left" else "left"
    b = d[(d.subject_id == sid) & (d.pair == core)]
    return dict(pos=pos, opos=opos, core=core, other=f"{opos} {core}", b=b, cov="cov_left_organ" if pos == "left" else "cov_right_organ",
                own_c=LEFT_C if pos == "left" else RIGHT_C, oth_c=RIGHT_C if pos == "left" else LEFT_C)


def header(cv, q, organ):
    y = cv.h - 0.12
    cv.text(W / 2, y, TITLE, 20, "bold", ha="center", va="top")
    y -= 0.5
    for ln in textwrap.wrap(q, int((W - 0.6) / 0.105)):
        cv.text(W / 2, y, ln, 14, color=INK2, ha="center", va="top")
        y -= 0.33
    cv.text(W / 2, y - 0.08, f"Organ: {organ}", 11.5, color=MUTED, ha="center", va="top")
    return y - 0.6


def out_dir(organ, num):
    return ROOT / organ.replace(" ", "_") / "rows13_16" / f"subject_{num}"


def figure_A(organ, d, sid, num):
    c = context(organ, d, sid)
    rows = []
    for a in (-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5):
        if a in (0, 1):
            r = c["b"][(c["b"].variant == "baseline") & (c["b"].dst == ("left" if a == 0 else "right"))].iloc[0]
            label, txt = f"The normal {'left' if a == 0 else 'right'} request", f"100% {'left' if a == 0 else 'right'}"
        else:
            r = c["b"][(c["b"].variant == "blend") & (c["b"].alpha == a)].iloc[0]
            label = "A mix of the two requests"
            lft, rgt = round((1 - a) * 100), round(a * 100)
            txt = f"{lft}% left - {abs(rgt)}% right" if a < 0 else (f"{rgt}% right - {abs(lft)}% left" if a > 1 else f"{lft}% left + {rgt}% right")
        rows.append(dict(label=label, emb_txt=txt, alpha=a, layers=[blend_color(a)] * 6, res=result_of(r, c["cov"], c["core"])))
    q = "Does the segmented side change gradually or abruptly when the text embedding is interpolated between the left and the right request?"
    desc_text = ("Before the model runs, the list of numbers that stands for the request (the text code) is replaced by a mix of the numbers "
                 "of the \"left\" request and the \"right\" request, for example 75% left + 25% right. Nothing is swapped inside the model: "
                 "the six layers then run normally on that mixed code.")
    fit_width(X_ON + 0.25 + max(text_width(r["res"][1], True) for r in rows))

    def build(cv):
        y = header(cv, q, organ)
        for ln in wrap(desc_text):
            cv.text(PAD, y, ln, color=INK2, va="top")
            y -= 0.3
        y -= 0.25
        y = draw_rows(cv, y, rows, "blend", organ, c["own_c"])
        return legend(cv, y - 0.4, [(LEFT_C, "the \"left\" request"), (RIGHT_C, "the \"right\" request"), ("mix", "a mix of the two")])
    return render(build, out_dir(organ, num) / "row14_A_mixed_request.png")


def figure_B(organ, d, sid, num):
    c = context(organ, d, sid)
    pos, opos, core, other, b, own, oth = c["pos"], c["opos"], c["core"], c["other"], c["b"], c["own_c"], c["oth_c"]

    def pick(var, layer=None):
        q = b[(b.dst == pos) & (b.variant == var)]
        if layer is not None:
            q = q[q.layer == layer]
        return q.iloc[0]
    rows = [dict(label="No change", layers=[own] * 6, res=result_of(pick("baseline"), c["cov"], core))]
    for k in range(6):
        rows.append(dict(label=f"Only layer {k + 1} switched to {opos}", layers=[oth if j == k else own for j in range(6)],
                         res=result_of(pick("single_layer_update", k), c["cov"], core)))
    rows.append(dict(label=f"Layers 4-6 switched to {opos}", layers=[own] * 3 + [oth] * 3, res=result_of(pick("updates_from_layer", 3), c["cov"], core)))
    rows.append(dict(label=f"All 6 layers switched to {opos}", layers=[oth] * 6, res=result_of(pick("updates_from_layer", 0), c["cov"], core)))
    q = f"Does the segmentation still follow the input request ({organ}) when the decoder layers are switched to the {other}?"
    chips = pb.chips_for(organ, "none", "")
    fit_width(X_ON + 0.25 + max(text_width(r["res"][1], True) for r in rows), PAD + 2.75 + chips_width(chips) + PAD)

    def build(cv):
        y = header(cv, q, organ)
        cv.text(PAD, y - 0.2, "Input given to VoxTell:", FS, "bold")
        cv.chips(PAD + 2.75, y - 0.2, chips)
        y -= 0.8
        y = draw_rows(cv, y, rows, "patch", organ, own)
        return legend(cv, y - 0.4, [(own, f"layer that works on the input request (\"{organ}\")"),
                                    (oth, f"layer switched to the \"{other}\" request")])
    return render(build, out_dir(organ, num) / f"row14_B_{pos}_to_{opos}_inside_the_model.png")


def main(names):
    for name in names:
        configure(name)
        d = read("row14_embedding_intervention")
        for sid in SUBJECTS:
            for o in ORGANS:
                for f in (figure_A, figure_B):
                    print("wrote", f(o, d, sid, NUMBERS[sid]).relative_to(REPO), flush=True)


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or ["AMOS", "OASIS"])
