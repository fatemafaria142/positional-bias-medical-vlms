"""
Row 15 for AMOS and OASIS: for every subject that was run and every organ, one figure that shows where VoxTell looks (its cross-attention)
in each of its six layers when it is given the request for that organ.

For each layer: attention on the requested organ (blue) and on the same organ on the other side of the body (dark), each as a multiple of
the average attention over the whole scan (dashed line = 1.0x = average).

Output: local_plots_groundedness/word_highlight/<AMOS|OASIS>/<organ>/rows13_16/subject_<N>/row15_cross_attention.png
        (N = subject number 1..10 as in the row01-row11 figures)
Input:  <AMOS|OASIS>-ROW(13 to 16)/row15_attention_enrichment_*.csv
Usage:  python scripts/make_row15_figures.py [AMOS] [OASIS]        (default: both)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_row14_figures as m  # noqa: E402  (canvas, colours, dataset configuration, two-pass sizing)
import plot_prompt_breakdown as pb  # noqa: E402

X_LABEL, X_BAR, W_BAR = 0.3, 1.3, 6.2
X_TXT = X_BAR + W_BAR + 0.15
DARK = "#2f3037"
PITCH = 0.72


def figure(organ, d, sid, num):
    pos, core = organ.split(None, 1)
    other = f"{'right' if pos == 'left' else 'left'} {core}"
    a = d[(d.subject_id == sid) & (d.prompt_organ == organ)].copy()
    q = a.set_index(["region_organ", "layer"]).enrichment
    layers = sorted(a.layer.unique())
    vmax = float(q.max()) * 1.05
    series = [(organ, pb.ORG_C), (other, DARK)]
    chips = pb.chips_for(organ, "none", "")
    m.fit_width(9.4, m.PAD + 2.5 + m.chips_width(chips) + m.PAD)
    question = f"When VoxTell is asked for the {organ}, does it look at the {organ} or at the {other}, layer by layer?"

    def build(cv):
        y = cv.h - 0.12
        cv.text(m.W / 2, y, m.TITLE, 20, "bold", ha="center", va="top")
        y -= 0.5
        import textwrap
        for ln in textwrap.wrap(question, int((m.W - 0.6) / 0.105)):
            cv.text(m.W / 2, y, ln, 14, color=m.INK2, ha="center", va="top")
            y -= 0.33
        cv.text(m.W / 2, y - 0.08, f"Organ: {organ}", 11.5, color=m.MUTED, ha="center", va="top")
        y -= 0.62
        cv.text(m.PAD, y - 0.2, "Input given to VoxTell:", m.FS, "bold")
        cv.chips(m.PAD + 2.5, y - 0.2, chips)
        y -= 0.55
        top = y
        for i, L in enumerate(layers):
            cy = y - PITCH * i - PITCH / 2
            cv.text(X_LABEL, cy, f"Layer {L}", m.FS, "bold")
            for k, (region, col) in enumerate(series):
                v = float(q[(region, L)])
                by = cy + 0.1 - k * 0.2
                cv.rect(X_BAR, by - 0.06, W_BAR, 0.12, m.TRACK, 0.05, 2)
                cv.rect(X_BAR, by - 0.06, max(W_BAR * min(v / vmax, 1), 0.06), 0.12, col, 0.05)
                cv.text(X_TXT, by, f"{v:.1f}×", 10.5, "bold" if k == 0 else "normal")
        bottom = y - PITCH * len(layers)
        rx = X_BAR + W_BAR * 1.0 / vmax
        cv.ax.plot([rx, rx], [top + 0.03, bottom], color=DARK, lw=1.2, linestyle=(0, (4, 3)), zorder=6)
        ly = bottom - 0.4
        for col, txt in [(pb.ORG_C, f"the requested organ ({organ})"), (DARK, f"the same organ on the other side ({other})"),
                         ("dash", "Attention on each region, compared with the average over the whole scan (1.0×)")]:
            if col == "dash":
                cv.ax.plot([m.PAD, m.PAD + 0.42], [ly, ly], color=DARK, lw=1.6, linestyle=(0, (4, 3)), zorder=6)
            else:
                cv.rect(m.PAD, ly - 0.11, 0.42, 0.22, col, 0.04)
            cv.text(m.PAD + 0.55, ly, txt, 11, color=m.INK2)
            ly -= 0.34
        return ly + 0.24
    return m.render(build, m.out_dir(organ, num) / "row15_cross_attention.png")


def main(names):
    for name in names:
        m.configure(name)
        d = m.read("row15_attention_enrichment")
        for sid in m.SUBJECTS:
            for o in m.ORGANS:
                print("wrote", figure(o, d, sid, m.NUMBERS[sid]).relative_to(m.REPO), flush=True)


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or ["AMOS", "OASIS"])
