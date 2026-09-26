"""
Row 16 for AMOS and OASIS: for every subject that was run and every organ, one figure that shows how different VoxTell's internal state is
for two requests ("left X" and "right X") on the same scan, layer by layer.

Nothing is changed inside the model: the two requests are run separately, each as normal, and their internal states are compared
(cosine distance: 0 = identical, larger = more different) before layer 1 and after each of the six layers. For scale, the same is
done for two requests about different organs (left X vs. left Y).

Output: local_plots_groundedness/word_highlight/<AMOS|OASIS>/<organ>/rows13_16/subject_<N>/row16_left_vs_right_inside_the_model.png
Input:  <AMOS|OASIS>-ROW(13 to 16)/row16_layer_distance_*.csv
Usage:  python scripts/make_row16_figures.py [AMOS] [OASIS]        (default: both)
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_row14_figures as m  # noqa: E402  (canvas, colours, dataset configuration, two-pass sizing)
import plot_prompt_breakdown as pb  # noqa: E402

X_LABEL, X_BAR, W_BAR = 0.3, 2.35, 5.3
X_TXT = X_BAR + W_BAR + 0.15
GREY = "#b9b8b2"
PITCH = 0.72
VMAX = 0.6                                              # same scale in every figure (the largest distance is about 0.53)


def figure(organ, d, sid, num):
    pos, core = organ.split(None, 1)
    other = f"{'right' if pos == 'left' else 'left'} {core}"
    l = d[(d.subject_id == sid) & (d.scope == "all_patches") & d.pair.str.startswith(core + ":")]
    lr = l[l.pair.str.contains("left vs right")].set_index("layer").cosine_distance
    ref_rows = l[l.pair.str.contains("reference")]
    ref_name = ref_rows.pair.iloc[0].split(": ", 1)[1].replace(" (reference)", "").replace(" vs ", " vs. ")
    rf = ref_rows.set_index("layer").cosine_distance
    layers = sorted(lr.index)
    chips1, chips2 = pb.chips_for(organ, "none", ""), pb.chips_for(other, "none", "")
    m.fit_width(9.4, m.PAD + 1.6 + max(m.chips_width(chips1), m.chips_width(chips2)) + m.PAD,
                X_LABEL + m.text_width(f"for scale: requests about two different organs ({ref_name})", size=11) + 0.55 + m.PAD)
    question = f"Inside VoxTell, how different are the requests for the {organ} and for the {other}, layer by layer?"
    note = ("Nothing is changed inside the model. Each request is run separately on the same scan, and what the model holds "
            "internally for the two requests is compared.")

    def build(cv):
        y = cv.h - 0.12
        cv.text(m.W / 2, y, m.TITLE, 20, "bold", ha="center", va="top")
        y -= 0.5
        for ln in textwrap.wrap(question, int((m.W - 0.6) / 0.105)):
            cv.text(m.W / 2, y, ln, 14, color=m.INK2, ha="center", va="top")
            y -= 0.33
        cv.text(m.W / 2, y - 0.08, f"Organ: {organ}", 11.5, color=m.MUTED, ha="center", va="top")
        y -= 0.55
        for ln in textwrap.wrap(note, int((m.W - 0.6) / 0.084)):
            cv.text(m.PAD, y - 0.1, ln, 11, color=m.INK2, va="top")
            y -= 0.28
        y -= 0.3
        for label, chips in (("Request 1:", chips1), ("Request 2:", chips2)):
            cv.text(m.PAD, y - 0.2, label, m.FS, "bold")
            cv.chips(m.PAD + 1.6, y - 0.2, chips)
            y -= 0.6
        y -= 0.2
        cv.text(X_BAR, y, "Difference between the two requests (0 = identical, larger = more different)", 10.5, color=m.INK2, va="bottom")
        y -= 0.1
        for i, k in enumerate(layers):
            cy = y - PITCH * i - PITCH / 2
            cv.text(X_LABEL, cy, "Before layer 1" if k == 0 else f"After layer {k}", m.FS, "bold")
            for j, (v, col) in enumerate(((float(lr[k]), pb.ORG_C), (float(rf[k]), GREY))):
                by = cy + 0.1 - j * 0.2
                cv.rect(X_BAR, by - 0.06, W_BAR, 0.12, m.TRACK, 0.05, 2)
                cv.rect(X_BAR, by - 0.06, max(W_BAR * min(v / VMAX, 1), 0.06), 0.12, col, 0.05)
                cv.text(X_TXT, by, f"{v:.2f}", 10.5, "bold" if j == 0 else "normal")
        ly = y - PITCH * len(layers) - 0.35
        for col, txt in [(pb.ORG_C, f"the two requests above ({core}: left vs. right)"),
                         (GREY, f"for scale: requests about two different organs ({ref_name})")]:
            cv.rect(m.PAD, ly - 0.11, 0.42, 0.22, col, 0.04)
            cv.text(m.PAD + 0.55, ly, txt, 11, color=m.INK2)
            ly -= 0.34
        return ly + 0.24
    return m.render(build, m.out_dir(organ, num) / "row16_left_vs_right_inside_the_model.png")


def main(names):
    for name in names:
        m.configure(name)
        d = m.read("row16_layer_distance")
        for sid in m.SUBJECTS:
            for o in m.ORGANS:
                print("wrote", figure(o, d, sid, m.NUMBERS[sid]).relative_to(m.REPO), flush=True)


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or ["AMOS", "OASIS"])
