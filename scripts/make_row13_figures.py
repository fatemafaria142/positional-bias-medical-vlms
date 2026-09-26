"""
Row 13 for AMOS and OASIS: for every subject that was run and every organ, one figure showing what VoxTell segments when the same request is
given on (1) the original image, (2) an image where the requested organ is replaced by random noise, (3) an image that is entirely noise.

Two bars per row:
  blue  = size of the segmented mask compared with the mask on the original image (100% = same size; it does not say the mask is correct)
  dark  = share of the true organ (the expert outline) that the segmented mask covers (100% = the whole organ)
The noise is Gaussian with the mean and standard deviation of the non-zero voxels of that scan (given in one line under the organ name).

Output: local_plots_groundedness/word_highlight/<AMOS|OASIS>/<organ>/rows13_16/subject_<N>/row13_organ_replaced_by_noise.png
Input:  <AMOS|OASIS>-ROW(13 to 16)/row13_organ_noise_*.csv  (+ the scan, for the noise mean / spread)
Usage:  python scripts/make_row13_figures.py [AMOS] [OASIS]        (default: both)
"""

import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_row14_figures as m  # noqa: E402  (canvas, colours, dataset configuration, two-pass sizing)
import plot_prompt_breakdown as pb  # noqa: E402

DARK = "#2f3037"
PITCH = 0.7
X_LABEL, W_BAR = 0.3, 4.0


def noise_stats(person, sid):
    import nibabel as nib
    path = m.REPO / person / "samples" / m.CFG["vol"] / "volumes" / f"{sid}_rot000_image.nii.gz"
    a = np.asarray(nib.load(path).dataobj, dtype="float32")
    nz = a[a != 0]
    return float(nz.mean()), float(nz.std())


def figure(organ, d, sid, num):
    s = d[(d.subject_id == sid) & (d.organ == organ)].set_index("condition")
    mean_, std_ = noise_stats(s.person.iloc[0], sid)
    conds = [("Original image", "original"), (f"{organ.capitalize()} replaced by random noise", "noise_target"),
             ("Whole image replaced by random noise", "noise_image")]
    vmax = max(1.0, max(float(s.loc[c, "size_vs_original"]) for _, c in conds))
    chips = pb.chips_for(organ, "none", "")
    x_bar = X_LABEL + max(m.text_width(l, True) for l, _ in conds) + 0.4
    x_txt = x_bar + W_BAR + 0.15
    legend_items = [(pb.ORG_C, "Size of the segmented mask, compared with the mask on the original image (100% = same size)"),
                    (DARK, "Share of the true organ covered by the segmented mask (100% = the whole organ)")]
    noise_line = f"Random noise: Gaussian, mean {mean_:.3g}, standard deviation {std_:.3g} (this scan's own average brightness and spread)"
    m.fit_width(x_txt + m.text_width("almost nothing (<1%)", size=10.5) + 0.3, m.PAD + 2.5 + m.chips_width(chips) + m.PAD,
                m.PAD + 0.55 + max(m.text_width(t, size=11) for _, t in legend_items) + m.PAD,
                m.PAD + m.text_width(noise_line, size=11) + m.PAD)
    question = f"Does VoxTell still segment the {organ} when the {organ} is replaced by random noise?"

    def build(cv):
        y = cv.h - 0.12
        cv.text(m.W / 2, y, m.TITLE, 20, "bold", ha="center", va="top")
        y -= 0.5
        for ln in textwrap.wrap(question, int((m.W - 0.6) / 0.105)):
            cv.text(m.W / 2, y, ln, 14, color=m.INK2, ha="center", va="top")
            y -= 0.33
        cv.text(m.W / 2, y - 0.08, f"Organ: {organ}", 11.5, color=m.MUTED, ha="center", va="top")
        cv.text(m.W / 2, y - 0.45, noise_line, 11, color=m.MUTED, ha="center", va="top")
        y -= 1.0
        cv.text(m.PAD, y - 0.2, "Input given to VoxTell:", m.FS, "bold")
        cv.chips(m.PAD + 2.5, y - 0.2, chips)
        y -= 0.7
        for i, (label, cond) in enumerate(conds):
            cy = y - PITCH * i - PITCH / 2
            cv.text(X_LABEL, cy, label, m.FS, "bold")
            v, n, cov = float(s.loc[cond, "size_vs_original"]), int(s.loc[cond, "n_pred_voxels"]), float(s.loc[cond, "cov_of_noised_region"])
            t1 = "nothing segmented" if n == 0 else ("almost nothing (<1%)" if v < 0.01 else f"{v * 100:.0f}%")
            for j, (val, vm, col, txt) in enumerate(((v, vmax, pb.ORG_C, t1), (cov, 1.0, DARK, f"{cov * 100:.0f}%"))):
                by = cy + 0.1 - j * 0.2
                cv.rect(x_bar, by - 0.06, W_BAR, 0.12, m.TRACK, 0.05, 2)
                if val > 0.005:
                    cv.rect(x_bar, by - 0.06, max(W_BAR * min(val / vm, 1), 0.06), 0.12, col, 0.05)
                cv.text(x_txt, by, txt, 10.5, "bold" if j == 0 else "normal")
        ly = y - PITCH * len(conds) - 0.3
        for col, txt in legend_items:
            cv.rect(m.PAD, ly - 0.11, 0.42, 0.22, col, 0.04)
            cv.text(m.PAD + 0.55, ly, txt, 11, color=m.INK2)
            ly -= 0.34
        return ly + 0.24
    return m.render(build, m.out_dir(organ, num) / "row13_organ_replaced_by_noise.png")


def main(names):
    for name in names:
        m.configure(name)
        d = m.read("row13_organ_noise")
        for sid in m.SUBJECTS:
            for o in m.ORGANS:
                print("wrote", figure(o, d, sid, m.NUMBERS[sid]).relative_to(m.REPO), flush=True)


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]] or ["AMOS", "OASIS"])
