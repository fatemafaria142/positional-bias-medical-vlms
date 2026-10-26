"""RQ11 image figures for the paper, one figure per organ, for the four largest organs that respond to the prompts.
Each figure is one compact row of panels side by side: the ground truth of the left organ, the ground truth of the right organ, then the predicted mask for each of the four prompt variants.
Under each predicted panel the side of the mask is written with two Dice bars (overlap with the true left organ and with the true right organ).
Green is the left side and red is the right side. In the predicted panels the white solid outline is the true left organ and the dashed outline is the true right organ."""
import base64, zlib, tarfile, os, textwrap
import numpy as np, pandas as pd, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(r"D:\Reasoning\results\neurite-oasis")
OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
TAR = Path(r"D:\Reasoning\data\raw\neurite-oasis.v1.0.tar")
SCRATCH = Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\3463f333-3745-4839-9350-42eda44b4d8b\scratchpad")
# organ -> (subject where every prompt gives a visible mask and the right-then-left prompt fails, crop scale, file); ordered from the biggest organ
ORGANS = {"cerebral white matter": ("OASIS_OAS1_0021_MR1", 1.0, "F3_cerebral_white_matter.png"), "cerebellum cortex": ("OASIS_OAS1_0012_MR1", 1.5, "F4_cerebellum_cortex.png"),
          "cerebellum white matter": ("OASIS_OAS1_0029_MR1", 1.3, "F5_cerebellum_white_matter.png"), "hippocampus": ("OASIS_OAS1_0020_MR1", 1.0, "F6_hippocampus.png")}
VARS = ["conflict_left_then_right", "conflict_right_then_left", "no_side_with_distractor", "side_word_in_injected_clause_only"]
HEADLAB = {"conflict_left_then_right": "Left named first, then right", "conflict_right_then_left": "Right named first, then left",
           "no_side_with_distractor": "No side named, distractor sentence", "side_word_in_injected_clause_only": "Side named only in an added sentence"}
CALLTXT = {"left": "mask on the left organ", "right": "mask on the right organ", "both": "mask on both organs", "neither": "mask on neither organ", "empty": "empty mask"}

avail = {x.name for x in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)})
INK = "#000000"; GREEN, RED = "#2f9e63", "#d6453d"

def load(subject):
    nii = SCRATCH / subject / "aligned_norm.nii.gz"
    if not nii.exists():
        nii.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TAR) as tf:
            for m in tf:
                if m.name == f"{subject}/aligned_norm.nii.gz":
                    m.name = os.path.basename(m.name); tf.extract(m, nii.parent); break
    return np.flip(np.transpose(np.asarray(nib.load(nii).dataobj).astype(np.float32), (1, 2, 0)), axis=(0, 2))
def dec(enc, shape):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool); pos, val = 0, False
    for r in runs:
        if val: flat[pos:pos + r] = True
        pos += int(r); val = not val
    return flat.reshape(shape)
M = pd.concat([ch[ch.subject_id.isin([v[0] for v in ORGANS.values()]) & ch.organ.isin(["left " + o for o in ORGANS])] for ch in pd.read_csv(ROOT / "RQ11" / "rq11_prompt_injection_bias_new.csv", chunksize=2000)])
ov = lambda ax_, m, col: ax_.imshow(np.ma.masked_where(~m, np.ones(m.shape)), cmap=matplotlib.colors.ListedColormap([col]), alpha=0.7, interpolation="nearest")

# layout in inches
PW, PH, PITCH, X0 = 1.95, 1.5, 2.1, 0.15
NC = 2 + len(VARS)
TITLE, HEAD, BELOW, LEG = 0.5, 1.0, 0.85, 0.75
FIGW = X0 + NC * PITCH
FIGH = TITLE + HEAD + PH + BELOW + LEG

def make(o):
    subject, k, fname = ORGANS[o]
    vol = load(subject); vmin, vmax = np.percentile(vol[vol > 0], [1, 99.5])
    q = M[(M.organ == "left " + o) & (M.subject_id == subject)].set_index("variant")
    shape = tuple(int(x) for x in q.volume_shape.iloc[0].split("x")); assert shape == vol.shape
    gl, gr = dec(q.iloc[0].gt_left_mask_rle, shape), dec(q.iloc[0].gt_right_mask_rle, shape)
    z = int(round(np.argwhere(gl | gr)[:, 0].mean())); cen = np.argwhere((gl | gr)[z - 6:z + 7].any(axis=0)).mean(0)
    k = min(k, shape[1] / 96, shape[2] / 124)  # the crop cannot be larger than the volume
    HA, HB = int(48 * k), int(62 * k)
    a0 = int(np.clip(round(cen[0]) - HA, 0, shape[1] - 2 * HA)); b0 = int(np.clip(round(cen[1]) - HB, 0, shape[2] - 2 * HB))
    cl, cr_ = np.argwhere(gl)[:, 2].mean(), np.argwhere(gr)[:, 2].mean(); assert cl < cr_
    zone_left = np.zeros(shape, bool); zone_left[:, :, :int(round((cl + cr_) / 2))] = True
    view = lambda a: np.flipud(a[z, a0:a0 + 2 * HA, b0:b0 + 2 * HB]); img = view(vol)
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor="white")
    txt = lambda x, y, s, **kw: fig.text(x / FIGW, (FIGH - y) / FIGH, s, color=INK, **kw)
    ytop = TITLE + HEAD
    def panel(ci):
        ax_ = fig.add_axes([(X0 + ci * PITCH) / FIGW, (FIGH - ytop - PH) / FIGH, PW / FIGW, PH / FIGH]); ax_.axis("off")
        ax_.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear"); return ax_
    def bar(x, y, letter, val, col):
        BX, BW = x + 0.2, 0.85
        txt(x, y, letter, ha="left", va="center", fontsize=9)
        fig.add_artist(Rectangle((BX / FIGW, (FIGH - y - 0.07) / FIGH), BW / FIGW, 0.14 / FIGH, fc="white", ec=INK, lw=0.8))
        if val > 0: fig.add_artist(Rectangle((BX / FIGW, (FIGH - y - 0.07) / FIGH), BW * val / FIGW, 0.14 / FIGH, fc=col, ec=INK, lw=0.8))
        txt(BX + BW + 0.06, y, f"{val:.2f}", ha="left", va="center", fontsize=9)
    txt(FIGW / 2, 0.08, f"Does VoxTell Choose the Left or the Right {o.title()} When the Prompt Names Conflicting or No Sides?", ha="center", va="top", fontsize=13.5, fontweight="bold")
    for ci, (gm, col, side) in enumerate(((gl, GREEN, "left"), (gr, RED, "right"))):
        ov(panel(ci), view(gm), col)
        txt(X0 + ci * PITCH + PW / 2, ytop - 0.12, f"Ground truth\ntrue {side} organ", ha="center", va="bottom", fontsize=10, linespacing=1.3)
    for vi, v in enumerate(VARS):
        ci = 2 + vi; r = q.loc[v]; mk = dec(r.pred_mask_rle, shape); ax_ = panel(ci); x = X0 + ci * PITCH
        for part, col in ((mk & zone_left, GREEN), (mk & ~zone_left, RED)): ov(ax_, view(part), col)
        for gm, ls in ((gl, "solid"), (gr, "dashed")): ax_.contour(view(gm).astype(float), levels=[0.5], colors="white", linewidths=1.0, linestyles=ls)
        prompt = "\u201c" + r.prompt.replace(o, "X") + "\u201d"
        txt(x + PW / 2, ytop - 0.12, HEADLAB[v] + "\n" + "\n".join(textwrap.wrap(prompt, 31)), ha="center", va="bottom", fontsize=8.6, linespacing=1.3)
        txt(x + PW / 2, ytop + PH + 0.17, CALLTXT[r.call], ha="center", va="center", fontsize=9.4)
        bar(x + 0.1, ytop + PH + 0.45, "L", r.dice_left_organ, GREEN); bar(x + 0.1, ytop + PH + 0.7, "R", r.dice_right_organ, RED)
    fig.add_artist(plt.Line2D([(X0 + 2 * PITCH - 0.08) / FIGW] * 2, [(FIGH - (TITLE + 0.05)) / FIGH, (FIGH - (ytop + PH + 0.85)) / FIGH], color=INK, lw=0.8))
    halo = [pe.Stroke(linewidth=4.0, foreground=INK), pe.Normal()]
    fig.legend(handles=[Patch(color=GREEN, alpha=0.85, label="Left organ or mask on the left side"), Patch(color=RED, alpha=0.85, label="Right organ or mask on the right side"),
                        Line2D([0], [0], color="white", lw=2, path_effects=halo, label="True left organ outline"),
                        Line2D([0], [0], color="white", lw=2, ls=(0, (2.2, 2.2)), path_effects=halo, label="True right organ outline")],
               loc="center", bbox_to_anchor=(0.5, 0.5 / FIGH), ncol=4, frameon=False, fontsize=9.8, handlelength=2.2, columnspacing=1.8)
    txt(FIGW / 2, FIGH - 0.14, "L and R are the Dice of the mask with the true left and the true right organ, from 0 (no overlap) to 1 (perfect match).",
        ha="center", va="center", fontsize=8.8)
    fig.savefig(OUT / fname, dpi=240, bbox_inches="tight", facecolor="white"); plt.close()

for o in ORGANS: make(o)
print("done")
