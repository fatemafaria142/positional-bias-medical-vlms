"""RQ3 prototype with images. One row per organ pair of one subject.
Panels: the organ image, and the true left organ (green) and true right organ (red) on the same slice.
Cells: attention KL divergence between the left and the right prompt in each of the six decoder layers, in both directions
(rq3_layer_distance_and_kl.csv, attention_kl_left_from_right and attention_kl_right_from_left). Masks come from the RQ6 coalitions file (full prompt row).
Saved in figures_with_images."""
import sys, base64, zlib
import numpy as np, pandas as pd, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize

SID = f"OASIS_OAS1_{int(sys.argv[1]) if len(sys.argv) > 1 else 14:04d}_MR1"
NII = Path(r"D:\Reasoning\data\raw\neurite-oasis-extracted") / SID / "aligned_norm.nii.gz"
KL = Path(r"D:\Reasoning\results\neurite-oasis\RQ3\rq3_layer_distance_and_kl.csv")
MASKS = Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\d543e881-113f-4f97-ad0b-09f0b81dbbfd\scratchpad") / f"{SID}_coalitions.pkl"
FIG = Path(__file__).parent / "figures_with_images"; FIG.mkdir(exist_ok=True)
A, B = "attention_kl_left_from_right", "attention_kl_right_from_left"

d = pd.read_csv(KL)
d = d[(d.subject_id == SID) & (d.scope == "organ_patches") & (d.pair_kind == "target") & (d.layer >= 1)].copy()
d["organ"] = d.pair.str.split(":").str[0]
M = pd.read_pickle(MASKS).set_index("organ")
SHAPE = tuple(int(x) for x in M.volume_shape.iloc[0].split("x"))

def dec(enc):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(SHAPE)), dtype=bool); pos, val = 0, False
    for r in runs:
        if val: flat[pos:pos + r] = True
        pos += int(r); val = not val
    return flat.reshape(SHAPE)

vol = np.flip(np.transpose(np.asarray(nib.load(NII).dataobj).astype(np.float32), (1, 2, 0)), axis=(0, 2)); assert vol.shape == SHAPE
vmin, vmax = np.percentile(vol[vol > 0], [1, 99.5])

avail = {x.name for x in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)})
INK = "#000000"; LEFT_COL, RIGHT_COL = "#2f9e63", "#d6453d"
cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cfe3f6", "#8fbfe8", "#4f97d6", "#2a73bf"])
norm = Normalize(0, 0.1)

PH, PW = 1.7, 2.7                                                           # panel height and width in inches

def panels(name):
    gl, gr = dec(M.loc["left " + name].gt_mask_rle), dec(M.loc["right " + name].gt_mask_rle)
    zs = [np.argwhere(g)[:, 0].mean() for g in (gl, gr)]; z = int(round(np.mean(zs)))
    pts = np.argwhere(gl[z] | gr[z])
    if len(pts) == 0: pts = np.argwhere(gl.any(axis=0) | gr.any(axis=0))
    lo, hi = pts.min(0), pts.max(0); cen = (lo + hi) / 2
    hw = max((hi[1] - lo[1]) / 2 + 14, (hi[0] - lo[0]) / 2 * PW / PH + 14, 34); hh = hw * PH / PW
    a0 = int(np.clip(round(cen[0] - hh), 0, SHAPE[1] - int(2 * hh))); b0 = int(np.clip(round(cen[1] - hw), 0, SHAPE[2] - int(2 * hw)))
    sl = lambda v: np.flipud(v[z, a0:a0 + int(2 * hh), b0:b0 + int(2 * hw)])
    pl, pr = dec(M.loc["left " + name].pred_mask_rle), dec(M.loc["right " + name].pred_mask_rle)
    return sl(vol), sl(gl), sl(gr), sl(pl), sl(pr)

def make(names, fname):
    n = len(names)
    W, rowh = 21.0, 2.3
    H = 1.2 + 0.5 + rowh * n + 1.6
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W / 2, H - 0.45, "How Does the KL Divergence Between Left and Right Prompt Attention Change Across the Prompt Decoder Layers?",
            ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
    top = H - 1.15
    XI = [2.7, 5.5, 8.3]; XC = 12.3; CW = 1.25
    ax.text(0.4, top, "Organ pair", ha="left", va="center", fontsize=12.5, color=INK, fontweight="semibold")
    for xi, t in zip(XI, ["Organ image", "True organs", "Predicted masks"]):
        ax.text(xi + PW / 2, top, t, ha="center", va="center", fontsize=12.5, color=INK, fontweight="semibold")
    for k in range(6):
        ax.text(XC + k * CW + CW / 2, top, f"Layer {k + 1}", ha="center", va="center", fontsize=12.5, color=INK, fontweight="semibold")
    top -= 0.3
    for i, name in enumerate(names):
        y0 = top - (i + 1) * rowh; yc = y0 + rowh / 2
        ax.add_patch(Rectangle((0.25, y0 + 0.05), W - 0.5, rowh - 0.12, fc="white", ec=INK, lw=1.4, zorder=1))
        ax.text(0.45, yc, ("Ventral DC" if name == "ventral DC" else name.capitalize()), ha="left", va="center", fontsize=14, color=INK, fontweight="semibold", zorder=4)
        img, gl, gr, pl, pr = panels(name)
        for xi, kind in zip(XI, ("img", "gt", "pr")):
            a = fig.add_axes([xi / W, (yc - PH / 2) / H, PW / W, PH / H]); a.axis("off")
            a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="auto")
            if kind == "gt":
                for g, c in ((gl, LEFT_COL), (gr, RIGHT_COL)):
                    if g.any():
                        rgba = np.zeros(g.shape + (4,)); rgba[g] = matplotlib.colors.to_rgba(c, 0.5); a.imshow(rgba, interpolation="nearest", aspect="auto")
                        a.contour(g.astype(float), levels=[0.5], colors=[c], linewidths=1.4)
            if kind == "pr":
                for g, t, c in ((pl, gl, LEFT_COL), (pr, gr, RIGHT_COL)):
                    if g.any():
                        rgba = np.zeros(g.shape + (4,)); rgba[g] = matplotlib.colors.to_rgba(c, 0.5); a.imshow(rgba, interpolation="nearest", aspect="auto")
                        a.contour(g.astype(float), levels=[0.5], colors=[c], linewidths=1.4)
                    if t.any(): a.contour(t.astype(float), levels=[0.5], colors=["white"], linewidths=1.0, linestyles="--")
                if not (pl.any() or pr.any()):
                    a.text(0.5, 0.06, "No mask predicted", transform=a.transAxes, ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold",
                           bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
        g = d[d.organ == name].sort_values("layer")
        for r, (col, lab) in enumerate(((A, "Left from right"), (B, "Right from left"))):
            yy = yc + 0.42 - r * 0.84
            ax.text(XC - 0.1, yy, lab, ha="right", va="center", fontsize=11, color=INK, zorder=4)
            for k, v in enumerate(g[col].values):
                x = XC + k * CW
                ax.add_patch(Rectangle((x + 0.04, yy - 0.3), CW - 0.08, 0.6, fc=cmap(norm(v)), ec=INK, lw=1.1, zorder=3))
                ax.text(x + CW / 2, yy, f"{v:.3f}", ha="center", va="center", fontsize=13, color=INK, fontweight="bold", zorder=4)
    # legend, one framed box as wide as the content, four columns
    LW = 1.55
    ax.add_patch(Rectangle((0.25, 0.15), W - 0.5, LW, fc="white", ec=INK, lw=1.4, zorder=1))
    xs = [W * 0.34, W * 0.50, W * 0.70]
    for xv in xs: ax.plot([xv, xv], [0.3, 0.15 + LW - 0.15], color=INK, lw=1.2)
    HY = 0.15 + LW - 0.3
    # column 1, colour scale of the KL divergence
    ax.text(0.6, HY, "KL divergence between the two prompts", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
    gx0, gx1, ly = 0.6, xs[0] - 0.4, 0.68
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(gx0, gx1, ly, ly + 0.22), cmap=cmap, vmin=0, vmax=1, aspect="auto", zorder=3)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.add_patch(Rectangle((gx0, ly), gx1 - gx0, 0.22, fc="none", ec=INK, lw=1.2, zorder=4))
    for v_ in (0, 0.02, 0.04, 0.06, 0.08, 0.1):
        xx = gx0 + (gx1 - gx0) * float(norm(v_)); ax.plot([xx] * 2, [ly - 0.05, ly], color=INK, lw=1.2, zorder=4)
        ax.text(xx, ly - 0.17, f"{v_:g}", ha="center", va="center", fontsize=9, color=INK)
    ax.text(gx0, 0.3, "0, same attention", ha="left", va="center", fontsize=9, color=INK)
    ax.text(gx1, 0.3, "larger, different attention", ha="right", va="center", fontsize=9, color=INK)
    # column 2, organs
    xm = xs[0] + 0.3
    ax.text(xm, HY, "Organs", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.add_patch(Rectangle((xm, 0.72), 0.4, 0.2, fc=LEFT_COL, alpha=0.6, ec=LEFT_COL, lw=1.2, zorder=3))
    ax.text(xm + 0.55, 0.82, "True left organ", ha="left", va="center", fontsize=9.5, color=INK)
    ax.add_patch(Rectangle((xm, 0.38), 0.4, 0.2, fc=RIGHT_COL, alpha=0.6, ec=RIGHT_COL, lw=1.2, zorder=3))
    ax.text(xm + 0.55, 0.48, "True right organ", ha="left", va="center", fontsize=9.5, color=INK)
    # column 3, predicted masks
    xo = xs[1] + 0.3
    ax.text(xo, HY, "Predicted masks", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(xo, 0.82, "Filled, from the left or the right prompt", ha="left", va="center", fontsize=9.5, color=INK)
    ax.plot([xo, xo + 0.45], [0.48, 0.48], color="white", lw=1.8, ls=(0, (2.2, 1.6)), path_effects=[pe.Stroke(linewidth=3.6, foreground=INK), pe.Normal()])
    ax.text(xo + 0.6, 0.48, "Dashed outline, true organ", ha="left", va="center", fontsize=9.5, color=INK)
    # column 4, what the two rows of cells mean
    xr = xs[2] + 0.3
    ax.text(xr, HY, "Direction of the comparison", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(xr, 0.98, "Left from right", ha="left", va="center", fontsize=9.5, color=INK, fontweight="semibold")
    ax.text(xr, 0.78, "counts mostly where the left prompt looks", ha="left", va="center", fontsize=9.2, color=INK)
    ax.text(xr, 0.50, "Right from left", ha="left", va="center", fontsize=9.5, color=INK, fontweight="semibold")
    ax.text(xr, 0.30, "counts mostly where the right prompt looks", ha="left", va="center", fontsize=9.2, color=INK)
    fig.savefig(FIG / fname, dpi=170, facecolor="white", bbox_inches="tight"); plt.close()

names = list(dict.fromkeys(d.organ))
names = [o for o in names if "left " + o in M.index and "right " + o in M.index]
order = d[d.layer == 1].set_index("organ")[A].reindex(names).sort_values(ascending=False).index.tolist()    # largest Layer 1 KL first
for k in range(0, len(order), 4):
    make(order[k:k + 4], f"RQ3_organ_pairs_group{k // 4 + 1}_subject{int(SID.split('_')[2])}.png")
print("done", SID, len(order))
