"""RQ6 four-player word attribution figures with the organ image, the ground truth and the predicted mask.
Each row is one organ of one subject. The three panels show the MRI slice through the organ, the true organ (ground truth) and the mask
predicted for the full prompt. The prompt is written next to them with the exact Shapley credit of each part in Dice units
(rq6_4player_word_shapley_summary.csv, metric f_dice_gt). Masks come from rq6_4player_word_shapley_coalitions.csv, full prompt row."""
import sys, base64, zlib
import numpy as np, pandas as pd, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SID = f"OASIS_OAS1_{int(sys.argv[1]) if len(sys.argv) > 1 else 14:04d}_MR1"
ROOT = Path(r"D:\Reasoning\results\neurite-oasis\RQ6")
NII = Path(r"D:\Reasoning\data\raw\neurite-oasis-extracted") / SID / "aligned_norm.nii.gz"
CACHE = Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\d543e881-113f-4f97-ad0b-09f0b81dbbfd\scratchpad") / f"{SID}_coalitions.pkl"
FIG = Path(__file__).parent / "figures"; FIG.mkdir(exist_ok=True)
FULL = "image+organ_word+other_words+position_word"
PL = ["image", "organ_word", "position_word", "other_words"]

s = pd.read_csv(ROOT / "rq6_4player_word_shapley_summary.csv")
SUBJECTS = sorted(s.subject_id.unique())[:50]
assert SID in SUBJECTS, SID
s = s[(s.metric == "f_dice_gt") & (s.kind == "individual")]
ind = s.pivot_table(index=["subject_id", "organ"], columns="role", values="shapley_value")[PL]
mean_order = ind.loc[SUBJECTS].groupby("organ").sum(axis=1).sum(axis=1) if False else ind.loc[SUBJECTS].groupby("organ").mean().sum(axis=1)

if CACHE.exists():
    C = pd.read_pickle(CACHE)
else:
    parts = [ch[(ch.subject_id == SID) & (ch.roles_present == FULL)] for ch in pd.read_csv(ROOT / "rq6_4player_word_shapley_coalitions.csv", chunksize=3000)]
    C = pd.concat(parts); CACHE.parent.mkdir(parents=True, exist_ok=True); C.to_pickle(CACHE)
C = C.set_index("organ")

arr = np.asarray(nib.load(NII).dataobj).astype(np.float32)
vol = np.flip(np.transpose(arr, (1, 2, 0)), axis=(0, 2))
SHAPE = tuple(int(x) for x in C.volume_shape.iloc[0].split("x")); assert SHAPE == vol.shape, (SHAPE, vol.shape)
def dec(enc):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(SHAPE)), dtype=bool); pos, val = 0, False
    for r in runs:
        if val: flat[pos:pos + r] = True
        pos += int(r); val = not val
    return flat.reshape(SHAPE)
vmin, vmax = np.percentile(vol[vol > 0], [1, 99.5])

avail = {x.name for x in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)})
INK = "#000000"
GT_COL, PR_COL = "#2f9e63", "#f28c28"
cmap = LinearSegmentedColormap.from_list("rb", [(0.0, "#d6453d"), (0.25, "#f0968d"), (0.5, "#ffffff"), (0.62, "#a9cdee"), (0.8, "#69a9e0"), (1.0, "#3a82cc")])
norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.5)

def panels(organ):
    gt, pr = dec(C.loc[organ].gt_mask_rle), dec(C.loc[organ].pred_mask_rle)
    z = int(round(np.argwhere(gt)[:, 0].mean()))
    pts = np.argwhere(gt[z])                                        # voxels of the true organ on this slice
    lo, hi = pts.min(0), pts.max(0); cen = (lo + hi) / 2
    H = int(np.clip(max(hi - lo) / 2 + 14, 24, min(SHAPE[1], SHAPE[2]) // 2))   # zoom on the organ, square window
    a0 = int(np.clip(round(cen[0]) - H, 0, SHAPE[1] - 2 * H)); b0 = int(np.clip(round(cen[1]) - H, 0, SHAPE[2] - 2 * H))
    view = lambda a: np.flipud(a[z, a0:a0 + 2 * H, b0:b0 + 2 * H])
    return view(vol), view(gt), view(pr)

def overlay(a, img, m, col, outline=None):
    a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
    if m is not None and m.any():
        rgba = np.zeros(m.shape + (4,)); rgba[m] = matplotlib.colors.to_rgba(col, 0.55); a.imshow(rgba, interpolation="nearest")
        a.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=1.4)
    if outline is not None and outline.any():
        a.contour(outline.astype(float), levels=[0.5], colors=[GT_COL], linewidths=1.4, linestyles="--")
    a.axis("off")

def make(organs, side, fname):
    n = len(organs)
    W, rowh = 22.0, 2.25
    H = 0.85 + 0.6 + rowh * n + 1.55
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    rend = fig.canvas.get_renderer()
    def tw(txt, fs):
        t = ax.text(0, 0, txt, fontsize=fs); w = t.get_window_extent(rend).width / fig.dpi; t.remove(); return w
    FS, PAD, GAP = 15, 0.16, 0.06
    def chip(x, y, label, val, number=True):
        w = tw(label, FS) + 2 * PAD
        ax.add_patch(FancyBboxPatch((x, y), w, 0.56, boxstyle="square,pad=0", fc=cmap(norm(val)), ec=INK, lw=1.2, zorder=3))
        ax.text(x + w / 2, y + 0.28, label, ha="center", va="center", fontsize=FS, color=INK, zorder=4)
        if number: ax.text(x + w / 2, y - 0.2, f"{(val if abs(val) >= 0.005 else 0.0):+.2f}", ha="center", va="center", fontsize=14, color=INK, fontweight="bold", zorder=4)
        return w
    ax.text(W / 2, H - 0.3, "Which Specific Word Contributes More, Organ Word, Position Word, Other Words or Image?", ha="center", va="center",
            fontsize=20, fontweight="bold", color=INK)
    top = H - 0.85
    IMG, XI = 1.6, [3.0, 4.85, 6.7]
    for xi, t in zip(XI, ["Organ image", "Ground truth", "Predicted mask"]):
        ax.text(xi + IMG / 2, top, t, ha="center", va="center", fontsize=12.5, color=INK, fontweight="semibold")
    ax.text(0.4, top, "Organ", ha="left", va="center", fontsize=12.5, color=INK, fontweight="semibold")
    ax.text(W - 3.3, top, "Final Dice score", ha="center", va="center", fontsize=12.5, color=INK, fontweight="semibold")
    ax.text(W - 1.3, top, "Outside the true\norgan or missed", ha="center", va="center", fontsize=12.5, color=INK, fontweight="semibold")
    top -= 0.35
    for i, (organ, v) in enumerate(organs):
        nm = organ.split(" ", 1)[1]
        y0 = top - (i + 1) * rowh                       # bottom of the row card
        ax.add_patch(FancyBboxPatch((0.25, y0 + 0.05), W - 0.5, rowh - 0.15, boxstyle="square,pad=0", fc="#ffffff", ec=INK, lw=1.4, zorder=1))
        yc = y0 + rowh / 2
        ax.text(0.45, yc, "Ventral DC" if nm == "ventral DC" else nm.capitalize(), ha="left", va="center", fontsize=14, color=INK, fontweight="semibold", zorder=4)
        img, gt, pr = panels(organ)
        for xi, kind in zip(XI, ("img", "gt", "pr")):
            a = fig.add_axes([xi / W, (yc - IMG / 2) / H, IMG / W, IMG / H])
            if kind == "img": overlay(a, img, None, None)
            elif kind == "gt": overlay(a, img, gt, GT_COL)
            else: overlay(a, img, pr, PR_COL, outline=gt)
        y = yc - 0.15
        x = 8.9
        x += chip(x, y, "Image", v["image"]) + 0.3
        ax.plot([x - 0.15] * 2, [y - 0.05, y + 0.62], color=INK, lw=1.2, zorder=4)
        x += 0.1; x0 = x
        for word, kind in [("Segment", "o"), ("the", "o"), (side, "p"), (nm, "w"), ("in", "o"), ("this", "o"), ("image.", "o")]:
            val = {"o": v["other_words"], "p": v["position_word"], "w": v["organ_word"]}[kind]
            x += chip(x, y, word, val, number=kind != "o") + GAP
        x -= GAP
        yb = y + 0.74
        ax.plot([x0, x0, x, x], [yb - 0.07, yb, yb, yb - 0.07], color=INK, lw=1.2, zorder=4)
        ax.text((x0 + x) / 2, yb + 0.2, f"Other words {(v['other_words'] if abs(v['other_words']) >= 0.005 else 0.0):+.2f}", ha="center", va="center", fontsize=14, color=INK, fontweight="bold", zorder=4)
        ax.text(W - 3.3, yc, f"{max(v[PL].sum(), 0) + 0.0:.2f}", ha="center", va="center", fontsize=20, color=INK, fontweight="bold", zorder=4)
        ax.text(W - 1.3, yc, f"{min(1 - v[PL].sum(), 1) + 0.0:.2f}", ha="center", va="center", fontsize=20, color=INK, fontweight="bold", zorder=4)
    # legend, one framed box with three groups
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.add_patch(plt.Rectangle((0.25, 0.1), W - 0.5, 1.35, fc="white", ec=INK, lw=1.4, zorder=1))
    ax.text(1.0, 1.2, "Shapley value of a part of the prompt", ha="left", va="center", fontsize=12, color=INK, fontweight="semibold")
    for xv in (9.6, 15.2):
        ax.plot([xv, xv], [0.2, 1.35], color=INK, lw=1.2)
    # group 1, credit color scale
    gx0, gx1, ly = 1.0, 8.6, 0.72
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(gx0, gx1, ly, ly + 0.26), cmap=cmap, vmin=0, vmax=1, aspect="auto", zorder=3)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.add_patch(plt.Rectangle((gx0, ly), gx1 - gx0, 0.26, fc="none", ec=INK, lw=1.2, zorder=4))
    for v_ in (-0.4, -0.2, 0, 0.2, 0.4):
        xx = gx0 + (gx1 - gx0) * float(norm(v_)); ax.plot([xx] * 2, [ly - 0.06, ly], color=INK, lw=1.2, zorder=4)
        ax.text(xx, ly - 0.2, f"{v_:g}", ha="center", va="center", fontsize=11, color=INK)
    ax.text(gx0, 0.27, "Red lowers the Dice score", ha="left", va="center", fontsize=11.5, color=INK)
    ax.text(gx1, 0.27, "Blue raises the Dice score", ha="right", va="center", fontsize=11.5, color=INK)
    # group 2, masks
    ax.text(10.0, 1.2, "Masks", ha="left", va="center", fontsize=12, color=INK, fontweight="semibold")
    ax.add_patch(plt.Rectangle((10.0, 0.72), 0.45, 0.22, fc=GT_COL, alpha=0.55, ec=GT_COL, lw=1.2, zorder=3))
    ax.text(10.65, 0.83, "True organ (ground truth)", ha="left", va="center", fontsize=11.5, color=INK)
    ax.add_patch(plt.Rectangle((10.0, 0.35), 0.45, 0.22, fc=PR_COL, alpha=0.55, ec=PR_COL, lw=1.2, zorder=3))
    ax.text(10.65, 0.46, "Predicted mask", ha="left", va="center", fontsize=11.5, color=INK)
    # group 3, outline
    ax.text(15.6, 1.2, "Outline", ha="left", va="center", fontsize=12, color=INK, fontweight="semibold")
    ax.plot([15.6, 16.15], [0.83, 0.83], color=GT_COL, lw=1.8, ls="--")
    ax.text(16.35, 0.83, "True organ", ha="left", va="center", fontsize=11.5, color=INK)
    ax.text(15.6, 0.46, "drawn on the predicted mask", ha="left", va="center", fontsize=11.5, color=INK)
    fig.savefig(FIG / fname, dpi=170, facecolor="white", bbox_inches="tight"); plt.close()

rows_all = ind.loc[SID]
for side in ("left", "right"):
    order = [o for o in mean_order.sort_values(ascending=False).index if o.split(" ")[0] == side]
    rows = [(o, rows_all.loc[o]) for o in order]
    for k, g in enumerate([rows[0:4], rows[4:8], rows[8:12], rows[12:16]]):
        make(g, side, f"F{1 if side == 'left' else 2}{'abcd'[k]}_{side}_organs_group{k + 1}_subject{int(SID.split('_')[2])}.png")
print("done", SID)
