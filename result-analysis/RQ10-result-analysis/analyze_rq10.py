"""RQ10 result analysis: tables + word-box figure (F1) + predicted-mask examples (F2) from results/neurite-oasis/RQ10.
Three players (image, organ word, position word), 2**3 = 8 prompt variants per organ, exact Shapley values on the Dice score against the ground truth.
The image is 'absent' when the organ's own region is replaced by Gaussian noise. The fixed scaffolding words are always present and are not scored."""
import base64, zlib, tarfile, os
import numpy as np, pandas as pd, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe

ROOT = Path(r"D:\Reasoning\results\neurite-oasis\RQ10")
OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
TAR = Path(r"D:\Reasoning\data\raw\neurite-oasis.v1.0.tar")
SCRATCH = Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\6133727f-a378-46df-9aa4-f78b1882019b\scratchpad")

s = pd.read_csv(ROOT / "rq10_word_shapley_summary.csv")
SUBJECTS = sorted(s.subject_id.unique())[:50]                  # first 50 subjects by ID, as in the other RQs
s = s[s.subject_id.isin(SUBJECTS) & (s.metric == "f_dice_gt")]
c = pd.read_csv(ROOT / "rq10_word_shapley_coalitions.csv", usecols=range(12))
c = c[c.subject_id.isin(SUBJECTS)]
core = lambda o: o.split(" ", 1)[1]
pretty = lambda o: "Ventral DC" if o == "ventral DC" else o.capitalize()

# ---------- checks ----------
ind = s[s.kind == "individual"].pivot_table(index=["subject_id", "organ"], columns="role", values="shapley_value")
full = c[c.roles_present == "image+organ_word+position_word"].set_index(["subject_id", "organ"]).dice_gt
none = c[c.roles_present == "none"].set_index(["subject_id", "organ"]).dice_gt
assert np.allclose(ind.sum(axis=1), (full - none).reindex(ind.index), atol=1e-4)      # efficiency axiom
assert c.groupby(["subject_id", "organ"]).size().eq(8).all() and c.other_words.isna().all()
N = len(ind); print("subject-organ cases:", N)

# ---------- T1: Dice of the 8 prompt variants ----------
prompts = {"image+organ_word+position_word": ("Intact", "Segment the left hippocampus in this image."),
           "organ_word+position_word": ("Organ region noised", "Segment the left hippocampus in this image."),
           "image+organ_word": ("Intact", "Segment the hippocampus in this image."),
           "organ_word": ("Organ region noised", "Segment the hippocampus in this image."),
           "image+position_word": ("Intact", "Segment the left structure in this image."),
           "position_word": ("Organ region noised", "Segment the left structure in this image."),
           "image": ("Intact", "Segment the structure in this image."),
           "none": ("Organ region noised", "Segment the structure in this image.")}
g = c.groupby("roles_present").dice_gt
t1 = pd.DataFrame({"Image": [prompts[k][0] for k in prompts], "Example prompt": [prompts[k][1] for k in prompts],
                   "Dice, mean": [round(g.mean()[k], 3) for k in prompts], "Dice, SD": [round(g.std()[k], 3) for k in prompts],
                   "Dice, median": [round(g.median()[k], 3) for k in prompts],
                   "Cases with Dice of 0.5 or more (%)": [int(round((c[c.roles_present == k].dice_gt >= 0.5).mean() * 100)) for k in prompts]})
t1.to_csv(TAB / "T1_dice_of_prompt_variants.csv", index=False); print(t1.to_string(index=False))

# ---------- T2: Shapley credit per player ----------
tot = ind.sum(axis=1)
rows = []
for r, name in (("image", "Image"), ("organ_word", "Organ word"), ("position_word", "Position word")):
    v = ind[r]
    rows.append({"Player": name, "Mean": round(v.mean(), 3), "SD": round(v.std(), 3), "Median": round(v.median(), 3),
                 "Share of the total credit (%)": round(v.mean() / tot.mean() * 100, 1),
                 "Cases with positive credit (%)": int(round((v > 0.001).mean() * 100))})
rows.append({"Player": "Total (full prompt minus empty prompt)", "Mean": round(tot.mean(), 3), "SD": round(tot.std(), 3),
             "Median": round(tot.median(), 3), "Share of the total credit (%)": 100.0, "Cases with positive credit (%)": int(round((tot > 0.001).mean() * 100))})
t2 = pd.DataFrame(rows); t2.to_csv(TAB / "T2_shapley_credit_per_player.csv", index=False); print(t2.to_string(index=False))

# ---------- T3: per organ ----------
ind2 = ind.reset_index(); ind2["core"] = ind2.organ.map(core); ind2["total"] = ind2[["image", "organ_word", "position_word"]].sum(axis=1)
t3 = ind2.groupby("core")[["image", "organ_word", "position_word", "total"]].mean()
t3 = t3.sort_values("total", ascending=False)
t3.insert(0, "Organ", [pretty(o) for o in t3.index])
t3.columns = ["Organ", "Image", "Organ word", "Position word", "Total credit (Dice of the full prompt)"]
mrow = {"Organ": "Mean of 16 organs", **{k: round(ind2[v].mean(), 3) for k, v in zip(t3.columns[1:], ["image", "organ_word", "position_word", "total"])}}
t3 = pd.concat([t3.round(3), pd.DataFrame([mrow])], ignore_index=True)
t3.to_csv(TAB / "T3_shapley_credit_per_organ.csv", index=False); print(t3.to_string(index=False))

# ---------- T4: interactions ----------
it = s[s.kind == "pairwise_interaction"]
lab = {"image+organ_word": "Image and organ word", "image+position_word": "Image and position word", "organ_word+position_word": "Organ word and position word"}
t4 = pd.DataFrame([{"Pair of players": lab[k], "Mean": round(v.shapley_value.mean(), 3), "SD": round(v.shapley_value.std(), 3),
                    "Median": round(v.shapley_value.median(), 3)} for k, v in it.groupby("role")])
t4.to_csv(TAB / "T4_pairwise_interaction.csv", index=False); print(t4.to_string(index=False))

# ---------- shared figure style ----------
avail = {x.name for x in font_manager.fontManager.ttflist}
FONT = next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)
plt.rcParams.update({"font.family": FONT})
INK = "#000000"
GREEN, RED = "#2f9e63", "#d6453d"

# ---------- F1: word boxes ----------
cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cfe3f6", "#8fbfe8", "#4f97d6"])    # light enough for black text
norm = Normalize(0, 0.7)
perorg = s[s.kind == "individual"].groupby(["organ", "role"]).shapley_value.mean().unstack()
tot_org = perorg.sum(axis=1)
mean_vals = ind.mean()
EX = [("All 32 organs", "left", "organ", mean_vals, tot.mean()),
      ("left hippocampus", "left", "hippocampus", perorg.loc["left hippocampus"], tot_org["left hippocampus"]),
      ("right putamen", "right", "putamen", perorg.loc["right putamen"], tot_org["right putamen"]),
      ("left cerebral white matter", "left", "cerebral white matter", perorg.loc["left cerebral white matter"], tot_org["left cerebral white matter"]),
      ("left thalamus", "left", "thalamus", perorg.loc["left thalamus"], tot_org["left thalamus"]),
      ("left lateral ventricle", "left", "lateral ventricle", perorg.loc["left lateral ventricle"], tot_org["left lateral ventricle"])]

W, rowh = 14.0, 1.18
H = 1.0 + rowh * len(EX) + 1.15
fig = plt.figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
rend = fig.canvas.get_renderer()
def tw(txt, fs, wt="normal"):
    t = ax.text(0, 0, txt, fontsize=fs, fontweight=wt); w = t.get_window_extent(rend).width / fig.dpi; t.remove(); return w
FS, PAD, GAP = 14, 0.17, 0.12
def box(x, y, label, val=None, scored=True):
    w = tw(label, FS, "semibold" if scored else "normal") + 2 * PAD
    fc = "white" if (val is None) else cmap(norm(max(val, 0))) if val >= 0 else "#f3c4c0"
    ax.add_patch(FancyBboxPatch((x, y), w, 0.52, boxstyle="round,pad=0,rounding_size=0.09", fc=fc, ec=INK,
                                lw=1.5 if scored else 1.0, ls="-" if scored else (0, (3, 2))))
    ax.text(x + w / 2, y + 0.26, label, ha="center", va="center", fontsize=FS, color=INK, fontweight="semibold" if scored else "normal")
    if scored: ax.text(x + w / 2, y - 0.17, f"{val:+.3f}", ha="center", va="center", fontsize=11.5, color=INK)
    return w
ax.text(W / 2, H - 0.32, "Credit Given to the Image, the Organ Word and the Position Word of a Prompt", ha="center", va="center",
        fontsize=14, fontweight="bold", color=INK)
ytop = H - 1.0
for i, (name, side, organ, v, total) in enumerate(EX):
    y = ytop - (i + 1) * rowh + 0.28
    if i == 0:
        ax.text(0.3, y + 0.38, "All 32 organs", ha="left", va="center", fontsize=11.5, color=INK)
        ax.text(0.3, y + 0.14, "(mean of the credit)", ha="left", va="center", fontsize=11.5, color=INK)
    else:
        ax.text(0.3, y + 0.26, name, ha="left", va="center", fontsize=11.5, color=INK)
    x = 3.3
    x += box(x, y, "Image", v["image"]) + 0.3
    ax.plot([x - 0.15] * 2, [y - 0.25, y + 0.77], color=INK, lw=1.0)
    for word in ["Segment", "the", side, organ, "in", "this", "image."]:
        if word == side:    x += box(x, y, side if i else "side", v["position_word"]) + GAP
        elif word == organ: x += box(x, y, organ if i else "organ", v["organ_word"]) + GAP
        else:               x += box(x, y, word, scored=False) + GAP
    ax.text(W - 0.3, y + 0.26, f"{total:.3f}", ha="right", va="center", fontsize=14, color=INK, fontweight="bold")
ax.text(W - 0.3, ytop - 0.02, "Total credit", ha="right", va="center", fontsize=11.5, color=INK)
ax.text(3.3, ytop - 0.02, "Players", ha="left", va="center", fontsize=11.5, color=INK)
ly = 0.42
gx0, gx1 = 3.3, 6.8
ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(gx0, gx1, ly, ly + 0.2), cmap=cmap, aspect="auto", zorder=2)
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.add_patch(plt.Rectangle((gx0, ly), gx1 - gx0, 0.2, fc="none", ec=INK, lw=1.0, zorder=3))
for v_ in (0, 0.2, 0.4, 0.6):
    xx = gx0 + (gx1 - gx0) * v_ / 0.7; ax.plot([xx] * 2, [ly - 0.05, ly], color=INK, lw=1.0); ax.text(xx, ly - 0.17, f"{v_:.1f}", ha="center", va="center", fontsize=10.5, color=INK)
ax.text(gx0 - 0.15, ly + 0.1, "Shapley credit in Dice units", ha="right", va="center", fontsize=11, color=INK)
ax.add_patch(FancyBboxPatch((8.3, ly - 0.02), 0.55, 0.24, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=INK, lw=1.0, ls=(0, (3, 2))))
ax.text(9.0, ly + 0.1, "Fixed word, always present and not scored", ha="left", va="center", fontsize=11, color=INK)
fig.savefig(FIG / "F1_word_boxes.png", dpi=250, facecolor="white", bbox_inches="tight"); plt.close()

# ---------- F2: predicted masks of the prompt variants on one subject ----------
SUBJECT, ORGANS = "OASIS_OAS1_0004_MR1", ["left hippocampus", "left thalamus"]
nii = next((p for p in (SCRATCH / SUBJECT / "aligned_norm.nii.gz",
                        Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\cda2fa20-8216-49b7-9dc5-f417374d9481\scratchpad") / SUBJECT / "aligned_norm.nii.gz") if p.exists()), None)
if nii is None:
    nii = SCRATCH / SUBJECT / "aligned_norm.nii.gz"; nii.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TAR) as tf:
        for m in tf:
            if m.name == f"{SUBJECT}/aligned_norm.nii.gz":
                m.name = os.path.basename(m.name); tf.extract(m, nii.parent); break
arr = np.asarray(nib.load(nii).dataobj).astype(np.float32)
vol = np.flip(np.transpose(arr, (1, 2, 0)), axis=(0, 2))
def dec(enc, shape):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool); pos, val = 0, False
    for r in runs:
        if val: flat[pos:pos + r] = True
        pos += int(r); val = not val
    return flat.reshape(shape)
parts = [ch[(ch.subject_id == SUBJECT) & ch.organ.isin(ORGANS)] for ch in pd.read_csv(ROOT / "rq10_word_shapley_coalitions.csv", chunksize=2000)]
M = pd.concat(parts); SHAPE = tuple(int(x) for x in M.volume_shape.iloc[0].split("x")); assert SHAPE == vol.shape, (SHAPE, vol.shape)
COLS = [("image+organ_word+position_word", "Segment the left {o}\nin this image."), ("image+organ_word", "Segment the {o}\nin this image."),
        ("image+position_word", "Segment the left structure\nin this image."), ("image", "Segment the structure\nin this image.")]
noised = {"image+organ_word+position_word": "organ_word+position_word", "image+organ_word": "organ_word", "image+position_word": "position_word", "image": "none"}
PW, PH, X0, PITCH = 2.3, 1.78, 1.5, 2.6                       # panel size, left margin and column pitch in inches
GROUP = 0.4 + 0.5 + PH + 0.3 + 0.1 + PH + 0.3 + 0.35          # group heading, prompt titles, two panel rows with Dice text, gap
figw, figh = X0 + 4 * PITCH, 0.7 + GROUP * len(ORGANS) + 0.55
fig = plt.figure(figsize=(figw, figh), facecolor="white")
axes = np.empty((2 * len(ORGANS), 4), dtype=object)
for oi in range(len(ORGANS)):
    top = figh - 0.7 - oi * GROUP
    for ri in range(2):
        for ci in range(4):
            y1 = top - 0.4 - 0.5 - ri * (PH + 0.4)
            axes[2 * oi + ri, ci] = fig.add_axes([(X0 + ci * PITCH) / figw, (y1 - PH) / figh, PW / figw, PH / figh])
vmin, vmax = np.percentile(vol[vol > 0], [1, 99.5])
for oi, organ in enumerate(ORGANS):
    q = M[M.organ == organ].set_index("roles_present")
    gt = dec(q.gt_mask_rle.iloc[0], SHAPE)
    z = int(round(np.argwhere(gt)[:, 0].mean())); cen = np.argwhere(gt[z - 3:z + 4].any(axis=0)).mean(0)
    HA, HB = (48, 62)
    a0 = int(np.clip(round(cen[0]) - HA, 0, SHAPE[1] - 2 * HA)); b0 = int(np.clip(round(cen[1]) - HB, 0, SHAPE[2] - 2 * HB))
    view = lambda a: np.flipud(a[z, a0:a0 + 2 * HA, b0:b0 + 2 * HB])
    img = view(vol)
    for ri in range(2):
        for ci, (key, tmpl) in enumerate(COLS):
            k = key if ri == 0 else noised[key]
            ax_ = axes[2 * oi + ri, ci]; ax_.axis("off")
            ax_.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
            m = dec(q.loc[k].pred_mask_rle, SHAPE)
            ax_.imshow(np.ma.masked_where(~view(m), np.ones_like(img)), cmap=matplotlib.colors.ListedColormap([GREEN]), alpha=0.7, interpolation="nearest")
            ax_.contour(view(gt).astype(float), levels=[0.5], colors="white", linewidths=1.1)
            ax_.text(0.5, -0.03, f"Dice {q.loc[k].dice_gt:.2f}", transform=ax_.transAxes, ha="center", va="top", fontsize=11.5, color=INK, fontweight="semibold")
            if ri == 0:
                ax_.set_title(tmpl.format(o=core(organ)), fontsize=11.5, color=INK, pad=5, fontweight="normal")
        lab_ = ["Image\nintact", "Organ region\nreplaced\nby noise"][ri]
        axes[2 * oi + ri, 0].text(-0.06, 0.5, lab_, transform=axes[2 * oi + ri, 0].transAxes, ha="right", va="center", fontsize=11, color=INK)
    fig.text(0.005, (figh - 0.7 - oi * GROUP - 0.12) / figh, f"Example {oi + 1}, the {core(organ)}", ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.5, 1 - 0.12 / figh, "Predicted Masks When the Image, the Organ Word or the Position Word Is Removed from the Prompt", ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
halo = [pe.Stroke(linewidth=4.2, foreground=INK), pe.Normal()]
fig.legend(handles=[Patch(color=GREEN, alpha=0.85, label="Predicted mask"), Line2D([0], [0], color="white", lw=2, path_effects=halo, label="True organ")],
           loc="center", bbox_to_anchor=(0.5, 0.28 / figh), ncol=2, frameon=False, fontsize=10.5, labelcolor=INK, handlelength=2.4, columnspacing=2.4)
fig.savefig(FIG / "F2_masks_of_prompt_variants.png", dpi=230, bbox_inches="tight", facecolor="white"); plt.close()
print("done")
