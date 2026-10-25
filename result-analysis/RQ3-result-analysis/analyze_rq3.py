"""RQ3 result analysis (KL divergence): one table + one figure from results/neurite-oasis/RQ3."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager

SRC = Path(r"D:\Reasoning\results\neurite-oasis\RQ3\rq3_layer_distance_and_kl.csv")
OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

d = pd.read_csv(SRC)
# attention KL exists only for the left-vs-right pair, layers 1-6; the two scopes hold identical values
d = d[(d.scope == "organ_patches") & (d.pair_kind == "target") & (d.layer >= 1)].copy()
A, B = "attention_kl_left_from_right", "attention_kl_right_from_left"
LAYERS = list(range(1, 7))

# ---------- table ----------
rows = []
for nm, col, fn in [("Left from right, mean \u00b1 SD", A, "ms"), ("Right from left, mean \u00b1 SD", B, "ms"),
                    ("Left from right, median", A, "med"), ("Right from left, median", B, "med")]:
    r = {"Attention KL divergence": nm}
    for l in LAYERS:
        x = d.loc[d.layer == l, col]
        r[f"Layer {l}"] = f"{x.mean():.3f} \u00b1 {x.std():.3f}" if fn == "ms" else f"{x.median():.3f}"
    rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(TAB / "T1_attention_kl_by_layer.csv", index=False, encoding="utf-8-sig")
print(tab.to_string(index=False)); print("n per layer:", d.groupby("layer").size().unique(), "subjects:", d.subject_id.nunique())

# ---------- figure ----------
avail = {x.name for x in font_manager.fontManager.ttflist}
FONT = next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)
plt.rcParams.update({"font.family": FONT})
INK, MUTED, LINE = "#000000", "#000000", "#000000"   # black everywhere, no gray text or lines
NAVY, LIGHT = "#2f9e63", "#d6453d"        # green for left from right and red for right from left, as in the RQ4 figures
GREEN_T, RED_T = "#1f7a49", "#b3342d"       # darker shades of the same colours for the text

ma, mb = d.groupby("layer")[A].mean(), d.groupby("layer")[B].mean()
fig, ax = plt.subplots(figsize=(8.6, 6.0), facecolor="white")
fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.97)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(LINE)
ax.set_xlim(0.15, 6.3); ax.set_ylim(0.0, 0.1)
ax.set_xticks(LAYERS, [f"Layer {l}" for l in LAYERS])
ax.tick_params(axis="x", length=0, labelsize=10, colors=INK, pad=6)
ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=3, color=LINE)
ax.set_ylabel("Mean attention KL divergence between the left and right prompts", fontsize=10, color=INK, labelpad=10)

ax.plot(LAYERS, mb.values, color=LIGHT, lw=3.4, marker="o", ms=7.5, mec="white", mew=1.2, zorder=3)
ax.plot(LAYERS, ma.values, color=NAVY, lw=3.4, marker="o", ms=7.5, mec="white", mew=1.2, zorder=4)
# each value is written next to its own point, the higher series above the point and the lower one below it
for x in LAYERS:
    g_up = ma[x] >= mb[x]
    for val, col, up in ((ma[x], GREEN_T, g_up), (mb[x], RED_T, not g_up)):
        if x == 1:            # steep lines leave this point to the right, so the labels go to its left
            off, ha, va = (-14, 0), "right", "center"
        elif x == 2:          # a steep line arrives from the upper left, so the labels go to the right of the point
            off, ha, va = (14, 13 if up else -13), "left", "bottom" if up else "top"
        else:
            off, ha, va = (0, 13 if up else -13), "center", "bottom" if up else "top"
        ax.annotate(f"{val:.3f}", (x, val), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=10, fontweight="bold", color=col, zorder=6)

fig.text(0.54, 0.965, "Attention Difference Between Left and Right Prompts Across the Decoder Layers",
         ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=INK)
fig.legend(handles=[plt.Line2D([0], [0], color=NAVY, lw=3.4, marker="o", ms=7, label="Left from right (mean over subjects and organs)"),
                    plt.Line2D([0], [0], color=LIGHT, lw=3.4, marker="o", ms=7, label="Right from left (mean over subjects and organs)")],
           loc="center", bbox_to_anchor=(0.54, 0.03), ncol=2, frameon=False, fontsize=10, columnspacing=2.4, handlelength=2.4)

fig.savefig(FIG / "F1_attention_kl_by_layer.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
