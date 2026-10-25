"""RQ2 result analysis: one table + one figure from results/neurite-oasis/RQ2 (and RQ1 fusion similarity)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager

ROOT = Path(r"D:\Reasoning\results\neurite-oasis")
OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

d = pd.read_csv(ROOT / "RQ2" / "rq2_text_embedding_similarity.csv")
f = pd.read_csv(ROOT / "RQ1" / "rq1_fusion_channel_similarity.csv")
# first 50 subjects (sorted by ID), the same convention as the other RQs
SUBJECTS = sorted(d.subject_id.unique())[:50]
d = d[d.subject_id.isin(SUBJECTS)]
f = f[f.subject_id.isin(SUBJECTS)]

# ---------- table: per organ ----------
t = d.groupby("organ_pair")[["cosine_similarity", "l2_distance"]].first()      # identical for every subject
c = f.pivot_table(index="organ_pair", columns="stage", values="cosine_mean")
t["stage0"] = c["intermediate_0"]
t["finest"] = c["finest_logit"]
t["gap"] = t["finest"] - t["cosine_similarity"]
t = t.sort_values("cosine_similarity")
img_ok = int(d.groupby("subject_id").image_feature_identical_check.all().sum())

pretty = lambda o: "Ventral DC" if o == "ventral DC" else o.capitalize()
tab = pd.DataFrame({
    "Organ": [pretty(o) for o in t.index],
    "Raw text cosine similarity": t.cosine_similarity.round(3), "Raw text L2 distance": t.l2_distance.round(1),
    "Decoder Stage 0 cosine similarity": t.stage0.round(3), "Decoder finest logit cosine similarity": t.finest.round(3),
    "Change (finest logit minus raw text)": t.gap.round(3)})
mean_row = {"Organ": "Mean of 16 organs", "Raw text cosine similarity": round(t.cosine_similarity.mean(), 3),
            "Raw text L2 distance": round(t.l2_distance.mean(), 1), "Decoder Stage 0 cosine similarity": round(t.stage0.mean(), 3),
            "Decoder finest logit cosine similarity": round(t.finest.mean(), 3), "Change (finest logit minus raw text)": round(t.gap.mean(), 3)}
tab = pd.concat([tab, pd.DataFrame([mean_row])], ignore_index=True)
tab.to_csv(TAB / "T1_text_vs_fusion_similarity.csv", index=False)
print(tab.to_string(index=False)); print("image check passed:", img_ok, "/", len(SUBJECTS))

# ---------- figure: vertical two-circle comparison per organ ----------
avail = {x.name for x in font_manager.fontManager.ttflist}
FONT = next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)
plt.rcParams.update({"font.family": FONT})
INK, MUTED, LINE = "#000000", "#000000", "#000000"   # black everywhere, no gray text or lines
BLUE, ORANGE = "#2f9e63", "#d6453d"   # green for the raw text embedding and red for the finest logit, as in the RQ4 figures

def lum(h):
    rgb = [int(h[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

n = len(t)
fig, ax = plt.subplots(figsize=(11.2, 7.6), facecolor="white")
fig.subplots_adjust(top=0.92, bottom=0.29, left=0.08, right=0.985)
ax.set_xlim(-0.7, n - 0.3); ax.set_ylim(0.72, 1.02)
for sp in ("top", "right", "bottom"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(LINE)
ax.set_xticks(range(n), [pretty(o) for o in t.index], rotation=40, ha="right", rotation_mode="anchor",
              fontsize=10, color=INK)
ax.tick_params(axis="x", length=0, pad=6)
ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
ax.set_ylabel("Cosine similarity between the left and right prompts", fontsize=10.5, color=INK, labelpad=8)
ax.set_xlabel("Organ, sorted by the raw text similarity from lowest to highest", fontsize=10.5, color=INK, labelpad=6)
ax.tick_params(axis="y", colors=MUTED, labelsize=9, length=3, color=LINE)

DOT = 780
for i, (o, r) in enumerate(t.iterrows()):
    ax.plot([i, i], [r.cosine_similarity, r.finest], color=LINE, lw=1.4, zorder=1, solid_capstyle="butt")
    for val, col in ((r.cosine_similarity, BLUE), (r.finest, ORANGE)):
        ax.scatter(i, val, s=DOT, color=col, zorder=3)
        ax.text(i, val, f"{val:.3f}", ha="center", va="center", fontsize=7.4, zorder=4,
                color="white" if lum(col) < 0.28 else INK)

fig.text(0.5, 0.965, "Left vs. Right Prompt Similarity of Raw Text and Final Decoder Output",
         ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
fig.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=BLUE, ms=11, label="Raw text embedding (same for every subject)"),
                    plt.Line2D([0], [0], marker="o", ls="", color=ORANGE, ms=11, label="Finest logit of the U-Net decoder (mean over subjects)")],
           loc="center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False, fontsize=10.5, columnspacing=3.0, handletextpad=0.4)

fig.savefig(FIG / "F1_text_vs_fusion_similarity.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
