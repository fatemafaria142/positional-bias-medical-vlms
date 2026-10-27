"""RQ6 result analysis: tables T1 to T5 and the summary figure F1 from results/neurite-oasis/RQ6.
Four players (image, organ word, position word, other words), 2**4 = 16 prompt variants per organ, exact Shapley values on the Dice score
against the true organ. First 50 subjects by ID, the same as the other RQs. The image is 'absent' when the organ's own region is replaced
by Gaussian noise."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SRC = Path(r"D:\Reasoning\results\neurite-oasis\RQ6")
OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
PL = ["image", "organ_word", "position_word", "other_words"]
NAMES = {"image": "Image", "organ_word": "Organ word", "position_word": "Position word", "other_words": "Other words"}
core = lambda o: o.split(" ", 1)[1]
pretty = lambda o: "Ventral DC" if o == "ventral DC" else o.capitalize()

s = pd.read_csv(SRC / "rq6_4player_word_shapley_summary.csv")
SUBJECTS = sorted(s.subject_id.unique())[:50]
s = s[s.subject_id.isin(SUBJECTS) & (s.metric == "f_dice_gt")]
c = pd.concat([ch[ch.subject_id.isin(SUBJECTS)] for ch in pd.read_csv(SRC / "rq6_4player_word_shapley_coalitions.csv",
               usecols=["subject_id", "organ", "roles_present", "dice_gt"], chunksize=200000)])
ind = s[s.kind == "individual"].pivot_table(index=["subject_id", "organ"], columns="role", values="shapley_value")[PL]
ind["total"] = ind.sum(axis=1)
full = c[c.roles_present == "image+organ_word+other_words+position_word"].set_index(["subject_id", "organ"]).dice_gt
none = c[c.roles_present == "none"].set_index(["subject_id", "organ"]).dice_gt
assert np.allclose(ind.total, (full - none).reindex(ind.index), atol=2e-4)              # efficiency axiom
assert len(ind) == 1600 and c.groupby(["subject_id", "organ"]).size().eq(16).all()

# ---------- T1: Dice of the 16 prompt variants ----------
def prompt(organ_word, position_word, other_words):
    words = (["left"] if position_word else []) + ["hippocampus" if organ_word else "structure"]
    if other_words: words = ["Segment", "the"] + words + ["in", "this", "image"]
    t = " ".join(words); return t[0].upper() + t[1:] + "."
rows = []
for ow in (True, False):
    for og in (True, False):
        for po in (True, False):
            for im in (True, False):
                key = "+".join(sorted([r for r, on in (("image", im), ("organ_word", og), ("position_word", po), ("other_words", ow)) if on])) or "none"
                x = c[c.roles_present == key].dice_gt
                rows.append({"Image": "Intact" if im else "Organ region replaced by noise", "Example prompt": prompt(og, po, ow),
                             "Dice, mean": round(x.mean(), 3), "Dice, SD": round(x.std(), 3), "Dice, median": round(x.median(), 3),
                             "Cases with Dice of 0.5 or more (%)": int(round((x >= 0.5).mean() * 100))})
pd.DataFrame(rows).to_csv(TAB / "T1_dice_of_prompt_variants.csv", index=False)

# ---------- T2: Shapley credit per player ----------
tot = ind.total
t2 = []
for r in PL + ["total"]:
    v = ind[r]
    t2.append({"Player": NAMES.get(r, "Total credit"), "Credit, mean": round(v.mean(), 3), "Credit, SD": round(v.std(), 3), "Credit, median": round(v.median(), 3),
               "Share of the total credit (%)": round(v.mean() / tot.mean() * 100, 1),
               "Cases with positive credit (%)": int(round((v > 0.001).mean() * 100)),
               "Cases with negative credit (%)": "" if r == "total" else int(round((v < -0.001).mean() * 100))})
pd.DataFrame(t2).to_csv(TAB / "T2_shapley_credit_per_player.csv", index=False)

# ---------- T3: per organ ----------
i2 = ind.reset_index(); i2["core"] = i2.organ.map(core)
t3 = i2.groupby("core")[PL + ["total"]].mean().sort_values("total", ascending=False)
t3.insert(0, "Organ", [pretty(o) for o in t3.index])
t3.columns = ["Organ", "Image", "Organ word", "Position word", "Other words", "Total credit"]
mean_row = {"Organ": "Mean of 16 organs", **{k: round(ind[v].mean(), 3) for k, v in zip(t3.columns[1:], PL + ["total"])}}
t3 = pd.concat([t3.round(3), pd.DataFrame([mean_row])], ignore_index=True)
t3.to_csv(TAB / "T3_shapley_credit_per_organ.csv", index=False)

# ---------- T4: pairwise interactions ----------
it = s[s.kind == "pairwise_interaction"]
lab = lambda k: " and ".join(NAMES[p].lower() if i else NAMES[p] for i, p in enumerate(sorted(k.split("+"), key=PL.index)))
t4 = pd.DataFrame([{"Pair of players": lab(k), "Interaction, mean": round(v.shapley_value.mean(), 3), "Interaction, SD": round(v.shapley_value.std(), 3),
                    "Interaction, median": round(v.shapley_value.median(), 3)} for k, v in it.groupby("role")])
t4["order"] = [sorted(k.split("+"), key=PL.index) for k in sorted(it.role.unique())]
t4 = t4.sort_values("order", key=lambda col: col.map(lambda p: [PL.index(x) for x in p])).drop(columns="order")
t4.to_csv(TAB / "T4_pairwise_interaction.csv", index=False)

# ---------- T5: spread across subjects ----------
sub = ind.groupby("subject_id")[PL + ["total"]].mean()
t5 = pd.DataFrame([{"Player": NAMES.get(r, "Total credit"), "Mean of the subject means": round(sub[r].mean(), 3), "SD over subjects": round(sub[r].std(), 3),
                    "Lowest subject": round(sub[r].min(), 3), "Highest subject": round(sub[r].max(), 3)} for r in PL + ["total"]])
t5.to_csv(TAB / "T5_credit_spread_over_subjects.csv", index=False)
for t in (t1 := pd.read_csv(TAB / "T1_dice_of_prompt_variants.csv"), pd.DataFrame(t2), t3, t4, t5): print(t.to_string(index=False)); print()

# ---------- F1: Shapley credit per organ and player, as a tile heatmap ----------
avail = {x.name for x in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": next(n for n in ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"] if n in avail)})
INK = "#000000"
cmap = LinearSegmentedColormap.from_list("rb", [(0.0, "#d6453d"), (0.25, "#f0968d"), (0.5, "#ffffff"), (0.62, "#a9cdee"), (0.8, "#69a9e0"), (1.0, "#3a82cc")])
norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.9)
body = t3.iloc[:-1]; mean = t3.iloc[-1]
data = body.iloc[:, 1:].astype(float).values
cols = ["Image", "Organ word", "Position word", "Other words", "Total credit"]
fig, ax = plt.subplots(figsize=(11.6, 9.0), facecolor="white")
fig.subplots_adjust(left=0.2, right=0.97, top=0.94, bottom=0.17)
ax.set_xlim(0, 5); ax.set_ylim(len(data) + 1.35, 0); ax.axis("off")
for j, name in enumerate(cols):
    ax.text(j + 0.5, 0.55, name, ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
for i in range(len(data)):
    ax.text(-0.08, i + 1.35 + 0.5, body.Organ.iloc[i], ha="right", va="center", fontsize=12, color=INK)
    for j in range(5):
        v = data[i, j]
        ax.add_patch(plt.Rectangle((j + 0.03, i + 1.35 + 0.04), 0.94, 0.92, fc=cmap(norm(v)), ec=INK, lw=1.0))
        ax.text(j + 0.5, i + 1.35 + 0.5, f"{(v if abs(v) >= 0.0005 else 0.0):+.3f}".replace("-", "\u2212"), ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold" if j == 4 else "normal")
y = len(data) + 1.35
ax.plot([-1.85, 5], [y + 0.0, y + 0.0], color=INK, lw=1.6)
ax.text(-0.08, y + 0.5, "Mean of 16 organs", ha="right", va="center", fontsize=12, fontweight="bold", color=INK)
for j, k in enumerate(cols):
    v = float(mean[k]); ax.add_patch(plt.Rectangle((j + 0.03, y + 0.04), 0.94, 0.92, fc=cmap(norm(v)), ec=INK, lw=1.6))
    ax.text(j + 0.5, y + 0.5, f"{v:+.3f}".replace("-", "\u2212"), ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
fig.text(0.5, 0.965, "Shapley Value of the Image, the Organ Word, the Position Word and the Other Words for Each Organ", ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)
lax = fig.add_axes([0.24, 0.085, 0.52, 0.025]); grid = np.linspace(-0.4, 0.9, 400); lax.imshow(cmap(norm(grid))[None, :, :], aspect="auto", extent=(-0.4, 0.9, 0, 1))
lax.set_yticks([]); [sp.set_edgecolor(INK) for sp in lax.spines.values()]
lax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]); lax.set_xticklabels(["\u22120.4", "\u22120.2", "0", "0.2", "0.4", "0.6", "0.8"], fontsize=11, color=INK)
lax.tick_params(colors=INK)
fig.text(0.5, 0.04, "Shapley value in Dice units. Blue raises the Dice score and red lowers it.", ha="center", va="center", fontsize=11.5, color=INK)
fig.savefig(FIG / "F1_shapley_value_per_organ.png", dpi=220, facecolor="white"); plt.close()
print("done")
