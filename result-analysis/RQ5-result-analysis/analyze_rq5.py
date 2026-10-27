"""RQ5 result analysis: tables from results/neurite-oasis/RQ5 (the figures are made by make_rq5_images.py).
A padded box around each organ is cut into 2 x 2 x 2 = 8 blocks. Absent blocks are replaced with Gaussian noise, and each block gets a Shapley credit
for the Dice score of the organ. The saved credits are Monte Carlo estimates (2 sampled subsets per block), not the exact 256-coalition values.
The main file and part 3 together give the first 50 subjects by ID, as in the other research questions."""
import base64, zlib
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(r"D:\Reasoning\results\neurite-oasis\RQ5")
OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
TAR = Path(r"D:\Reasoning\data\raw\neurite-oasis.v1.0.tar")
SCRATCH = Path(r"C:\Users\FATEMA~1\AppData\Local\Temp\claude\D--Reasoning\6133727f-a378-46df-9aa4-f78b1882019b\scratchpad")

main = pd.read_csv(ROOT / "rq5_grid_occlusion_shapley.csv")
KEY = ["subject_id", "organ", "block_id"]
for part in ("part1", "part2"):                                         # these parts repeat rows that are already in the main file
    pp = pd.read_csv(ROOT / f"rq5_grid_occlusion_shapley_{part}.csv", usecols=KEY + ["shapley_value"])
    chk = main.merge(pp, on=KEY, suffixes=("", "_p"))
    assert len(chk) == len(pp) and np.allclose(chk.shapley_value, chk.shapley_value_p)
p3 = pd.read_csv(ROOT / "rq5_grid_occlusion_shapley_part3.csv")        # part 3 adds the 15 subjects that are not in the main file
assert not set(p3.subject_id) & set(main.subject_id)
b = pd.concat([main, p3], ignore_index=True)
SUBJECTS = sorted(b.subject_id.unique()); NS = len(SUBJECTS)
assert b.groupby(["subject_id", "organ"]).size().eq(8).all()
pretty = lambda o: "Ventral DC" if o == "ventral DC" else o.capitalize()
core = lambda o: o.split(" ", 1)[1]

def dec(enc, shape):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool); pos, val = 0, False
    for r in runs:
        if val: flat[pos:pos + r] = True
        pos += int(r); val = not val
    return flat.reshape(shape)
sl_of = lambda s: tuple(slice(*map(int, p.split(":"))) for p in s.split("x"))

# ---------- share of the organ inside every block ----------
rows = []
for (sid, org), g in b.groupby(["subject_id", "organ"]):
    shape = tuple(int(x) for x in g.volume_shape.iloc[0].split("x")); gt = dec(g.gt_mask_rle.iloc[0], shape); n = gt.sum()
    for _, r in g.iterrows():
        rows.append((sid, org, r.block_id, r.shapley_value, gt[sl_of(r.block_bbox)].sum() / max(n, 1)))
d = pd.DataFrame(rows, columns=["sid", "organ", "block", "sh", "frac"]); d["core"] = d.organ.map(core)
assert np.allclose(d.groupby(["sid", "organ"]).frac.sum(), 1)
print("subjects", NS, "cases", d.groupby(["sid", "organ"]).ngroups, "block rows", len(d))

# ---------- per case: block with the most organ voxels vs the other seven ----------
def case(g):
    i = g.frac.idxmax()
    return pd.Series({"top": g.sh[i], "others": g.sh.drop(i).mean(), "total": g.sh.sum(), "maxabs_share": g.sh.abs().max() / max(g.sh.abs().sum(), 1e-12),
                      "neg": (g.sh < 0).mean()})
cs = d.groupby(["sid", "organ", "core"]).apply(case, include_groups=False).reset_index()
cs["higher"] = cs.top > cs.others

# ---------- T1: per organ ----------
def rho(g):
    return spearmanr(g.sh, g.frac)[0] if g.sh.nunique() > 1 else np.nan
t1 = cs.groupby("core").agg(total=("total", "mean"), top=("top", "mean"), others=("others", "mean"), higher=("higher", "mean"))
t1["rho"] = d.groupby("core").apply(rho, include_groups=False)
t1 = t1.sort_values("total", ascending=False)
tab1 = pd.DataFrame({"Organ": [pretty(o) for o in t1.index], "Total credit of the 8 blocks": t1.total.round(3).values,
                     "Credit of the block with most of the organ": t1.top.round(3).values, "Mean credit of the other 7 blocks": t1.others.round(3).values,
                     "Cases where the first is higher (%)": (t1.higher * 100).round(0).astype(int).values,
                     "Spearman correlation of block credit with the organ share of the block": t1.rho.round(2).values})
mean = {"Organ": "Mean of 16 organs", "Total credit of the 8 blocks": round(cs.total.mean(), 3), "Credit of the block with most of the organ": round(cs.top.mean(), 3),
        "Mean credit of the other 7 blocks": round(cs.others.mean(), 3), "Cases where the first is higher (%)": int(round(cs.higher.mean() * 100)),
        "Spearman correlation of block credit with the organ share of the block": round(spearmanr(d.sh, d.frac)[0], 2)}
kept = t1.index[t1.total > 0.1]                                        # organs for which the blocks receive any credit at all
ac = cs[cs.core.isin(kept)]
mean12 = {"Organ": "Mean of the 12 organs with a Shapley value", "Total credit of the 8 blocks": round(ac.total.mean(), 3), "Credit of the block with most of the organ": round(ac.top.mean(), 3),
          "Mean credit of the other 7 blocks": round(ac.others.mean(), 3), "Cases where the first is higher (%)": int(round(ac.higher.mean() * 100)), "Spearman correlation of block credit with the organ share of the block": np.nan}
tab1 = pd.concat([tab1, pd.DataFrame([mean, mean12])], ignore_index=True).drop(columns="Spearman correlation of block credit with the organ share of the block")
tab1.to_csv(TAB / "T1_credit_per_organ.csv", index=False); print(tab1.to_string(index=False))

# ---------- T2: credit by organ share of the block, with a 95% interval from resampling subjects ----------
rng = np.random.default_rng(0)
BINS = [(-1e-9, 1e-9, "No voxel of the organ"), (1e-9, 0.05, "Up to 5%"), (0.05, 0.15, "5% to 15%"), (0.15, 0.30, "15% to 30%"), (0.30, 1.01, "More than 30%")]
kept = t1.index[t1.total > 0.1]                                        # organs for which the blocks receive any credit at all
act = d[d.core.isin(kept)]
print("organs kept:", sorted(act.core.unique()))
subj_idx = {s: i for i, s in enumerate(SUBJECTS)}
def boot_mean(df, B=2000):
    g = df.groupby("sid").sh.agg(["sum", "count"]).reindex(SUBJECTS).fillna(0)
    s_, c_ = g["sum"].values, g["count"].values
    idx = rng.integers(0, NS, (B, NS)); tot = c_[idx].sum(1)
    m = np.where(tot > 0, s_[idx].sum(1) / np.maximum(tot, 1), np.nan)
    return np.nanpercentile(m, [2.5, 97.5])
rows2 = []
for lo, hi, lab in BINS:
    q = act[(act.frac > lo) & (act.frac <= hi)]
    ci = boot_mean(q)
    rows2.append({"Share of the organ inside the block": lab, "Blocks": len(q), "Credit, mean": round(q.sh.mean(), 3), "Credit, median": round(q.sh.median(), 3),
                  "95% interval of the mean": f"{ci[0]:.3f} to {ci[1]:.3f}", "Blocks with negative credit (%)": int(round((q.sh < 0).mean() * 100))})
t2 = pd.DataFrame(rows2); t2.to_csv(TAB / "T2_credit_by_organ_share.csv", index=False); print(t2.to_string(index=False))
rho_act = spearmanr(act.sh, act.frac)[0]
bs = []
for _ in range(1000):
    pick = rng.integers(0, NS, NS); q = pd.concat([act[act.sid == SUBJECTS[i]] for i in pick]); bs.append(spearmanr(q.sh, q.frac)[0])
print("rho (all)", round(spearmanr(d.sh, d.frac)[0], 3), "rho (organs with credit)", round(rho_act, 3), np.percentile(bs, [2.5, 97.5]).round(3))
actcs = cs[cs.core.isin(act.core.unique())]
print("cases with credit:", len(actcs), "top>others (%):", round(actcs.higher.mean() * 100, 1), "top mean", round(actcs.top.mean(), 3), "others", round(actcs.others.mean(), 3),
      "largest block share of abs credit", round(actcs.maxabs_share.mean(), 3), "negative blocks (%)", round(actcs.neg.mean() * 100, 1), "total>1:", (actcs.total > 1).mean().round(3))
# (the Spearman correlation and the other summary measures are not used in the report)

# the tables use the name Shapley value for the credit of a block
r1 = pd.read_csv(TAB / "T1_credit_per_organ.csv"); r2 = pd.read_csv(TAB / "T2_credit_by_organ_share.csv")
r1.columns = ["Organ", "Sum of the 8 Shapley values", "Shapley value of the organ block", "Mean Shapley value of the 7 surrounding blocks", "Cases where the organ block is higher (%)"]
r1.insert(4, "Difference, organ block minus surrounding blocks", (r1["Shapley value of the organ block"] - r1["Mean Shapley value of the 7 surrounding blocks"]).round(3))
r1 = r1[["Organ", "Shapley value of the organ block", "Mean Shapley value of the 7 surrounding blocks", "Difference, organ block minus surrounding blocks", "Cases where the organ block is higher (%)", "Sum of the 8 Shapley values"]]
r2.columns = ["Share of the organ inside the block", "Number of blocks", "Mean Shapley value", "Median Shapley value", "95% interval of the mean", "Blocks with a negative Shapley value (%)"]
r1.to_csv(TAB / "T1_shapley_value_per_organ.csv", index=False); r2.to_csv(TAB / "T2_shapley_value_by_organ_share.csv", index=False)
(TAB / "T1_credit_per_organ.csv").unlink(); (TAB / "T2_credit_by_organ_share.csv").unlink()
print("done")
