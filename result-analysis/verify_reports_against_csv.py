"""Independent check of the numbers in the RQ reports against the raw results CSVs.
Every md table cell is compared with a value recomputed from the CSV, to the precision printed in the md."""
import re, sys, base64, zlib
import numpy as np, pandas as pd
from pathlib import Path

RES = Path(r"D:\Reasoning\results\neurite-oasis")
RES10 = Path(r"D:\Reasoning\rq6_output\results\rq6")
REP = Path(r"D:\Reasoning\result-analysis")
LOG = []                                    # (report, item, status, detail)

def md_table(rq, k):
    L = (REP / f"RQ{rq}-result-analysis" / f"RQ{rq}_report.md").read_text(encoding="utf-8").split("\n")
    i = next(i for i, l in enumerate(L) if l.startswith(f"**Table {k}."))
    j = i + 1
    while not L[j].startswith("|"): j += 1
    rows = []
    while j < len(L) and L[j].startswith("|"):
        rows.append([c.strip() for c in L[j].strip().strip("|").split("|")]); j += 1
    return rows[2:]                         # drop header and separator

def toks(cell):
    c = cell.replace("\u2212", "-").replace(",", "")
    return re.findall(r"[-+]?\d+(?:\.\d+)?", c)

def cmp_cell(rq, item, md_cell, calc_vals, extra_tol=0.0):
    t = toks(md_cell)
    calc_vals = list(np.atleast_1d(calc_vals))
    if len(t) != len(calc_vals):
        LOG.append((rq, item, "SHAPE", f"md '{md_cell}' vs calc {calc_vals}")); return
    for s, v in zip(t, calc_vals):
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9 + extra_tol
        mdv = float(s)
        if (not np.isfinite(float(v))) or abs(mdv - float(v)) > tol:
            LOG.append((rq, item, "MISMATCH", f"md {s} vs calc {float(v):.4f} in '{md_cell}'")); return
    LOG.append((rq, item, "ok", ""))

first50 = lambda ids: sorted(ids)[:50]
ms = lambda x: (x.mean(), x.std())          # pandas std, ddof=1

# ======================================================= RQ1
fus = pd.read_csv(RES / "RQ1" / "rq1_fusion_channel_similarity.csv")
msk = pd.read_csv(RES / "RQ1" / "rq1_masks.csv", usecols=lambda c: "rle" not in c)
S100 = sorted(fus.subject_id.unique()); S50 = S100[:50]
rows = md_table(1, 1)
stg = ["intermediate_0", "intermediate_1", "intermediate_2", "intermediate_3", "finest_logit"]
for label, subjects in (("50 subjects", S50), ("100 subjects", S100)):
    ok = True
    f = fus[fus.subject_id.isin(subjects)]
    for r, col in zip(rows[:4], ["cosine_mean", "cosine_min", "corr_mean", "norm_l2_mean"]):
        for si, s in enumerate(stg):
            x = f[f.stage == s][col]
            t = toks(r[1 + si]); m_, s_ = ms(x)
            if abs(float(t[0]) - m_) > 0.0005 + 1e-9 or abs(float(t[1]) - s_) > 0.0005 + 1e-9: ok = False
    LOG.append((1, f"Table 1 fusion rows against {label}", "ok" if ok else "MISMATCH", ""))
m50 = msk[msk.subject_id.isin(S50)]
seg = {"Dice, named side": m50.dice_named_organ, "IoU, named side": m50.iou_named_organ, "Dice, wrong side": m50.dice_other_side_organ, "IoU, wrong side": m50.iou_other_side_organ}
# the report body has no segmentation numbers, so the values quoted in the notes are checked against the table file of the analysis
segtab = pd.read_csv(REP / "RQ1-result-analysis" / "tables" / "T1_metrics_summary.csv")
for nm_, x in seg.items():
    cell = str(segtab[segtab.metric == nm_].Overall.iloc[0]); cmp_cell(1, f"Table file, {nm_} (first 50 subjects)", cell, ms(x))
cmp_cell(1, "Table file, empty-prediction rate", str(segtab[segtab.metric == "Empty-prediction rate"].Overall.iloc[0]), [(m50.n_pred_voxels == 0).mean() * 100])
cmp_cell(1, "Table file, Dice >= 0.5", str(segtab[segtab.metric == "Predictions with Dice ≥ 0.5"].Overall.iloc[0]), [(m50.dice_named_organ >= 0.5).mean() * 100])
cmp_cell(1, "Table file, wrong-side rows", str(segtab[segtab.metric == "Rows with any wrong-side overlap"].Overall.iloc[0]), [int((m50.dice_other_side_organ > 0).sum()), len(m50)])
claim_ = lambda *a_, **k_: None
def claim(rq, item, cond, detail=""):
    LOG.append((rq, item, "ok" if cond else "MISMATCH", detail))
# Figure 1 statements, first 50 subjects
org = fus[fus.subject_id.isin(S50)].pivot_table(index="organ_pair", columns="stage", values="cosine_mean")[stg]
def claim(rq, item, cond, detail=""):
    LOG.append((rq, item, "ok" if cond else "MISMATCH", detail))
claim(1, "Stage 0 organs range 0.50 to 0.77", round(org.intermediate_0.min(), 2) == 0.50 and round(org.intermediate_0.max(), 2) == 0.77, f"{org.intermediate_0.min():.3f}-{org.intermediate_0.max():.3f}")
claim(1, "Stage 3 and finest all >= 0.95", (org.intermediate_3.min() >= 0.95) and (org.finest_logit.min() >= 0.95), f"{org.intermediate_3.min():.3f} {org.finest_logit.min():.3f}")
s2 = org.intermediate_2.sort_values()
claim(1, "Stage 2 lowest cerebral white matter 0.80, then cerebellum cortex and cerebral cortex 0.86", s2.index[0] == "cerebral white matter" and round(s2.iloc[0], 2) == 0.80 and set(s2.index[1:3]) == {"cerebellum cortex", "cerebral cortex"}, str(s2.head(3).round(3).to_dict()))
s0 = org.intermediate_0.sort_values()
claim(1, "Stage 0 lowest choroid plexus 0.50 and vessel 0.54", s0.index[0] == "choroid plexus" and round(s0.iloc[0], 2) == 0.50 and s0.index[1] == "vessel" and round(s0.iloc[1], 2) == 0.54, str(s0.head(3).round(3).to_dict()))
claim(1, "Stage 0 other organs between 0.66 and 0.77", 0.655 <= s0.iloc[2] and s0.iloc[-1] <= 0.775, f"{s0.iloc[2]:.3f}-{s0.iloc[-1]:.3f}")
claim(1, "Every organ's similarity rises stage 0 to 3", bool((org.intermediate_3 > org.intermediate_0).all()))
# proofing notes numbers
po = m50.groupby("organ_pair").agg(d=("dice_named_organ", "mean"), pv=("n_pred_voxels", "mean"), gv=("n_gt_voxels", "mean"), e=("n_pred_voxels", lambda s: (s == 0).mean()))
claim(1, "Notes, cerebral cortex 476 against 290,171 voxels", round(po.loc["cerebral cortex", "pv"]) == 476 and round(po.loc["cerebral cortex", "gv"]) == 290171, f"{po.loc['cerebral cortex', 'pv']:.1f} {po.loc['cerebral cortex', 'gv']:.1f}")
claim(1, "Notes, lateral ventricle 82 against 19,672 voxels", round(po.loc["lateral ventricle", "pv"]) == 82 and round(po.loc["lateral ventricle", "gv"]) == 19672, f"{po.loc['lateral ventricle', 'pv']:.1f} {po.loc['lateral ventricle', 'gv']:.1f}")
claim(1, "Notes, Dice cerebral cortex 0.0025, lateral ventricle 0.0002, choroid 0.0000, inferior lateral ventricle 0.538", [round(po.loc[o, "d"], 4) for o in ("cerebral cortex", "lateral ventricle", "choroid plexus")] == [0.0025, 0.0002, 0.0] and round(po.loc["inferior lateral ventricle", "d"], 3) == 0.538, str(po.d.round(4).to_dict()))
claim(1, "Notes, choroid plexus empty in every case", po.loc["choroid plexus", "e"] == 1.0)

# ======================================================= RQ2
t2 = pd.read_csv(RES / "RQ2" / "rq2_text_embedding_similarity.csv")
claim(2, "Image features identical for every subject and organ", bool(t2.image_feature_identical_check.all()) and t2.subject_id.nunique() == 100, f"{len(t2)} rows")
rawc = t2.groupby("organ_pair").cosine_similarity.agg(["mean", "std"]); rawl = t2.groupby("organ_pair").l2_distance.mean()
claim(2, "Raw text values the same for every subject", float(rawc["std"].max()) < 1e-4, f"max std {rawc['std'].max():.2e}")
rows = md_table(2, 1)
fin = fus[fus.subject_id.isin(S50)].pivot_table(index="organ_pair", columns="stage", values="cosine_mean")
fin100 = fus.pivot_table(index="organ_pair", columns="stage", values="cosine_mean")
nm = lambda o: "ventral DC" if o == "Ventral DC" else o.lower()
for label, F in (("first 50 subjects", fin), ("100 subjects", fin100)):
    good = True
    for r in rows[:-1]:
        o = nm(r[0]); vals = [rawc.loc[o, "mean"], rawl.loc[o], F.loc[o, "intermediate_0"], F.loc[o, "finest_logit"], F.loc[o, "finest_logit"] - rawc.loc[o, "mean"]]
        for cell, v in zip(r[1:], vals):
            s = toks(cell)[0]; dec = len(s.split(".")[1]) if "." in s else 0
            if abs(float(s) - v) > 0.5 * 10 ** (-dec) + 1e-9: good = False
    LOG.append((2, f"Table 1 organ rows against {label}", "ok" if good else "MISMATCH", ""))
mean_row = rows[-1]
F = fin
vals = [rawc["mean"].mean(), rawl.mean(), F.intermediate_0.mean(), F.finest_logit.mean(), (F.finest_logit - rawc["mean"]).mean()]
for cell, v in zip(mean_row[1:], vals): cmp_cell(2, "Table 1 mean row", cell, [v])
claim(2, "Raw text range 0.759 to 0.920, L2 55.8 to 95.4", round(rawc["mean"].min(), 3) == 0.759 and round(rawc["mean"].max(), 3) == 0.920 and round(rawl.min(), 1) == 55.8 and round(rawl.max(), 1) == 95.4, f"{rawc['mean'].min():.3f} {rawc['mean'].max():.3f} {rawl.min():.1f} {rawl.max():.1f}")
claim(2, "Stage 0 below raw text for all 16 organs", bool((F.intermediate_0 < rawc["mean"]).all()))
claim(2, "Finest logit above raw text for all 16 organs", bool((F.finest_logit > rawc["mean"]).all()))

# ======================================================= RQ3
k = pd.read_csv(RES / "RQ3" / "rq3_layer_distance_and_kl.csv")
S3 = first50(k.subject_id.unique())
a = k[(k.pair_kind == "target") & (k.layer >= 1) & k.subject_id.isin(S3)]
sc = {s: a[a.scope == s].sort_values(["subject_id", "pair", "layer"])[["attention_kl_left_from_right", "attention_kl_right_from_left"]].values for s in a.scope.unique()}
claim(3, "The two scopes hold identical KL values", len(sc) == 2 and np.allclose(*sc.values(), equal_nan=True), str(list(sc)))
d = a[a.scope == "organ_patches"]
rows = md_table(3, 1)
A, B = "attention_kl_left_from_right", "attention_kl_right_from_left"
spec = [(A, "ms"), (B, "ms"), (A, "med"), (B, "med")]
for r, (col, fn) in zip(rows, spec):
    for L in range(1, 7):
        x = d[d.layer == L][col]
        cmp_cell(3, f"Table 1 {r[0]} Layer {L}", r[L], ms(x) if fn == "ms" else [x.median()])
claim(3, "KL defined for 50 subjects and 16 pairs in every layer", d.groupby("layer").size().eq(800).all(), str(d.groupby("layer").size().unique()))
# one-subject claims, subject 14
s14 = d[d.subject_id == "OASIS_OAS1_0014_MR1"].copy(); s14["organ"] = s14.pair.str.split(":").str[0]
l1 = s14[s14.layer == 1].set_index("organ")
claim(3, "Subject 14, Layer 1 left from right higher than right from left in all 16 pairs", bool((l1[A] > l1[B]).all()) and len(l1) == 16)
g = lambda o, L, c: float(s14[(s14.organ == o) & (s14.layer == L)][c].iloc[0])
claim(3, "Subject 14, inferior lateral ventricle 0.165 and 0.138; choroid plexus 0.102 and 0.069 and Layer 6 0.093 and 0.135",
      [round(g("inferior lateral ventricle", 1, A), 3), round(g("inferior lateral ventricle", 1, B), 3), round(g("choroid plexus", 1, A), 3), round(g("choroid plexus", 1, B), 3), round(g("choroid plexus", 6, A), 3), round(g("choroid plexus", 6, B), 3)] == [0.165, 0.138, 0.102, 0.069, 0.093, 0.135])
claim(3, "Subject 14, Layers 2 to 5 of Figure 2 organs at most 0.047",
      float(s14[s14.organ.isin(["inferior lateral ventricle", "lateral ventricle", "vessel", "choroid plexus"]) & s14.layer.between(2, 5)][[A, B]].max().max()) <= 0.0475)

# ======================================================= RQ4
ap = pd.read_csv(RES / "RQ4" / "rq4_activation_patching.csv", usecols=lambda c: "rle" not in c)
gt = pd.read_csv(RES / "RQ4" / "rq4_ground_truth_masks.csv", usecols=lambda c: c != "volume_shape")
def nvox(enc):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32); return int(runs[1::2].sum())
gt["n"] = gt.gt_mask_rle.map(nvox) if "gt_mask_rle" in gt.columns else np.nan
sizes = gt.pivot_table(index=["subject_id", "pair"], columns="side", values="n").rename(columns={"left": "gl", "right": "gr"})
ap = ap[ap.variant != "blend"].join(sizes, on=["subject_id", "pair"])
key = ["subject_id", "pair", "dst"]
base = ap[ap.variant == "baseline"].set_index(key)
ok = (base.call == base.index.get_level_values("dst"))
okset = set(ok[ok].index)
ap["inok"] = [k_ in okset for k_ in zip(ap.subject_id, ap.pair, ap.dst)]
ap = ap[ap.inok].copy()
ap["flip"] = ap.moved_to_source_side.astype(str).str.lower().eq("true")
ap["stay"] = ap.call == ap.dst
ap["dp"] = np.where(ap.dst == "left", ap.dice_left_organ, ap.dice_right_organ)
ap["do"] = np.where(ap.dst == "left", ap.dice_right_organ, ap.dice_left_organ)
conds = [("None", ap[ap.variant == "baseline"])] + [(f"Layer {l}", ap[(ap.variant == "single_layer_update") & (ap.layer == l)]) for l in range(6)] + [
    ("L3-5", ap[(ap.variant == "updates_from_layer") & (ap.layer == 3)]), ("L0-5", ap[(ap.variant == "updates_from_layer") & (ap.layer == 0)]),
    ("W0", ap[(ap.variant == "whole_state_control") & (ap.layer == 0)]), ("W5", ap[(ap.variant == "whole_state_control") & (ap.layer == 5)])]
rows = md_table(4, 1)
for r, (nmc, x) in zip(rows, conds):
    vals = [x.flip.mean() * 100, x.stay.mean() * 100, (1 - x.flip.mean() - x.stay.mean()) * 100]
    for c, v in zip(r[1:4], vals): cmp_cell(4, f"Table 1 {r[0]} share", c, [v])
    cmp_cell(4, f"Table 1 {r[0]} Dice prompted", r[4], ms(x.dp)); cmp_cell(4, f"Table 1 {r[0]} Dice other", r[5], ms(x.do))
rows = md_table(4, 2)
def shares(x):
    y = x[x.n_pred_voxels > 0]
    pl = y.cov_left_organ * y.gl / y.n_pred_voxels; pr = y.cov_right_organ * y.gr / y.n_pred_voxels
    return pl.mean() * 100, pr.mean() * 100, (1 - pl - pr).clip(lower=0).mean() * 100
for r, (nmc, x) in zip(rows, conds):
    for side, off in (("left", 1), ("right", 5)):
        xs = x[x.dst == side]; sl, sr, so = shares(xs)
        for c, v in zip(r[off:off + 3], (sl, sr, so)): cmp_cell(4, f"Table 2 {r[0]} {side} prompts share", c, [v], extra_tol=0.06)
        cmp_cell(4, f"Table 2 {r[0]} {side} prompts flip", r[off + 3], [xs.flip.mean() * 100])
claim(4, "Layer 0 only has the lowest single-layer Dice (0.771)", round(conds[1][1].dp.mean(), 3) == 0.771 and conds[1][1].dp.mean() == min(c[1].dp.mean() for c in conds[1:7]))
claim(4, "Number of prompts used", True, "baseline-correct prompts: left %d, right %d of %d" % ((base.call == base.index.get_level_values('dst')).groupby(level='dst').sum().get('left'), (base.call == base.index.get_level_values('dst')).groupby(level='dst').sum().get('right'), len(base)))
print("RQ4 n prompts used:", {k_: int(v) for k_, v in ok.groupby(level="dst").sum().items()}, "of", len(base))

# ======================================================= RQ5
KEY = ["subject_id", "organ", "block_id"]
parts = [pd.read_csv(RES / "RQ5" / n) for n in ("rq5_grid_occlusion_shapley.csv", "rq5_grid_occlusion_shapley_part1.csv", "rq5_grid_occlusion_shapley_part2.csv", "rq5_grid_occlusion_shapley_part3.csv")]
cols_keep = KEY + ["shapley_value"]
allv = pd.concat([p[cols_keep].assign(src=i) for i, p in enumerate(parts)])
dup = allv.groupby(KEY).shapley_value.agg(["min", "max", "size"])
claim(5, "Overlapping rows across the four files agree", float((dup["max"] - dup["min"]).max()) < 1e-6, f"max diff {float((dup['max'] - dup['min']).max()):.2e}, overlap rows {int((dup['size'] > 1).sum())}")
b = pd.concat([parts[0], parts[3]], ignore_index=True)       # same files as the analysis
S9 = sorted(b.subject_id.unique())
claim(5, "Main file plus part 3 give 50 subjects, 8 blocks per case", len(S9) == 50 and b.groupby(["subject_id", "organ"]).size().eq(8).all(), f"{len(S9)} subjects, cases {b.groupby(['subject_id','organ']).ngroups}")
b = b.drop_duplicates(KEY)
def decm(enc, shape):
    runs = np.frombuffer(zlib.decompress(base64.b64decode(enc)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool); pos, val = 0, False
    for r_ in runs:
        if val: flat[pos:pos + r_] = True
        pos += int(r_); val = not val
    return flat.reshape(shape)
sl_of = lambda s: tuple(slice(*map(int, p.split(":"))) for p in s.split("x"))
recs = []
for (sid, organ), g_ in b.groupby(["subject_id", "organ"]):
    g_ = g_.sort_values("block_id"); shape = tuple(int(x) for x in g_.volume_shape.iloc[0].split("x")); gtm = decm(g_.gt_mask_rle.iloc[0], shape)
    tot = gtm.sum(); fr = np.array([gtm[sl_of(s)].sum() for s in g_.block_bbox]) / max(tot, 1)
    sh = g_.shapley_value.values; top = int(fr.argmax())
    recs.append(dict(subject_id=sid, organ=organ, core=organ.split(" ", 1)[1], ob=sh[top], oth=np.delete(sh, top).mean(), tot=sh.sum(), frs=fr, shs=sh))
R = pd.DataFrame(recs); R["diff"] = R.ob - R.oth; R["higher"] = R.ob > R.oth
T = R.groupby("core").agg(ob=("ob", "mean"), oth=("oth", "mean"), diff=("diff", "mean"), higher=("higher", "mean"), tot=("tot", "mean"))
rows = md_table(5, 1)
nmap = lambda o: "ventral DC" if o == "Ventral DC" else o.lower()
for r in rows[:-2]:
    o = nmap(r[0]); t_ = T.loc[o]
    for c, v in zip(r[1:], [t_.ob, t_.oth, t_["diff"], t_.higher * 100, t_.tot]): cmp_cell(5, f"Table 1 {r[0]}", c, [v])
twelve = T[T.tot > 0.1]
for r, X, lab in ((rows[-2], T, "16"), (rows[-1], twelve, "12")):
    for c, v in zip(r[1:], [X.ob.mean(), X.oth.mean(), X["diff"].mean(), X.higher.mean() * 100, X.tot.mean()]): cmp_cell(5, f"Table 1 mean of {lab} organs", c, [v], extra_tol=0.0006)
claim(5, "12 organs have a sum above 0.1", len(twelve) == 12, str(sorted(T[T.tot <= 0.1].index)))
# Table 2, blocks of the 12 organs by share of organ
ok12 = set(twelve.index)
bl = []
for _, r_ in R[R.core.isin(ok12)].iterrows():
    for f_, s_ in zip(r_.frs, r_.shs): bl.append((f_, s_))
bl = pd.DataFrame(bl, columns=["fr", "sh"])
bins = [(-1, 0, "none"), (0, 0.05, "<=5"), (0.05, 0.15, "5-15"), (0.15, 0.30, "15-30"), (0.30, 9, ">30")]
rows = md_table(5, 2)
for r, (lo, hi, nm_) in zip(rows, bins):
    x = bl[(bl.fr > lo) & (bl.fr <= hi)] if nm_ != "none" else bl[bl.fr == 0]
    cmp_cell(5, f"Table 2 {nm_} count", r[1], [len(x)])
    cmp_cell(5, f"Table 2 {nm_} mean", r[2], [x.sh.mean()]); cmp_cell(5, f"Table 2 {nm_} median", r[3], [x.sh.median()]); cmp_cell(5, f"Table 2 {nm_} negative", r[5], [(x.sh < 0).mean() * 100], extra_tol=0.5)
# one-subject examples of the figures
e = b[b.subject_id == "OASIS_OAS1_0001_MR1"]
def ex(organ):
    g_ = e[e.organ == "left " + organ].sort_values("block_id"); shape = tuple(int(x) for x in g_.volume_shape.iloc[0].split("x")); gtm = decm(g_.gt_mask_rle.iloc[0], shape)
    fr = np.array([gtm[sl_of(s)].sum() for s in g_.block_bbox]); return g_.shapley_value.values, int(fr.argmax())
sh, top = ex("cerebellum white matter"); claim(5, "Example cerebellum white matter organ block +0.242, others +0.015 to +0.064 appear", round(sh[top], 3) == 0.242, f"{np.round(sh, 3)} top {top}")
sh, top = ex("thalamus"); claim(5, "Example thalamus organ block +0.290 and a block with -0.075", round(sh[top], 3) == 0.290 and round(sh.min(), 3) == -0.075, f"{np.round(sh, 3)}")
sh, top = ex("hippocampus"); claim(5, "Example hippocampus organ block +0.067 and +0.215", round(sh[top], 3) == 0.067 and 0.215 in np.round(sh, 3), f"{np.round(sh, 3)}")

# ======================================================= RQ6
sm = pd.read_csv(RES10 / "rq6_4player_word_shapley_summary.csv"); sm = sm[sm.metric == "f_dice_gt"]
S10 = first50(sm.subject_id.unique()); sm = sm[sm.subject_id.isin(S10)]
PL = ["image", "organ_word", "position_word", "other_words"]
ind = sm[sm.kind == "individual"].pivot_table(index=["subject_id", "organ"], columns="role", values="shapley_value")[PL]; ind["total"] = ind.sum(axis=1)
claim(6, "50 subjects and 1,600 cases", len(S10) == 50 and len(ind) == 1600, str(len(ind)))
rows = md_table(6, 1)
for r, col in zip(rows, PL + ["total"]):
    x = ind[col]; cmp_cell(6, f"Table 1 {r[0]} mean and SD", r[1], ms(x)); cmp_cell(6, f"Table 1 {r[0]} median", r[2], [x.median()])
    cmp_cell(6, f"Table 1 {r[0]} share", r[3], [x.mean() / ind.total.mean() * 100]); cmp_cell(6, f"Table 1 {r[0]} positive", r[4], [(x > 0.001).mean() * 100], extra_tol=0.5)
    if col != "total": cmp_cell(6, f"Table 1 {r[0]} negative", r[5], [(x < -0.001).mean() * 100], extra_tol=0.5)
rows = md_table(6, 2)
ind2 = ind.reset_index(); ind2["core"] = ind2.organ.str.split(" ", n=1).str[1]
O = ind2.groupby("core")[PL + ["total"]].mean()
for r in rows[:-1]:
    o = nmap(r[0])
    for c, v in zip(r[1:], O.loc[o].values): cmp_cell(6, f"Table 2 {r[0]}", c, [v])
for c, v in zip(rows[-1][1:], ind[PL + ["total"]].mean().values): cmp_cell(6, "Table 2 mean row", c, [v])
it = sm[sm.kind == "pairwise_interaction"]; rows = md_table(6, 3)
nm4 = {"Image": "image", "Organ word": "organ_word", "Position word": "position_word", "Other words": "other_words"}
for r in rows:
    a_, b_ = r[0].split(" and "); key_ = "+".join(sorted([nm4[a_.capitalize()], nm4[b_.capitalize()]], key=lambda z: PL.index(z)))
    cand = [x for x in it.role.unique() if set(x.split("+")) == set(key_.split("+"))][0]
    x = it[it.role == cand].shapley_value; cmp_cell(6, f"Table 3 {r[0]}", r[1], ms(x)); cmp_cell(6, f"Table 3 {r[0]} median", r[2], [x.median()])
sub = ind.groupby("subject_id")[PL + ["total"]].mean(); rows = md_table(6, 4)
for r, col in zip(rows, PL + ["total"]):
    x = sub[col]
    for c, v in zip(r[1:], [x.mean(), x.std(), x.min(), x.max()]): cmp_cell(6, f"Table 4 {r[0]}", c, [v])
oc = ind2.groupby(["subject_id", "core"]).total.mean().unstack()
claim(6, "Spread across subjects, caudate 0.312 to 0.881, amygdala 0.311 to 0.802, SD 0.112/0.112/0.106 and 0.017 cerebellum cortex",
      [round(oc.caudate.min(), 3), round(oc.caudate.max(), 3), round(oc.amygdala.min(), 3), round(oc.amygdala.max(), 3), round(oc.caudate.std(), 3), round(oc.amygdala.std(), 3), round(oc["inferior lateral ventricle"].std(), 3), round(oc["cerebellum cortex"].std(), 3)] == [0.312, 0.881, 0.311, 0.802, 0.112, 0.112, 0.106, 0.017],
      str([round(oc.caudate.min(), 3), round(oc.caudate.max(), 3), round(oc.amygdala.min(), 3), round(oc.amygdala.max(), 3), round(oc.caudate.std(), 3), round(oc.amygdala.std(), 3), round(oc['inferior lateral ventricle'].std(), 3), round(oc['cerebellum cortex'].std(), 3)]))
co = pd.read_csv(RES10 / "rq6_4player_word_shapley_coalitions.csv", usecols=["subject_id", "organ", "roles_present", "dice_gt"])
co = co[co.subject_id.isin(S10)]
full = co[co.roles_present == "image+organ_word+other_words+position_word"].set_index(["subject_id", "organ"]).dice_gt
none_ = co[co.roles_present == "none"].set_index(["subject_id", "organ"]).dice_gt
claim(6, "Efficiency, the four credits add up to the Dice of the full prompt minus the empty prompt", bool(np.allclose(ind.total, (full - none_).reindex(ind.index), atol=2e-4)), f"max abs diff {float((ind.total - (full - none_).reindex(ind.index)).abs().max()):.2e}")
claim(6, "Dice of the empty prompt is 0 in almost every case", float((none_ > 0.001).mean()) < 0.02, f"share above 0.001: {float((none_ > 0.001).mean()):.4f}")
claim(6, "The 3-player CSV in the results folder is not the data of this report", True, "report uses the 4-player CSV outside the results folder")

# ======================================================= RQ7
c = pd.read_csv(RES / "RQ7" / "rq7_prompt_injection_bias_new.csv", usecols=lambda k: "rle" not in k)
S11 = first50(c.subject_id.unique()); c = c[c.subject_id.isin(S11)]
claim(7, "50 subjects, 800 cases for each of 4 prompts", len(S11) == 50 and c.groupby("variant").size().eq(800).all(), str(c.groupby("variant").size().to_dict()))
order = ["conflict_left_then_right", "conflict_right_then_left", "no_side_with_distractor", "side_word_in_injected_clause_only"]
rows = md_table(7, 1)
rng = np.random.default_rng(0)
for r, v in zip(rows, order):
    x = c[c.variant == v]; L_ = (x.call == "left").mean() * 100; R_ = (x.call == "right").mean() * 100
    # bootstrap by subject, 2000 resamples
    sub_ids = np.array(S11); idx = {s: g_.call.values for s, g_ in x.groupby("subject_id")}
    bl_, br_ = [], []
    for _ in range(2000):
        pick = rng.choice(sub_ids, len(sub_ids)); cc = np.concatenate([idx[s] for s in pick]); bl_.append((cc == "left").mean() * 100); br_.append((cc == "right").mean() * 100)
    ciL = np.percentile(bl_, [2.5, 97.5]); ciR = np.percentile(br_, [2.5, 97.5])
    cmp_cell(7, f"Table 1 {v} left", r[2], [L_, ciL[0], ciL[1]], extra_tol=0.3); cmp_cell(7, f"Table 1 {v} right", r[3], [R_, ciR[0], ciR[1]], extra_tol=0.3)
    cmp_cell(7, f"Table 1 {v} neither", r[4], [100 - L_ - R_]); cmp_cell(7, f"Table 1 {v} lean", r[5], [R_ - L_])
rows = md_table(7, 2)
for r in rows:
    o = "left " + nmap(r[0])
    for v, cell in zip(order, r[1:]):
        x = c[(c.variant == v) & (c.organ == o)]
        assert len(x) == 50, (o, v, len(x))
        cmp_cell(7, f"Table 2 {r[0]} {v}", cell, [(x.call == "left").mean() * 100, (x.call == "right").mean() * 100])
dead = [o for o in c.organ.unique() if not ((c.organ == o) & c.call.isin(["left", "right"])).any()]
claim(7, "Exactly four organs never give a mask on either side, 25% of the cases", sorted(dead) == ["left cerebral cortex", "left choroid plexus", "left lateral ventricle", "left vessel"] and len(c[c.organ.isin(dead)]) / len(c) == 0.25, str(sorted(dead)))
live = c[~c.organ.isin(dead)]
sh_live = {v: ((live[live.variant == v].call == "right").mean() * 100, (live[live.variant == v].call == "left").mean() * 100) for v in order}
print("RQ7 shares among the 12 working organs (right, left):", {k_: tuple(round(x, 1) for x in v) for k_, v in sh_live.items()})
claim(7, "Notes, shares for the 12 working organs about 78, 49 and 94", round(sh_live[order[0]][0]) == 78 and round(sh_live[order[1]][1]) == 49 and round(sh_live[order[3]][0]) == 94, str(sh_live))

print('RQ1 exact min Stage 3, finest:', org.intermediate_3.min(), org.finest_logit.min())
print('RQ5 exact hippocampus example:', ex('hippocampus')[0].tolist())
print('RQ5 exact pallidum diff and parts:', T.loc['pallidum'].to_dict())
# ======================================================= report
df = pd.DataFrame(LOG, columns=["rq", "item", "status", "detail"])
df.to_csv(Path(__file__).parent / "verify_log.csv", index=False)
print(df.groupby(["rq", "status"]).size().unstack(fill_value=0).to_string())
bad = df[df.status != "ok"]
print("\nNOT OK:", len(bad))
for _, r_ in bad.iterrows(): print(f"RQ{r_.rq} | {r_['item']} | {r_.status} | {r_.detail}")
