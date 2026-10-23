"""
Per-subject example figures for 20 subjects per experiment -> results/<exp>/experimental_plot/
plus a README.md per folder (image, then bullet points with that subject's numbers).

The 20 subjects are the same for every experiment: the 100 subjects are sorted by their Experiment I mean Dice
(canonical base / left / right prompts, 7 structures) and 20 are taken at evenly spaced ranks, from the hardest
to the best scan.

Run:  cd scripts && ../.venv/bin/python plot_axis2_experimental_plots.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import axis2_plot_style as S
import plot_axis2_exp1 as E1
import plot_axis2_exp2_3 as E23
import plot_axis2_exp4 as E4
import plot_axis2_exp5 as E5

N_SUBJECTS = 20
FOLDER = "experimental_plot"


def f2(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def select_subjects():
    m = pd.read_csv(E1.EXP_DIR / "metrics.csv", usecols=["subject_id", "template_style", "condition", "dice_vs_matched_gt"])
    d = (m[(m.template_style == "canonical_sentence") & m.condition.isin(["base", "left", "right"])]
         .groupby("subject_id").dice_vs_matched_gt.mean().sort_values())
    idx = np.unique(np.round(np.linspace(0, len(d) - 1, N_SUBJECTS)).astype(int))
    out = []
    for k, i in enumerate(idx, start=1):
        sid = d.index[i]
        out.append(dict(sid=sid, tag=S.subj_tag(sid), rank=i + 1, n=len(d), dice=d.iloc[i], order=k,
                        label=f"rank {i + 1}/{len(d)} by Exp I Dice"))
    return out


def header(title, intro_lines, subjects):
    lines = [f"# {title}", ""] + intro_lines + [
        "",
        "## How the 20 subjects were chosen",
        "",
        "All 100 subjects were sorted by their mean Dice in Experiment I (plain base / left / right sentences, 7 structures). "
        "20 subjects were taken at evenly spaced ranks, so the list runs from the **hardest scan (rank 1)** to the **best scan (rank 100)**. "
        "The same 20 subjects are used in every experiment's `experimental_plot/` folder, so a patient can be followed across experiments.",
        "",
        "| # | Subject | Rank (1 = hardest) | Exp I mean Dice |",
        "|---:|---|---:|---:|",
    ]
    lines += [f"| {s['order']} | OAS1_{s['tag']} | {s['rank']} | {s['dice']:.3f} |" for s in subjects]
    lines += ["", "**Colour key:** purple = left prompt / left GT, orange = right prompt / right GT, teal = no-side prompt, "
              "red = contradictory prompt, white outline = ground truth. Image left = patient left (L / R tags in each panel).", "", "---", ""]
    return lines


def write_md(out_dir, lines):
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {(out_dir / 'README.md').relative_to(S.REPO_ROOT)}")


# ============================================================================== Exp I
def exp1(subjects):
    out = S.RESULTS_DIR / "results_axis2_exp1" / FOLDER
    T = ["hippocampus", "thalamus", "caudate", "putamen"]
    m = pd.read_csv(E1.EXP_DIR / "metrics.csv", usecols=["subject_id", "target", "template_style", "condition",
                                                          "dice_vs_matched_gt", "n_pred_voxels", "frac_voxels_right_of_midline"])
    m = m[(m.template_style == "canonical_sentence") & m.target.isin(T)]
    lines = header("Experiment I — per-subject example masks (20 subjects)", [
        "Each figure shows one subject. Rows = hippocampus, thalamus, caudate, putamen (coronal slice through the structure). "
        "Columns = ground truth, then the prompts *'Segment the X in this image.'* (no side), *'... the left X ...'*, *'... the right X ...'*, "
        "*'left X located on the right hemisphere'* and *'right X located on the left hemisphere'*.",
        "",
        "Under each figure: the Dice of every prompt for that subject (vs. the GT the prompt asks for), the share of the no-side mask lying in the right hemisphere, and the automatic observations.",
    ], subjects)
    for s in subjects:
        fname = f"{s['order']:02d}_exp1_OAS1_{s['tag']}.png"
        E1.fig_examples(s["sid"], s["label"], out_dir=out, fname=fname)
        d = m[m.subject_id == s["sid"]]
        piv = d.pivot_table(index="target", columns="condition", values="dice_vs_matched_gt")
        fr = d[d.condition == "base"].set_index("target").frac_voxels_right_of_midline
        lines += [f"## {s['order']}. Subject OAS1_{s['tag']} — rank {s['rank']}/100 (Exp I mean Dice {s['dice']:.3f})", "",
                  f"![]({fname})", ""]
        for t in T:
            r = piv.loc[t]
            lines.append(f"- **{S.short(t)}:** no side {f2(r.get('base'))} ({fr.get(t, np.nan) * 100:.0f} % right of midline) · "
                         f"left {f2(r.get('left'))} · right {f2(r.get('right'))} · "
                         f"left-on-right {f2(r.get('contradictory_left_on_right'))} · right-on-left {f2(r.get('contradictory_right_on_left'))}")
        lat = piv[["left", "right"]]
        ok = (lat >= 0.5).all(axis=None)
        lines.append("- **Left/right prompts:** " + ("all 8 land on the correct structure (Dice ≥ 0.5)." if ok else
                     "weak (Dice < 0.5): " + ", ".join(f"{S.short(t)} {c} ({lat.loc[t, c]:.2f})" for t in T for c in ["left", "right"] if lat.loc[t, c] < 0.5) + "."))
        lean_r = [S.short(t) for t in T if fr.get(t, 0.5) > 0.6]
        lean_l = [S.short(t) for t in T if fr.get(t, 0.5) < 0.4]
        weak = [f"{S.short(t)} ({piv.loc[t, 'base']:.2f})" for t in T if piv.loc[t, "base"] < 0.5]
        txt = []
        if lean_r:
            txt.append("leans **right** for " + ", ".join(lean_r))
        if lean_l:
            txt.append("leans **left** for " + ", ".join(lean_l))
        if weak:
            txt.append("weak (Dice < 0.5) for " + ", ".join(weak))
        lines.append("- **No-side sentence:** " + ("; ".join(txt) + "." if txt else "balanced and accurate for all 4 structures."))
        lor = piv["contradictory_left_on_right"]
        rol = piv["contradictory_right_on_left"]
        lines.append(f"- **Contradictions:** 'left X on right hemisphere' mean Dice {lor.mean():.2f} "
                     f"({int((lor < 0.1).sum())}/4 structures fail < 0.1) vs 'right X on left hemisphere' {rol.mean():.2f} "
                     f"({int((rol < 0.1).sum())}/4 fail)" + (" → the usual asymmetry." if lor.mean() < rol.mean() - 0.1 else "."))
        lines += ["", "---", ""]
    write_md(out, lines)


# ============================================================================== Exp II
def exp2(subjects):
    out = S.RESULTS_DIR / "results_axis2_exp2_3" / FOLDER
    items = {"caudate": [("OASIS_P0057", "'left caudate nucleus'"), ("OASIS_P0060", "'l. caudate'"),
                         ("OASIS_P0062", "'Find the caudate that is on the left.'"), ("OASIS_P0065", "C-shaped description"),
                         ("OASIS_P0066", "contradictory")],
             "thalamus": [("OASIS_P0031", "'left dorsal thalamus'"), ("OASIS_P0033", "'left thalamic nucleus'"),
                          ("OASIS_P0036", "'Find the thalamus that is on the left.'"), ("OASIS_P0039", "'deep relay nucleus' description"),
                          ("OASIS_P0040", "contradictory")]}
    m = pd.read_csv(E23.EXP2 / "metrics_no_masks.csv", usecols=["subject_id", "prompt_id", "target", "dice_vs_gt", "ref_dice_vs_gt", "dice_vs_opposite_gt"])
    lines = header("Experiments II — per-subject example masks for re-worded prompts (20 subjects)", [
        "Each figure shows one subject and the same LEFT request worded six ways. Top row = left caudate, bottom row = left thalamus. "
        "Columns: the reference sentence *'Segment the left X in this image.'*, a clinical synonym, a second synonym / abbreviation, "
        "a paraphrase, an anatomical location description and the contradictory prompt. The Dice in each panel title is vs. the left GT.",
        "",
        "Experiment III (layer localization) produces no masks, so it has no per-subject figures.",
    ], subjects)
    for s in subjects:
        fname = f"{s['order']:02d}_exp2_OAS1_{s['tag']}.png"
        E23.exp2_examples(s["sid"], s["label"], out_dir=out, fname=fname)
        d = m[m.subject_id == s["sid"]].set_index("prompt_id")
        lines += [f"## {s['order']}. Subject OAS1_{s['tag']} — rank {s['rank']}/100 (Exp I mean Dice {s['dice']:.3f})", "",
                  f"![]({fname})", ""]
        for t, its in items.items():
            ref = d.loc[its[0][0], "ref_dice_vs_gt"]
            parts = [f"reference {f2(ref)}"] + [f"{lab} {f2(d.loc[p, 'dice_vs_gt'])}" for p, lab in its]
            lines.append(f"- **Left {t}:** " + " · ".join(parts))
            bad = [lab for p, lab in its if d.loc[p, "dice_vs_gt"] < 0.3]
            spill = [lab for p, lab in its if d.loc[p, "dice_vs_opposite_gt"] > 0.3]
            note = []
            if ref < 0.3:
                note.append(f"even the reference wording fails on this scan (Dice {ref:.2f}) — a left-side failure")
            if bad:
                note.append("wordings that break it (Dice < 0.3): " + ", ".join(bad))
            if spill:
                note.append("mask spills into the RIGHT " + t + " for: " + ", ".join(spill))
            lines.append(f"  - {S.short(t)}: " + ("; ".join(note) + "." if note else "all wordings work (Dice ≥ 0.3)."))
        lines += ["", "---", ""]
    write_md(out, lines)


# ============================================================================== Exp IV
def exp4(subjects):
    out = S.RESULTS_DIR / "results_axis2_exp4" / FOLDER
    items = {"hippocampus": [("OASIS_P0001", "no side"), ("OASIS_P0003", "left"), ("OASIS_P0015", "right"),
                             ("OASIS_P0013", "left location description"), ("OASIS_P0014", "left-on-right contradiction")],
             "caudate": [("OASIS_P0053", "no side"), ("OASIS_P0055", "left"), ("OASIS_P0067", "right"),
                         ("OASIS_P0065", "left location description"), ("OASIS_P0066", "left-on-right contradiction")]}
    m = pd.read_csv(E4.EXP_DIR / "metrics.csv", usecols=["subject_id", "prompt_id", "dice_mean_mask_vs_gt", "roi_epistemic_mi_bits", "inter_pass_mean_dice"])
    lines = header("Experiment IV — per-subject MC-dropout uncertainty maps (20 subjects)", [
        "Each figure shows one subject. Rows 1–2 = hippocampus, rows 3–4 = caudate. For each structure the first row is the "
        "**mean probability** over 5 dropout passes (yellow = confident foreground) and the second row the **epistemic variance** "
        "(bright = the passes disagree). Columns: no side, left, right, left location description, left-on-right contradiction.",
        "",
        "Under each figure: Dice of the averaged mask and the epistemic uncertainty (mutual information, bits) for every prompt. "
        "Typical values over all 100 subjects: MI ≈ 0.04–0.08 for normal prompts, 0.25–0.32 for descriptions / contradictions.",
    ], subjects)
    for s in subjects:
        fname = f"{s['order']:02d}_exp4_OAS1_{s['tag']}.png"
        E4.fig_maps(s["sid"], s["label"], out_dir=out, fname=fname)
        d = m[m.subject_id == s["sid"]].set_index("prompt_id")
        lines += [f"## {s['order']}. Subject OAS1_{s['tag']} — rank {s['rank']}/100 (Exp I mean Dice {s['dice']:.3f})", "",
                  f"![]({fname})", ""]
        for t, its in items.items():
            parts = [f"{lab} Dice {f2(d.loc[p, 'dice_mean_mask_vs_gt'])} / MI {f2(d.loc[p, 'roi_epistemic_mi_bits'])}" for p, lab in its]
            lines.append(f"- **{S.short(t)}:** " + " · ".join(parts))
            flagged = [lab for p, lab in its if d.loc[p, "dice_mean_mask_vs_gt"] < 0.5 and d.loc[p, "roi_epistemic_mi_bits"] >= 0.2]
            silent = [lab for p, lab in its if d.loc[p, "dice_mean_mask_vs_gt"] < 0.5 and d.loc[p, "roi_epistemic_mi_bits"] < 0.2]
            notes = []
            if flagged:
                notes.append("failures correctly **flagged** by high uncertainty: " + ", ".join(flagged))
            if silent:
                notes.append("**silent** failures (low uncertainty, bad mask): " + ", ".join(silent))
            lines.append(f"  - {S.short(t)}: " + ("; ".join(notes) + "." if notes else "no failures (all Dice ≥ 0.5)."))
        lines += ["", "---", ""]
    write_md(out, lines)


# ============================================================================== Exp V
def exp5(subjects):
    out = S.RESULTS_DIR / "results_axis2_exp5" / FOLDER
    m = pd.read_csv(E5.EXP_DIR / "metrics_no_masks.csv",
                    usecols=["subject_id", "ablation_setting", "target", "side", "variant_subtype", "n_pred_voxels", "dice_vs_matched_gt"])
    m = m[(m.variant_subtype == "canonical_sentence") & m.target.isin(["hippocampus", "caudate"])]
    lines = header("Experiment V — per-subject decoder-ablation masks (20 subjects)", [
        "Each figure shows one subject. Rows 1–3 = hippocampus, rows 4–6 = caudate, each with the no-side, left and right sentence. "
        "Columns = the 4 decoder settings: both ON (normal VoxTell), image decoder only (prompt decoder OFF), prompt decoder only "
        "(image decoder OFF) and both OFF. The number in each panel is the 3D voxel count of the mask.",
        "",
        "Under each figure: Dice (vs. the requested GT) and voxel count per setting, averaged over the no-side, left and right prompts.",
    ], subjects)
    for s in subjects:
        fname = f"{s['order']:02d}_exp5_OAS1_{s['tag']}.png"
        E5.fig_examples_combined(s["sid"], s["label"], out_dir=out, fname=fname)
        d = m[m.subject_id == s["sid"]]
        lines += [f"## {s['order']}. Subject OAS1_{s['tag']} — rank {s['rank']}/100 (Exp I mean Dice {s['dice']:.3f})", "",
                  f"![]({fname})", ""]
        for t in ["hippocampus", "caudate"]:
            dt = d[d.target == t]
            parts = []
            for st in E5.SETTINGS:
                x = dt[dt.ablation_setting == st]
                parts.append(f"{E5.SLABEL[st].split(chr(10))[0]} Dice {x.dice_vs_matched_gt.mean():.2f} ({int(x.n_pred_voxels.mean()):,} vox)")
            lines.append(f"- **{S.short(t)}:** " + " · ".join(parts))
            on = dt[dt.ablation_setting == "both_on"].set_index("side")
            io = dt[dt.ablation_setting == "image_only"]
            notes = []
            bad = [sd for sd in ["none", "left", "right"] if on.loc[sd, "dice_vs_matched_gt"] < 0.5]
            if bad:
                notes.append("even the full model is weak for: " + ", ".join(f"{sd} ({on.loc[sd, 'dice_vs_matched_gt']:.2f})" for sd in bad))
            ne = int((io.n_pred_voxels == 0).sum())
            notes.append(f"prompt decoder OFF → {ne}/3 masks empty")
            notes.append("image decoder OFF → diffuse masks "
                         f"{int(dt[dt.ablation_setting == 'text_only'].n_pred_voxels.mean() / max(on.n_pred_voxels.mean(), 1))}× the normal size")
            lines.append(f"  - {S.short(t)}: " + "; ".join(notes) + ".")
        lines += ["", "---", ""]
    write_md(out, lines)


def main():
    S.apply_style()
    subjects = select_subjects()
    print("subjects:", [s["tag"] for s in subjects])
    exp1(subjects)
    exp2(subjects)
    exp4(subjects)
    exp5(subjects)


if __name__ == "__main__":
    main()
