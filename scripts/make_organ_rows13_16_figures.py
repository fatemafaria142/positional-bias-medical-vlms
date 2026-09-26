"""
Rows 13-16 for AMOS, one figure per organ AND per subject (nothing is averaged or shown together), in the same style as
the word-highlight figure `all_prompts.png`: the exact prompt as coloured word chips (position word = magenta, organ name =
blue), the organ and its position named in the header, then one row per condition with its result.

    local_plots_groundedness/word_highlight/AMOS/<organ>/rows13_16/subject_<N>/
        row13_organ_replaced_by_noise.png   the prompt is unchanged, the image is changed: does the mask survive?
        row14_blend_left_right.png          the "left" and "right" text embeddings are blended: does the mask follow the side?
        row14_patching_layers.png           decoder layers get the other prompt's update: does the mask move?
        row15_cross_attention.png           where the prompt looks inside the model, layer by layer
        row16_left_right_distance.png       how different the left and the right prompt are inside the model, layer by layer

<N> is the subject number in the row01-row11 figures (1..10 = the ten AMOS subjects in sorted order); rows 13-16 were run
for 1 = amos_0508, 3 = amos_0530 and 8 = amos_0555.

Input: AMOS-ROW(13 to 16)/*.csv (from voxtell-amos-mechanistic-rows13-16-kaggle.ipynb).
Usage: python scripts/make_organ_rows13_16_figures.py
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_prompt_breakdown as pb  # noqa: E402  (fonts, palette, chip helpers)
from make_plan_question_figures import DATASETS, load  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "AMOS-ROW(13 to 16)"
ROOT = REPO / "local_plots_groundedness" / "word_highlight" / "AMOS"
CFG = DATASETS["AMOS"]
RUN = ["amos_0508"]                                    # subject 1; add "amos_0530", "amos_0555" to draw those too
POS_C, ORG_C = pb.POS_C, pb.ORG_C
INK, INK2, MUTED, SURFACE, GREY, GREY_TEXT, FS = pb.INK, pb.INK2, pb.MUTED, pb.SURFACE, pb.GREY, pb.GREY_TEXT, pb.FS
TRACK = pb.mix(ORG_C, SURFACE, 0.22)
DARK, LIGHT = "#2f3037", "#b9b8b2"


def prompt_chips(organ, extra=None, position_text=None):
    """The exact prompt as chips; `position_text` replaces the position word (for blended embeddings)."""
    chips = pb.chips_for(organ, "none", "")
    if position_text:
        chips = [(position_text, k) if k == "position" else (t, k) for t, k in chips]
    if extra:
        chips.append((extra, "replacement"))
    return chips


# ------------------------------------------------------------------------------------------------
def draw(spec):
    """spec: title, question, subline, header=[{label, chips}], rows=[{label, chips|None, bars=[{value, vmax, color, text}]}],
    legend=[(kind, text)], bar_w, out, ref=(value, vmax, text)|None, numbered."""
    header, rows, bar_w = spec.get("header", []), spec["rows"], spec["bar_w"]
    pad, gap = 0.16, 0.09
    scratch = plt.figure(figsize=(20, 5), dpi=200)
    sax = scratch.add_subplot(111)

    def width(text, bold):
        t = sax.text(0, 0, text, fontsize=FS, fontweight="bold" if bold else "normal")
        scratch.canvas.draw()
        w = t.get_window_extent(scratch.canvas.get_renderer()).width / scratch.dpi
        t.remove()
        return w

    def chip_ws(chips):
        return [width(t, k != "fixed") + 2 * pad for t, k in chips]

    for r in header + rows:
        r["ws"] = chip_ws(r["chips"]) if r.get("chips") else []
    label_w = max(width(r["label"], True) for r in header + rows) + 0.75
    plt.close(scratch)

    chips_w = max([sum(r["ws"]) + gap * (len(r["ws"]) - 1) for r in header + rows if r["ws"]] or [0])
    chip_rows = any(r.get("chips") for r in rows)
    left, right, text_w = 0.35, 0.35, 3.0
    chips_x0 = left + label_w
    bar_x0 = chips_x0 + chips_w + 0.55 if chip_rows else chips_x0
    fig_w = bar_x0 + bar_w + text_w + right if chip_rows else chips_x0 + max(chips_w, bar_w) + text_w + right
    n_bars = max(len(r["bars"]) for r in rows)
    pitch = 0.66 if n_bars == 1 else 0.3 + 0.17 * n_bars
    chip_h = 0.4
    how = []                                   # the 'how to read' paragraph under the question is no longer shown
    note_lines = [textwrap.wrap(n, max(40, int((fig_w - 0.9) / 0.082))) for n in spec.get('notes', [])]
    n_note = sum(len(x) for x in note_lines)
    fig_h = 2.05 + 0.3 * len(how) + 0.66 * len(header) + (0.35 if header else 0) + pitch * len(rows) + 1.25 + (0.5 + 0.27 * n_note + 0.1 * len(note_lines) if n_note else 0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top = fig_h - 0.12
    ax.text(fig_w / 2, top, "AMOS22 MRI", fontsize=20, fontweight="bold", color=INK, ha="center", va="top")
    ax.text(fig_w / 2, top - 0.44, spec["question"], fontsize=13, color=INK2, ha="center", va="top")
    for i, line in enumerate(how):
        ax.text(fig_w / 2, top - 0.86 - 0.3 * i, line, fontsize=11.5, color=INK2, ha="center", va="top")
    sub_y = top - 0.86 - 0.3 * len(how) - 0.1
    ax.text(fig_w / 2, sub_y, spec["subline"], fontsize=11.5, color=MUTED, ha="center", va="top")

    def chips_at(r, cy):
        x = chips_x0
        for (text, kind), w in zip(r["chips"], r["ws"]):
            box = dict(boxstyle="round,pad=0,rounding_size=0.06")
            if kind == "fixed":
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=0, facecolor=GREY, zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, color=GREY_TEXT, ha="left", va="center", zorder=5)
            elif kind in ("position", "organ"):
                col = POS_C if kind == "position" else ORG_C
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=0, facecolor=col, zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, fontweight="bold", color=pb.text_on(col), ha="left", va="center", zorder=5)
            else:                                                  # replacement / patch note
                ax.add_patch(mpatches.FancyBboxPatch((x, cy - chip_h / 2), w, chip_h, linewidth=1.5, edgecolor=ORG_C,
                                                     facecolor=pb.mix(ORG_C, SURFACE, 0.15), zorder=3, **box))
                ax.text(x + pad, cy, text, fontsize=FS, fontweight="bold", color=pb.mix(ORG_C, "#000000", 0.6),
                        ha="left", va="center", zorder=5)
            x += w + gap

    y = sub_y - 0.65
    for i, r in enumerate(header):                                 # the prompt(s), shown once
        cy = y - 0.66 * i - 0.33
        ax.text(left, cy, r["label"], fontsize=FS, fontweight="bold", color=INK, ha="left", va="center")
        chips_at(r, cy)
    y -= 0.66 * len(header) + (0.35 if header else 0)
    ax.text(bar_x0, y + 0.06, spec.get("bar_title", ""), fontsize=10.5, color=INK2, ha="left", va="bottom")
    for i, r in enumerate(rows):
        cy = y - pitch * i - pitch / 2
        lx = left
        if spec.get("numbered", True):
            ax.add_patch(mpatches.Circle((left + 0.16, cy), 0.16, facecolor=INK, linewidth=0, zorder=3))
            ax.text(left + 0.16, cy, str(i + 1), fontsize=11, fontweight="bold", color="white", ha="center", va="center", zorder=4)
            lx = left + 0.5
        ax.text(lx, cy, r["label"], fontsize=FS, fontweight="bold", color=INK, ha="left", va="center")
        if r.get("chips"):
            chips_at(r, cy)
        nb = len(r["bars"])
        for k, b in enumerate(r["bars"]):
            by = cy + (nb - 1) / 2 * 0.17 - k * 0.17
            h = 0.2 if nb == 1 else 0.12
            ax.add_patch(mpatches.FancyBboxPatch((bar_x0, by - h / 2), bar_w, h, linewidth=0, facecolor=TRACK, zorder=2,
                                                 boxstyle="round,pad=0,rounding_size=0.05"))
            if b["value"] > 0.005:
                ax.add_patch(mpatches.FancyBboxPatch((bar_x0, by - h / 2), max(bar_w * min(b["value"] / b["vmax"], 1), 0.08), h,
                                                     linewidth=0, facecolor=b["color"], zorder=3,
                                                     boxstyle="round,pad=0,rounding_size=0.05"))
            ax.text(bar_x0 + bar_w + 0.14, by, b["text"], fontsize=FS + (1 if nb == 1 else -1.5),
                    fontweight="bold" if nb == 1 else "normal", color=INK, ha="left", va="center")
    if spec.get("ref"):
        v, vmax, txt = spec["ref"]
        rx = bar_x0 + bar_w * v / vmax
        ax.plot([rx, rx], [y + 0.02, y - pitch * len(rows)], color=DARK, lw=1.2, linestyle=(0, (4, 3)), zorder=6)
        ax.text(rx, y - pitch * len(rows) - 0.06, txt, fontsize=9.5, color=DARK, ha="center", va="top")

    ly = y - pitch * len(rows) - (0.75 if spec.get('ref') else 0.4)
    lx = left
    for kind, txt in spec["legend"]:
        w = 0.44
        box = dict(boxstyle="round,pad=0,rounding_size=0.04")
        if kind == "position":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=0, facecolor=POS_C, **box))
        elif kind == "organ":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=0, facecolor=ORG_C, **box))
        elif kind == "note":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=1.2, edgecolor=ORG_C,
                                                 facecolor=pb.mix(ORG_C, SURFACE, 0.15), **box))
        elif kind == "fixed":
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=0, facecolor=GREY, **box))
        else:                                                      # a bar colour
            ax.add_patch(mpatches.FancyBboxPatch((lx, ly - 0.09), w, 0.18, linewidth=0, facecolor=kind, **box))
        ax.text(lx + w + 0.12, ly, txt, fontsize=10.5, color=INK2, ha="left", va="center")
        lx += w + 0.12 + 0.088 * len(txt) + 0.4
    ny = ly - 0.5
    for lines in note_lines:
        for ln in lines:
            ax.text(left, ny, ln, fontsize=10.5, color=INK2, ha="left", va="center")
            ny -= 0.27
        ny -= 0.1
    out = spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def read(name):
    d = pd.read_csv(DATA / f"{name}_amos22_mri.csv")
    return d[d.subject_id.isin(RUN)].copy()


def noise_stats(person, sid):
    """Mean / standard deviation of the non-zero voxels of the scan = the Gaussian used for the noise (as in run_erasure)."""
    import nibabel as nib
    a = np.asarray(nib.load(DATA.parent / person / "samples" / "amos22_mri" / "volumes" / f"{sid}_rot000_image.nii.gz").dataobj, dtype="float32")
    nz = a[a != 0]
    return float(nz.mean()), float(nz.std()), float(a.min()), float(a.max())


def figures(organ, sid, num, d13, d14, d15, d16):
    pos, core = organ.split(None, 1)
    other = ("right " if pos == "left" else "left ") + core
    opos = "right" if pos == "left" else "left"
    cov = "cov_left_organ" if pos == "left" else "cov_right_organ"
    out = ROOT / organ.replace(" ", "_") / "rows13_16"
    sub = f"Organ: {organ}   (position: {pos}, organ name: {core})"
    prompt = f"Segment the {organ} in this image."
    hdr = [{"label": "Prompt", "chips": prompt_chips(organ)}]
    L1 = [("position", "Position word"), ("organ", "Organ name"), ("fixed", "Word kept fixed")]
    MASK = 'Mask = the region VoxTell marks as "this is the organ" in the scan (its segmentation).'
    TRUE = "True organ = the outline drawn by experts in the dataset (the ground truth)."
    mean_, std_, lo_, hi_ = noise_stats(d13[d13.subject_id == sid].person.iloc[0], sid)
    notes14b = [MASK, TRUE,
                "Blend = the model's internal representation of the two words is mixed, for example 75% of \"left\" and 25% of \"right\"; it is not a new sentence.",
                f"Bar = how much of the true {organ} the mask covers (100% = all of it).",
                '"Mask on: neither / both" means the mask is not clearly on just one of the two organs.']
    notes14p = [MASK, TRUE,
                f'Patching = while the model handles the request "{organ}", the result of the chosen layer(s) is swapped for what the request "{other}" would have produced there.',
                f"Bar = how much of the true {organ} the mask covers (100% = all of it; 0% = the mask is no longer on the organ).",
                '"Mask on" says which organ the mask ended up on.']
    notes15 = [TRUE, "Attention = how strongly the model looks at a part of the scan while it decides what to segment.",
               "Each layer is one processing step inside the model; layer 1 is the first, layer 6 the last.",
               "The dark bar is the same organ on the other side of the body; the grey bar is the other organ type."]
    notes16 = ["Cosine distance = how different two internal representations are: 0 means identical, larger means more different.",
               "The grey bar is a scale for comparison: the difference between two different organs.",
               "Input to the decoder = before any processing layer; After layer k = after the k-th layer."]

    # ---- row 13 ----
    s = d13[(d13.subject_id == sid) & (d13.organ == organ)].set_index("condition")
    conds = [("Original image", "original"), (f"{organ.capitalize()} replaced by random noise", "noise_target"),
             ("Whole image replaced by random noise", "noise_image")]
    vmax = max(1.0, max(float(s.loc[c, "size_vs_original"]) for _, c in conds))
    def r13(label, cond):
        v, n, cv = float(s.loc[cond, "size_vs_original"]), int(s.loc[cond, "n_pred_voxels"]), float(s.loc[cond, "cov_of_noised_region"])
        t1 = "nothing segmented" if n == 0 else ("almost nothing (<1%)" if v < 0.01 else f"{v * 100:.0f}%")
        return {"label": label, "chips": prompt_chips(organ),
                "bars": [{"value": v, "vmax": vmax, "color": ORG_C, "text": t1},
                         {"value": cv, "vmax": 1.0, "color": DARK, "text": f"{cv * 100:.0f}%"}]}
    notes13 = [MASK, TRUE,
               "Blue bar = size of the mask compared with the mask on the original image. 100% means the same size as on the original image; "
               "it does not say the mask is correct.",
               "Dark bar = how much of the true organ the mask covers. 100% would mean the whole true organ is covered. It is below 100% "
               "even on the original image, because VoxTell is not perfect.",
               "A large blue bar with a tiny dark bar means VoxTell segmented something, but not the organ.",
               f"Random noise = every voxel of the hidden region (or of the whole image) is replaced by a random value from a Gaussian "
               f"(bell-curve) distribution with mean {mean_:.0f} and standard deviation {std_:.0f}: the average brightness and spread of this "
               f"scan (its values range from {lo_:.0f} to {hi_:.0f}). The original voxels are removed, not added to. One random draw is used."]
    draw(dict(question=f"If the {organ} is hidden under random noise, does VoxTell still segment it?",
              howto="Every row gives VoxTell the same request (the words below). Only the image changes. Compare the blue bar (is a mask "
                    "still segmented, and how large) with the dark bar (does it still cover the true organ).",
              subline=sub, bar_title="Blue: mask size vs. the original image's mask     Dark: how much of the true organ is covered",
              bar_w=2.4, out=out / "row13_organ_replaced_by_noise.png", rows=[r13(l, c) for l, c in conds], legend=L1 + [(ORG_C, "Mask size"), (DARK, "True organ covered")],
              notes=notes13))

    # ---- row 14 blend ----
    b = d14[(d14.subject_id == sid) & (d14.pair == core)]
    def blend_row(a):
        if a in (0, 1):                                             # the two plain prompts
            r = b[(b.variant == "baseline") & (b.dst == ("left" if a == 0 else "right"))].iloc[0]
            label, chips = f"Prompt: {'left' if a == 0 else 'right'}", prompt_chips(("left " if a == 0 else "right ") + core)
        else:                                                       # blended embedding
            r = b[(b.variant == "blend") & (b.alpha == a)].iloc[0]
            label = f"Blend {a:g}"
            lft, rgt = round((1 - a) * 100), round(a * 100)
            txt = f"{lft}% left − {abs(rgt)}% right" if a < 0 else (f"{rgt}% right − {abs(lft)}% left" if a > 1 else f"{lft}% left + {rgt}% right")
            chips = prompt_chips("left " + core, position_text=txt)
        return {"label": label, "chips": chips,
                "bars": [{"value": float(r[cov]), "vmax": 1.0, "color": ORG_C, "text": f"{r[cov]:.2f}  mask on: {r['call']}"}]}
    draw(dict(question='What if the request is a mix of "left" and "right"? Which side does VoxTell segment?',
              howto='The rows marked "Prompt" are the two normal requests. The other rows blend the meaning of "left" and "right" '
                    f'inside the model (the percentages). The bar shows how much of the true {organ} the mask covers; the words next to it '
                    'say which side the mask landed on.', subline=sub,
              bar_title=f"Fraction of the {organ} covered by the mask", bar_w=2.4, out=out / "row14_blend_left_right.png", notes=notes14b,
              rows=[blend_row(a) for a in (-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5)],
              legend=[("position", "Position word (or the blend of the two)"), ("organ", "Organ name"), ("fixed", "Word kept fixed")]))

    # ---- row 14 patching ----
    p = d14[(d14.subject_id == sid) & (d14.pair == core) & (d14.dst == pos)]
    def patch_row(label, var, layer=None):
        q = p[p.variant == var]
        if layer is not None:
            q = q[q.layer == layer]
        r = q.iloc[0]
        extra = None if var == "baseline" else f'update from "{other}"'
        return {"label": label, "chips": prompt_chips(organ, extra=extra),
                "bars": [{"value": float(r[cov]), "vmax": 1.0, "color": ORG_C, "text": f"{r[cov]:.2f}  mask on: {r['call']}"}]}
    rows = [patch_row("Unmodified", "baseline")] + [patch_row(f"Layer {k + 1} patched", "single_layer_update", k) for k in range(6)] + \
           [patch_row("Layers 4-6 patched", "updates_from_layer", 3), patch_row("Layers 1-6 patched", "updates_from_layer", 0)]
    draw(dict(question=f'Can one step inside the model turn a request for the "{organ}" into the "{other}"?',
              howto=f'The request stays "{organ}", but the chosen processing layers are made to use the signal of the "{other}" request. '
                    f'The bar shows how much of the {organ} the mask still covers; 0 means the mask left the organ.', subline=sub,
              bar_title=f"Fraction of the {organ} covered by the mask", bar_w=2.4, out=out / "row14_patching_layers.png", notes=notes14p, rows=rows,
              legend=L1 + [("note", "Which prompt the patched layers take their update from")]))

    # ---- row 15 ----
    a = d15[(d15.subject_id == sid) & (d15.prompt_organ == organ)].copy()
    a["kind"] = np.where(a.is_matched, "named", np.where(a.is_wrong_side, "other_side", "other_type"))
    q = a.groupby(["kind", "layer"]).enrichment.mean()
    layers = sorted(a.layer.unique())
    vmax = float(q.max()) * 1.05
    cols = {"named": ORG_C, "other_side": DARK, "other_type": LIGHT}
    rows = [{"label": f"Layer {l}", "chips": None,
             "bars": [{"value": float(q[(k, l)]), "vmax": vmax, "color": cols[k], "text": f"{q[(k, l)]:.1f}"}
                      for k in ("named", "other_side", "other_type")]} for l in layers]
    draw(dict(question=f'When asked for the "{organ}", which part of the scan does VoxTell look at?',
              howto="For each processing layer, the bars show how much attention the model pays to three regions. A longer bar means "
                    "more attention. The dashed line is the average attention over the whole scan, so a bar beyond it means the model "
                    "looks there more than average.", subline=sub, header=hdr, rows=rows, numbered=False,
              bar_title="Attention on each region, relative to the scan average", bar_w=6.0,
              out=out / "row15_cross_attention.png", notes=notes15, ref=(1.0, vmax, "average over the scan"),
              legend=[(ORG_C, f"On the named {organ}"), (DARK, f"On the {other}"), (LIGHT, "On the other organ type")]))

    # ---- row 16 ----
    l = d16[(d16.subject_id == sid) & (d16.scope == "all_patches") & d16.pair.str.startswith(core)]
    lr = l[l.pair.str.contains("left vs right")].set_index("layer").cosine_distance
    ref_row = l[l.pair.str.contains("reference")]
    ref_name = ref_row.pair.iloc[0].split(": ", 1)[1].replace(" (reference)", "")
    rf = ref_row.set_index("layer").cosine_distance
    rows = [{"label": "Input to the decoder" if k == 0 else f"After layer {k}", "chips": None,
             "bars": [{"value": float(lr[k]), "vmax": 0.6, "color": ORG_C, "text": f"{lr[k]:.2f}"},
                      {"value": float(rf[k]), "vmax": 0.6, "color": LIGHT, "text": f"{rf[k]:.2f}"}]} for k in sorted(lr.index)]
    hdr16 = [{"label": "Prompt 1", "chips": prompt_chips("left " + core)}, {"label": "Prompt 2", "chips": prompt_chips("right " + core)}]
    draw(dict(question="Inside the model, how different do the left and the right request look?",
              howto="A short bar means the model sees the two requests as almost the same; a long bar means it sees them as very "
                    "different. Blue compares the left and right request. Grey compares two different organs, to give a sense of scale.",
              subline=sub, header=hdr16, rows=rows,
              numbered=False, bar_title="Cosine distance between the two representations", bar_w=6.0,
              out=out / "row16_left_right_distance.png", notes=notes16,
              legend=[(ORG_C, f"Prompt 1 vs. Prompt 2 (left {core} vs. right {core})"), (LIGHT, f"Reference: {ref_name}")]))


def main():
    allsubs = sorted(load(CFG, "baseline_predictions").subject_id.unique())
    d13, d14 = read("row13_organ_noise"), read("row14_embedding_intervention")
    d15, d16 = read("row15_attention_enrichment"), read("row16_layer_distance")
    for organ in CFG["organs"]:
        for sid in RUN:
            figures(organ, sid, allsubs.index(sid) + 1, d13, d14, d15, d16)
        print("wrote 15 figures for", organ)


if __name__ == "__main__":
    main()
