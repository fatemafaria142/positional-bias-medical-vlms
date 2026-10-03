"""
Local (no GPU, no torch, no voxtell) visualization for PLAN ROWS 17-21 (the
image-region attribution suite) of voxtell-oasis-rows17-21-kaggle.ipynb /
voxtell-oasis-mechanistic-rows13-21-kaggle.ipynb.

Run the Kaggle notebook, download its results zip from the Output tab, unzip
it, then point this script at the unzipped `results/` directory -- it reads
the small CSV/.npz outputs and renders one set of figures per organ, using the
ORIGINAL scan + ground truth that are already present locally in this repo
(<person>/samples/<dataset>/volumes/), so nothing GPU-side needs to be
downloaded or re-run.

Rows covered (mirrors scripts/voxtell_experiments.py's own docstrings):
  17  voxel-grid occlusion Shapley map            (IMAGE)  -- which image region matters most
  18  per-layer attention / KL divergence           (IMAGE)  -- line chart + attention heatmap
  19  gradient saliency + Grad-CAM                   (IMAGE)  -- enrichment bars + saliency/CAM heatmaps
  20  fused spatial feature map                      (IMAGE)  -- IoU-with-GT bars + fused-feature heatmaps
  21  joint image / organ-word / position-word Shapley (IMAGE *and* WORD together)
      -- the one row that puts image contribution and word contribution on the
      same scale; also rendered as one summary chart across every organ.

Registration note: only row 17's voxel map lives in the FULL, un-cropped scan
grid, so only it is resampled and overlaid, pixel-registered, on the real scan
(same trick as scripts/plot_groundedness_results.py: transpose(2,1,0) + the
RAS-canonical affine + nibabel resampling -- the model's array axes are
reversed relative to the raw NIfTI, see build_groundedness_investigation_notebooks.py's
AXCODES comment). Rows 18-20's maps live in VoxTell's own cropped +
downsampled internal working grid, and the notebook does not export that
crop's bounding box -- so those are shown as their own standalone heatmap
(labeled as such) next to, not registered onto, a separately sliced GT/image
reference panel.

Output: local_plots_mechanistic/rows17_21/<organ>/row{17,18,19,20,21}_*.png

Usage:
    python scripts/make_rows17_21_figures.py RESULTS_DIR [--suffix neurite_oasis] [--out-dir DIR]

    # e.g. after unzipping the Kaggle output of voxtell-oasis-rows17-21-kaggle.ipynb
    # (or voxtell-oasis-mechanistic-rows13-21-kaggle.ipynb -- same file names):
    python scripts/make_rows17_21_figures.py voxtell_neurite_oasis_rows17_21_results
"""

import argparse
import base64
import csv
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed categorical order + diverging/sequential hues, validated for CVD safety
# (dataviz skill's reference palette) -- reused as-is from the rest of this
# repo's plotting scripts (GT_C/PRED_C already use slots 3/8 below).
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
GRAY_MID = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, GRAY_MID, RED])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", BLUE, "#0d366b"])


# --------------------------------------------------------------------------------------------------
# shared decode / resample helpers (same approach as scripts/plot_groundedness_results.py)
# --------------------------------------------------------------------------------------------------
def decode_mask_rle(encoded: str, shape) -> np.ndarray:
    runs = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    idx, value = 0, False
    for run_len in runs:
        if value and run_len:
            flat[idx:idx + run_len] = True
        idx += int(run_len)
        value = not value
    return flat.reshape(tuple(shape), order="C")


def canonical_affine(path: Path):
    """Affine of the RAS-reoriented (nib.as_closest_canonical) volume -- pairs with the model's own
    (reversed-axis) array space, the same way plot_groundedness_results.py's canonical_affine does."""
    return nib.as_closest_canonical(nib.load(str(path))).affine


def resample_model_array(arr, ref_path: Path, target_nii, order: int, valid: np.ndarray = None):
    """`arr` is in the model's own array space (same shape/orientation as voxtell_experiments.py's
    label_vol -- axes REVERSED relative to canonical RAS, see this file's module docstring).
    Transposes it back, resamples onto `target_nii`'s own (raw, un-reoriented) grid, and -- if
    `valid` (a boolean mask of "defined" voxels, e.g. row 17's padded box) is given -- returns NaN
    outside it in the resampled frame too, instead of silently extrapolating zeros."""
    img = nib.Nifti1Image(np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(np.float32), canonical_affine(ref_path))
    out = nibabel.processing.resample_from_to(img, target_nii, order=order).get_fdata()
    if valid is not None:
        valid_img = nib.Nifti1Image(np.ascontiguousarray(valid.transpose(2, 1, 0)).astype(np.uint8), canonical_affine(ref_path))
        valid_out = nibabel.processing.resample_from_to(valid_img, target_nii, order=0).get_fdata().astype(bool)
        out = np.where(valid_out, out, np.nan)
    return out


def load_manifest_label_ids(repo: Path, person: str, dataset: str, subject_id: str) -> dict:
    """organ name -> label_id, read straight from the per-person manifest (not the Kaggle CSVs) --
    lets ground truth be loaded directly from the local label volume, no RLE decode needed for it."""
    path = repo / person / "samples" / dataset / f"manifest_{dataset}.csv"
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["subject_id"] == subject_id and int(r["rotation_angle_deg"]) == 0]
    return {r["organ"]: int(r["label_id"]) for r in rows}


def slice_axis_of(img_nii):
    axcodes = nib.aff2axcodes(img_nii.affine)
    return next((a for a, code in enumerate(axcodes) if code in ("S", "I")), int(np.argmin(img_nii.shape)))


def side_labels(img_nii, slice_axis):
    axcodes = nib.aff2axcodes(img_nii.affine)
    rem = [a for a in range(3) if a != slice_axis]
    h_code = axcodes[rem[0]]
    return ("R", "L") if h_code == "L" else ("L", "R")


def bbox_2d(mask_2d, pad, shape):
    coords = np.nonzero(mask_2d)
    if len(coords[0]) == 0:
        return 0, shape[0], 0, shape[1]
    r0, r1 = max(coords[0].min() - pad, 0), min(coords[0].max() + pad + 1, shape[0])
    c0, c1 = max(coords[1].min() - pad, 0), min(coords[1].max() + pad + 1, shape[1])
    return r0, r1, c0, c1


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel_image(ax, img2, title, left_label, right_label):
    ax.imshow(img2, cmap="gray", origin="lower")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=5)
    clean_axes(ax)
    ax.text(0.02, 0.5, left_label, transform=ax.transAxes, color="white", fontsize=10, fontweight="bold", va="center", zorder=6)
    ax.text(0.98, 0.5, right_label, transform=ax.transAxes, color="white", fontsize=10, fontweight="bold", va="center", ha="right", zorder=6)


def overlay(ax, mask_2d, color, alpha=0.5):
    rgba = np.zeros((*mask_2d.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.where(mask_2d, alpha, 0.0)
    ax.imshow(rgba, origin="lower")
    if mask_2d.any():
        ax.contour(mask_2d, levels=[0.5], colors=color, linewidths=1.6)


def heat_panel(ax, arr_2d, title, cmap, vmin=None, vmax=None, center_zero=False):
    a = np.ma.masked_invalid(arr_2d)
    if center_zero:
        v = np.nanmax(np.abs(arr_2d)) if np.isfinite(arr_2d).any() else 1.0
        vmin, vmax = -v, v
    im = ax.imshow(a, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=5)
    clean_axes(ax)
    return im


def savefig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def style_bars(ax, title, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------------------
# per-subject context: raw image, GT masks, label ids, slice axis -- loaded once, reused by all rows
# --------------------------------------------------------------------------------------------------
class Subject:
    def __init__(self, repo, person, dataset, subject_id, image_path, label_path):
        self.repo, self.person, self.dataset, self.subject_id = repo, person, dataset, subject_id
        base = repo / person / "samples" / dataset
        self.img_nii = nib.load(str(base / image_path))
        self.image = self.img_nii.get_fdata().astype(np.float32)
        self.label = nib.load(str(base / label_path)).get_fdata().astype(int)
        self.slice_axis = slice_axis_of(self.img_nii)
        self.left_label, self.right_label = side_labels(self.img_nii, self.slice_axis)
        self.label_ids = load_manifest_label_ids(repo, person, dataset, subject_id)
        self.label_path_abs = base / label_path

    def gt_mask(self, organ):
        return self.label == self.label_ids[organ]

    def slice_of(self, organ, arr_3d=None):
        gt = self.gt_mask(organ)
        coords = np.nonzero(gt)
        idx3 = [slice(None)] * 3
        idx3[self.slice_axis] = int(round(coords[self.slice_axis].mean())) if len(coords[0]) else self.image.shape[self.slice_axis] // 2
        idx3 = tuple(idx3)
        arr = self.image if arr_3d is None else arr_3d
        return arr[idx3], gt[idx3], idx3

    def crop_window(self, organ, pad=18):
        _, gt2, _ = self.slice_of(organ)
        return bbox_2d(gt2.T, pad, gt2.T.shape)


def crop(arr2d, win):
    r0, r1, c0, c1 = win
    return arr2d[r0:r1, c0:c1]


# --------------------------------------------------------------------------------------------------
# row 17 -- voxel-grid occlusion Shapley map (full-volume frame -> exactly registered on the scan)
# --------------------------------------------------------------------------------------------------
def figure_row17(subj, organ, shap_rows, voxel_map, out_dir):
    """One story, top to bottom: (1) where this organ sits in the WHOLE scan, (2) zoomed in on the
    organ itself -- real picture vs. which parts of it the model needed, (3) the bottom-line answer
    in a plain sentence, with the block ranking underneath for anyone who wants the detail."""
    img2, gt2, idx3 = subj.slice_of(organ)
    valid = ~np.isnan(voxel_map)
    map_full = resample_model_array(voxel_map, subj.label_path_abs, subj.img_nii, order=1, valid=valid)
    map2 = map_full[idx3]
    win = subj.crop_window(organ, pad=14)
    full_img, full_gt = img2.T, gt2.T
    img2c, gt2c, map2c = crop(full_img, win), crop(full_gt, win), crop(map2.T, win)
    r0, r1, c0, c1 = win

    df = pd.DataFrame(shap_rows).sort_values("shapley_value", ascending=False)
    best, worst = df.iloc[0], df.iloc[-1]
    v_range = np.nanmax(np.abs(map2c)) if np.isfinite(map2c).any() else 1e-6
    n_helps = int((df.shapley_value >= 0).sum())
    takeaway = (
        f'"{organ}": every one of the 8 image blocks around this organ HELPED the model get it right '
        f"(none hurt it) -- block {int(best.block_id)} mattered most, block {int(worst.block_id)} mattered "
        f"least. The model is genuinely reading the real scan here, not ignoring it."
        if n_helps == 8 else
        f'"{organ}": {n_helps} of 8 image blocks helped, {8 - n_helps} actually hurt the result when kept real '
        f"(vs. being replaced with noise) -- block {int(best.block_id)} helped most, block {int(worst.block_id)} "
        f"hurt most."
    )

    fig = plt.figure(figsize=(10.5, 12), dpi=170)
    fig.patch.set_facecolor(SURFACE)

    # --- step 1: whole scan, with a box showing where we're about to zoom in -----------------------
    fig.text(0.5, 0.975, "ROW 17 -- which part of the picture did the model actually need?", ha="center",
              fontsize=15, fontweight="bold", color=INK)
    fig.text(0.5, 0.953, f"organ: {organ}", ha="center", fontsize=11.5, color=MUTED)
    fig.text(0.5, 0.918, "1. Where is it in the full scan?", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax0 = fig.add_axes([0.32, 0.66, 0.36, 0.22])
    ax0.imshow(full_img, cmap="gray", origin="lower")
    clean_axes(ax0)
    ax0.text(0.02, 0.5, subj.left_label, transform=ax0.transAxes, color="white", fontsize=10, fontweight="bold", va="center", zorder=6)
    ax0.text(0.98, 0.5, subj.right_label, transform=ax0.transAxes, color="white", fontsize=10, fontweight="bold", va="center", ha="right", zorder=6)
    overlay(ax0, full_gt, AQUA, alpha=0.45)
    from matplotlib.patches import Rectangle
    pad_in = max(1, int(0.015 * (c1 - c0)))
    ax0.add_patch(Rectangle((c0 + pad_in, r0 + pad_in), c1 - c0 - 2 * pad_in, r1 - r0 - 2 * pad_in,
                             fill=False, edgecolor=ORANGE, linewidth=2.2, zorder=5))

    # --- step 2: zoomed scan vs. zoomed "what mattered" heatmap ------------------------------------
    fig.text(0.5, 0.615, "2. Zoomed in: the real picture, and which parts of it the model needed",
              ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax1 = fig.add_axes([0.06, 0.345, 0.40, 0.225])
    panel_image(ax1, img2c, "The real scan here", subj.left_label, subj.right_label)
    overlay(ax1, gt2c, AQUA)
    ax2 = fig.add_axes([0.54, 0.345, 0.40, 0.225])
    im = heat_panel(ax2, map2c, "Red = helped, blue = hurt", DIVERGING, vmin=-v_range, vmax=v_range)
    if gt2c.any():
        ax2.contour(gt2c, levels=[0.5], colors=INK, linewidths=1.2, linestyles="--")
    fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.04, label="Shapley credit (dice)")

    # --- step 3: the bottom line, in one sentence, plus the detail bar chart -----------------------
    fig.text(0.5, 0.275, "3. The bottom line", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.245, "\n".join(_wrap(takeaway, 92)), ha="center", va="top", fontsize=11, color=INK2)
    ax3 = fig.add_axes([0.12, 0.035, 0.76, 0.12])
    colors = [RED if v >= 0 else BLUE for v in df.shapley_value]
    ax3.bar(df.block_id.astype(str), df.shapley_value, color=colors, width=0.62, zorder=3)
    ax3.axhline(0, color=INK2, linewidth=1)
    style_bars(ax3, "", "Shapley credit\n(per block)")
    ax3.set_xlabel("block id (detail, for reference)", fontsize=9.5, color=MUTED)

    out = savefig(fig, out_dir / organ.replace(" ", "_") / "row17_story.png")
    return (out,)


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width)


# --------------------------------------------------------------------------------------------------
# row 18 -- per-layer attention / KL divergence (map NOT registered: VoxTell's own cropped grid)
# --------------------------------------------------------------------------------------------------
def _organ_core_side(organ):
    parts = organ.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in ("left", "right"):
        return parts[1], parts[0].lower()
    return organ, None


def figure_row18_pair(subj, core, kl_left, kl_right, maps, out_dir):
    """THE positional-bias version of row 18: does the model's internal attention match the organ
    EQUALLY WELL regardless of which side is asked? A single-organ KL curve (the old version of this
    figure) cannot answer that -- it has no "other side" to compare against. This one puts left and
    right side by side everywhere: the two scans, the two KL curves, the two attention patterns, and
    a bottom line that states the actual side gap in plain language."""
    left_organ, right_organ = f"left {core}", f"right {core}"
    img2l, gt2l, _ = subj.slice_of(left_organ)
    img2r, gt2r, _ = subj.slice_of(right_organ)
    winl, winr = subj.crop_window(left_organ, pad=14), subj.crop_window(right_organ, pad=14)
    img2cl, gt2cl = crop(img2l.T, winl), crop(gt2l.T, winl)
    img2cr, gt2cr = crop(img2r.T, winr), crop(gt2r.T, winr)

    dfl = pd.DataFrame(kl_left).sort_values("layer")
    dfr = pd.DataFrame(kl_right).sort_values("layer")
    col = "kl_divergence_vs_generic_normalized"
    gap = (dfl[col].values - dfr[col].values)
    mean_gap = float(np.mean(gap))
    worse_side = "left" if mean_gap > 0 else "right" if mean_gap < 0 else None
    similar = abs(mean_gap) < 0.15 * max(dfl[col].mean(), dfr[col].mean(), 1e-6)
    takeaway = (
        f'"{core}": the model\'s attention matches the LEFT and RIGHT organ about EQUALLY well '
        f"(average gap {abs(mean_gap):.2f} KL) -- no sign of a side-specific attention bias here."
        if similar else
        f'"{core}": the model\'s attention matches the {("RIGHT" if worse_side == "left" else "LEFT")} organ '
        f"noticeably better than the {worse_side.upper()} one (average gap {abs(mean_gap):.2f} KL) -- "
        f"a real positional asymmetry for this organ."
    )

    fig = plt.figure(figsize=(12, 12.5), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.5, 0.975, "ROW 18 -- does attention match the organ EQUALLY on both sides?", ha="center",
              fontsize=15, fontweight="bold", color=INK)
    fig.text(0.5, 0.953, f"organ pair: {core}  (left vs. right)", ha="center", fontsize=11.5, color=MUTED)

    fig.text(0.5, 0.918, "1. The two organs, as they actually sit in the scan", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax0 = fig.add_axes([0.10, 0.70, 0.33, 0.195])
    panel_image(ax0, img2cl, "Left " + core, subj.left_label, subj.right_label)
    overlay(ax0, gt2cl, AQUA)
    ax0b = fig.add_axes([0.57, 0.70, 0.33, 0.195])
    panel_image(ax0b, img2cr, "Right " + core, subj.left_label, subj.right_label)
    overlay(ax0b, gt2cr, AQUA)

    fig.text(0.5, 0.645, "2. At each processing step, does attention match the LEFT organ as well as the RIGHT one?",
              ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax1 = fig.add_axes([0.10, 0.37, 0.80, 0.21])
    ax1.plot(dfl.layer, dfl[col], "-o", color=BLUE, label=f"left {core}", linewidth=2.2, markersize=6)
    ax1.plot(dfr.layer, dfr[col], "-o", color=ORANGE, label=f"right {core}", linewidth=2.2, markersize=6)
    ax1.fill_between(dfl.layer, dfl[col], dfr[col], color=GRAY_MID, alpha=0.6, zorder=1)
    style_bars(ax1, "", "KL divergence  (lower = attention matches that organ better)")
    ax1.set_xlabel("processing step (decoder layer)", fontsize=10.5, color=INK2)
    ax1.legend(frameon=False, fontsize=9.5, loc="upper right")

    key_l, key_r = f"{left_organ}__last_layer", f"{right_organ}__last_layer"
    if key_l in maps and key_r in maps:
        fig.text(0.5, 0.320, "3. What each side's attention pattern actually looks like", ha="center",
                  fontsize=12.5, fontweight="bold", color=INK)
        ax2 = fig.add_axes([0.20, 0.145, 0.22, 0.125])
        ax3 = fig.add_axes([0.58, 0.145, 0.22, 0.125])
        for ax, key, label in [(ax2, key_l, f"left {core}"), (ax3, key_r, f"right {core}")]:
            m = np.asarray(maps[key])
            proj = m.max(axis=0) if m.ndim == 3 else m
            heat_panel(ax, proj, label, SEQ_BLUE)
        fig.text(0.5, 0.148, "(the model's own internal grid -- shown for pattern only, not lined up pixel-for-pixel with the scans above)",
                  ha="center", fontsize=8.5, color=MUTED)
        bottom_y = 0.095
    else:
        bottom_y = 0.26

    fig.text(0.5, bottom_y, "The bottom line", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, bottom_y - 0.03, "\n".join(_wrap(takeaway, 100)), ha="center", va="top", fontsize=11, color=INK2)

    return savefig(fig, out_dir / core.replace(" ", "_") / "row18_left_vs_right_story.png")


PAPER_INK = "#000000"
PAPER_MUTED = "#555555"
PAPER_GRID = "#dddddd"


def _paper_axes_style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PAPER_INK)
    ax.spines["bottom"].set_color(PAPER_INK)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=PAPER_INK, labelsize=9, width=0.8)


def _panel_label(ax, letter):
    ax.text(-0.06, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=PAPER_INK, va="top")


def figure_row18_paper(subj, core, kl_left, kl_right, out_dir):
    """Publication-ready version of the row-18 left-vs-right comparison: white background, minimal
    spines, lettered panels (a/b), no explanatory prose baked into the image -- that belongs in the
    external figure caption (returned as a string alongside the saved PNG, so it can be pasted
    straight into a paper). Two panels: (a) both organs' true location on one anatomical slice, for
    spatial context; (b) the actual data -- per-layer attention-GT match quality, left vs. right."""
    left_organ, right_organ = f"left {core}", f"right {core}"
    img2, _, idx3 = subj.slice_of(left_organ)
    gt_l = subj.gt_mask(left_organ)[idx3]
    gt_r = subj.gt_mask(right_organ)[idx3]
    img2t, gt_lt, gt_rt = img2.T, gt_l.T, gt_r.T
    union = gt_lt | gt_rt
    win = bbox_2d(union, 16, union.shape)
    img2c, gt_lc, gt_rc = crop(img2t, win), crop(gt_lt, win), crop(gt_rt, win)

    dfl = pd.DataFrame(kl_left).sort_values("layer")
    dfr = pd.DataFrame(kl_right).sort_values("layer")
    col = "kl_divergence_vs_generic_normalized"

    fig = plt.figure(figsize=(8.0, 3.4), dpi=300)
    fig.patch.set_facecolor("white")

    # --- panel (a): anatomical context, both organs on one slice -----------------------------------
    ax_a = fig.add_axes([0.04, 0.10, 0.30, 0.80])
    ax_a.imshow(img2c, cmap="gray", origin="lower")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    if gt_lc.any():
        ax_a.contour(gt_lc, levels=[0.5], colors=BLUE, linewidths=1.3)
    if gt_rc.any():
        ax_a.contour(gt_rc, levels=[0.5], colors=ORANGE, linewidths=1.3)
    ax_a.text(0.03, 0.04, subj.left_label, transform=ax_a.transAxes, color="white", fontsize=8, fontweight="bold")
    ax_a.text(0.97, 0.04, subj.right_label, transform=ax_a.transAxes, color="white", fontsize=8,
              fontweight="bold", ha="right")
    from matplotlib.lines import Line2D
    ax_a.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.5, label=f"left {core}"),
                          Line2D([0], [0], color=ORANGE, lw=1.5, label=f"right {core}")],
                loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=8)
    _panel_label(ax_a, "a")

    # --- panel (b): the actual data -----------------------------------------------------------------
    ax_b = fig.add_axes([0.46, 0.17, 0.50, 0.73])
    _paper_axes_style(ax_b)
    ax_b.plot(dfl.layer, dfl[col], "-o", color=BLUE, linewidth=1.4, markersize=4, label="Left")
    ax_b.plot(dfr.layer, dfr[col], "-o", color=ORANGE, linewidth=1.4, markersize=4, label="Right")
    ax_b.set_xlabel("Decoder layer", fontsize=9.5, color=PAPER_INK)
    ax_b.set_ylabel("KL divergence\n(attention vs. ground truth)", fontsize=9.5, color=PAPER_INK)
    ax_b.yaxis.grid(True, color=PAPER_GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, fontsize=9, loc="best", handlelength=1.6)
    _panel_label(ax_b, "b")

    out = savefig(fig, out_dir / core.replace(" ", "_") / "row18_paper.png")

    mean_gap = float(np.mean(dfl[col].values - dfr[col].values))
    caption = (
        f'Figure. Cross-attention alignment with ground truth for left vs. right {core}, by decoder '
        f'layer. (a) Ground-truth location of the left (blue) and right (orange) {core} on a single '
        f'representative slice. (b) KL divergence between the ground-truth mask and the cross-attention '
        f'density at each decoder layer, after normalizing out a generic, non-spatial prompt\'s attention '
        f'(lower = better alignment). Mean left-right gap: {abs(mean_gap):.2f} KL '
        f'({"left" if mean_gap > 0 else "right"} worse). '
        f'Subject: {subj.subject_id} ({subj.person}).'
    )
    return out, caption


def figure_row18_explainer(subj, core, kl_left, kl_right, out_dir):
    """Self-contained version: one clean figure with a title, subtitle, legend, axis labels and a
    one-line definition note -- everything a reader needs is ON the image, no external caption or
    verbal explanation required. Same clean two-panel layout as figure_row18_paper (not the numbered
    "1/2/3" story version), but with enough built-in text to stand alone."""
    left_organ, right_organ = f"left {core}", f"right {core}"
    img2, _, idx3 = subj.slice_of(left_organ)
    gt_l = subj.gt_mask(left_organ)[idx3]
    gt_r = subj.gt_mask(right_organ)[idx3]
    img2t, gt_lt, gt_rt = img2.T, gt_l.T, gt_r.T
    union = gt_lt | gt_rt
    win = bbox_2d(union, 16, union.shape)
    img2c, gt_lc, gt_rc = crop(img2t, win), crop(gt_lt, win), crop(gt_rt, win)

    dfl = pd.DataFrame(kl_left).sort_values("layer")
    dfr = pd.DataFrame(kl_right).sort_values("layer")
    col = "kl_divergence_vs_generic_normalized"
    mean_gap = float(np.mean(dfl[col].values - dfr[col].values))
    similar = abs(mean_gap) < 0.15 * max(dfl[col].mean(), dfr[col].mean(), 1e-6)
    worse_side = "left" if mean_gap > 0 else "right"
    verdict = (f"No meaningful left/right gap (Δ={abs(mean_gap):.2f}) — attention is side-symmetric for this organ."
               if similar else
               f"{worse_side.capitalize()} side matches worse (Δ={abs(mean_gap):.2f} KL) — a possible positional asymmetry.")

    fig = plt.figure(figsize=(9.6, 4.6), dpi=220)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, f"Does VoxTell's attention find the {core} equally well on both sides?",
              ha="center", fontsize=14.5, fontweight="bold", color=PAPER_INK)
    fig.text(0.5, 0.918, "How closely attention matches the organ, layer by layer — left vs. right",
              ha="center", fontsize=10.5, color=PAPER_MUTED)

    # one shared legend for both panels (colors mean the same thing throughout) -- avoids repeating
    # the organ name in two separate legend boxes, and no per-panel titles below it either (the
    # title + subtitle above already say what both panels are) -- one clear statement, not three.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color=BLUE, lw=2.2, label=f"Left {core}"),
               Line2D([0], [0], color=ORANGE, lw=2.2, label=f"Right {core}"),
               Patch(facecolor=GRAY_MID, label="Gap between sides")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.855), ncol=3, frameon=False, fontsize=9.5)

    # --- panel (a): anatomical context --------------------------------------------------------------
    ax_a = fig.add_axes([0.03, 0.14, 0.30, 0.62])
    ax_a.imshow(img2c, cmap="gray", origin="lower")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    if gt_lc.any():
        ax_a.contour(gt_lc, levels=[0.5], colors=BLUE, linewidths=1.6)
    if gt_rc.any():
        ax_a.contour(gt_rc, levels=[0.5], colors=ORANGE, linewidths=1.6)
    ax_a.text(0.03, 0.04, subj.left_label, transform=ax_a.transAxes, color="white", fontsize=9, fontweight="bold")
    ax_a.text(0.97, 0.04, subj.right_label, transform=ax_a.transAxes, color="white", fontsize=9,
              fontweight="bold", ha="right")

    # --- panel (b): the data -------------------------------------------------------------------------
    ax_b = fig.add_axes([0.42, 0.24, 0.53, 0.52])
    _paper_axes_style(ax_b)
    ax_b.plot(dfl.layer, dfl[col], "-o", color=BLUE, linewidth=1.8, markersize=5.5)
    ax_b.plot(dfr.layer, dfr[col], "-o", color=ORANGE, linewidth=1.8, markersize=5.5)
    ax_b.fill_between(dfl.layer, dfl[col], dfr[col], color=GRAY_MID, alpha=0.5, zorder=0)
    ax_b.set_xlabel("Decoder layer", fontsize=10, color=PAPER_INK)
    ax_b.set_ylabel("KL divergence (lower = better match)", fontsize=10, color=PAPER_INK)
    ax_b.yaxis.grid(True, color=PAPER_GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)

    fig.text(0.5, 0.115, f"Verdict: {verdict}", ha="center", fontsize=11, fontweight="bold",
              color=(INK if similar else RED))
    fig.text(0.5, 0.055,
              f"Subject {subj.subject_id} ({subj.person}) • KL divergence measures how far the model's attention "
              "is from the organ's true shape; 0 would mean a perfect match.",
              ha="center", fontsize=8.5, color=PAPER_MUTED)

    return savefig(fig, out_dir / core.replace(" ", "_") / "row18_explainer.png")


# --------------------------------------------------------------------------------------------------
# row 19 -- gradient saliency + Grad-CAM (maps NOT registered: VoxTell's own cropped patch)
# --------------------------------------------------------------------------------------------------
def figure_row19(subj, organ, row, maps, out_dir):
    img2, gt2, _ = subj.slice_of(organ)
    win = subj.crop_window(organ, pad=14)
    img2c, gt2c = crop(img2.T, win), crop(gt2.T, win)

    fig = plt.figure(figsize=(11.5, 4.6), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax1 = fig.add_axes([0.07, 0.16, 0.26, 0.68])
    bars = ["saliency_enrichment", "cam_enrichment"]
    vals = [row[b] for b in bars]
    ax1.bar(["vanilla gradient", "Grad-CAM"], vals, color=[BLUE, ORANGE], width=0.55, zorder=3)
    ax1.axhline(1.0, color=INK2, linewidth=1, linestyle="--", zorder=2)
    style_bars(ax1, f"{organ} -- saliency inside vs. outside the organ", "enrichment (in / out)")
    ax1.text(0.5, -0.17, "dashed line = 1.0 (no enrichment)", transform=ax1.transAxes,
              fontsize=8.5, color=MUTED, ha="center", va="top")

    ax2 = fig.add_axes([0.40, 0.16, 0.26, 0.68])
    ax3 = fig.add_axes([0.70, 0.16, 0.26, 0.68])
    sal_key, cam_key = f"{organ}__saliency", f"{organ}__cam"
    for ax, key, title in [(ax2, sal_key, "Vanilla-gradient saliency\n(model's own cropped patch)"),
                            (ax3, cam_key, "Grad-CAM\n(model's own cropped patch)")]:
        if key in maps:
            m = maps[key]
            mid = m.shape[subj.slice_axis] // 2 if subj.slice_axis < m.ndim else m.shape[0] // 2
            idx = tuple(slice(None) if a != subj.slice_axis else mid for a in range(m.ndim))
            heat_panel(ax, m[idx], title, SEQ_BLUE)
    fig.suptitle(f"{organ}: gradient-based saliency (cheap counterpart to row 17's exact map)",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    return savefig(fig, out_dir / organ.replace(" ", "_") / "row19_gradient_saliency.png")


# --------------------------------------------------------------------------------------------------
# row 20 -- fused spatial feature map, matched vs. mismatched prompt (maps NOT registered)
# --------------------------------------------------------------------------------------------------
def figure_row20(subj, organ, rows, maps, out_dir):
    df = pd.DataFrame(rows)
    order = ["matched", "mismatched_side", "mismatched_organ"]
    df["condition"] = pd.Categorical(df["condition"], categories=[c for c in order if c in set(df.condition)], ordered=True)
    df = df.sort_values("condition")
    colors = {"matched": BLUE, "mismatched_side": ORANGE, "mismatched_organ": AQUA}

    fig = plt.figure(figsize=(11.5, 4.6), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax1 = fig.add_axes([0.07, 0.18, 0.26, 0.66])
    ax1.bar(df.condition.astype(str), df.iou_with_gt, color=[colors[c] for c in df.condition], width=0.6, zorder=3)
    style_bars(ax1, f"{organ} -- fused feature map vs. GT", "IoU with ground truth")
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df.condition.astype(str), rotation=15, ha="right")

    panels = [(c, maps.get(f"{organ}__{c}")) for c in df.condition]
    xs = [0.37, 0.595, 0.82]
    for x, (cond, m) in zip(xs, panels):
        ax = fig.add_axes([x, 0.18, 0.17, 0.66])
        if m is not None:
            mid = m.shape[subj.slice_axis] // 2 if subj.slice_axis < m.ndim else m.shape[0] // 2
            idx = tuple(slice(None) if a != subj.slice_axis else mid for a in range(m.ndim))
            heat_panel(ax, m[idx], f"{cond}\n(model's own cropped grid)", SEQ_BLUE)
        else:
            ax.axis("off")
    fig.suptitle(f"{organ}: does the fused image+text feature map move with the prompt?",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    return savefig(fig, out_dir / organ.replace(" ", "_") / "row20_fused_activation_overlap.png")


# --------------------------------------------------------------------------------------------------
# row 21 -- joint image / organ-word / position-word Shapley (IMAGE *and* WORD, same scale)
# --------------------------------------------------------------------------------------------------
ROLE_LABEL = {"image": "image", "organ_word": "organ word", "position_word": "position word (left/right)"}
ROLE_COLOR = {"image": BLUE, "organ_word": ORANGE, "position_word": AQUA}


def figure_row21(organ, rows, out_dir):
    df = pd.DataFrame(rows)
    roles = ["image", "organ_word", "position_word"]
    metrics = ["f_dice_gt", "f_dice_self"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    width = 0.32
    x = np.arange(len(roles))
    for i, metric in enumerate(metrics):
        vals = [float(df[(df.role == r) & (df.metric == metric)].shapley_value.iloc[0]) if
                ((df.role == r) & (df.metric == metric)).any() else 0.0 for r in roles]
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=[BLUE, ORANGE][i],
               label="vs. ground truth (f_dice_gt)" if metric == "f_dice_gt" else "vs. the full-prompt prediction (f_dice_self)",
               zorder=3)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([ROLE_LABEL[r] for r in roles])
    style_bars(ax, f"{organ} -- image vs. word contribution (joint Shapley)", "Shapley credit (dice)")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.tight_layout()
    return savefig(fig, out_dir / organ.replace(" ", "_") / "row21_joint_image_word_shapley.png")


def figure_row21_summary(all_rows, out_dir):
    """One chart across every organ in the subject: f_dice_gt Shapley credit, image vs. organ word
    vs. position word -- the single cross-organ view of "does the model lean on the picture or the
    words, and does that change by organ?"."""
    df = pd.DataFrame(all_rows)
    df = df[df.metric == "f_dice_gt"]
    organs = sorted(df.organ.unique(), key=lambda o: (o.split(None, 1)[1], o.split(None, 1)[0]) if " " in o else (o,))
    roles = ["image", "organ_word", "position_word"]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * len(organs) + 2), 5.2), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(organs))
    width = 0.27
    for i, r in enumerate(roles):
        vals = [float(df[(df.organ == o) & (df.role == r)].shapley_value.iloc[0]) if
                ((df.organ == o) & (df.role == r)).any() else 0.0 for o in organs]
        ax.bar(x + (i - 1) * width, vals, width=width, color=ROLE_COLOR[r], label=ROLE_LABEL[r], zorder=3)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(organs, rotation=45, ha="right", fontsize=9)
    style_bars(ax, "Row 21 summary -- image vs. word contribution, every organ (f_dice_gt)", "Shapley credit (dice)")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.tight_layout()
    return savefig(fig, out_dir / "row21_summary_all_organs.png")


def image_vs_word_totals(all_rows) -> pd.DataFrame:
    """Collapses row 21's 3 roles into the 2-way question people actually ask: does the model lean
    on the picture, or on the words ("organ_word" + "position_word" summed into one "word" bucket)?
    Shapley credits are additive (they sum to the gap between the full coalition and the empty one),
    so summing the two word roles is the correct way to get a total "word" contribution. Returns one
    row per organ x metric: organ, metric, image, word, word_wins (bool)."""
    df = pd.DataFrame(all_rows)
    out = []
    for (organ, metric), g in df.groupby(["organ", "metric"]):
        image = float(g[g.role == "image"].shapley_value.sum())
        word = float(g[g.role.isin(["organ_word", "position_word"])].shapley_value.sum())
        out.append({"organ": organ, "metric": metric, "image": round(image, 6), "word": round(word, 6),
                     "word_wins": word > image})
    return pd.DataFrame(out)


def figure_row21_image_vs_word(all_rows, out_dir):
    """The direct answer to "is image or word contributing more?": one 2-bar-per-organ chart
    (f_dice_gt only -- the metric scored against real ground truth, not the model's own prediction),
    plus a printed verdict per organ and overall."""
    totals = image_vs_word_totals(all_rows)
    t = totals[totals.metric == "f_dice_gt"].copy()
    organs = sorted(t.organ.unique(), key=lambda o: (o.split(None, 1)[1], o.split(None, 1)[0]) if " " in o else (o,))
    t = t.set_index("organ").loc[organs]

    fig, ax = plt.subplots(figsize=(max(7.5, 0.6 * len(organs) + 2), 5.2), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(organs))
    width = 0.36
    ax.bar(x - width / 2, t["image"], width=width, color=BLUE, label="image", zorder=3)
    ax.bar(x + width / 2, t["word"], width=width, color=ORANGE, label="word (organ word + position word)", zorder=3)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(organs, rotation=45, ha="right", fontsize=9)
    style_bars(ax, "Row 21 -- image vs. word, 2-way (f_dice_gt)", "Shapley credit (dice)")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.tight_layout()
    out_path = savefig(fig, out_dir / "row21_image_vs_word_2way.png")

    print("\n=== Image vs. word contribution (f_dice_gt), per organ ===")
    for organ, row in t.iterrows():
        winner = "WORD" if row.word_wins else "IMAGE"
        print(f"  {organ:28s} image={row['image']:+.4f}  word={row['word']:+.4f}  -> {winner} contributes more")
    n_word, n_image = int(t.word_wins.sum()), int((~t.word_wins).sum())
    mean_image, mean_word = float(t["image"].mean()), float(t["word"].mean())
    overall = "WORD" if mean_word > mean_image else "IMAGE"
    print(f"\n  Across {len(t)} organ(s): word wins on {n_word}, image wins on {n_image}.")
    print(f"  Mean Shapley credit: image={mean_image:+.4f}, word={mean_word:+.4f}  -> overall, {overall} contributes more.")
    return out_path


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------
def rel(p: Path, repo: Path) -> str:
    try:
        return str(p.relative_to(repo))
    except ValueError:
        return str(p)


def read_csv(results_dir: Path, name: str, suffix: str) -> pd.DataFrame:
    path = results_dir / f"{name}_{suffix}.csv"
    if not path.exists():
        print(f"[skip] not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="unzipped Kaggle results/ directory (rows17-21 or rows13-21 notebook output)")
    ap.add_argument("--suffix", default="neurite_oasis", help="dataset suffix used in the CSV/zip file names (default: neurite_oasis)")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help="repo root containing <person>/samples/<dataset>/ (default: this repo)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <repo>/local_plots_mechanistic/rows17_21")
    args = ap.parse_args()
    out_dir = args.out_dir or (args.repo / "local_plots_mechanistic" / "rows17_21")

    row0 = read_csv(args.results_dir, "row0_baseline_masks", args.suffix)
    if row0.empty:
        print("row0_baseline_masks CSV is required (image_path/label_path/organ list) -- nothing to plot.")
        return

    row17 = read_csv(args.results_dir, "row17_grid_shapley", args.suffix)
    row18 = read_csv(args.results_dir, "row18_attention_kl", args.suffix)
    row19 = read_csv(args.results_dir, "row19_gradient_saliency", args.suffix)
    row20 = read_csv(args.results_dir, "row20_fused_activation_overlap", args.suffix)
    row21 = read_csv(args.results_dir, "row21_joint_shapley", args.suffix)

    all_row21_rows = []
    for (person, subject_id), sub0 in row0.groupby(["person", "subject_id"]):
        sub0 = sub0.set_index("organ")
        first = sub0.iloc[0]
        subj = Subject(args.repo, person, first["dataset"] if "dataset" in sub0.columns else args.suffix,
                        subject_id, first["image_path"], first["label_path"])
        maps17 = _load_npz(_find_npz(args.results_dir, 17, "row17_shapley_maps", subject_id))
        maps18 = _load_npz(_find_npz(args.results_dir, 18, "row18_attention_maps", subject_id))
        maps19 = _load_npz(_find_npz(args.results_dir, 19, "row19_saliency_maps", subject_id))
        maps20 = _load_npz(_find_npz(args.results_dir, 20, "row20_fused_maps", subject_id))

        # row 18 is a LEFT vs RIGHT comparison, so it's handled once per organ PAIR, not per organ
        # (the old per-organ version couldn't say anything about positional bias -- no "other side"
        # to compare against).
        if not row18.empty:
            sub18 = row18[row18.subject_id == subject_id]
            cores = sorted({_organ_core_side(o)[0] for o in sub18.organ.unique() if _organ_core_side(o)[1]})
            captions = []
            for core in cores:
                left_rows = sub18[sub18.organ == f"left {core}"].to_dict("records")
                right_rows = sub18[sub18.organ == f"right {core}"].to_dict("records")
                if left_rows and right_rows and f"left {core}" in subj.label_ids and f"right {core}" in subj.label_ids:
                    try:
                        print("wrote", rel(figure_row18_explainer(subj, core, left_rows, right_rows, out_dir), args.repo))
                        paper_path, caption = figure_row18_paper(subj, core, left_rows, right_rows, out_dir)
                        print("wrote", rel(paper_path, args.repo))
                        captions.append(caption)
                    except Exception as e:
                        print(f"[error] {subject_id}/row18/{core}: {e}")
            if captions:
                cap_path = out_dir / "row18_paper_captions.txt"
                cap_path.parent.mkdir(parents=True, exist_ok=True)
                cap_path.write_text("\n\n".join(captions), encoding="utf-8")
                print("wrote", rel(cap_path, args.repo))

        for organ in sub0.index:
            if organ not in subj.label_ids:
                print(f"[skip] {subject_id}/{organ}: not in local manifest, can't locate ground truth")
                continue
            try:
                if not row17.empty and organ in maps17:
                    r = row17[(row17.subject_id == subject_id) & (row17.organ == organ)].to_dict("records")
                    if r:
                        for p in figure_row17(subj, organ, r, maps17[organ], out_dir):
                            print("wrote", rel(p, args.repo))
                if not row19.empty:
                    r = row19[(row19.subject_id == subject_id) & (row19.organ == organ)]
                    if len(r):
                        print("wrote", rel(figure_row19(subj, organ, r.iloc[0], maps19, out_dir), args.repo))
                if not row20.empty:
                    r = row20[(row20.subject_id == subject_id) & (row20.organ == organ)].to_dict("records")
                    if r:
                        print("wrote", rel(figure_row20(subj, organ, r, maps20, out_dir), args.repo))
                if not row21.empty:
                    r = row21[(row21.subject_id == subject_id) & (row21.organ == organ)].to_dict("records")
                    if r:
                        all_row21_rows += r
                        print("wrote", rel(figure_row21(organ, r, out_dir), args.repo))
            except Exception as e:
                print(f"[error] {subject_id}/{organ}: {e}")

    if all_row21_rows:
        print("wrote", rel(figure_row21_summary(all_row21_rows, out_dir), args.repo))
        print("wrote", rel(figure_row21_image_vs_word(all_row21_rows, out_dir), args.repo))


def _find_npz(results_dir: Path, row: int, folder_name: str, subject_id: str) -> Path:
    """The notebook writes these to `<folder_name>/<subject_id>.npz` (e.g. `row17_shapley_maps/
    OASIS_OAS1_0057_MR1.npz`), but a manual download from Kaggle's flat Output tab often lands as
    `Row<row>-<subject_id>.npz` (or lowercase, or with underscores) directly in the results dir
    instead -- try the real structure first, then those common flattened variants, case-insensitively,
    before giving up."""
    candidates = [
        results_dir / folder_name / f"{subject_id}.npz",
        results_dir / f"Row{row}-{subject_id}.npz",
        results_dir / f"row{row}-{subject_id}.npz",
        results_dir / f"Row{row}_{subject_id}.npz",
        results_dir / f"row{row}_{subject_id}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    if results_dir.is_dir():
        for p in results_dir.iterdir():
            if p.is_file() and p.suffix == ".npz" and p.stem.lower() in (f"row{row}-{subject_id}".lower(), f"row{row}_{subject_id}".lower()):
                return p
    return candidates[0]  # doesn't exist -- _load_npz will just return {}


def _load_npz(path: Path) -> dict:
    if not path.exists():
        return {}
    with np.load(path) as z:
        flat = {k: z[k] for k in z.files}
    # row17's keys are plain organ names; rows 18-20 use "<organ>__<suffix>" -- group row17's as-is,
    # leave the others keyed by their full "organ__suffix" string (consumed directly above).
    return flat


if __name__ == "__main__":
    main()
