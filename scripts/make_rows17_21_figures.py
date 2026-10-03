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
from scipy.ndimage import zoom as _ndi_zoom

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


def reconstruct_cropped_map(maps: dict, key: str, label_shape) -> np.ndarray:
    """Row 18's attention maps (unlike row 17's) are saved downsampled AND cropped to a bounding
    box (`__bbox__`) that was not previously exported -- `voxtell_experiments.run_attention_kl` now
    also saves `__bbox__` and `__crop_shape__` alongside the maps, which is everything needed to put
    a map back into the FULL label-volume frame (upsample to `__crop_shape__`, then place at
    `__bbox__`), after which it can go through the exact same `resample_model_array` registration
    row 17's map already uses. Returns None if `key`, `__bbox__` or `__crop_shape__` are missing
    (e.g. a map downloaded before this was added) -- callers must fall back gracefully, not guess."""
    if key not in maps or "__bbox__" not in maps or "__crop_shape__" not in maps:
        return None
    small = np.asarray(maps[key], dtype=np.float32)
    crop_shape = tuple(int(s) for s in np.asarray(maps["__crop_shape__"]))
    zoom_factors = [c / s for c, s in zip(crop_shape, small.shape)]
    upsampled = _ndi_zoom(small, zoom_factors, order=1)
    upsampled = upsampled[tuple(slice(0, c) for c in crop_shape)]  # guard rounding overshoot
    bbox = np.asarray(maps["__bbox__"]).astype(int)
    full = np.full(tuple(label_shape), np.nan, dtype=np.float32)
    sl = tuple(slice(int(bbox[a, 0]), int(bbox[a, 0]) + upsampled.shape[a]) for a in range(3))
    full[sl] = upsampled
    return full


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
        self._model_space_shape = None

    @property
    def model_space_shape(self):
        """Shape of the array frame VoxTell's own code (voxtell_mechanistic/voxtell_experiments)
        works in -- REVERSED axis order relative to the canonical-RAS shape (verified empirically:
        this is `nib.as_closest_canonical(...).shape[::-1]`, same convention as row 17's `label_vol`
        and now row 18's reconstructed attention maps)."""
        if self._model_space_shape is None:
            canon_shape = nib.as_closest_canonical(nib.load(str(self.label_path_abs))).shape
            self._model_space_shape = tuple(int(s) for s in canon_shape[::-1])
        return self._model_space_shape

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


def _rounded_card(fig, rect, facecolor="white", edgecolor="#e4e6ec", lw=1.0, shadow=True, pad_in=0.0, zorder=-2):
    """A rounded white "card" behind a panel, drawn in figure-fraction coordinates. `rect` =
    (x0, y0, w, h) in figure fraction, same convention as fig.add_axes. A faint offset duplicate
    underneath fakes a soft drop shadow -- the one cheap trick that reliably makes a flat matplotlib
    layout read as "designed" instead of a bare default chart."""
    from matplotlib.patches import FancyBboxPatch
    x0, y0, w, h = rect
    if shadow:
        shadow_patch = FancyBboxPatch((x0 - pad_in + 0.004, y0 - pad_in - 0.006), w + 2 * pad_in, h + 2 * pad_in,
                                       boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                       linewidth=0, facecolor="#000000", alpha=0.06, zorder=zorder)
        fig.add_artist(shadow_patch)
    card = FancyBboxPatch((x0 - pad_in, y0 - pad_in), w + 2 * pad_in, h + 2 * pad_in,
                           boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                           linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder + 1)
    fig.add_artist(card)


def figure_row18_explainer(subj, core, kl_left, kl_right, out_dir):
    """Self-contained, designed version: declarative title, Left/Right/Gap legend only (the organ
    is already named once in the title), a compact statistics card instead of a prose "Verdict:"
    sentence, and rounded card panels on a soft page background instead of a flat white chart --
    the floating per-point KL value labels were dropped (the exact numbers already live in the
    stats card; repeating all 12 of them on the chart was clutter, not clarity)."""
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
    mean_l, mean_r = float(dfl[col].mean()), float(dfr[col].mean())
    mean_gap = mean_l - mean_r
    similar = abs(mean_gap) < 0.15 * max(mean_l, mean_r, 1e-6)
    worse_side = "Left" if mean_gap > 0 else "Right"

    PAGE_BG = "#f4f5f8"
    fig = plt.figure(figsize=(10.4, 4.9), dpi=220)
    fig.patch.set_facecolor(PAGE_BG)
    fig.text(0.5, 0.955, f"Attention–Ground-Truth Alignment by Decoder Layer: {core.title()} (Left vs. Right)",
              ha="center", fontsize=13.5, fontweight="bold", color=PAPER_INK)
    fig.text(0.5, 0.905,
              "KL divergence between cross-attention density and the ground-truth mask, normalized against a generic prompt",
              ha="center", fontsize=9.5, color=PAPER_MUTED)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color=BLUE, lw=2.6, label="Left", solid_capstyle="round"),
               Line2D([0], [0], color=ORANGE, lw=2.6, label="Right", solid_capstyle="round"),
               Patch(facecolor=GRAY_MID, label="Gap")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.845), ncol=3, frameon=False, fontsize=9.5)

    # --- panel (a): anatomical context, on its own card ---------------------------------------------
    card_a = (0.035, 0.12, 0.26, 0.60)
    _rounded_card(fig, card_a, pad_in=0.012)
    ax_a = fig.add_axes([card_a[0] + 0.012, card_a[1] + 0.03, card_a[2] - 0.024, card_a[3] - 0.05])
    ax_a.imshow(img2c, cmap="gray", origin="lower")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    if gt_lc.any():
        ax_a.contour(gt_lc, levels=[0.5], colors=BLUE, linewidths=1.8)
    if gt_rc.any():
        ax_a.contour(gt_rc, levels=[0.5], colors=ORANGE, linewidths=1.8)

    # --- panel (b): the data, on its own card --------------------------------------------------------
    card_b = (0.345, 0.12, 0.415, 0.60)
    _rounded_card(fig, card_b, pad_in=0.012)
    ax_b = fig.add_axes([card_b[0] + 0.045, card_b[1] + 0.13, card_b[2] - 0.08, card_b[3] - 0.19])
    _paper_axes_style(ax_b)
    ax_b.fill_between(dfl.layer, dfl[col], dfr[col], color=GRAY_MID, alpha=0.6, zorder=0)
    ax_b.plot(dfl.layer, dfl[col], "-", color=BLUE, linewidth=2.4, zorder=3, solid_capstyle="round")
    ax_b.plot(dfr.layer, dfr[col], "-", color=ORANGE, linewidth=2.4, zorder=3, solid_capstyle="round")
    ax_b.scatter(dfl.layer, dfl[col], s=46, color=BLUE, edgecolor="white", linewidth=1.3, zorder=4)
    ax_b.scatter(dfr.layer, dfr[col], s=46, color=ORANGE, edgecolor="white", linewidth=1.3, zorder=4)
    ax_b.margins(y=0.15)
    ax_b.set_xlabel("Decoder layer", fontsize=10, color=PAPER_INK)
    ax_b.set_ylabel("KL divergence (lower = better match)", fontsize=9.5, color=PAPER_INK)
    ax_b.yaxis.grid(True, color="#e9eaef", linewidth=0.7, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_facecolor("white")

    # --- panel (c): compact statistics card, not a prose sentence ------------------------------------
    card_c = (0.80, 0.12, 0.165, 0.60)
    _rounded_card(fig, card_c, pad_in=0.012)
    ax_c = fig.add_axes([card_c[0] + 0.015, card_c[1] + 0.02, card_c[2] - 0.03, card_c[3] - 0.04])
    ax_c.axis("off")
    stat_rows = [
        ("Mean KL — Left", f"{mean_l:.3f}", BLUE),
        ("Mean KL — Right", f"{mean_r:.3f}", ORANGE),
        ("Mean gap (L−R)", f"{mean_gap:+.3f}", PAPER_INK),
        ("Larger gap on", worse_side if not similar else "— (n.s.)", PAPER_INK),
    ]
    y0 = 0.95
    for i, (label, value, color) in enumerate(stat_rows):
        ax_c.text(0.0, y0, label, fontsize=8.6, color=PAPER_MUTED, transform=ax_c.transAxes, va="top")
        ax_c.text(0.0, y0 - 0.115, value, fontsize=13, fontweight="bold", color=color, transform=ax_c.transAxes, va="top")
        y0 -= 0.245
        if i < len(stat_rows) - 1:
            ax_c.plot([0, 1], [y0 + 0.065, y0 + 0.065], color="#edeef2", linewidth=1, transform=ax_c.transAxes)

    fig.text(0.5, 0.045,
              f"Subject {subj.subject_id} ({subj.person}). KL divergence = distance between the model's cross-attention "
              "and the true organ shape (0 = perfect match).",
              ha="center", fontsize=8.3, color=PAPER_MUTED)

    return savefig(fig, out_dir / core.replace(" ", "_") / "row18_explainer.png")


def figure_row18_attention_overlay(subj, core, maps, out_dir):
    """THE direct answer to "where is the model looking": the actual cross-attention density,
    registered and overlaid on the real scan (same registration trick as row 17's Shapley map --
    reconstruct_cropped_map + resample_model_array), for both the left and right organ side by
    side. Requires `__bbox__`/`__crop_shape__` in `maps` (added to voxtell_experiments.py's
    run_attention_kl -- re-run row 18 on Kaggle to get these in your .npz if they're missing).
    Returns None (and prints why) if the map can't be reconstructed, rather than silently drawing
    something misleading."""
    left_organ, right_organ = f"left {core}", f"right {core}"
    shape = subj.model_space_shape
    panels = []
    for organ in (left_organ, right_organ):
        full = reconstruct_cropped_map(maps, f"{organ}__last_layer", shape)
        if full is None:
            print(f"[skip] row18 attention overlay for {organ}: __bbox__/__crop_shape__ not in this "
                  "npz (re-run row 18 on Kaggle with the updated code to get them)")
            return None
        valid = ~np.isnan(full)
        registered = resample_model_array(full, subj.label_path_abs, subj.img_nii, order=1, valid=valid)
        panels.append(registered)
    reg_l, reg_r = panels

    img2, _, idx3 = subj.slice_of(left_organ)
    gt_l = subj.gt_mask(left_organ)[idx3]
    gt_r = subj.gt_mask(right_organ)[idx3]
    map_l, map_r = reg_l[idx3], reg_r[idx3]
    img2t, gt_lt, gt_rt, map_lt, map_rt = img2.T, gt_l.T, gt_r.T, map_l.T, map_r.T
    union = gt_lt | gt_rt
    win = bbox_2d(union, 20, union.shape)
    img2c = crop(img2t, win)
    gt_lc, gt_rc = crop(gt_lt, win), crop(gt_rt, win)
    map_lc, map_rc = crop(map_lt, win), crop(map_rt, win)

    fig = plt.figure(figsize=(10.4, 5.2), dpi=220)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, f"Where Does VoxTell's Attention Actually Land? {core.title()} (Left vs. Right)",
              ha="center", fontsize=13.5, fontweight="bold", color=PAPER_INK)
    fig.text(0.5, 0.918,
              "Last-decoder-layer cross-attention density, registered onto the real scan; dashed outline = true organ shape",
              ha="center", fontsize=9.5, color=PAPER_MUTED)

    vmax = float(np.nanmax([np.nanmax(map_lc) if np.isfinite(map_lc).any() else 0,
                             np.nanmax(map_rc) if np.isfinite(map_rc).any() else 0]))
    vmax = vmax if vmax > 0 else 1.0
    axes_specs = [(0.05, "Left " + core, map_lc, gt_lc), (0.53, "Right " + core, map_rc, gt_rc)]
    last_im = None
    for x0, label, m, gtc in axes_specs:
        ax = fig.add_axes([x0, 0.14, 0.40, 0.68])
        ax.imshow(img2c, cmap="gray", origin="lower")
        heat = np.ma.masked_invalid(m)
        last_im = ax.imshow(heat, cmap=SEQ_BLUE, origin="lower", alpha=0.65, vmin=0, vmax=vmax)
        if gtc.any():
            ax.contour(gtc, levels=[0.5], colors=RED, linewidths=1.6, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(label, fontsize=11, fontweight="bold", color=PAPER_INK, pad=6)
        ax.text(0.03, 0.04, subj.left_label, transform=ax.transAxes, color="white", fontsize=9, fontweight="bold")
        ax.text(0.97, 0.04, subj.right_label, transform=ax.transAxes, color="white", fontsize=9,
                fontweight="bold", ha="right")
    cax = fig.add_axes([0.945, 0.18, 0.015, 0.60])
    fig.colorbar(last_im, cax=cax, label="Attention density")

    fig.text(0.5, 0.045,
              f"Subject {subj.subject_id} ({subj.person}). Blue = where the model's cross-attention concentrated; "
              "red dashed outline = true organ location. Overlap = attention genuinely finds the organ.",
              ha="center", fontsize=8.3, color=PAPER_MUTED)

    return savefig(fig, out_dir / core.replace(" ", "_") / "row18_attention_overlay.png")


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
    style_bars(ax, f"{organ} -- image vs. word contribution to Dice score",
               "Dice-score contribution\n(Shapley credit, in Dice units)")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.tight_layout()
    return savefig(fig, out_dir / organ.replace(" ", "_") / "row21_joint_image_word_shapley.png")


_ROLE_SET_ORDER = ["none", "image", "organ_word", "position_word", "image+organ_word",
                   "image+position_word", "organ_word+position_word", "image+organ_word+position_word"]
_ROLE_SET_LABEL = {
    "none": "Nothing\n(blank prompt, noised image)",
    "image": "Image only",
    "organ_word": "Organ word only",
    "position_word": "Position word only",
    "image+organ_word": "Image +\norgan word",
    "image+position_word": "Image +\nposition word",
    "organ_word+position_word": "Organ word +\nposition word",
    "image+organ_word+position_word": "All three\n(full prompt, real image)",
}


def figure_row21_coalition_dice(organ, coalition_rows, out_dir):
    """The literal version of "which matters more, by Dice score": one bar per one of the 8
    image/organ-word/position-word present-or-absent combinations, showing the ACTUAL Dice score
    VoxTell achieves with exactly that information available -- no Shapley math, no derived credit,
    just "if you only give it X, what Dice do you get." Professional, declarative, self-contained,
    matching the row 18 explainer style: title states the finding's subject, legend has no redundant
    repetition, every bar labeled with its exact value."""
    df = pd.DataFrame(coalition_rows).set_index("roles_present").loc[_ROLE_SET_ORDER].reset_index()
    singleton_or_none = df.roles_present.isin(["none", "image", "organ_word", "position_word"])
    colors = [BLUE if ("image" in r and "organ_word" not in r and "position_word" not in r) else
              ORANGE if ("organ_word" in r and "image" not in r and "position_word" not in r) else
              AQUA if ("position_word" in r and "image" not in r and "organ_word" not in r) else
              PAPER_MUTED if r == "none" else "#9c9c94" for r in df.roles_present]

    full = float(df.loc[df.roles_present == "image+organ_word+position_word", "dice_gt"].iloc[0])
    baseline = float(df.loc[df.roles_present == "none", "dice_gt"].iloc[0])
    singles = {r: float(df.loc[df.roles_present == r, "dice_gt"].iloc[0]) for r in ["image", "organ_word", "position_word"]}
    best_single = max(singles, key=singles.get)
    best_single_label = {"image": "the image alone", "organ_word": "the organ word alone", "position_word": "the position word alone"}[best_single]

    fig = plt.figure(figsize=(10.5, 5.2), dpi=220)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, f"Which Input Matters Most for Segmenting {organ.title()}, by Dice Score",
              ha="center", fontsize=13.5, fontweight="bold", color=PAPER_INK)
    fig.text(0.5, 0.918,
              "Actual Dice score when VoxTell is given only this combination of image / organ word / position word",
              ha="center", fontsize=9.5, color=PAPER_MUTED)

    ax = fig.add_axes([0.08, 0.20, 0.88, 0.62])
    _paper_axes_style(ax)
    x = np.arange(len(df))
    ax.bar(x, df.dice_gt, color=colors, width=0.62, zorder=3)
    for xi, v in zip(x, df.dice_gt):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 5), ha="center",
                     fontsize=9, fontweight="bold", color=PAPER_INK)
    ax.axhline(full, color=PAPER_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.axhline(baseline, color=PAPER_MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_ROLE_SET_LABEL[r] for r in df.roles_present], fontsize=8.6)
    ax.set_ylabel("Dice score vs. ground truth", fontsize=10.5, color=PAPER_INK)
    ax.set_ylim(0, max(1.0, df.dice_gt.max() * 1.15))
    ax.yaxis.grid(True, color=PAPER_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=BLUE, lw=6, label="Image"),
               Line2D([0], [0], color=ORANGE, lw=6, label="Organ word"),
               Line2D([0], [0], color=AQUA, lw=6, label="Position word"),
               Line2D([0], [0], color="#9c9c94", lw=6, label="Combination"),
               Line2D([0], [0], color=PAPER_MUTED, lw=6, label="Nothing")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.6, ncol=1)

    fig.text(0.5, 0.075,
              f"Best single input: {best_single_label} ({singles[best_single]:.3f} Dice) — "
              f"full prompt + real image reaches {full:.3f}, nothing reaches {baseline:.3f}.",
              ha="center", fontsize=10.5, fontweight="bold", color=PAPER_INK)
    fig.text(0.5, 0.025,
              "Dashed line = Dice with everything (full prompt + real image). Dotted line = Dice with nothing "
              "(blank prompt + noised image, i.e. the model's uninformed baseline).",
              ha="center", fontsize=8.3, color=PAPER_MUTED)

    return savefig(fig, out_dir / organ.replace(" ", "_") / "row21_coalition_dice.png")


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
    style_bars(ax, "Row 21 summary -- image vs. word contribution to Dice score, every organ",
               "Dice-score contribution (Shapley credit, Dice units)")
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
    """The headline figure for the project's core question: professional style matching the row 18
    explainer (declarative title, exact values where it matters, a stats box with the overall
    verdict) -- not just a bare per-organ bar chart left for the reader to average in their head."""
    totals = image_vs_word_totals(all_rows)
    t = totals[totals.metric == "f_dice_gt"].copy()
    organs = sorted(t.organ.unique(), key=lambda o: (o.split(None, 1)[1], o.split(None, 1)[0]) if " " in o else (o,))
    t = t.set_index("organ").loc[organs]
    n_word, n_image = int(t.word_wins.sum()), int((~t.word_wins).sum())
    mean_image, mean_word = float(t["image"].mean()), float(t["word"].mean())
    ratio = mean_word / mean_image if mean_image > 1e-9 else float("inf")
    overall = "WORD" if mean_word > mean_image else "IMAGE"

    fig = plt.figure(figsize=(max(10.5, 0.55 * len(organs) + 3), 7.2), dpi=220)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, "Word Information Outweighs Image Content for Most Organs",
              ha="center", fontsize=14, fontweight="bold", color=PAPER_INK)
    fig.text(0.5, 0.925,
              "Dice-score contribution of the image vs. the words (organ name + position), per organ, joint Shapley decomposition",
              ha="center", fontsize=9.8, color=PAPER_MUTED)

    ax = fig.add_axes([0.06, 0.30, 0.72, 0.60])
    _paper_axes_style(ax)
    x = np.arange(len(organs))
    width = 0.36
    ax.bar(x - width / 2, t["image"], width=width, color=BLUE, label="Image", zorder=3)
    ax.bar(x + width / 2, t["word"], width=width, color=ORANGE, label="Word (organ + position)", zorder=3)
    ax.axhline(0, color=PAPER_INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(organs, rotation=45, ha="right", fontsize=8.6)
    ax.set_ylabel("Dice-score contribution\n(Shapley credit)", fontsize=10, color=PAPER_INK)
    ax.yaxis.grid(True, color=PAPER_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    ax_c = fig.add_axes([0.82, 0.30, 0.16, 0.60])
    ax_c.axis("off")
    stat_rows = [
        ("Mean contribution — Image", f"{mean_image:.3f}", BLUE),
        ("Mean contribution — Word", f"{mean_word:.3f}", ORANGE),
        ("Word / Image ratio", f"{ratio:.1f}×", PAPER_INK),
        ("Organs where word wins", f"{n_word} / {len(t)}", PAPER_INK),
        ("Overall", overall, RED if overall == "WORD" else BLUE),
    ]
    y0 = 0.94
    for label, value, color in stat_rows:
        ax_c.text(0.0, y0, label, fontsize=8.4, color=PAPER_MUTED, transform=ax_c.transAxes, va="top", wrap=True)
        ax_c.text(0.0, y0 - 0.10, value, fontsize=13, fontweight="bold", color=color, transform=ax_c.transAxes, va="top")
        y0 -= 0.205
    ax_c.add_patch(plt.Rectangle((-0.10, 0.02), 1.18, 0.98, transform=ax_c.transAxes, fill=False,
                                  edgecolor=PAPER_GRID, linewidth=1.0))

    fig.text(0.5, 0.045,
              "Shapley decomposition over 3 roles (image, organ word, position word), scored against ground truth (f_dice_gt). "
              "\"Word\" = organ word + position word combined.",
              ha="center", fontsize=8.2, color=PAPER_MUTED)

    out_path = savefig(fig, out_dir / "row21_image_vs_word_2way.png")

    print("\n=== Image vs. word contribution (f_dice_gt), per organ ===")
    for organ, row in t.iterrows():
        winner = "WORD" if row.word_wins else "IMAGE"
        print(f"  {organ:28s} image={row['image']:+.4f}  word={row['word']:+.4f}  -> {winner} contributes more")
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
    row21_coal = read_csv(args.results_dir, "row21_coalition_dice", args.suffix)

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
                        overlay_path = figure_row18_attention_overlay(subj, core, maps18, out_dir)
                        if overlay_path:
                            print("wrote", rel(overlay_path, args.repo))
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
                if not row21_coal.empty:
                    rc = row21_coal[(row21_coal.subject_id == subject_id) & (row21_coal.organ == organ)].to_dict("records")
                    if rc:
                        print("wrote", rel(figure_row21_coalition_dice(organ, rc, out_dir), args.repo))
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
