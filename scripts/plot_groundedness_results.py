"""
Local (no GPU, no torch, no voxtell) visualization of the model-internals
investigation results (voxtell-{amos,oasis}-groundedness-investigation-kaggle.ipynb).

Run those notebooks on Kaggle, download `baseline_predictions_<dataset>.csv` from
the Output tab, then run this script against it -- it decodes the RLE-encoded
ground-truth/predicted masks (pure Python/numpy/zlib, same decode_mask_rle() as
the notebook) and overlays them on the ORIGINAL image, which is already present
locally in this repo (<person>/samples/<dataset>/volumes/), so no re-download of
image data is needed -- only the small CSV.

Mirrors scripts/visualize_rotation_gt_pred.py's resampling approach: the decoded
mask lives in the CSV's own RAS-reoriented (canonical) voxel grid, which is a
different array orientation than the raw un-reoriented image loaded here, so the
mask is resampled (nearest-neighbor) onto the raw image's grid before slicing --
guaranteeing the ground-truth and predicted overlays line up with the same
image slice.

Also plots word/role-importance bar charts from the notebooks' attribution CSVs
(word_attribution_/shap_attribution_/lime_attribution_<dataset>.csv) -- pass 1-3 of
them together for a cross-method (exact Shapley vs. SHAP vs. LIME) comparison chart,
auto-detected from the CSV's columns.

Output: local_plots_groundedness/<dataset_name>/
    <subject_id>_<organ>.png             -- mask plots: Original / GT / Prediction, Dice titled
    <subject_id>_<organ>_<combo>.png     -- ablation_masks.csv: one per plotted combo
    word_importance_<csv names>.png      -- attribution CSVs: word/role importance bar chart

Usage:
    # mask plotting
    python scripts/plot_groundedness_results.py baseline_predictions_amos22_mri.csv
    python scripts/plot_groundedness_results.py baseline_predictions_amos22_mri.csv --subject amos_0530
    python scripts/plot_groundedness_results.py ablation_masks_amos22_mri.csv --organ "right kidney" --dropped position

    # word-importance bar chart, single method
    python scripts/plot_groundedness_results.py word_attribution_amos22_mri.csv --organ "right kidney"

    # cross-method comparison chart (exact Shapley vs. SHAP vs. LIME, same organ)
    python scripts/plot_groundedness_results.py word_attribution_amos22_mri.csv shap_attribution_amos22_mri.csv lime_attribution_amos22_mri.csv --organ "right kidney"
"""

import argparse
import base64
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Lexend.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

MASK_GREEN = "#22c55e"
MASK_GREEN_DARK = "#15803d"
MASK_RED = "#ef4444"
MASK_RED_DARK = "#b91c1c"
DICE_NONZERO_COLOR = "#FF7145"
DICE_ZERO_COLOR = "#1E74FF"


def decode_mask_rle(encoded: str, shape) -> np.ndarray:
    """Inverse of the notebooks' encode_mask_rle(); shape = volume_shape_ras split on 'x'."""
    runs = np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=np.uint32)
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    idx = 0
    value = False
    for run_len in runs:
        if value and run_len:
            flat[idx:idx + run_len] = True
        idx += int(run_len)
        value = not value
    return flat.reshape(shape, order="C")


def canonical_affine(path: Path):
    """Affine of the RAS-reoriented (nib.as_closest_canonical) volume -- pairs
    with volume_shape_ras / the decoded masks, cheap since it's header-only."""
    return nib.as_closest_canonical(nib.load(str(path))).affine


def overlay_mask(ax, mask_2d, color, color_dark):
    rgba = np.zeros((*mask_2d.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.where(mask_2d, 0.55, 0.0)
    ax.imshow(rgba, origin="lower")
    if mask_2d.any():
        ax.contour(mask_2d, levels=[0.5], colors=color_dark, linewidths=1.8)


def plot_row(row: pd.Series, out_dir: Path, volume_cache: dict):
    """Works with either baseline_predictions_<dataset>.csv (columns
    pred_mask_rle_b64/dice_gt, one row per subject x organ) or
    ablation_masks_<dataset>.csv (columns mask_rle_b64/f_dice_gt/roles_dropped,
    one row per subject x organ x word-ablation combo) -- auto-detected below."""
    image_path = REPO_ROOT / row["image_path"]
    label_path = REPO_ROOT / row["label_path"]
    if not image_path.exists():
        print(f"[skip] {row['subject_id']}/{row['organ']}: image not found locally at {image_path}")
        return

    is_ablation_row = "mask_rle_b64" in row.index
    pred_mask_col = "mask_rle_b64" if is_ablation_row else "pred_mask_rle_b64"
    dice = float(row["f_dice_gt"] if is_ablation_row else row["dice_gt"])

    cache_key = str(image_path)
    if cache_key not in volume_cache:
        img_nii = nib.load(str(image_path))
        image = img_nii.get_fdata().astype(np.float32)
        axcodes = nib.aff2axcodes(img_nii.affine)
        slice_axis = next((a for a, code in enumerate(axcodes) if code in ("S", "I")), int(np.argmin(image.shape)))
        label = nib.load(str(label_path)).get_fdata().astype(int)
        volume_cache[cache_key] = (img_nii, image, label, slice_axis)
    img_nii, image, label, slice_axis = volume_cache[cache_key]
    gt_mask = label == int(row["label_id"])

    # Slice through the GT mask's own centroid (computed locally, not from the
    # CSV) -- works the same for both CSV formats and keeps every organ/combo
    # centered in frame regardless of where the ablated prediction lands.
    gt_coords = np.nonzero(gt_mask)
    slice_idx = int(round(gt_coords[slice_axis].mean())) if len(gt_coords[0]) else image.shape[slice_axis] // 2
    idx = [slice(None)] * 3
    idx[slice_axis] = slice_idx
    img_slice = image[tuple(idx)]
    gt_slice = gt_mask[tuple(idx)]

    # Predicted mask lives in the CSV's own RAS-reoriented voxel grid (a
    # different orientation than the raw `image` above) -- resample onto the
    # raw image's grid so it lines up with the same slice as ground truth.
    pred_shape = tuple(int(s) for s in row["volume_shape_ras"].split("x"))
    pred_mask_full = decode_mask_rle(row[pred_mask_col], pred_shape)
    affine_key = ("affine", row["image_path"])
    if affine_key not in volume_cache:
        volume_cache[affine_key] = canonical_affine(image_path)
    pred_affine = volume_cache[affine_key]
    pred_mask_canon_img = nib.Nifti1Image(pred_mask_full.transpose(2, 1, 0).astype(np.uint8), pred_affine)
    pred_mask_raw = nibabel.processing.resample_from_to(pred_mask_canon_img, img_nii, order=0).get_fdata().astype(bool)
    pred_slice = pred_mask_raw[tuple(idx)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, label_text in zip(axes, ("Original", "Ground Truth", "Prediction")):
        ax.imshow(img_slice.T, cmap="gray", origin="lower")
        ax.axis("off")
        ax.set_title(label_text, fontsize=13, fontweight="bold", pad=10)

    overlay_mask(axes[1], gt_slice.T, MASK_GREEN, MASK_GREEN_DARK)
    overlay_mask(axes[2], pred_slice.T, MASK_RED, MASK_RED_DARK)

    dice_color = DICE_NONZERO_COLOR if dice > 0 else DICE_ZERO_COLOR
    axes[2].text(0.5, -0.08, f"Dice = {dice:.3f}", transform=axes[2].transAxes,
                 ha="center", fontsize=12, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=dice_color, edgecolor="none", alpha=0.95))

    title = f'"{row["prompt_text"]}"  (subject={row["subject_id"]})'
    organ_slug = str(row["organ"]).replace(" ", "_")
    if is_ablation_row:
        dropped = row["roles_dropped"] or "(none -- baseline)"
        title += f"\ndropped roles: {dropped}"
        combo_tag = "baseline" if row["is_baseline"] else f"drop-{str(row['roles_dropped']).replace(',', '-') or 'none'}"
        out_name = f"{row['subject_id']}_{organ_slug}_{combo_tag}.png"
    else:
        out_name = f"{row['subject_id']}_{organ_slug}.png"

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.12, wspace=0.05)

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


# Which value column identifies each attribution CSV, and its legend label.
ATTRIBUTION_VALUE_COL = {
    "shapley_value": "Exact Shapley (row 5)",
    "shap_value": "SHAP KernelExplainer (row 7)",
    "lime_weight": "LIME (row 6)",
}


def load_attribution_series(csv_path: Path, subject, organ, metric):
    """Returns (word/role -> mean |importance|, legend label), or (None, None) if
    csv_path isn't a recognized attribution CSV (word_attribution/shap_attribution/
    lime_attribution_<dataset>.csv)."""
    df = pd.read_csv(csv_path)
    value_col = next((c for c in ATTRIBUTION_VALUE_COL if c in df.columns), None)
    if value_col is None:
        return None, None
    if subject:
        df = df[df["subject_id"] == subject]
    if organ:
        df = df[df["organ"] == organ]
    if "metric" in df.columns:
        df = df[df["metric"] == (metric or "f_dice_gt")]
    word_col = "word" if "word" in df.columns else "role"
    if df.empty:
        return None, ATTRIBUTION_VALUE_COL[value_col]
    agg = df.groupby(word_col)[value_col].apply(lambda s: s.abs().mean()).sort_values(ascending=False)
    return agg, ATTRIBUTION_VALUE_COL[value_col]


def plot_word_importance(csv_paths, subject, organ, metric, out_dir: Path):
    """Grouped bar chart of mean |importance| per word/role, one bar-group color per
    method -- pass 1 attribution CSV for a single method's chart, or up to 3
    (word_attribution + shap_attribution + lime_attribution) for a side-by-side
    cross-method comparison chart."""
    series_list = []
    for p in csv_paths:
        agg, label = load_attribution_series(p, subject, organ, metric)
        if label is None:
            print(f"[skip] {p}: not a recognized attribution CSV (no shapley_value/shap_value/lime_weight column)")
            continue
        if agg is None:
            print(f"[skip] {p}: no rows match the given filters")
            continue
        series_list.append((label, agg))
    if not series_list:
        raise SystemExit("no attribution data to plot from the given CSV(s)/filters")

    all_words = sorted(set().union(*(s.index for _, s in series_list)), key=lambda w: -max(
        (s.get(w, 0.0) for _, s in series_list)))
    colors = ["#2563eb", "#ef4444", "#16a34a", "#f59e0b"]
    n = len(series_list)
    x = np.arange(len(all_words))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(6, len(all_words) * 0.9 * n / 2), 5))
    for i, (label, agg) in enumerate(series_list):
        vals = [agg.get(w, 0.0) for w in all_words]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=label, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(all_words, rotation=45, ha="right")
    ax.set_ylabel(f"mean |importance|  ({metric or 'f_dice_gt'})")
    title = "Word/role importance"
    if subject:
        title += f" -- {subject}"
    if organ:
        title += f" / {organ}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    if n > 1:
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()

    tag = "_".join(Path(p).stem for p in csv_paths)
    out_path = out_dir / f"word_importance_{tag}.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_paths", type=Path, nargs="+",
                     help="one CSV for mask plotting (baseline_predictions_<dataset>.csv or "
                          "ablation_masks_<dataset>.csv), OR one-to-three attribution CSVs "
                          "(word_attribution_/shap_attribution_/lime_attribution_<dataset>.csv) "
                          "for a word-importance bar chart -- pass several together for a "
                          "cross-method comparison chart")
    ap.add_argument("--subject", default=None, help="only plot this subject_id")
    ap.add_argument("--organ", default=None, help="only plot this organ")
    ap.add_argument("--metric", default=None,
                     help="attribution CSVs only: which of f_dice_gt/f_dice_self/f_centroid_shift/f_volume "
                          "to chart (default: f_dice_gt)")
    ap.add_argument("--dropped", default=None,
                     help="ablation_masks.csv only: only plot combos whose roles_dropped contains this role "
                          "(e.g. 'position' for 'what did dropping left/right look like'); "
                          "pass '' (empty string) to plot only baseline rows")
    ap.add_argument("--all-combos", action="store_true",
                     help="ablation_masks.csv only: plot EVERY combo (can be dozens of PNGs per organ) "
                          "instead of just the baseline row (the default, to avoid an accidental huge batch)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: local_plots_groundedness/<dataset_name>/")
    args = ap.parse_args()

    first_df = pd.read_csv(args.csv_paths[0])
    dataset_name = first_df["dataset"].iloc[0]
    out_dir = args.out_dir or (REPO_ROOT / "local_plots_groundedness" / dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if any(c in first_df.columns for c in ATTRIBUTION_VALUE_COL):
        plot_word_importance(args.csv_paths, args.subject, args.organ, args.metric, out_dir)
        return

    if len(args.csv_paths) > 1:
        raise SystemExit("mask plotting takes exactly one CSV (baseline_predictions_* or ablation_masks_*)")
    df = first_df
    is_ablation_csv = "mask_rle_b64" in df.columns
    if args.subject:
        df = df[df["subject_id"] == args.subject]
    if args.organ:
        df = df[df["organ"] == args.organ]

    if is_ablation_csv:
        if args.dropped is not None:
            df = df[df["roles_dropped"].fillna("").apply(lambda s: args.dropped in str(s).split(","))] \
                if args.dropped else df[df["is_baseline"]]
        elif not args.all_combos:
            print("(ablation_masks.csv detected -- plotting only the baseline row per subject/organ by default; "
                  "pass --all-combos for every ablation combo, or --dropped <role> for a specific one)")
            df = df[df["is_baseline"]]

    if df.empty:
        raise SystemExit("no rows match the given filters")

    volume_cache = {}
    print(f"plotting {len(df)} rows from {args.csv_paths[0].name}...")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"({i}/{len(df)}) {row['subject_id']} / {row['organ']}", flush=True)
        plot_row(row, out_dir, volume_cache)

    print(f"\n[done] wrote plots -> {out_dir}")


if __name__ == "__main__":
    main()
