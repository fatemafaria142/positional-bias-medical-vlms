"""
Aggregates every model's per-row results into one dataset-wise summary table
(rows = models, columns = the RQ1-RQ3 bias metrics from
position_bias_research_plan.md's Comparison Specification).

Every model is scored by the SAME score() function here, computed directly
from raw centroid/bbox geometry plus image size. Earlier versions of this
script trusted the per-model *_no_shift columns already baked into
molmo2/owlv2/qwen25vl/moondream2's CSVs for 4 of the 11 models, and
recomputed the other 7 independently -- those two paths turned out to use
different shift-tolerance definitions (the baked-in columns used ~5% of the
image DIAGONAL, this script and scripts/score_predictions.py both use 5% of
image width/height), which made the "does it shift" numbers not comparable
across all 11 models. This version ignores every precomputed *_correct_side
/ *_no_shift / *_dist_norm / *_iou column and derives everything from raw
geometry only, so all 11 rows are computed identically.

Output: Results/Findings/FINDINGS_DATA.md, one table per dataset.
"""

from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "Results"
FINDINGS_DIR = RESULTS_DIR / "Findings"
DATASETS = ["lgg", "kvasir", "isic"]
SHIFT_TOLERANCE_FRAC = 0.05  # 5% of image width/height, per position_bias_research_plan.md

TARGET_NOUN = {"lgg": "tumor", "kvasir": "polyp", "isic": "lesion"}

OUTPUT_TIER = {
    "CLIPSeg": "mask",
    "Florence-2": "bbox",
    "Kosmos-2": "bbox",
    "PaliGemma 2": "mask",
    "Grounding DINO": "bbox",
    "LocateAnything-3B": "bbox",
    "NorthMicroVision": "bbox",
    "Molmo2-8B": "point",
    "OWLv2": "bbox",
    "Qwen2.5-VL-7B": "bbox",
    "Moondream2 (detect)": "bbox",
    "Moondream2 (point)": "point",
}


def image_size_lookup(ds: str):
    manifest = pd.read_csv(REPO_ROOT / "samples" / ds / f"manifest_{ds}.csv")
    sizes = {}
    for _, r in manifest.iterrows():
        if r["sample_id"] in sizes:
            continue
        with Image.open(REPO_ROOT / "samples" / ds / r["image_path"]) as im:
            sizes[r["sample_id"]] = im.size
    return sizes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else float("nan")


def score(df, cx_col, cy_col, sizes, bbox_cols=None, valid_mask=None):
    """Scores one model's raw predictions against gt_centroid_x/y (and
    gt_bbox_* if bbox_cols is given), independent of any *_correct_side /
    *_no_shift / *_dist_norm / *_iou columns already in df."""
    df = df.copy()
    if valid_mask is None:
        valid_mask = df[cx_col].notna()
    df["_valid"] = valid_mask
    df["_w"] = df["sample_id"].map(lambda s: sizes[s][0])
    df["_h"] = df["sample_id"].map(lambda s: sizes[s][1])
    df["_diag"] = (df["_w"] ** 2 + df["_h"] ** 2) ** 0.5

    df["_pred_lr"] = pd.NA
    df["_pred_ud"] = pd.NA
    v = df["_valid"]
    df.loc[v, "_pred_lr"] = (df.loc[v, cx_col] < df.loc[v, "_w"] / 2).map({True: "left", False: "right"})
    df.loc[v, "_pred_ud"] = (df.loc[v, cy_col] < df.loc[v, "_h"] / 2).map({True: "upper", False: "lower"})

    prompted_lr = df["prompt_id"].map({"P1": "left", "P2": "right"})
    prompted_ud = df["prompt_id"].map({"P3": "upper", "P4": "lower"})
    is_positional = df["prompt_id"] != "P0"

    side_correct = pd.Series(pd.NA, index=df.index, dtype="object")
    m1 = prompted_lr.notna() & v
    m2 = prompted_ud.notna() & v
    side_correct[m1] = df.loc[m1, "_pred_lr"] == prompted_lr[m1]
    side_correct[m2] = df.loc[m2, "_pred_ud"] == prompted_ud[m2]
    df["side_correct"] = side_correct

    df["dist_norm"] = pd.NA
    df.loc[v, "dist_norm"] = (
        ((df.loc[v, cx_col] - df.loc[v, "gt_centroid_x"]) ** 2 +
         (df.loc[v, cy_col] - df.loc[v, "gt_centroid_y"]) ** 2) ** 0.5 / df.loc[v, "_diag"]
    )
    # A normalized distance above ~sqrt(2) (corner-to-corner) is only possible if the
    # predicted coordinates were parsed wrong (e.g. NorthMicroVision occasionally emits
    # coordinates far outside the image); treat those as unusable rather than real outliers.
    df.loc[df["dist_norm"] > 2.0, "dist_norm"] = pd.NA

    p0 = df[df["prompt_id"] == "P0"].set_index(["dataset", "sample_id"])[[cx_col, cy_col]]
    p0 = p0[~p0.index.duplicated()]
    key = list(zip(df["dataset"], df["sample_id"]))
    p0x = pd.Series([p0[cx_col].get(k, pd.NA) for k in key], index=df.index)
    p0y = pd.Series([p0[cy_col].get(k, pd.NA) for k in key], index=df.index)
    shift_px = ((df[cx_col] - p0x) ** 2 + (df[cy_col] - p0y) ** 2) ** 0.5
    tolerance = SHIFT_TOLERANCE_FRAC * df[["_w", "_h"]].max(axis=1)
    df["no_shift"] = pd.NA
    nm = is_positional & v
    df.loc[nm, "no_shift"] = shift_px[nm] <= tolerance[nm]

    if bbox_cols:
        df["iou"] = pd.NA
        df.loc[v, "iou"] = df.loc[v].apply(
            lambda r: iou(
                (r[bbox_cols[0]], r[bbox_cols[1]], r[bbox_cols[2]], r[bbox_cols[3]]),
                (r["gt_bbox_xmin"], r["gt_bbox_ymin"], r["gt_bbox_xmax"], r["gt_bbox_ymax"]),
            ),
            axis=1,
        )
    else:
        df["iou"] = pd.NA

    df["detected"] = v
    return df


PLAN_FORMAT_FILES = {
    "CLIPSeg": "results_clipseg_{ds}.csv",
    "Florence-2": "results_florence2_{ds}.csv",
    "Kosmos-2": "results_kosmos2_{ds}.csv",
    "PaliGemma 2": "results_paligemma_{ds}.csv",
    "Grounding DINO": "results_grounding_dino_{ds}.csv",
    "LocateAnything-3B": "results_locateanything3b_{ds}.csv",
    "NorthMicroVision": "results_northmicrovision_{ds}.csv",
}


def load_all(ds: str, sizes: dict) -> dict:
    out = {}

    for model, pattern in PLAN_FORMAT_FILES.items():
        df = pd.read_csv(RESULTS_DIR / ds / pattern.format(ds=ds))
        has_bbox = {"pred_bbox_xmin", "pred_bbox_ymin", "pred_bbox_xmax", "pred_bbox_ymax", "gt_bbox_xmin"}.issubset(df.columns)
        bbox_cols = ("pred_bbox_xmin", "pred_bbox_ymin", "pred_bbox_xmax", "pred_bbox_ymax") if has_bbox else None
        out[model] = score(df, "pred_centroid_x", "pred_centroid_y", sizes, bbox_cols=bbox_cols)

    molmo = pd.read_csv(RESULTS_DIR / ds / f"{ds}_molmo2_results.csv")
    out["Molmo2-8B"] = score(molmo, "molmo_x", "molmo_y", sizes)

    owlv2 = pd.read_csv(RESULTS_DIR / ds / f"{ds}_owlv2_results.csv")
    out["OWLv2"] = score(owlv2, "owlv2_cx", "owlv2_cy", sizes,
                          bbox_cols=("owlv2_xmin", "owlv2_ymin", "owlv2_xmax", "owlv2_ymax"),
                          valid_mask=~owlv2["owlv2_is_degenerate"].astype(bool))

    qwen = pd.read_csv(RESULTS_DIR / ds / f"{ds}_qwen25vl_results.csv")
    out["Qwen2.5-VL-7B"] = score(qwen, "qwen_cx", "qwen_cy", sizes,
                                  bbox_cols=("qwen_xmin", "qwen_ymin", "qwen_xmax", "qwen_ymax"),
                                  valid_mask=~qwen["qwen_is_degenerate"].astype(bool))

    md = pd.read_csv(RESULTS_DIR / ds / f"{ds}_moondream2_results.csv")
    out["Moondream2 (detect)"] = score(md, "md_detect_cx", "md_detect_cy", sizes,
                                        bbox_cols=("md_detect_xmin", "md_detect_ymin", "md_detect_xmax", "md_detect_ymax"),
                                        valid_mask=~md["md_detect_is_degenerate"].astype(bool))
    out["Moondream2 (point)"] = score(md, "md_point_x", "md_point_y", sizes,
                                       valid_mask=md["md_point_x"].notna())

    return out


def summarize(df: pd.DataFrame, model: str) -> dict:
    positional = df[df["prompt_id"] != "P0"]
    n_side_known = positional["side_correct"].notna().sum()
    n_shift_known = positional["no_shift"].notna().sum()

    wrong_side_rate = (1 - positional["side_correct"].mean()) if n_side_known else float("nan")
    no_shift_rate = positional["no_shift"].mean() if n_shift_known else float("nan")
    mean_dist = df["dist_norm"].mean()
    mean_iou = df["iou"].mean() if df["iou"].notna().any() else float("nan")
    detect_rate = df["detected"].mean()

    return {
        "Model": model,
        "Output": OUTPUT_TIER[model],
        "Detect Rate": detect_rate,
        "Wrong-Side Rate (P1-P4)": wrong_side_rate,
        "No-Shift Rate (P1-P4)": no_shift_rate,
        "Mean Norm. Dist.": mean_dist,
        "Mean IoU": mean_iou,
    }


def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.1%}"


def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.3f}"


def build_dataset_table(ds: str) -> tuple[pd.DataFrame, str]:
    sizes = image_size_lookup(ds)
    rows = [summarize(df, model) for model, df in load_all(ds, sizes).items()]

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("Wrong-Side Rate (P1-P4)", na_position="last")

    lines = [
        f"### {ds.upper()}: target is \"the {TARGET_NOUN[ds]}\"",
        "",
        "| Model | Output | Detect Rate | Wrong-Side Rate (P1-P4) | No-Shift Rate (P1-P4) | Mean Norm. Distance | Mean IoU |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['Model']} | {r['Output']} | {fmt_pct(r['Detect Rate'])} | "
            f"{fmt_pct(r['Wrong-Side Rate (P1-P4)'])} | {fmt_pct(r['No-Shift Rate (P1-P4)'])} | "
            f"{fmt_num(r['Mean Norm. Dist.'])} | {fmt_num(r['Mean IoU'])} |"
        )
    return summary, "\n".join(lines)


def main():
    sections = []
    for ds in DATASETS:
        _, table_md = build_dataset_table(ds)
        sections.append(table_md)

    FINDINGS_DIR.mkdir(exist_ok=True)
    out_path = FINDINGS_DIR / "FINDINGS_DATA.md"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(sections) + "\n")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
