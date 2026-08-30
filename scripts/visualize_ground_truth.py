"""
Draws the ground-truth centroid, bbox, and side label directly onto each
sampled image, so you can visually confirm gt_side_lr / gt_side_ud are
correct -- rather than trusting numbers in a CSV.

Usage:
    python scripts/visualize_ground_truth.py lgg
    python scripts/visualize_ground_truth.py kvasir
    python scripts/visualize_ground_truth.py isic

Output: samples/<dataset>/visual_check/<sample_id>.png -- the original
image with a red dot at the centroid, a red box at the bbox, and the
side_lr/side_ud labels printed in the corner, plus a line marking the
image's true midline and midheight for reference.
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent


def visualize(dataset_name: str):
    sample_dir = REPO_ROOT / "samples" / dataset_name
    manifest_path = sample_dir / f"manifest_{dataset_name}.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}\nRun scripts/sample_{dataset_name}.py first.")

    out_dir = sample_dir / "visual_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))

    # Only need one row per image (P0) -- ground truth is the same across all 5 prompts.
    seen = set()
    p0_rows = [r for r in rows if r["prompt_id"] == "P0" and not (r["sample_id"] in seen or seen.add(r["sample_id"]))]

    for r in p0_rows:
        img_path = sample_dir / r["image_path"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Reference lines: true midline (vertical) and midheight (horizontal).
        draw.line([(w / 2, 0), (w / 2, h)], fill="yellow", width=1)
        draw.line([(0, h / 2), (w, h / 2)], fill="yellow", width=1)

        # Ground-truth bbox.
        bbox = (float(r["gt_bbox_xmin"]), float(r["gt_bbox_ymin"]), float(r["gt_bbox_xmax"]), float(r["gt_bbox_ymax"]))
        draw.rectangle(bbox, outline="red", width=3)

        # Ground-truth centroid.
        cx, cy = float(r["gt_centroid_x"]), float(r["gt_centroid_y"])
        radius = 5
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="red")

        # Label: what the code claims this image's ground truth side is.
        label = f"{r['sample_id']} | {r['gt_side_lr']} / {r['gt_side_ud']}"
        draw.rectangle((0, 0, draw.textlength(label) + 8, 16), fill="black")
        draw.text((4, 2), label, fill="white")

        img.save(out_dir / f"{r['sample_id']}.png")

    print(f"[ok] {dataset_name}: wrote {len(p0_rows)} annotated images -> {out_dir}")
    print("Open a few of these and check: does the red dot/box actually sit on the "
          "left/right and upper/lower side the label claims, relative to the yellow "
          "midline/midheight lines?")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("lgg", "kvasir", "isic"):
        raise SystemExit("Usage: python scripts/visualize_ground_truth.py [lgg|kvasir|isic]")
    visualize(sys.argv[1])
