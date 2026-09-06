"""Build clean, presentation-ready ground-truth overlay PNGs for a dataset.

Usage:
    python scripts/visualize_dataset.py acdc
    python scripts/visualize_dataset.py isic_task2
    python scripts/visualize_dataset.py chaos

For each sample_id in dataset/<name>/manifest_<name>.csv, this reads every distinct
region's mask (skipping the combined "all"/P_ALL row, since that has no single mask),
overlays each region in its own color on top of the original image, marks each region's
centroid with a dot, draws the left/right and upper/lower midlines for the wrong-side
check, and renders a title + a horizontal row of color-coded legend chips underneath --
styled to be dropped straight into a slide. Output goes to
dataset/<name>/visual_check/<sample_id>.png. This is a viewing aid only, not part of the
manifest/scoring pipeline; the original (unannotated) image already lives alongside it
at dataset/<name>/<sample_id>/<sample_id>.png.
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
]

BG_COLOR = (255, 255, 255)
TITLE_COLOR = (30, 30, 30)


def load_manifest(dataset_name):
    manifest_path = os.path.join("dataset", dataset_name, f"manifest_{dataset_name}.csv")
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def region_color_map(regions):
    return {region: PALETTE[i % len(PALETTE)] for i, region in enumerate(sorted(regions))}


def get_font(size, bold=False):
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def text_size(draw, text, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def readable_text_color(bg_color):
    r, g, b = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20) if luminance > 150 else (255, 255, 255)


def display_label(region):
    return region.replace("_", " ")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/visualize_dataset.py <dataset_name>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    dataset_dir = os.path.join("dataset", dataset_name)
    out_dir = os.path.join(dataset_dir, "visual_check")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_manifest(dataset_name)
    isolated_rows = [r for r in rows if r["prompt_id"] == "P0"]

    by_sample = {}
    for r in isolated_rows:
        by_sample.setdefault(r["sample_id"], []).append(r)

    all_regions = {r["region"] for r in isolated_rows}
    color_map = region_color_map(all_regions)

    title_text = "Ground truth overlay"

    for sample_id, sample_rows in sorted(by_sample.items()):
        image_path = os.path.join(dataset_dir, sample_rows[0]["image_path"])
        base_img = Image.open(image_path).convert("RGB")
        width, height = base_img.size

        # scale everything to this image's resolution so it stays legible whether the
        # source is a ~256px MRI slice or a ~4000px dermoscopy photo
        scale = max(1, round(min(width, height) / 300))
        title_font = get_font(16 * scale, bold=True)
        chip_font = get_font(13 * scale)
        dot_r = 4 * scale
        line_w = max(1, scale)
        pad = 10 * scale

        # --- build the overlay image itself ---
        arr = np.array(base_img).astype(np.float32)
        centroids = []
        for r in sample_rows:
            region = r["region"]
            mask_path = os.path.join(dataset_dir, r["mask_path"])
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            color = np.array(color_map[region], dtype=np.float32)
            arr[mask] = 0.4 * arr[mask] + 0.6 * color
            centroids.append((float(r["gt_centroid_x"]), float(r["gt_centroid_y"]), color_map[region]))

        overlay = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(overlay)
        draw.line([(width // 2, 0), (width // 2, height)], fill=(255, 255, 255), width=line_w)
        draw.line([(0, height // 2), (width, height // 2)], fill=(255, 255, 255), width=line_w)
        for cx, cy, color in centroids:
            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                outline=(255, 255, 255),
                fill=color,
                width=line_w,
            )

        # --- measure the legend row (color square + plain text, side by side) ---
        tmp_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        swatch = chip_font.size
        item_gap = 6 * scale   # space between swatch and its label
        group_gap = 18 * scale  # space between legend items
        legend_h = swatch
        legend_specs = []
        for region in sorted({r["region"] for r in sample_rows}):
            color = color_map[region]
            label = display_label(region)
            tw, th = text_size(tmp_draw, label, chip_font)
            legend_h = max(legend_h, th)
            legend_specs.append((label, color, tw))
        legend_row_w = sum(swatch + item_gap + tw for _, _, tw in legend_specs) + group_gap * (
            len(legend_specs) - 1
        )

        title_w, title_h = text_size(tmp_draw, title_text, title_font)

        canvas_w = max(width, legend_row_w, title_w) + 2 * pad
        title_block_h = title_h + int(pad * 0.5)
        legend_block_h = legend_h + int(pad * 0.5)
        canvas_h = title_block_h + height + legend_block_h + int(pad * 1.5)

        canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
        cdraw = ImageDraw.Draw(canvas)

        # title, centered
        cdraw.text(((canvas_w - title_w) // 2, pad // 2), title_text, fill=TITLE_COLOR, font=title_font)

        # image, centered
        img_x = (canvas_w - width) // 2
        img_y = pad // 2 + title_block_h
        canvas.paste(overlay, (img_x, img_y))
        cdraw.rectangle(
            [img_x - 1, img_y - 1, img_x + width, img_y + height],
            outline=(210, 210, 210),
            width=1,
        )

        # legend: color square + plain text, side by side
        x = (canvas_w - legend_row_w) // 2
        y = img_y + height + int(pad * 0.5)
        for label, color, tw in legend_specs:
            cdraw.rectangle([x, y, x + swatch, y + swatch], fill=color)
            cdraw.text((x + swatch + item_gap, y - 1), label, fill=TITLE_COLOR, font=chip_font)
            x += swatch + item_gap + tw + group_gap

        canvas.save(os.path.join(out_dir, f"{sample_id}.png"))

    print(f"Wrote {len(by_sample)} visual-check images to {out_dir}")


if __name__ == "__main__":
    main()
