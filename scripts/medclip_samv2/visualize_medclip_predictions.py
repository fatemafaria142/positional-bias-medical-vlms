"""Build polished 4-panel (Original | Mask | Ground Truth | Prediction) comparison PNGs
from MedCLIP-SAMv2 results.

Usage:
    python scripts/medclip_samv2/visualize_medclip_predictions.py acdc
    python scripts/medclip_samv2/visualize_medclip_predictions.py isic_task2
    python scripts/medclip_samv2/visualize_medclip_predictions.py chaos

Reads Results/medclip_samv2/<dataset>_results.csv (written by run_medclip_samv2.py).

Color formation (fixed, consistent across every image so it's learnable at a glance):
  - Ground truth = GREEN
  - Prediction   = RED
Each panel's label is tinted to match what it shows, so the color key doubles as the
panel title -- no separate legend needed.

Panels (each a rounded white "card" with a soft drop shadow):
  1. Original      -- the unannotated image
  2. Mask          -- all of that image's region masks merged into ONE plain mask
                       (white-on-black), no image underneath, so the true combined
                       shape is legible on its own
  3. Ground Truth  -- that same combined mask overlaid on the image in green
  4. Prediction    -- the model's single predicted mask overlaid on the image in red

Title = the exact prompt sent to the model (wrapped to fit). Below it, a colored Dice
badge (green/amber/red by score) is the one score kept front and center -- IoU dropped,
sample id dropped, per request.

Output: Results/medclip_samv2/<dataset>/visual_check/<sample_id>.png
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GT_COLOR = (39, 174, 96)       # green -- ground truth, everywhere
PRED_COLOR = (231, 76, 60)     # red   -- prediction, everywhere
MASK_FG = (255, 255, 255)      # plain mask panel: white shape
MASK_BG = (24, 24, 27)         # plain mask panel: near-black background
PAGE_BG = (241, 243, 246)      # whole-canvas background (soft gray, not stark white)
CARD_BG = (255, 255, 255)      # each panel's card background
TITLE_COLOR = (24, 24, 27)
LABEL_NEUTRAL = (55, 65, 81)
SHADOW_COLOR = (15, 23, 42)

DICE_GOOD = (39, 174, 96)      # >= 0.5
DICE_MID = (243, 156, 18)      # 0.2 - 0.5
DICE_BAD = (192, 57, 43)       # < 0.2


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


def union_mask(dataset_dir, mask_paths):
    masks = [np.array(Image.open(os.path.join(dataset_dir, p)).convert("L")) > 0 for p in mask_paths.split(";")]
    out = masks[0]
    for m in masks[1:]:
        out = np.logical_or(out, m)
    return out


def mask_centroid(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def dice_color(dice):
    if dice >= 0.5:
        return DICE_GOOD
    if dice >= 0.2:
        return DICE_MID
    return DICE_BAD


def draw_crosshair_and_dot(panel, mask_bool, dot_color, line_w, dot_r):
    width, height = panel.size
    draw = ImageDraw.Draw(panel)
    draw.line([(width // 2, 0), (width // 2, height)], fill=(255, 255, 255), width=line_w)
    draw.line([(0, height // 2), (width, height // 2)], fill=(255, 255, 255), width=line_w)
    if mask_bool is not None:
        c = mask_centroid(mask_bool)
        if c is not None:
            cx, cy = c
            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                outline=(255, 255, 255), fill=dot_color, width=line_w,
            )


def make_overlay_panel(base_img, mask_bool, color, line_w, dot_r):
    arr = np.array(base_img).astype(np.float32)
    if mask_bool is not None and mask_bool.any():
        arr[mask_bool] = 0.4 * arr[mask_bool] + 0.6 * np.array(color, dtype=np.float32)
    panel = Image.fromarray(arr.astype(np.uint8))
    draw_crosshair_and_dot(panel, mask_bool, color, line_w, dot_r)
    return panel


def make_plain_mask_panel(mask_bool, size, line_w, dot_r):
    width, height = size
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = MASK_BG
    if mask_bool is not None:
        arr[mask_bool] = MASK_FG
    panel = Image.fromarray(arr)
    draw_crosshair_and_dot(panel, mask_bool, GT_COLOR, line_w, dot_r)
    return panel


def rounded_mask(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return m


def paste_card(canvas, content_img, x, y, radius, shadow_pad, shadow_blur):
    """Paste `content_img` at (x, y) as a rounded card with a soft drop shadow."""
    w, h = content_img.size

    shadow = Image.new("RGBA", (w + shadow_pad * 2, h + shadow_pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [shadow_pad, shadow_pad, shadow_pad + w, shadow_pad + h], radius=radius, fill=(*SHADOW_COLOR, 90)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    canvas.paste(shadow, (x - shadow_pad, y - shadow_pad + shadow_blur // 2), shadow)

    card = Image.new("RGB", (w, h), CARD_BG)
    card.paste(content_img, (0, 0))
    m = rounded_mask((w, h), radius)
    canvas.paste(card, (x, y), m)


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_size(draw, candidate, font)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/medclip_samv2/visualize_medclip_predictions.py <dataset_name>")
        sys.exit(1)
    dataset_name = sys.argv[1]

    dataset_dir = os.path.join(REPO_ROOT, "dataset", dataset_name)
    results_path = os.path.join(REPO_ROOT, "Results", "medclip_samv2", f"{dataset_name}_results.csv")
    out_dir = os.path.join(REPO_ROOT, "Results", "medclip_samv2", dataset_name, "visual_check")
    os.makedirs(out_dir, exist_ok=True)

    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_written = 0
    for row in rows:
        if not row["pred_mask_path"]:
            continue

        image_path = os.path.join(dataset_dir, row["image_path"])
        base_img = Image.open(image_path).convert("RGB")
        width, height = base_img.size

        if row["prompt_id"] == "P_ALL":
            gt_mask = union_mask(dataset_dir, row["gt_mask_path"])
        else:
            gt_mask = np.array(Image.open(os.path.join(dataset_dir, row["gt_mask_path"])).convert("L")) > 0

        pred_path = os.path.join(REPO_ROOT, row["pred_mask_path"])
        pred_mask = np.array(Image.open(pred_path).convert("L")) > 127
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.fromarray(pred_mask.astype(np.uint8) * 255).resize((width, height))
            ) > 127

        scale = max(1, round(min(width, height) / 300))
        title_font = get_font(17 * scale, bold=True)
        badge_font = get_font(20 * scale, bold=True)
        label_font = get_font(18 * scale, bold=True)
        line_w = max(1, scale)
        dot_r = 4 * scale
        pad = 16 * scale
        gap = 22 * scale
        radius = 10 * scale
        shadow_pad = 8 * scale
        shadow_blur = 6 * scale
        label_gap = 8 * scale

        panels = [
            ("Original", LABEL_NEUTRAL, make_overlay_panel(base_img, None, GT_COLOR, line_w, dot_r)),
            ("Mask", LABEL_NEUTRAL, make_plain_mask_panel(gt_mask, (width, height), line_w, dot_r)),
            ("Ground Truth", GT_COLOR, make_overlay_panel(base_img, gt_mask, GT_COLOR, line_w, dot_r)),
            ("Prediction", PRED_COLOR, make_overlay_panel(base_img, pred_mask, PRED_COLOR, line_w, dot_r)),
        ]
        n_cols, n_rows = 2, 2

        tmp_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        row_total_w = width * n_cols + gap * (n_cols - 1)
        max_title_width = max(row_total_w, 480 * scale)

        title_lines = wrap_text(tmp_draw, row["prompt_text"], title_font, max_title_width)
        dice_val = float(row["dice"])
        badge_text = f"Dice {dice_val:.3f}"
        badge_color = dice_color(dice_val)

        title_line_h = text_size(tmp_draw, "Ay", title_font)[1]
        title_h = title_line_h * len(title_lines) + int(4 * scale) * (len(title_lines) - 1)
        badge_text_w, badge_text_h = text_size(tmp_draw, badge_text, badge_font)
        badge_pad_x, badge_pad_y = 18 * scale, 8 * scale
        badge_w = badge_text_w + 2 * badge_pad_x
        badge_h = badge_text_h + 2 * badge_pad_y
        label_h = text_size(tmp_draw, "Ay", label_font)[1]

        canvas_w = max(row_total_w, badge_w) + 2 * pad + 2 * shadow_pad
        header_h = title_h + int(pad * 0.6) + badge_h
        label_block_h = label_h + label_gap
        card_h = label_block_h + height
        canvas_h = (
            pad + header_h + int(pad * 0.8)
            + card_h * n_rows + gap * (n_rows - 1)
            + pad + 2 * shadow_pad
        )

        canvas = Image.new("RGB", (canvas_w, canvas_h), PAGE_BG)
        cdraw = ImageDraw.Draw(canvas)

        y = pad
        for line in title_lines:
            lw, _ = text_size(cdraw, line, title_font)
            cdraw.text(((canvas_w - lw) // 2, y), line, fill=TITLE_COLOR, font=title_font)
            y += title_line_h + int(4 * scale)
        y += int(pad * 0.3)

        badge_x = (canvas_w - badge_w) // 2
        cdraw.rounded_rectangle(
            [badge_x, y, badge_x + badge_w, y + badge_h], radius=badge_h // 2, fill=badge_color
        )
        cdraw.text(
            (badge_x + badge_pad_x, y + badge_pad_y - int(1 * scale)),
            badge_text, fill=(255, 255, 255), font=badge_font,
        )

        grid_x0 = (canvas_w - row_total_w) // 2
        grid_y0 = pad + header_h + int(pad * 0.8) + shadow_pad

        for idx, (label, label_color, panel_img) in enumerate(panels):
            col = idx % n_cols
            grid_row = idx // n_cols
            x = grid_x0 + col * (width + gap)
            y_top = grid_y0 + grid_row * (card_h + gap)

            lw, _ = text_size(cdraw, label, label_font)
            cdraw.text((x + (width - lw) // 2, y_top), label, fill=label_color, font=label_font)

            y_img = y_top + label_block_h
            paste_card(canvas, panel_img, x, y_img, radius, shadow_pad, shadow_blur)

        out_path = os.path.join(out_dir, f"{row['sample_id']}.png")
        canvas.save(out_path)
        n_written += 1

    print(f"Wrote {n_written} comparison images to {out_dir}")


if __name__ == "__main__":
    main()
