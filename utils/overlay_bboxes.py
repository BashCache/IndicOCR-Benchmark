"""
Overlays bounding boxes from an OCR/Layout JSON file onto document images.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_OUTPUT_DIR = Path("overlay_outputs")

LINE_WIDTH = 2
COLOR = "red"
FILL_ALPHA = 40
DRAW_INDICES = True
SHOW_CONF = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay OCR/layout bounding boxes onto images from a results JSON file."
    )
    parser.add_argument(
        "--images",
        "-i",
        dest="images_path",
        required=True,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "--json",
        "-j",
        dest="json_path",
        required=True,
        help="Path to the results JSON file containing bounding boxes.",
    )
    return parser.parse_args()


def load_all_boxes_from_json(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Loads JSON and maps image filenames and stems to their bounding box elements."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_boxes: Dict[str, List[Dict[str, Any]]] = {}

    if isinstance(data, dict) and "images" in data:
        for img in data["images"]:
            lines = img.get("lines", [])
            img_id = img.get("image_id", "")
            img_path = img.get("image_path", "")
            keys = {
                Path(img_id).name,
                Path(img_id).stem,
                Path(img_path).name,
                Path(img_path).stem,
                img_id,
            }
            for k in keys:
                if k:
                    image_boxes[k] = lines

    elif isinstance(data, dict) and "results" in data:
        for res in data["results"]:
            elements = res.get("elements", [])
            converted_lines = []
            for elem in elements:
                bbox = elem.get("polygon") or elem.get("bbox")
                if bbox and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                    x1, y1, x2, y2 = bbox
                    bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                item_data = dict(elem)
                item_data["bbox"] = bbox
                item_data["text"] = elem.get("label", elem.get("text", ""))
                item_data["confidence"] = elem.get("score", elem.get("confidence"))
                converted_lines.append(item_data)
            img_name = res.get("image_name", "")
            img_path = res.get("image_path", "")
            keys = {
                Path(img_name).name,
                Path(img_name).stem,
                Path(img_path).name,
                Path(img_path).stem,
                img_name,
            }
            for k in keys:
                if k:
                    image_boxes[k] = converted_lines

    elif isinstance(data, dict) and "elements" in data:
        elements = data.get("elements", [])
        converted_lines = []
        for elem in elements:
            bbox = elem.get("polygon") or elem.get("bbox")
            if bbox and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                x1, y1, x2, y2 = bbox
                bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            item_data = dict(elem)
            item_data["bbox"] = bbox
            item_data["text"] = elem.get("label", elem.get("text", ""))
            item_data["confidence"] = elem.get("score", elem.get("confidence"))
            converted_lines.append(item_data)
        img_name = data.get("image_name", "")
        img_path = data.get("image_path", "")
        keys = {
            Path(img_name).name,
            Path(img_name).stem,
            Path(img_path).name,
            Path(img_path).stem,
            img_name,
            "*",
        }
        for k in keys:
            if k:
                image_boxes[k] = converted_lines

    elif isinstance(data, dict) and "lines" in data:
        image_boxes["*"] = data["lines"]

    return image_boxes


def overlay_bounding_boxes(
    image_path: Path,
    lines: List[Dict[str, Any]],
    output_path: Path,
    line_width: int = LINE_WIDTH,
    color: str = COLOR,
    fill_alpha: int = FILL_ALPHA,
    draw_indices: bool = DRAW_INDICES,
    show_conf: bool = SHOW_CONF,
) -> Path:
    """Overlays bounding boxes and index labels onto the image and saves the result."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_image = Image.open(image_path).convert("RGB")
    width, height = base_image.size

    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 100, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    rgb = color_map.get(color.lower(), (255, 0, 0))
    fill_color = (*rgb, max(0, min(255, fill_alpha)))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, item in enumerate(lines, start=1):
        raw_bbox = item.get("bbox")
        if not raw_bbox:
            continue

        if len(raw_bbox) == 4 and all(isinstance(v, (int, float)) for v in raw_bbox):
            x1, y1, x2, y2 = raw_bbox
            raw_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        points: List[Tuple[int, int]] = []
        for pt in raw_bbox:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                points.append((int(round(pt[0])), int(round(pt[1]))))

        if len(points) < 3:
            continue

        if fill_alpha > 0:
            overlay_draw.polygon(points, fill=fill_color)

        poly_closed = points + [points[0]]
        overlay_draw.line(poly_closed, fill=rgb + (255,), width=line_width)

        if draw_indices:
            top_left = points[0]
            order_val = item.get("_corrected_order")
            if order_val is None:
                order_val = item.get("layout_id")
            if order_val is None:
                order_val = idx
            label = str(order_val)
            if show_conf and "confidence" in item and item["confidence"] is not None:
                label += f" ({item['confidence']:.2f})"

            pad = 2
            if hasattr(overlay_draw, "textbbox"):
                bbox_text = overlay_draw.textbbox(top_left, label, font=font)
                badge_box = [
                    bbox_text[0] - pad,
                    bbox_text[1] - pad,
                    bbox_text[2] + pad,
                    bbox_text[3] + pad,
                ]
            else:
                badge_box = [
                    top_left[0] - pad,
                    top_left[1] - pad,
                    top_left[0] + 18 + pad,
                    top_left[1] + 12 + pad,
                ]

            overlay_draw.rectangle(badge_box, fill=(0, 0, 0, 200))
            overlay_draw.text(top_left, label, fill=(255, 255, 255, 255), font=font)

    base_rgba = base_image.convert("RGBA")
    combined = Image.alpha_composite(base_rgba, overlay).convert("RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=95)
    return output_path


def collect_images(path: Path) -> List[Path]:
    """Collects all images from a file or directory."""
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        return []
    if path.is_dir():
        files: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(list(set(files)))
    return []


def main() -> None:
    args = parse_args()
    input_path = Path(args.images_path).resolve()
    json_path = Path(args.json_path).resolve()
    output_dir = DEFAULT_OUTPUT_DIR.resolve()

    print(f"Loading JSON annotations from: {json_path}")
    image_boxes_map = load_all_boxes_from_json(json_path)

    images = collect_images(input_path)
    if not images:
        print(f"No valid image files found at: {input_path}")
        return

    print(f"Found {len(images)} image(s) to process. Output directory: {output_dir}")

    success_count = 0
    for img_path in images:
        lines = (
            image_boxes_map.get(img_path.name)
            or image_boxes_map.get(img_path.stem)
            or image_boxes_map.get(str(img_path))
            or image_boxes_map.get("*")
        )

        if not lines:
            print(f"  [SKIPPED] {img_path.name}: no matching annotations found in JSON.")
            continue

        out_file = output_dir / f"{img_path.stem}_bbox_overlay.jpg"
        result_path = overlay_bounding_boxes(
            image_path=img_path,
            lines=lines,
            output_path=out_file,
        )
        print(f"  [DONE] {img_path.name} ({len(lines)} boxes) -> {result_path.name}")
        success_count += 1

    print(f"\nCompleted: {success_count}/{len(images)} image overlays created in {output_dir}")


if __name__ == "__main__":
    main()
