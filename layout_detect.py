#!/usr/bin/env python3
"""
layout_detect.py
----------------
Layout Detection script using PaddleOCR's LayoutDetection engine (PP-DocLayout).

Identifies document layout regions in an image, such as:
  - Paragraph Titles / Document Titles
  - Text blocks (paragraphs, columns)
  - Tables
  - Images / Figures / Charts
  - Headers / Footers
  - Formulas / Footnotes / Aside text

Features:
  1. Detects bounding boxes [xmin, ymin, xmax, ymax], labels, and confidence scores.
  2. Reading-order sorting (top-to-bottom, column-aware).
  3. Visual layout overlay with color-coded bounding boxes and badges.
  4. Optional cropping of each identified layout element into separate image files.
  5. JSON report export for downstream processing (e.g. feeding text regions into OCR).
  6. Reusable Python API (`PaddleLayoutDetector`) and rich CLI interface.

Usage:
------
# Detect layout for a single image with visual overlay
python layout_detect.py --images data/rasi-1.png --visualize

# Detect layout and extract cropped blocks for all images in a folder
python layout_detect.py --images data/ --visualize --save-crops

# Custom threshold and custom output directory
python layout_detect.py --images data/tamil-2column.png --threshold 0.6 --output-dir outputs/layout_results
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("layout_detect")

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Distinct color palette per layout category for clean visual overlays
LAYOUT_COLORS: Dict[str, str] = {
    "text": "#2ECC71",              # Emerald green
    "paragraph_title": "#3498DB",   # Sky blue
    "doc_title": "#2980B9",         # Deep blue
    "content": "#1ABC9C",           # Teal
    "table": "#E67E22",             # Orange
    "image": "#E74C3C",             # Coral red
    "figure_title": "#C0392B",      # Crimson
    "chart": "#D35400",             # Rust
    "header": "#9B59B6",            # Purple
    "footer": "#8E44AD",            # Dark purple
    "formula": "#F39C12",           # Amber
    "formula_number": "#F1C40F",    # Yellow
    "abstract": "#16A085",          # Dark teal
    "reference": "#7F8C8D",         # Gray
    "reference_content": "#95A5A6", # Light gray
    "footnote": "#34495E",          # Navy
    "aside_text": "#2C3E50",        # Dark slate
    "seal": "#C0392B",              # Red
    "algorithm": "#27AE60",         # Forest green
    "number": "#E67E22",            # Tangerine
    "default": "#34495E",           # Charcoal
}


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """Return an RGB tuple for a given layout label."""
    hex_color = LAYOUT_COLORS.get(label.lower(), LAYOUT_COLORS["default"])
    return ImageColor.getrgb(hex_color)


def collect_image_paths(sources: Sequence[str]) -> List[str]:
    """
    Expand a list of files, directories, or glob patterns into
    a sorted, deduplicated list of valid image file paths.
    """
    image_paths: List[str] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                image_paths.extend(str(f) for f in p.glob(f"*{ext}"))
                image_paths.extend(str(f) for f in p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(str(p))
        else:
            expanded = glob.glob(src)
            for item in expanded:
                ip = Path(item)
                if ip.is_file() and ip.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(str(ip))
                elif ip.is_dir():
                    for ext in SUPPORTED_EXTENSIONS:
                        image_paths.extend(str(f) for f in ip.glob(f"*{ext}"))

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for path in image_paths:
        abs_p = os.path.abspath(path)
        if abs_p not in seen:
            seen.add(abs_p)
            unique_paths.append(abs_p)

    return unique_paths


class PaddleLayoutDetector:
    """
    Layout detection engine using PaddleOCR's LayoutDetection API (PP-DocLayout).

    Parameters:
        device      : 'cpu' or 'gpu'.
        threshold   : Minimum confidence threshold (0.0 to 1.0) to retain a detected layout box.
        cpu_threads : Number of CPU inference threads.
    """

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.3,
        cpu_threads: int = 4,
    ):
        self.device = device.lower()
        self.threshold = float(threshold)
        self.cpu_threads = int(cpu_threads)
        self._model = self._init_model()

    def _init_model(self):
        """Initialise PaddleOCR LayoutDetection with safe runtime flags."""
        logger.info(
            "Loading PaddleOCR LayoutDetection model (device=%s, threshold=%.2f)...",
            self.device,
            self.threshold,
        )
        try:
            from paddleocr import LayoutDetection
        except ImportError as err:
            raise ImportError(
                "PaddleOCR is not installed in the environment.\n"
                "Install via: pip install paddleocr"
            ) from err

        # enable_mkldnn=False avoids ConvertPirAttribute2RuntimeAttribute crashes on modern CPU
        detector = LayoutDetection(
            device=self.device,
            enable_mkldnn=False,
            cpu_threads=self.cpu_threads,
            threshold=self.threshold,
        )
        logger.info("LayoutDetection model initialized successfully.")
        return detector

    def detect_single(
        self,
        image_path: str,
        reorder_layout: bool = True,
        save_vis: bool = False,
        save_crops: bool = False,
        output_dir: Optional[str] = None,
        sort_reading_order: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run layout detection on a single image.

        Args:
            image_path     : Path to the image file.
            reorder_layout : If True, applies column-aware reading-order reordering (via fix_layout_reordering).
            save_vis       : If True, saves annotated visual image with layout bounding boxes.
            save_crops     : If True, saves each detected layout region as an individual cropped image.
            output_dir     : Directory where visual images and crops will be saved.

        Returns:
            Dictionary containing image metadata, detected layout elements, and summary counts.
        """
        if sort_reading_order is not None:
            reorder_layout = sort_reading_order

        path_obj = Path(image_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open image to get dimensions and verify integrity
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
            # Convert to RGB to ensure 3 channels for visualization/crops
            img_rgb = img.convert("RGB")

        logger.info("Detecting layouts in: %s (%dx%d)", path_obj.name, orig_w, orig_h)

        # Run model inference
        raw_results = self._model.predict(image_path)
        if not raw_results:
            logger.warning("No output from LayoutDetection for %s", path_obj.name)
            raw_boxes = []
        else:
            raw_boxes = raw_results[0].get("boxes", [])

        # Parse detected boxes
        elements: List[Dict[str, Any]] = []
        for box in raw_boxes:
            score = float(box.get("score", 0.0))
            if score < self.threshold:
                continue

            coord = box.get("coordinate", [])
            if len(coord) != 4:
                continue

            xmin, ymin, xmax, ymax = [float(c) for c in coord]

            # Clamp coordinates to image boundaries
            xmin = max(0.0, min(float(orig_w), xmin))
            ymin = max(0.0, min(float(orig_h), ymin))
            xmax = max(0.0, min(float(orig_w), xmax))
            ymax = max(0.0, min(float(orig_h), ymax))

            width = round(xmax - xmin, 1)
            height = round(ymax - ymin, 1)

            if width <= 1 or height <= 1:
                continue

            label = str(box.get("label", "unknown"))
            cls_id = int(box.get("cls_id", -1))

            elements.append({
                "label": label,
                "cls_id": cls_id,
                "score": round(score, 4),
                "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
                "polygon": [
                    [round(xmin, 1), round(ymin, 1)],
                    [round(xmax, 1), round(ymin, 1)],
                    [round(xmax, 1), round(ymax, 1)],
                    [round(xmin, 1), round(ymax, 1)],
                ],
                "dimensions": {
                    "width": width,
                    "height": height,
                    "area": round(width * height, 1),
                },
                "center": {
                    "x": round((xmin + xmax) / 2.0, 1),
                    "y": round((ymin + ymax) / 2.0, 1),
                },
            })

        # Assign 1-based layout ID in detection order
        for idx, elem in enumerate(elements, start=1):
            elem["layout_id"] = idx

        # Generate summary counts by label
        label_summary: Dict[str, int] = {}
        for elem in elements:
            lbl = elem["label"]
            label_summary[lbl] = label_summary.get(lbl, 0) + 1

        result_dict: Dict[str, Any] = {
            "image_name": path_obj.name,
            "image_path": str(path_obj.resolve()),
            "image_shape": {"width": orig_w, "height": orig_h},
            "total_layouts": len(elements),
            "layout_summary": label_summary,
            "elements": elements,
        }

        # Prepare output directories if visualization or crops requested
        if output_dir:
            out_base = Path(output_dir)
        else:
            out_base = Path("outputs/layout_detection")

        vis_path = None
        crop_paths = []

        if save_vis:
            out_base.mkdir(parents=True, exist_ok=True)
            vis_filename = f"{path_obj.stem}_layout_vis.jpg"
            vis_path = str(out_base / vis_filename)
            self._draw_layout_overlay(img_rgb, elements, vis_path)
            result_dict["visualization_path"] = vis_path
            logger.info("Saved visual layout overlay to: %s", vis_path)

        # After visualization: reorder the layout based on fix_layout_reordering.py
        if reorder_layout:
            reordered_elements = self.reorder_layout(elements, page_width=orig_w)
            result_dict["reordered_elements"] = reordered_elements

            if save_vis:
                reordered_vis_filename = f"{path_obj.stem}_reordered_layout_vis.jpg"
                reordered_vis_path = str(out_base / reordered_vis_filename)
                self._draw_layout_overlay(img_rgb, reordered_elements, reordered_vis_path)
                result_dict["reordered_visualization_path"] = reordered_vis_path
                logger.info("Saved reordered visual layout overlay to: %s", reordered_vis_path)

            elements_to_crop = reordered_elements
        else:
            elements_to_crop = elements

        if save_crops:
            crops_dir = out_base / f"{path_obj.stem}_crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            crop_paths = self._save_element_crops(img_rgb, elements_to_crop, crops_dir, path_obj.stem)
            result_dict["crops_directory"] = str(crops_dir)
            result_dict["cropped_files"] = crop_paths
            logger.info("Saved %d cropped regions to: %s", len(crop_paths), crops_dir)

        return result_dict

    def detect_batch(
        self,
        image_paths: Sequence[str],
        reorder_layout: bool = True,
        save_vis: bool = False,
        save_crops: bool = False,
        output_dir: Optional[str] = None,
        sort_reading_order: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Run layout detection over multiple images."""
        if sort_reading_order is not None:
            reorder_layout = sort_reading_order
        batch_results: List[Dict[str, Any]] = []
        for img_path in image_paths:
            try:
                res = self.detect_single(
                    image_path=img_path,
                    reorder_layout=reorder_layout,
                    save_vis=save_vis,
                    save_crops=save_crops,
                    output_dir=output_dir,
                )
                batch_results.append(res)
            except Exception as err:
                logger.error("Failed layout detection on %s: %s", img_path, err)
                batch_results.append({
                    "image_name": Path(img_path).name,
                    "image_path": str(Path(img_path).resolve()),
                    "error": str(err),
                    "total_layouts": 0,
                    "elements": [],
                })
        return batch_results

    @staticmethod
    def _sort_reading_order(
        elements: List[Dict[str, Any]], page_width: int
    ) -> List[Dict[str, Any]]:
        """
        Sort layout elements into top-to-bottom, column-aware reading order.
        Columns are detected by analyzing horizontal centers and overlaps.
        """
        if len(elements) <= 1:
            return elements

        # Group elements into vertical columns if multi-column document
        # Check if elements are predominantly side-by-side
        cols: List[List[Dict[str, Any]]] = []
        # Sort primarily by vertical position first
        sorted_by_y = sorted(elements, key=lambda e: (e["bbox"][1], e["bbox"][0]))

        # Check if there are distinct columns (e.g. 2-column layout)
        x_centers = [e["center"]["x"] for e in sorted_by_y]
        mid_x = page_width / 2.0

        left_side = [e for e in sorted_by_y if e["bbox"][2] <= mid_x * 1.15]
        right_side = [e for e in sorted_by_y if e["bbox"][0] >= mid_x * 0.85]

        # If a significant number of blocks fall distinctly on left and right
        is_multi_column = (
            len(left_side) >= 1
            and len(right_side) >= 1
            and (len(left_side) + len(right_side)) >= len(elements) * 0.7
        )

        if is_multi_column:
            # Full-width blocks (spanning across center, like header/title) stay in global Y order
            spanning = [
                e for e in sorted_by_y
                if e not in left_side and e not in right_side
            ]

            # Sort: Spanning top titles -> Left column blocks -> Right column blocks -> Spanning bottom footers
            top_spanning = [e for e in spanning if e["center"]["y"] < page_width * 0.3]
            bottom_spanning = [e for e in spanning if e["center"]["y"] >= page_width * 0.3]

            left_sorted = sorted(left_side, key=lambda e: e["bbox"][1])
            right_sorted = sorted(right_side, key=lambda e: e["bbox"][1])

            sorted_elements = top_spanning + left_sorted + right_sorted + bottom_spanning
            return sorted_elements

        # Single-column fallback: sort by Y-top, then X-left
        return sorted_by_y

    @staticmethod
    def _draw_layout_overlay(
        image: Image.Image,
        elements: List[Dict[str, Any]],
        save_path: str,
    ) -> None:
        """
        Draw high-contrast, polished bounding boxes and label tags on the image.
        Uses local PIL rendering with no external font server dependencies.
        """
        overlay_img = image.copy()
        draw = ImageDraw.Draw(overlay_img, "RGBA")

        # Load default or available font
        try:
            # Try to load a clean TrueType font if present on Linux
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
            badge_font = font

        for elem in elements:
            lid = elem.get("_corrected_order") or elem.get("layout_id", "")
            label = elem["label"]
            score = elem["score"]
            xmin, ymin, xmax, ymax = elem["bbox"]

            rgb = get_color_for_label(label)

            # Draw semi-transparent fill inside the bounding box
            fill_color = (*rgb, 35)  # 15% opacity tint
            draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)

            # Draw solid outline border
            outline_color = (*rgb, 255)
            line_width = max(2, int(min(image.size) / 400))
            draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=line_width)

            # Draw label badge header
            badge_text = f"#{lid} {label.upper()} ({int(score * 100)}%)"

            # Compute text size
            try:
                bbox_text = draw.textbbox((0, 0), badge_text, font=badge_font)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]
            except AttributeError:
                # Pillow < 8 compatibility
                text_w, text_h = 100, 16

            pad = 4
            badge_x1 = xmin
            badge_y1 = max(0, ymin - text_h - pad * 2)
            badge_x2 = xmin + text_w + pad * 2
            badge_y2 = badge_y1 + text_h + pad * 2

            # Solid background for label tag
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=outline_color)
            # White text
            draw.text(
                (badge_x1 + pad, badge_y1 + pad),
                badge_text,
                fill=(255, 255, 255, 255),
                font=badge_font,
            )

        # Save annotated image
        overlay_img.save(save_path, quality=95)

    @staticmethod
    def reorder_layout(
        elements: List[Dict[str, Any]],
        page_width: int,
        fullwidth_ratio: float = 0.6,
        rtl: bool = False,
        spanning_width_ratio: float = 1.5,
    ) -> List[Dict[str, Any]]:
        """
        Reorder layout elements into natural reading order based on fix_layout_reordering.py.
        Uses column-gap clustering with Otsu thresholding and full-width header/footer separation.
        """
        from fix_layout_reordering import fix_layout_reading_order
        return fix_layout_reading_order(
            elements=elements,
            page_width=page_width,
            fullwidth_ratio=fullwidth_ratio,
            rtl=rtl,
            spanning_width_ratio=spanning_width_ratio,
        )

    @staticmethod
    def _save_element_crops(
        image: Image.Image,
        elements: List[Dict[str, Any]],
        crops_dir: Path,
        stem: str,
    ) -> List[str]:
        """Crop each detected bounding box region and save to disk."""
        saved_paths: List[str] = []
        for elem in elements:
            lid = elem.get("_corrected_order") or elem.get("layout_id", 0)
            label = elem["label"]
            score = elem["score"]
            xmin, ymin, xmax, ymax = elem["bbox"]

            # Crop box with integer bounds
            crop_box = (int(xmin), int(ymin), int(xmax), int(ymax))
            cropped = image.crop(crop_box)

            crop_filename = f"{stem}_region_{lid:02d}_{label}_{int(score*100)}pct.png"
            crop_filepath = crops_dir / crop_filename
            cropped.save(str(crop_filepath))
            saved_paths.append(str(crop_filepath))

        return saved_paths


def print_layout_table(result: Dict[str, Any]) -> None:
    """Print a clean CLI summary table of detected layout components."""
    img_name = result.get("image_name", "Image")
    shape = result.get("image_shape", {})
    total = result.get("total_layouts", 0)
    summary = result.get("layout_summary", {})
    elements = result.get("elements", [])

    print("\n" + "=" * 78)
    print(f"LAYOUT DETECTION RESULTS: {img_name}")
    print(f"Dimensions: {shape.get('width', '?')}x{shape.get('height', '?')} px | Total Layouts: {total}")
    print("-" * 78)

    if summary:
        summary_str = ", ".join(f"{k}: {v}" for k, v in sorted(summary.items()))
        print(f"Summary: {summary_str}")
        print("-" * 78)

    if not elements:
        print("  No layout components detected above the confidence threshold.")
        print("=" * 78 + "\n")
        return

    # Table headers
    header = f"{'ID':<4} | {'Label':<18} | {'Score':<7} | {'Bounding Box [x1, y1, x2, y2]':<30} | {'Dimensions'}"
    print(header)
    print("-" * 78)

    for elem in elements:
        lid = elem.get("layout_id", "-")
        lbl = elem.get("label", "unknown")
        sc = f"{elem.get('score', 0.0)*100:.1f}%"
        bbox = elem.get("bbox", [])
        bbox_str = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
        dim = elem.get("dimensions", {})
        dim_str = f"{int(dim.get('width', 0))}x{int(dim.get('height', 0))} px"

        print(f"{lid:<4} | {lbl:<18} | {sc:<7} | {bbox_str:<30} | {dim_str}")

    print("=" * 78)
    if "visualization_path" in result:
        print(f"Annotated visualization: {result['visualization_path']}")
    if "reordered_visualization_path" in result:
        print(f"Reordered visualization: {result['reordered_visualization_path']}")
    if "crops_directory" in result:
        print(f"Cropped regions folder : {result['crops_directory']}")
    print("")


def str2bool(val: Any) -> bool:
    """Parse boolean strings into a boolean value."""
    if isinstance(val, bool):
        return val
    val_str = str(val).strip().lower()
    if val_str in ("yes", "true", "t", "y", "1"):
        return True
    if val_str in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected (true/false), got {val!r}")


def main():
    parser = argparse.ArgumentParser(
        prog="layout_detect.py",
        description="Detect document layouts (tables, text blocks, titles, figures) using PaddleOCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--images", "-i",
        nargs="+",
        required=True,
        help="One or more image file paths, wildcard patterns, or folders containing images.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold for layout box detection (0.0 to 1.0).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Compute device for inference.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Number of threads for CPU inference.",
    )
    parser.add_argument(
        "--visualize", "--save-vis",
        action="store_true",
        default=True,
        help="Save an annotated image with color-coded layout bounding boxes and labels.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Disable saving annotated visual layout images.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        default=False,
        help="Crop and save each detected layout region as a separate image file.",
    )
    parser.add_argument(
        "--reorder-layout",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable or disable column-aware layout reordering: true or false (default: true).",
    )
    parser.add_argument(
        "--no-reorder-layout",
        action="store_false",
        dest="reorder_layout",
        help="Disable layout reordering.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/layout_detection",
        help="Directory to save visual overlays, crops, and JSON reports.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output JSON report. If omitted, saves to <output-dir>/layout_results_<timestamp>.json",
    )

    args = parser.parse_args()

    # Collect images
    image_paths = collect_image_paths(args.images)
    if not image_paths:
        logger.error("No valid image files found matching: %s", args.images)
        sys.exit(1)

    logger.info("Found %d image(s) to process.", len(image_paths))

    # Initialize layout detector
    detector = PaddleLayoutDetector(
        device=args.device,
        threshold=args.threshold,
        cpu_threads=args.cpu_threads,
    )

    # Run batch detection
    results = detector.detect_batch(
        image_paths=image_paths,
        reorder_layout=args.reorder_layout,
        save_vis=args.visualize,
        save_crops=args.save_crops,
        output_dir=args.output_dir,
    )

    # Print summary tables to stdout
    for res in results:
        print_layout_table(res)

    # Prepare JSON report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        json_output_path = Path(args.output)
    else:
        json_output_path = out_dir / f"layout_results_{timestamp}.json"

    report = {
        "metadata": {
            "timestamp": timestamp,
            "engine": "PaddleOCR LayoutDetection (PP-DocLayout)",
            "device": args.device,
            "threshold": args.threshold,
            "image_count": len(image_paths),
        },
        "results": results,
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Complete layout analysis saved to JSON: %s", json_output_path)


if __name__ == "__main__":
    main()
