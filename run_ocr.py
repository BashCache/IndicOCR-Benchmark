#!/usr/bin/env python3
"""
CLI entry point for the Tamil PP-OCRv5 full document OCR pipeline (detect + recognise).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from ocr_pipeline.full_pipeline import FullPipelineEngine
from ocr_pipeline.result_writer import save_json_report
from single_col_fix_reading_order import fix_reading_order
from utils.overlay_bboxes import overlay_bounding_boxes

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(sources: List[str]) -> List[str]:
    """Expands files, directories, and glob patterns into a sorted, deduplicated path list."""
    paths: List[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for ext in IMAGE_EXTENSIONS:
                paths.extend(sorted(p.glob(f"*{ext}")))
                paths.extend(sorted(p.glob(f"*{ext.upper()}")))
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
        else:
            expanded = [Path(f) for f in sorted(glob.glob(src))]
            if expanded:
                paths.extend(expanded)
            else:
                logger.warning("Skipping unrecognised input: %s", src)

    seen, unique = set(), []
    for p in paths:
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ocr.py",
        description="Full-pipeline Tamil OCR (detect + recognise) using PP-OCRv5 (PaddleOCR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--images", "-i",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more image files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--rec-model-dir",
        default=None,
        metavar="DIR",
        help="Optional path to custom text recognition model directory.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run inference on GPU.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        metavar="N",
        help="CPU threads for Paddle inference engine.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Confidence threshold; results below this are flagged low-confidence.",
    )
    parser.add_argument(
        "--no-lightweight",
        action="store_true",
        help="Load all 5 models (including doc-orientation, unwarping, textline-orientation).",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1920,
        metavar="PX",
        help="Auto-downscale images exceeding this max dimension before detection. Set 0 to disable.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Explicit path for JSON output. Defaults to outputs/ocr_results_<timestamp>.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        metavar="DIR",
        help="Directory for auto-saved JSON results when --output is omitted.",
    )
    parser.add_argument(
        "--overlay-dir",
        default="overlay_outputs",
        metavar="DIR",
        help="Directory to save overlaid bounding box images. Default: overlay_outputs.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip generating bounding box overlay images.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    images = collect_images(args.images)
    if not images:
        logger.error("No valid images found in: %s", args.images)
        return 1
    logger.info("Found %d image(s) to process.", len(images))

    logger.info("Mode: FULL PIPELINE (detect + recognise)")
    try:
        engine = FullPipelineEngine(
            lang="ta",
            rec_model_dir=args.rec_model_dir,
            use_gpu=args.gpu,
            cpu_threads=args.threads,
            score_threshold=args.threshold,
            lightweight=not args.no_lightweight,
            max_side=args.max_side,
        )
    except ImportError as exc:
        logger.error(str(exc))
        return 2

    full_results = engine.run(images)

    config_dict = {
        "mode": "full_pipeline",
        "lang": "ta",
        "use_gpu": args.gpu,
        "cpu_threads": args.threads,
        "threshold": args.threshold,
        "lightweight": not args.no_lightweight,
        "max_side": args.max_side,
    }
    json_report = {
        "run": {
            "timestamp": datetime.now(tz=timezone.utc).astimezone().isoformat(),
            "model": "ta_PP-OCRv5_mobile_rec_infer (full pipeline)",
            "total_images": len(full_results),
            "config": config_dict,
        },
        "images": full_results,
    }

    for r in full_results:
        logger.info("  %-30s → %d lines detected", r["image_id"], r["line_count"])
        for ln in r["lines"]:
            flag = "  ⚠" if ln["low_confidence"] else ""
            logger.info("      %r  (conf=%.4f)%s", ln["text"], ln["confidence"], flag)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(json_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Results saved → %s", out_path)
    else:
        out_path = save_json_report(
            report=json_report,
            output_dir=args.output_dir,
        )

    print(f"Results saved → {out_path}")

    # 1. Generate initial bounding box overlays
    if not args.no_overlay and args.overlay_dir:
        overlay_dir = Path(args.overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_count = 0
        for r in full_results:
            img_path = Path(r["image_path"])
            lines = r.get("lines", [])
            if img_path.exists() and lines:
                out_overlay = overlay_dir / f"{img_path.stem}_bbox_overlay.jpg"
                try:
                    overlay_bounding_boxes(
                        image_path=img_path,
                        lines=lines,
                        output_path=out_overlay,
                    )
                    logger.info("Initial overlay saved → %s", out_overlay)
                    overlay_count += 1
                except Exception as exc:
                    logger.warning("Failed to generate initial overlay for %s: %s", img_path.name, exc)
        if overlay_count > 0:
            print(f"Initial overlays saved → {overlay_dir.resolve()} ({overlay_count} image(s))")

    # 2. Fix reading order using single_col_fix_reading_order
    logger.info("Applying single-column reading order correction...")
    for r in full_results:
        lines = r.get("lines", [])
        if lines:
            try:
                corrected_lines = fix_reading_order(lines)
                r["lines"] = corrected_lines
                r["full_text"] = "\n".join(b.get("text", "") for b in corrected_lines)
            except Exception as exc:
                logger.warning("Failed reading order fix for %s: %s", r.get("image_id"), exc)

    # 3. Redraw overlays with the changed ordering
    if not args.no_overlay and args.overlay_dir:
        overlay_dir = Path(args.overlay_dir)
        reordered_count = 0
        for r in full_results:
            img_path = Path(r["image_path"])
            lines = r.get("lines", [])
            if img_path.exists() and lines:
                out_reordered = overlay_dir / f"{img_path.stem}_reordered_overlay.jpg"
                try:
                    overlay_bounding_boxes(
                        image_path=img_path,
                        lines=lines,
                        output_path=out_reordered,
                    )
                    logger.info("Reordered overlay saved → %s", out_reordered)
                    reordered_count += 1
                except Exception as exc:
                    logger.warning("Failed to generate reordered overlay for %s: %s", img_path.name, exc)
        if reordered_count > 0:
            print(f"Reordered overlays saved → {overlay_dir.resolve()} ({reordered_count} image(s))")

    # 4. Update JSON file with the corrected reading order
    out_path.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Updated results JSON with corrected order → %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
