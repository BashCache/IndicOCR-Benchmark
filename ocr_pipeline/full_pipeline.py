"""
full_pipeline.py
----------------
Full OCR pipeline (text detection + recognition) using the high-level
paddleocr.PaddleOCR API (PaddleOCR v3 / paddleocr ≥ 2.8).

This is the correct way to handle full-page or paragraph images.
The PP-OCRv5 recognition model (ta_PP-OCRv5_mobile_rec_infer) expects
pre-cropped text LINE STRIPS (~48 px tall).  For full-page images you
MUST run a text detector first to find those line regions.

PaddleOCR's high-level API does exactly that in one call:
  detect text regions → crop → recognise each crop → return results

Memory notes
------------
The full pipeline loads up to 5 models simultaneously:
  1. PP-LCNet_x1_0_doc_ori      (doc orientation classify) ~150 MB  — optional
  2. UVDoc                       (doc unwarping)            ~350 MB  — optional
  3. PP-LCNet_x1_0_textline_ori (textline orientation)     ~150 MB  — optional
  4. PP-OCRv5_server_det         (text detection)           ~300 MB  — required
  5. ta_PP-OCRv5_mobile_rec      (text recognition)         ~100 MB  — required

Use lightweight=True (default) to skip models 1–3 and save ~650 MB RAM.
Use max_side to auto-downscale images before detection to further cap memory.

Usage
-----
    engine = FullPipelineEngine(lang="ta")
    results = engine.run(["page.png", "doc.jpg"])
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path]

# Default: auto-downscale if max side exceeds this (pixels)
DEFAULT_MAX_SIDE = 1920


# ─────────────────────────────────────────────────────────────────────────────
# Inline reading-order sort (whitespace projection)
# ─────────────────────────────────────────────────────────────────────────────

def _sort_into_reading_order(
    lines: List[Dict[str, Any]],
    page_width: float,
    min_gap_pct: float = 0.03,
    floor_px: int = 10,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Sort OCR lines into reading order by detecting column boundaries via
    x-axis whitespace projection, then sorting by (column, y_top).

    The effective gap threshold adapts to page width:
        threshold = max(floor_px, page_width * min_gap_pct)

    Returns:
        (sorted_lines, col_ids)  — col_ids are 0-based.
    """
    if not lines:
        return [], []

    # Build bbox intervals
    def xleft(bbox):  return min(p[0] for p in bbox)
    def xright(bbox): return max(p[0] for p in bbox)
    def ytop(bbox):   return min(p[1] for p in bbox)
    def xcentre(bbox): return (xleft(bbox) + xright(bbox)) / 2.0

    # Merge overlapping x-intervals to find whitespace corridors
    intervals = sorted((xleft(ln["bbox"]), xright(ln["bbox"])) for ln in lines)
    merged: List[Tuple[int, int]] = []
    for xl, xr in intervals:
        if merged and xl <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], xr))
        else:
            merged.append((xl, xr))

    pw = page_width or (merged[-1][1] if merged else 1)
    threshold = max(floor_px, int(pw * min_gap_pct))

    # Find gap midpoints that exceed threshold
    boundaries: List[float] = []
    for i in range(len(merged) - 1):
        gap_w = merged[i+1][0] - merged[i][1]
        if gap_w >= threshold:
            boundaries.append((merged[i][1] + merged[i+1][0]) / 2.0)

    n_cols = len(boundaries) + 1
    if n_cols > 1:
        logger.info(
            "Detected %d columns. Boundaries at x≈%s  (threshold=%dpx)",
            n_cols, [round(b) for b in boundaries], threshold,
        )
    else:
        logger.info("Single column detected (threshold=%dpx).", threshold)

    # Assign column by x-centre
    def assign_col(bbox):
        cx = xcentre(bbox)
        for i, b in enumerate(boundaries):
            if cx < b:
                return i
        return len(boundaries)

    annotated = [(assign_col(ln["bbox"]), ytop(ln["bbox"]), ln) for ln in lines]
    annotated.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in annotated], [t[0] for t in annotated]


def _downscale_if_needed(
    img_path: str,
    max_side: int,
) -> Tuple[str, float]:
    """
    If either dimension of the image exceeds *max_side*, downscale it
    proportionally and save to a temp file.

    Returns:
        (path_to_use, scale_factor)  — scale_factor < 1.0 means image was resized.
    """
    img = Image.open(img_path)
    w, h = img.size
    longest = max(w, h)

    if longest <= max_side:
        return img_path, 1.0

    scale = max_side / longest
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(
        "Downscaling %s from %dx%d → %dx%d (scale=%.3f) to fit max_side=%d",
        Path(img_path).name, w, h, new_w, new_h, scale, max_side,
    )
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Save to a temp file (same format as original)
    suffix = Path(img_path).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    resized.save(tmp.name)
    tmp.close()
    return tmp.name, scale


def _scale_bbox(bbox: Any, scale: float) -> Any:
    """Scale polygon coordinates back to original image space."""
    if scale == 1.0 or not bbox:
        return bbox
    return [[int(x / scale), int(y / scale)] for x, y in bbox]


class FullPipelineEngine:
    """
    Wraps paddleocr.PaddleOCR (v3 API) for the full detect + recognise pipeline.

    Args:
        lang            : Language code. 'ta' for Tamil.
        rec_model_dir   : Optional path to a custom recognition model directory.
        use_gpu         : Run on GPU (sets device='gpu').
        cpu_threads     : Number of CPU threads.
        score_threshold : Minimum recognition confidence to include in results.
        lightweight     : Skip doc-orientation, unwarping, and textline-orientation
                          models (~650 MB saved).  Default True.
        max_side        : Auto-downscale images whose longest side exceeds this
                          value before detection. Reduces peak memory usage.
                          Set to 0 to disable. Default 1920.
    """

    def __init__(
        self,
        lang: str = "ta",
        rec_model_dir: Optional[str] = None,
        use_gpu: bool = False,
        cpu_threads: int = 4,
        score_threshold: float = 0.5,
        lightweight: bool = True,
        max_side: int = DEFAULT_MAX_SIDE,
    ):
        self.score_threshold = score_threshold
        self.max_side = max_side
        self._tmp_files: List[str] = []
        self._engine = self._build_engine(
            lang, rec_model_dir, use_gpu, cpu_threads, lightweight
        )

    @staticmethod
    def _build_engine(
        lang: str,
        rec_model_dir: Optional[str],
        use_gpu: bool,
        cpu_threads: int,
        lightweight: bool,
    ):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is not installed. Install it with:\n"
                "  pip install paddleocr\n"
                "Then re-run."
            ) from exc

        device = "gpu" if use_gpu else "cpu"

        kwargs: Dict[str, Any] = {
            "lang":          lang,
            "device":        device,
            # MKL-DNN causes a ConvertPirAttribute2RuntimeAttribute crash
            # with PIR-format detection models — keep it off.
            "enable_mkldnn": False,
            "cpu_threads":   cpu_threads,
        }

        if lightweight:
            # Disable the three optional heavyweight preprocessing models
            # to save ~650 MB of RAM.
            kwargs["use_doc_orientation_classify"] = False
            kwargs["use_doc_unwarping"]            = False
            kwargs["use_textline_orientation"]     = False
            logger.info(
                "Lightweight mode: skipping doc-orientation, unwarping, "
                "and textline-orientation models (~650 MB saved)."
            )

        # Memory pre-check — warn if available RAM is dangerously low
        try:
            mem_mb = int(
                open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]
            ) // 1024
            if mem_mb < 1500:
                logger.warning(
                    "LOW MEMORY: only %d MB available. "
                    "Model loading may be killed by the OS (exit 137). "
                    "Close other applications or reduce --max-side to lower peak usage.",
                    mem_mb,
                )
            else:
                logger.info("Available RAM before model load: %d MB", mem_mb)
        except Exception:
            pass

        if rec_model_dir:
            kwargs["text_recognition_model_dir"] = rec_model_dir
            logger.info("Using custom rec model from: %s", rec_model_dir)

        logger.info(
            "Initialising PaddleOCR full pipeline "
            "(lang=%s, device=%s, lightweight=%s)",
            lang, device, lightweight,
        )
        return PaddleOCR(**kwargs)

    def run(
        self,
        images: Sequence[ImageInput],
    ) -> List[Dict[str, Any]]:
        """
        Run detect + recognise on a list of images.

        Returns a list of per-image dicts::

            {
                "image_id"      : str,
                "image_path"    : str,
                "original_shape": {"height": H, "width": W},
                "lines"         : [
                    {
                        "bbox"          : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
                        "text"          : "வணக்கம்",
                        "confidence"    : 0.9731,
                        "low_confidence": False,
                    },
                    ...
                ],
                "full_text"  : "வணக்கம்\\nதமிழ்\\n...",
                "line_count" : int,
                "downscaled" : bool,
            }
        """
        results = []
        for img_path in images:
            img_str = str(img_path)

            # Record original dimensions
            orig_img = Image.open(img_str)
            orig_w, orig_h = orig_img.size
            orig_img.close()

            # Auto-downscale if needed
            path_to_use, scale = _downscale_if_needed(img_str, self.max_side) \
                if self.max_side > 0 else (img_str, 1.0)
            downscaled = scale < 1.0
            if downscaled:
                self._tmp_files.append(path_to_use)

            logger.info(
                "Running full pipeline on: %s%s",
                img_str,
                f" (downscaled ×{scale:.3f})" if downscaled else "",
            )

            try:
                raw_list = self._engine.predict(path_to_use)
            except Exception as exc:
                logger.error("PaddleOCR failed on %s: %s", img_str, exc)
                results.append(self._empty_result(img_str, orig_h, orig_w))
                continue
            finally:
                # Clean up temp file immediately after use
                if downscaled and os.path.exists(path_to_use):
                    os.unlink(path_to_use)
                    self._tmp_files = [f for f in self._tmp_files if f != path_to_use]

            lines = self._parse_raw(raw_list, scale)

            # Sort into reading order using whitespace projection
            sorted_lines, col_ids = _sort_into_reading_order(
                lines, page_width=float(orig_w)
            )
            n_cols = max(col_ids) + 1 if col_ids else 1

            # Build full_text: left col first, then right col, separated by blank line
            parts: List[str] = []
            prev_col = col_ids[0] if col_ids else 0
            for ln, col in zip(sorted_lines, col_ids):
                if col != prev_col:
                    parts.append("")
                parts.append(ln["text"])
                prev_col = col
            full_text = "\n".join(parts)

            results.append(
                {
                    "image_id":       Path(img_str).name,
                    "image_path":     str(Path(img_str).resolve()),
                    "original_shape": {"height": orig_h, "width": orig_w},
                    "columns_detected": n_cols,
                    "lines":          sorted_lines,
                    "full_text":      full_text,
                    "line_count":     len(sorted_lines),
                    "downscaled":     downscaled,
                }
            )
        return results

    def _parse_raw(
        self, raw_list: Any, scale: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Convert PaddleOCR v3 predict() output to our flat line format.

        PaddleOCR v3 result.json["res"] contains:
          rec_texts  : list[str]
          rec_scores : list[float]
          rec_polys  : list of 4-point polygon [[x,y], ...]
        """
        lines: List[Dict[str, Any]] = []
        if not raw_list:
            return lines

        for result in raw_list:
            if result is None:
                continue
            try:
                data = result.json.get("res", {})
                texts  = data.get("rec_texts",  [])
                scores = data.get("rec_scores", [])
                polys  = data.get("rec_polys",  [])

                for text, score, poly in zip(texts, scores, polys):
                    conf = float(score)
                    # Scale bbox coords back to original image space
                    scaled_poly = _scale_bbox(poly, scale)
                    lines.append(
                        {
                            "bbox":           scaled_poly,
                            "text":           text,
                            "confidence":     round(conf, 4),
                            "low_confidence": conf < self.score_threshold,
                        }
                    )
            except (TypeError, AttributeError, KeyError) as exc:
                logger.debug("Could not parse result: %s", exc)
                continue

        return lines

    @staticmethod
    def _empty_result(
        img_path: str, h: int = 0, w: int = 0
    ) -> Dict[str, Any]:
        return {
            "image_id":       Path(img_path).name,
            "image_path":     str(Path(img_path).resolve()),
            "original_shape": {"height": h, "width": w},
            "lines":          [],
            "full_text":      "",
            "line_count":     0,
            "downscaled":     False,
        }

    def __del__(self):
        """Clean up any leftover temp files."""
        for f in self._tmp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except OSError:
                pass
