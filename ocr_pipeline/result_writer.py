"""
Serialises OCR pipeline results to a structured JSON file.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .pipeline import OCRResult

logger = logging.getLogger(__name__)


def _image_meta(image_path: str) -> Dict[str, Any]:
    """Returns file size in KB and resolved path for an image file."""
    p = Path(image_path)
    meta: Dict[str, Any] = {
        "image_path": str(p.resolve()),
        "file_size_kb": None,
    }
    if p.exists():
        meta["file_size_kb"] = round(p.stat().st_size / 1024, 2)
    return meta


def build_json_report(
    results: List[OCRResult],
    image_paths: List[str],
    config_dict: Optional[Dict[str, Any]] = None,
    model_name: str = "ta_PP-OCRv5_mobile_rec_infer",
) -> Dict[str, Any]:
    """Assembles structured report dictionary from recognition results."""
    ts = datetime.now(tz=timezone.utc).astimezone().isoformat()

    image_entries: List[Dict[str, Any]] = []
    for idx, (result, img_path) in enumerate(zip(results, image_paths)):
        h, w = result.meta.get("original_shape", (None, None))
        entry: Dict[str, Any] = {
            "index": idx,
            "image_id": result.image_id,
        }
        if isinstance(img_path, str) and Path(img_path).exists():
            entry.update(_image_meta(img_path))
        else:
            entry["image_path"] = str(img_path)
            entry["file_size_kb"] = None

        entry["original_shape"] = {"height": h, "width": w}
        entry["ocr_text"] = result.text
        entry["confidence"] = result.confidence
        entry["low_confidence"] = result.low_confidence

        image_entries.append(entry)

    return {
        "run": {
            "timestamp": ts,
            "model": model_name,
            "total_images": len(results),
            "config": config_dict or {},
        },
        "images": image_entries,
    }


def save_json_report(
    report: Dict[str, Any],
    output_dir: str = "outputs",
    filename: Optional[str] = None,
) -> Path:
    """Writes report dictionary as formatted JSON to output_dir."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ocr_results_{ts_str}.json"

    out_path = out_dir / filename
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("JSON results saved → %s", out_path)
    return out_path
