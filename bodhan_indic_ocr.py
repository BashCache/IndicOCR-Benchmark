"""
Document Layout Detection and Text Recognition using Bodhan AI's IndicOCR
(https://huggingface.co/bodhan-ai/indic-ocr).

Architecture:
  1. Layout Detection (IndicDocLayout):
     Based on PP-DocLayoutV3 (33M params). Detects document regions
     (Paragraph, Title, Table, Equation, Header, Footer, Caption, MCQ, etc.)
     with pixel bounding boxes [x0, y0, x1, y1], class labels, and confidence scores.
  2. Intrinsic Reading Order:
     Uses the model's INTRINSIC neural reading-order head (trained via pairwise query
     order logits). Extrinsic techniques like Otsu thresholding, projection profiles,
     XY-cut, or heuristic column clustering are completely unnecessary.
  3. Text Recognition (IndicBlockOCR):
     Based on fine-tuned Qwen3.5-0.8B vision-language model with Sarvam-30B tokenizer.
     Transcribes English and 22 Indian languages, converting mathematical formulas
     to LaTeX and tables to HTML or Markdown.
  4. Reading-Ordered Markdown Assembly:
     Transcribed blocks are sequenced in the model's intrinsic reading order,
     with dehyphenation and math repair applied automatically.

Usage:
------
# 1. Full pipeline (Layout Detection + Text Recognition in Intrinsic Reading Order)
python bodhan_indic_ocr.py --images data/rasi-1.png --visualize

# 2. Layout Detection only (rapid detection of regions and reading-order sequence)
python bodhan_indic_ocr.py --images data/rasi-1.png --layout-only --visualize

# 3. Process a folder of images and save crops
python bodhan_indic_ocr.py --images data/ --visualize --save-crops --output-dir outputs/bodhan_ocr

# 4. Use specific device and custom confidence threshold
python bodhan_indic_ocr.py --images data/tamil-2column.png --device cpu --conf-threshold 0.4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw, ImageFont
import torch
from huggingface_hub import snapshot_download

# Default Hugging Face repo ID
HF_REPO_ID = "bodhan-ai/indic-ocr"

# Automatically resolve HF model snapshot path and inject into sys.path for top-level idp_* imports
try:
    _model_snap_dir = snapshot_download(repo_id=HF_REPO_ID, local_files_only=False)
    if _model_snap_dir not in sys.path:
        sys.path.insert(0, _model_snap_dir)
except Exception:
    pass

try:
    from idp_types import (
        LayoutConfig,
        DedupConfig,
        RecognizerConfig,
        CropConfig,
        TableFormat,
        Block,
    )
    from idp_layout import IndicDocLayoutBackend
    from idp_recognizer import HfRecognizer, CropRequest
    from idp_offline import IndicBlockOCR
    from idp_crops import area_clamp
    from idp_contract import prompt_for, is_transcribed, DROP_TYPES
    from idp_reconstruct import reconstruct
except ImportError:
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bodhan_indic_ocr")

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Distinct color palette for visual overlays (RGBA format for PIL drawing)
LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    # Textual types
    "paragraph": (37, 99, 235),       # Blue
    "text": (37, 99, 235),            # Blue
    "title": (180, 83, 9),            # Amber / Gold
    "chapter-title": (180, 83, 9),    # Amber
    "section-title": (217, 119, 6),   # Orange
    "sub-section-title": (245, 158, 11),
    "header": (13, 148, 136),         # Teal
    "footer": (13, 148, 136),         # Teal
    "page-number": (107, 114, 128),   # Gray
    "footnote": (124, 58, 237),       # Purple
    "caption": (16, 185, 129),        # Emerald
    "table-caption": (16, 185, 129),
    "image-caption": (16, 185, 129),
    # Structural & Special
    "table": (225, 29, 72),           # Rose / Red
    "equation": (147, 51, 234),       # Purple
    "expression": (147, 51, 234),
    "mcq": (5, 150, 105),             # Green
    "question": (2, 132, 199),        # Sky Blue
    "answer": (22, 163, 74),          # Green
    "code": (71, 85, 105),            # Slate
    "list": (79, 70, 229),            # Indigo
    "diagram": (234, 88, 12),         # Orange-Red
    "image": (219, 39, 119),          # Pink
    "chart": (234, 88, 12),
    "reference": (100, 116, 139),     # Slate Gray
}
DEFAULT_COLOR = (99, 102, 241)        # Indigo default


def get_label_color(label: str) -> Tuple[int, int, int]:
    """Returns RGB color for a given label string."""
    clean_label = str(label).strip().lower()
    return LABEL_COLORS.get(clean_label, DEFAULT_COLOR)


def collect_images(sources: List[str]) -> List[str]:
    """Expands files, directories, and glob patterns into a sorted, deduplicated path list."""
    paths: List[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                paths.extend(sorted(p.glob(f"*{ext}")))
                paths.extend(sorted(p.glob(f"*{ext.upper()}")))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            paths.append(p)
        else:
            expanded = [Path(f) for f in sorted(glob.glob(src))]
            if expanded:
                paths.extend(expanded)
            else:
                logger.warning("No image files matched pattern or path: %s", src)

    seen, unique = set(), []
    for p in paths:
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


class BodhanIndicOCR:
    """
    High-level interface for Bodhan AI IndicOCR.
    Handles layout detection and block text recognition without requiring reading order first.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        table_format: str = "html",
        load_layout: bool = True,
        load_ocr: bool = True,
        batch_size: int = 4,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.table_format = table_format
        self.batch_size = batch_size

        # Resolve device (cuda vs cpu)
        if device is None:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device.lower()

        logger.info("Initializing BodhanIndicOCR on device: %s", self.device)

        # Download or locate model repository
        self.model_dir = Path(self._resolve_model_dir(model_dir))
        logger.info("Using model weights directory: %s", self.model_dir)

        # Inject repository into Python sys.path so its vendored modules are importable
        repo_str = str(self.model_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # Initialize internal stage references (lazy loaded on demand)
        self._layout_backend = None
        self._ocr_backend = None

        if load_layout:
            self._init_layout_backend()
        if load_ocr:
            self._init_ocr_backend()

    def _resolve_model_dir(self, model_dir: Optional[Union[str, Path]]) -> Path:
        """Resolves local model directory or downloads snapshot via huggingface_hub."""
        if model_dir is not None and Path(model_dir).exists():
            return Path(model_dir)

        logger.info("Downloading/verifying Bodhan AI IndicOCR snapshot from Hugging Face Hub (%s)...", HF_REPO_ID)

        try:
            downloaded_path = snapshot_download(
                repo_id=HF_REPO_ID,
                local_files_only=False,
            )
            return Path(downloaded_path)
        except Exception as exc:
            logger.error("Failed to download model snapshot from Hugging Face: %s", exc)
            raise RuntimeError(
                f"Could not load '{HF_REPO_ID}'. Please verify your Hugging Face token and internet connection."
            ) from exc

    def _init_layout_backend(self) -> None:
        """Initializes the IndicDocLayout detection model."""
        if self._layout_backend is not None:
            return

        logger.info("Loading IndicDocLayout detection model...")
        layout_ckpt = self.model_dir / "weights" / "layout"
        if not layout_ckpt.exists():
            raise FileNotFoundError(f"Layout checkpoint not found at {layout_ckpt}")

        try:
            cfg = LayoutConfig(
                device=self.device,
                conf=self.conf_threshold,
            )
            dedup = DedupConfig()
            self._layout_backend = IndicDocLayoutBackend(
                ckpt=str(layout_ckpt),
                config=cfg,
                dedup=dedup,
            )
            logger.info("IndicDocLayout loaded successfully.")
        except Exception as e:
            logger.error("Error initializing layout backend: %s", e)
            raise

    def _init_ocr_backend(self) -> None:
        """Initializes the IndicBlockOCR text recognition model."""
        if self._ocr_backend is not None:
            return

        logger.info("Loading IndicBlockOCR text recognition model (Qwen3.5-0.8B)...")
        ocr_ckpt = self.model_dir / "weights" / "ocr"
        if not ocr_ckpt.exists():
            raise FileNotFoundError(f"OCR checkpoint not found at {ocr_ckpt}")

        try:
            rec_cfg = RecognizerConfig(
                table_format=TableFormat(self.table_format),
                dtype="float32" if self.device == "cpu" else "bfloat16",
            )
            hf_rec = HfRecognizer(
                ckpt=str(ocr_ckpt),
                config=rec_cfg,
                device=self.device,
                batch_size=self.batch_size,
            )
            self._ocr_backend = IndicBlockOCR(
                backend=hf_rec,
                config=rec_cfg,
                dedup=DedupConfig(),
                crop=CropConfig(),
            )
            logger.info("IndicBlockOCR loaded successfully.")
        except Exception as e:
            logger.error("Error initializing OCR backend: %s", e)
            raise

    def detect_layout(
        self,
        image: Union[str, Path, Any],
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs layout detection on the input image.
        Uses the model's INTRINSIC neural reading-order head (PP-DocLayoutV3 pairwise query logits).
        Extrinsic techniques (like Otsu thresholding, projection profiles, or geometric column cuts)
        are completely bypassed.

        Returns a list of detected blocks in intrinsic reading order (reading_order=0, 1, 2, ...).
        """
    def detect_layout(
        self,
        image: Union[str, Path, Any],
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs layout detection on the input image.
        Uses the model's INTRINSIC neural reading-order head (PP-DocLayoutV3 pairwise query logits).
        Extrinsic techniques (like Otsu thresholding, projection profiles, or geometric column cuts)
        are completely bypassed.

        Returns a list of detected blocks in intrinsic reading order (reading_order=0, 1, 2, ...).
        """
        self._init_layout_backend()

        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        if conf_threshold is not None:
            self._layout_backend.config.conf = conf_threshold

        # self._layout_backend.detect runs inference, deduplication, and sorts directly
        # by the model's intrinsic pairwise reading-order logits (_densify).
        backend_blocks = self._layout_backend.detect(pil_img)

        blocks: List[Dict[str, Any]] = []
        for b in backend_blocks:
            blocks.append({
                "block_id": f"block_{b.order:03d}",
                "reading_order": b.order,
                "label": b.label,
                "type": b.type,
                "bbox_xyxy": b.bbox_xyxy,
                "conf": b.conf,
                "text": "",
            })

        logger.info(
            "Detected %d layout blocks in intrinsic reading order (conf >= %.2f)",
            len(blocks),
            self._layout_backend.config.conf,
        )
        return blocks

    def draw_text_recognition_bbox(
        self,
        image_or_crop: Union[str, Path, Any],
        block: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        is_crop: bool = True,
        pad_px: int = 0,
        show_text: bool = True,
        line_width: int = 2,
    ) -> Any:
        """
        Draws the text recognition bounding box and metadata badge on a cropped image or full page.
        """
        return draw_text_recognition_bbox(
            image_or_crop=image_or_crop,
            block=block,
            output_path=output_path,
            is_crop=is_crop,
            pad_px=pad_px,
            show_text=show_text,
            line_width=line_width,
        )

    def recognize_blocks(
        self,
        image: Union[str, Path, Any],
        blocks: List[Dict[str, Any]],
        max_blocks: Optional[int] = None,
        save_crops_dir: Optional[Union[str, Path]] = None,
        draw_bboxes: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Performs block OCR on the given list of blocks.
        Blocks are transcribed in their intrinsic reading order.

        When each cropped image is passed to IndicBlockOCR, a bounding box
        for text recognition is drawn (and saved if save_crops_dir is provided).
        """
        self._init_ocr_backend()

        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        width, height = pil_img.size
        crop_cfg = self._ocr_backend.crop
        table_fmt = self._ocr_backend.config.table_format

        # Prepare list of blocks to transcribe
        transcribable_indices = []
        requests: List[CropRequest] = []
        crop_entries = []

        if save_crops_dir:
            Path(save_crops_dir).mkdir(parents=True, exist_ok=True)

        for idx, block in enumerate(blocks):
            if max_blocks is not None and len(requests) >= max_blocks:
                break

            btype = block.get("type", "Text")
            label = block.get("label", "text")

            # Check if block is eligible for text recognition
            if btype in DROP_TYPES or not is_transcribed(label):
                continue

            x0, y0, x1, y1 = [round(v) for v in block["bbox_xyxy"]]
            # Add padding if configured
            pad_px = crop_cfg.pad_px or 0
            if pad_px:
                x0 = max(0, x0 - pad_px)
                y0 = max(0, y0 - pad_px)
                x1 = min(width, x1 + pad_px)
                y1 = min(height, y1 + pad_px)

            if x1 <= x0 or y1 <= y0:
                continue

            crop = pil_img.crop((x0, y0, x1, y1)).convert("RGB")
            clamped_crop = area_clamp(crop, crop_cfg)
            prompt = prompt_for(btype, table_fmt)

            bid = block.get("block_id", f"block_{idx:03d}")
            clean_label = label.replace("/", "_").replace(" ", "_")

            # Draw text recognition bounding box when passing cropped image
            if draw_bboxes and save_crops_dir:
                bbox_crop_path = Path(save_crops_dir) / f"{bid}_{clean_label}_bbox.png"
                self.draw_text_recognition_bbox(
                    image_or_crop=clamped_crop,
                    block=block,
                    output_path=bbox_crop_path,
                    is_crop=True,
                    pad_px=pad_px,
                    show_text=False,
                )
                # Also save raw crop image
                raw_crop_path = Path(save_crops_dir) / f"{bid}_{clean_label}.png"
                clamped_crop.save(str(raw_crop_path))
                logger.info(
                    "Drew text recognition bounding box for crop %s [#%d %s] -> %s",
                    bid,
                    block.get("reading_order", 0),
                    label,
                    bbox_crop_path.name,
                )

            requests.append(CropRequest(image=clamped_crop, prompt=prompt))
            transcribable_indices.append(idx)
            crop_entries.append((idx, clamped_crop, block, pad_px))

        if not requests:
            logger.info("No transcribable text blocks found to recognize.")
            return blocks

        logger.info("Transcribing %d eligible blocks using IndicBlockOCR...", len(requests))
        start_t = time.time()
        transcriptions = self._ocr_backend.backend.transcribe(requests)
        elapsed = time.time() - start_t
        logger.info("Transcribed %d blocks in %.2fs (%.2fs/block)", len(requests), elapsed, elapsed / len(requests))

        # Map transcribed texts back to the original blocks by index
        for block_idx, text in zip(transcribable_indices, transcriptions):
            blocks[block_idx]["text"] = text.strip()

        # Update saved bounding box crops with the transcribed text
        if draw_bboxes and save_crops_dir:
            for b_idx, clamped_crop, block, pad_px in crop_entries:
                if block.get("text"):
                    bid = block.get("block_id", f"block_{b_idx:03d}")
                    clean_label = block.get("label", "text").replace("/", "_").replace(" ", "_")
                    bbox_crop_path = Path(save_crops_dir) / f"{bid}_{clean_label}_bbox.png"
                    self.draw_text_recognition_bbox(
                        image_or_crop=clamped_crop,
                        block=block,
                        output_path=bbox_crop_path,
                        is_crop=True,
                        pad_px=pad_px,
                        show_text=True,
                    )

        return blocks

    def process_image(
        self,
        image_path: Union[str, Path],
        layout_only: bool = False,
        max_blocks: Optional[int] = None,
        save_crops_dir: Optional[Union[str, Path]] = None,
        draw_bboxes: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing for a single document image:
        1. Layout Detection & Intrinsic Reading Order (IndicDocLayout)
        2. Block Text Recognition (IndicBlockOCR, unless layout_only=True)
           (draws text recognition bounding boxes on crops when passed)
        3. Reading-Ordered Markdown Synthesis
        """
        img_path = Path(image_path)
        pil_img = Image.open(str(img_path)).convert("RGB")
        width, height = pil_img.size

        logger.info("Processing document: %s (%dx%d px)", img_path.name, width, height)

        # Stage 1: Layout Detection in intrinsic reading order
        blocks = self.detect_layout(pil_img)

        # Stage 2: Text Recognition per block
        markdown_text = ""
        if not layout_only:
            blocks = self.recognize_blocks(
                pil_img,
                blocks,
                max_blocks=max_blocks,
                save_crops_dir=save_crops_dir,
                draw_bboxes=draw_bboxes,
            )

            # Reconstruct document markdown following intrinsic reading order
            try:
                block_objs = [
                    Block(
                        order=b.get("reading_order", idx),
                        label=b.get("label", "text"),
                        type=b.get("type", "Text"),
                        bbox_xyxy=b["bbox_xyxy"],
                        conf=b.get("conf", 1.0),
                        text=b.get("text", ""),
                    )
                    for idx, b in enumerate(blocks)
                ]
                markdown_text = reconstruct(block_objs)
            except Exception as exc:
                logger.warning("Reconstruct helper unavailable: %s. Using basic sequence join.", exc)
                markdown_text = "\n\n".join(b["text"] for b in blocks if b.get("text"))

        result: Dict[str, Any] = {
            "image": img_path.name,
            "image_path": str(img_path.resolve()),
            "width": width,
            "height": height,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": HF_REPO_ID,
            "device": self.device,
            "total_blocks": len(blocks),
            "reading_order_source": "intrinsic_neural_head (PP-DocLayoutV3 pairwise queries, no extrinsic thresholding)",
            "blocks": blocks,
            "markdown": markdown_text,
        }

        return result


def draw_text_recognition_bbox(
    image_or_crop: Union[str, Path, Any],
    block: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    is_crop: bool = True,
    pad_px: int = 0,
    show_text: bool = True,
    line_width: int = 2,
) -> Any:
    """
    Renders an annotated bounding box for a text recognition block onto a cropped block or full image.

    Parameters:
        image_or_crop : PIL.Image or path (cropped block image if is_crop=True, or full page if is_crop=False).
        block         : Dictionary containing block details:
                        - 'bbox_xyxy': [x0, y0, x1, y1]
                        - 'block_id': str (e.g. 'block_001')
                        - 'reading_order': int
                        - 'label': str (e.g. 'paragraph', 'title', 'table')
                        - 'conf': float (confidence score)
                        - 'text': str (recognized text, optional)
        output_path   : Optional filesystem path to save the resulting image.
        is_crop       : True if image_or_crop is the cropped block; False if it is the full document.
        pad_px        : Padding applied to the crop in pixels (delineates inner text area).
        show_text     : If True, draws a preview overlay of the transcribed text if available.
        line_width    : Width of the bounding box outline.

    Returns:
        Annotated PIL.Image (RGB).
    """
    if isinstance(image_or_crop, (str, Path)):
        base_img = Image.open(str(image_or_crop)).convert("RGBA")
    else:
        base_img = image_or_crop.convert("RGBA")

    w, h = base_img.size
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Scale font size appropriately for crop vs full-page
    try:
        font_size = max(11, min(16, int(h * 0.08))) if is_crop else 14
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        font_small = ImageFont.truetype("DejaVuSans.ttf", max(9, font_size - 3))
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    label = block.get("label", "text")
    block_id = block.get("block_id", "block")
    reading_order = block.get("reading_order", 0)
    conf = block.get("conf", 1.0)
    text = block.get("text", "")

    rgb = get_label_color(label)
    outline_color = (*rgb, 240)
    fill_color = (*rgb, 35)

    if is_crop:
        # Bounding box on the cropped image
        x0 = max(0, pad_px)
        y0 = max(0, pad_px)
        x1 = min(w - 1, w - pad_px) if pad_px < w else w - 1
        y1 = min(h - 1, h - pad_px) if pad_px < h else h - 1

        # Draw semi-transparent inner fill and boundary outline
        draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=line_width)

        # If padding was present, draw subtle outer border
        if pad_px > 0 and (x0 > 0 or y0 > 0 or x1 < w - 1 or y1 < h - 1):
            draw.rectangle([0, 0, w - 1, h - 1], outline=(150, 150, 150, 120), width=1)

        # Header Badge: #{reading_order} {block_id} [{label}] {conf:.2f}
        badge_text = f"#{reading_order} {block_id} [{label}] {conf:.2f}"
        badge_bbox = draw.textbbox((x0, y0), badge_text, font=font)
        badge_w = badge_bbox[2] - badge_bbox[0]
        badge_h = badge_bbox[3] - badge_bbox[1]

        badge_rect = [x0, y0, min(w - 1, x0 + badge_w + 8), min(h - 1, y0 + badge_h + 6)]
        draw.rectangle(badge_rect, fill=(*rgb, 235))
        draw.text((x0 + 4, y0 + 3), badge_text, fill=(255, 255, 255, 255), font=font)

        # Transcribed text badge overlay
        if show_text and text:
            preview = text.strip().replace("\n", " ")
            if len(preview) > 55:
                preview = preview[:52] + "..."
            txt_label = f"OCR: {preview}"
            txt_bbox = draw.textbbox((x0, y1 - 20), txt_label, font=font_small)
            txt_w = txt_bbox[2] - txt_bbox[0]
            txt_h = txt_bbox[3] - txt_bbox[1]
            txt_rect = [x0, max(y0 + badge_h + 8, y1 - txt_h - 6), min(w - 1, x0 + txt_w + 8), y1]
            draw.rectangle(txt_rect, fill=(0, 0, 0, 205))
            draw.text((x0 + 4, txt_rect[1] + 3), txt_label, fill=(255, 255, 255, 240), font=font_small)
    else:
        # Full-page image bounding box
        x0, y0, x1, y1 = [int(round(v)) for v in block["bbox_xyxy"]]
        draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color, width=line_width)

        badge_text = f"#{reading_order} {block_id} [{label}] {conf:.2f}"
        badge_y = max(0, y0 - 20)
        badge_bbox = draw.textbbox((x0, badge_y), badge_text, font=font)
        draw.rectangle(
            [badge_bbox[0] - 2, badge_bbox[1] - 2, badge_bbox[2] + 4, badge_bbox[3] + 2],
            fill=(*rgb, 240),
        )
        draw.text((x0 + 1, badge_y + 1), badge_text, fill=(255, 255, 255, 255), font=font)

        if show_text and text:
            preview = text.strip().replace("\n", " ")
            if len(preview) > 40:
                preview = preview[:37] + "..."
            txt_bbox = draw.textbbox((x0, y0 + 4), preview, font=font_small)
            draw.rectangle(
                [txt_bbox[0] - 2, txt_bbox[1] - 1, txt_bbox[2] + 2, txt_bbox[3] + 1],
                fill=(0, 0, 0, 190),
            )
            draw.text((x0, y0 + 4), preview, fill=(255, 255, 255, 240), font=font_small)

    annotated = Image.alpha_composite(base_img, overlay).convert("RGB")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(str(out_p))

    return annotated


def draw_visual_overlay(
    image: Union[str, Path, Any],
    blocks: List[Dict[str, Any]],
    output_path: Union[str, Path],
    show_text: bool = True,
) -> None:
    """
    Renders an annotated visual overlay with color-coded bounding boxes,
    block ID badges, and labels onto the image.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGBA")
    else:
        pil_img = image.convert("RGBA")

    overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Try loading a readable default font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for block in blocks:
        x0, y0, x1, y1 = [int(round(v)) for v in block["bbox_xyxy"]]
        label = block.get("label", "text")
        block_id = block.get("block_id", "")
        conf = block.get("conf", 1.0)
        text = block.get("text", "")

        rgb = get_label_color(label)
        box_color = (*rgb, 220)
        fill_color = (*rgb, 35)

        # Draw semi-transparent rectangle and border
        draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=box_color, width=2)

        # Draw label badge at top-left of box with intrinsic reading order rank
        reading_order = block.get("reading_order", 0)
        badge_text = f"#{reading_order} {block_id} [{label}] {conf:.2f}"
        badge_bbox = draw.textbbox((x0, max(0, y0 - 18)), badge_text, font=font)
        draw.rectangle(
            [badge_bbox[0] - 2, badge_bbox[1] - 1, badge_bbox[2] + 2, badge_bbox[3] + 1],
            fill=(*rgb, 240),
        )
        draw.text((x0, max(0, y0 - 18)), badge_text, fill=(255, 255, 255, 255), font=font)

        # Optional preview of recognized text
        if show_text and text:
            preview = text[:40].replace("\n", " ") + ("..." if len(text) > 40 else "")
            txt_bbox = draw.textbbox((x0, y0 + 3), preview, font=font_small)
            draw.rectangle(
                [txt_bbox[0] - 1, txt_bbox[1] - 1, txt_bbox[2] + 1, txt_bbox[3] + 1],
                fill=(0, 0, 0, 180),
            )
            draw.text((x0, y0 + 3), preview, fill=(255, 255, 255, 230), font=font_small)

    annotated = Image.alpha_composite(pil_img, overlay).convert("RGB")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(str(out_p))
    logger.info("Saved visual overlay to: %s", out_p)


def save_individual_crops(
    image: Union[str, Path, Any],
    blocks: List[Dict[str, Any]],
    crops_dir: Union[str, Path],
    draw_bboxes: bool = True,
) -> None:
    """
    Saves each detected block as a separate cropped image file.
    Also saves an annotated version with bounding boxes and labels drawn.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    crops_path = Path(crops_dir)
    crops_path.mkdir(parents=True, exist_ok=True)

    for block in blocks:
        x0, y0, x1, y1 = [int(round(v)) for v in block["bbox_xyxy"]]
        if x1 <= x0 or y1 <= y0:
            continue
        crop = pil_img.crop((x0, y0, x1, y1))
        bid = block.get("block_id", "block")
        label = block.get("label", "unknown").replace("/", "_").replace(" ", "_")

        # 1. Save raw crop
        crop_filename = crops_path / f"{bid}_{label}.png"
        crop.save(str(crop_filename))

        # 2. Save crop with bounding box for text recognition
        if draw_bboxes:
            bbox_crop_filename = crops_path / f"{bid}_{label}_bbox.png"
            draw_text_recognition_bbox(
                image_or_crop=crop,
                block=block,
                output_path=bbox_crop_filename,
                is_crop=True,
                pad_px=0,
                show_text=bool(block.get("text")),
            )

    logger.info("Saved %d block crops into: %s", len(blocks), crops_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bodhan_indic_ocr.py",
        description="Layout detection and text recognition using Bodhan AI IndicOCR (PP-DocLayout + Qwen3.5-0.8B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--images", "-i",
        nargs="+",
        required=True,
        help="One or more image files, folder paths, or glob patterns.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs/bodhan_ocr",
        help="Directory to save JSON reports, visual overlays, and crops.",
    )

    # Execution modes
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Run only layout detection (skip text recognition).",
    )
    parser.add_argument(
        "--ocr-only",
        type=str,
        default=None,
        help="Path to an existing layout JSON report; runs text recognition on those blocks directly.",
    )

    # Model & Hardware configurations
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to local Bodhan AI IndicOCR model snapshot. If omitted, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Compute device (defaults to 'cuda' if GPU is present, otherwise 'cpu').",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="Layout detection confidence threshold.",
    )
    parser.add_argument(
        "--table-format",
        choices=["html", "markdown"],
        default="html",
        help="Output format for detected table blocks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for block text transcription.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Maximum number of blocks to transcribe (useful for quick testing on CPU).",
    )

    # Visualization and Artifacts
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save color-coded visual overlay images.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save individual cropped images for each detected layout block.",
    )
    parser.add_argument(
        "--no-crop-bboxes",
        action="store_true",
        help="Disable drawing bounding boxes on cropped images for text recognition.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    image_paths = collect_images(args.images)
    if not image_paths:
        logger.error("No valid image files found matching inputs: %s", args.images)
        return 1

    logger.info("Found %d image(s) to process.", len(image_paths))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the Bodhan IndicOCR engine
    try:
        engine = BodhanIndicOCR(
            model_dir=args.model_dir,
            device=args.device,
            conf_threshold=args.conf_threshold,
            table_format=args.table_format,
            load_layout=True,
            load_ocr=not args.layout_only,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error("Failed to initialize Bodhan IndicOCR: %s", e)
        return 1

    # Process images
    for idx, img_path_str in enumerate(image_paths, 1):
        img_p = Path(img_path_str)
        logger.info("[%d/%d] Processing %s...", idx, len(image_paths), img_p.name)

        crops_dir = (output_dir / f"{img_p.stem}_crops") if args.save_crops else None

        try:
            # Mode A: Run with existing layout JSON
            if args.ocr_only:
                with open(args.ocr_only, encoding="utf-8") as f:
                    layout_data = json.load(f)
                blocks = layout_data.get("blocks", [])
                blocks = engine.recognize_blocks(
                    img_p,
                    blocks,
                    max_blocks=args.max_blocks,
                    save_crops_dir=crops_dir,
                    draw_bboxes=not args.no_crop_bboxes,
                )
                result = {**layout_data, "blocks": blocks}
            else:
                # Mode B: Full or Layout-only pipeline
                result = engine.process_image(
                    img_p,
                    layout_only=args.layout_only,
                    max_blocks=args.max_blocks,
                    save_crops_dir=crops_dir,
                    draw_bboxes=not args.no_crop_bboxes,
                )

            # Save JSON report
            json_filename = output_dir / f"{img_p.stem}_bodhan_ocr.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("Saved JSON report to: %s", json_filename)

            # Save reading-ordered Markdown text if OCR was performed
            if result.get("markdown"):
                md_filename = output_dir / f"{img_p.stem}_bodhan_text.md"
                with open(md_filename, "w", encoding="utf-8") as f:
                    f.write(result["markdown"])
                logger.info("Saved reading-ordered markdown to: %s", md_filename)

            # Generate visual overlay
            if args.visualize:
                overlay_filename = output_dir / f"{img_p.stem}_overlay.png"
                draw_visual_overlay(
                    img_p,
                    result["blocks"],
                    overlay_filename,
                    show_text=not args.layout_only,
                )

            # Save individual block crops (ensuring any non-text blocks like images are also saved)
            if args.save_crops and crops_dir:
                save_individual_crops(
                    img_p,
                    result["blocks"],
                    crops_dir,
                    draw_bboxes=not args.no_crop_bboxes,
                )

        except Exception as err:
            logger.error("Error processing %s: %s", img_p.name, err, exc_info=True)

    logger.info("Processing complete. Results saved in %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
