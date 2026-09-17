"""
Orchestrates the end-to-end Tamil OCR recognition pipeline:
    ImageBatch -> Preprocessor -> PaddlePredictor -> CTCDecoder -> Results
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .config import PipelineConfig
from .model_loader import PaddlePredictor, load_model_config
from .postprocessor import CTCDecoder
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray]


class OCRResult:
    """Container for per-image recognition output."""

    __slots__ = ("image_id", "text", "confidence", "low_confidence", "meta")

    def __init__(
        self,
        image_id: str,
        text: str,
        confidence: float,
        low_confidence: bool,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.image_id = image_id
        self.text = text
        self.confidence = confidence
        self.low_confidence = low_confidence
        self.meta = meta or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "text": self.text,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            **self.meta,
        }

    def __repr__(self) -> str:
        flag = " [LOW-CONF]" if self.low_confidence else ""
        return f"OCRResult({self.image_id!r}: {self.text!r} conf={self.confidence:.4f}{flag})"


class OCRPipeline:
    """Full recognition pipeline for Tamil PP-OCRv5."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._setup_logging()

        model_dir = self.cfg.model_dir
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_cfg = load_model_config(model_dir)
        char_dict = model_cfg["character_dict"]
        image_shape = model_cfg["image_shape"]
        logger.info("Character dict size: %d", len(char_dict))
        logger.info("Model input shape (C,H,W): %s", image_shape)

        self.preprocessor = Preprocessor(
            image_shape=image_shape,
            mean=self.cfg.rec_image_mean,
            std=self.cfg.rec_image_std,
        )
        self.predictor = PaddlePredictor(
            model_dir=model_dir,
            use_gpu=self.cfg.use_gpu,
            enable_mkldnn=self.cfg.enable_mkldnn,
            cpu_threads=self.cfg.cpu_threads,
        )
        self.decoder = CTCDecoder(
            character_dict=char_dict,
            score_threshold=self.cfg.rec_score_threshold,
        )

        logger.info("OCRPipeline ready.")

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    @staticmethod
    def _image_id(img: ImageInput, idx: int) -> str:
        if isinstance(img, (str, Path)):
            return Path(img).name
        return f"image_{idx:04d}"

    def _run_batch(
        self,
        batch_images: Sequence[ImageInput],
        batch_ids: List[str],
    ) -> List[OCRResult]:
        """Runs one batch through preprocessor -> predictor -> decoder."""
        batch_tensor, valid_ratios, orig_shapes = self.preprocessor(batch_images)
        logits = self.predictor.run(batch_tensor)
        decoded = self.decoder(logits, valid_ratios)

        results = []
        for i, (dec, img) in enumerate(zip(decoded, batch_images)):
            h, w = orig_shapes[i]
            results.append(
                OCRResult(
                    image_id=batch_ids[i],
                    text=dec["text"],
                    confidence=dec["confidence"],
                    low_confidence=dec["low_confidence"],
                    meta={"original_shape": (h, w)},
                )
            )
        return results

    def run(self, images: Sequence[ImageInput]) -> List[OCRResult]:
        """Runs the OCR pipeline on a list of images in batches."""
        all_results: List[OCRResult] = []
        batch_size = self.cfg.batch_size

        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            ids = [self._image_id(img, start + i) for i, img in enumerate(batch)]
            logger.info(
                "Processing batch %d–%d / %d",
                start + 1,
                min(start + batch_size, len(images)),
                len(images),
            )
            all_results.extend(self._run_batch(batch, ids))

        return all_results
