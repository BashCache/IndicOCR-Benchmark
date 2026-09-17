"""
Central configuration for the Tamil PP-OCRv5 recognition pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PipelineConfig:
    # Model path
    model_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "ta_PP-OCRv5_mobile_rec_infer",
    )

    # Pre-processing
    rec_image_shape: Tuple[int, int, int] = (3, 48, 320)
    rec_image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    rec_image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    # Inference backend
    use_gpu: bool = False
    enable_mkldnn: bool = True
    cpu_threads: int = 4
    batch_size: int = 8

    # Post-processing
    rec_score_threshold: float = 0.5

    # Output & Logging
    log_level: str = "INFO"
    output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "outputs"
    )
