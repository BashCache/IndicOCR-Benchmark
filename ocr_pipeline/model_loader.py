"""
Parses inference.yml and constructs the PaddlePaddle inference predictor.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

try:
    import paddle.inference as paddle_infer
except ImportError:
    paddle_infer = None

logger = logging.getLogger(__name__)


def load_model_config(model_dir: str) -> Dict[str, Any]:
    """Parses inference.yml and returns character dictionary and image shape."""
    yml_path = Path(model_dir) / "inference.yml"
    if not yml_path.exists():
        raise FileNotFoundError(f"inference.yml not found in {model_dir}")

    with open(yml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    char_dict: List[str] = cfg.get("PostProcess", {}).get("character_dict", [])
    if not char_dict:
        raise ValueError("character_dict is empty in inference.yml")

    transform_ops = cfg.get("PreProcess", {}).get("transform_ops", [])
    image_shape = (3, 48, 320)
    for op in transform_ops:
        if isinstance(op, dict) and "RecResizeImg" in op:
            shape = op["RecResizeImg"].get("image_shape", image_shape)
            image_shape = tuple(shape)
            break

    return {
        "character_dict": char_dict,
        "image_shape": image_shape,
        "raw_yaml": cfg,
    }


class PaddlePredictor:
    """Wrapper around paddle.inference.Predictor for PIR format models."""

    def __init__(
        self,
        model_dir: str,
        use_gpu: bool = False,
        enable_mkldnn: bool = True,
        cpu_threads: int = 4,
    ):
        self.model_dir = model_dir
        self._predictor = self._build_predictor(
            model_dir, use_gpu, enable_mkldnn, cpu_threads
        )
        self._input_names = self._predictor.get_input_names()
        self._output_names = self._predictor.get_output_names()
        logger.info(
            "Predictor ready. Inputs: %s Outputs: %s",
            self._input_names,
            self._output_names,
        )

    @staticmethod
    def _build_predictor(
        model_dir: str,
        use_gpu: bool,
        enable_mkldnn: bool,
        cpu_threads: int,
    ):
        if paddle_infer is None:
            raise ImportError(
                "PaddlePaddle is not installed. "
                "Install via: pip install paddlepaddle (CPU) or "
                "pip install paddlepaddle-gpu (GPU)."
            )

        json_path = os.path.join(model_dir, "inference.json")
        params_path = os.path.join(model_dir, "inference.pdiparams")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"inference.json not found in {model_dir}")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"inference.pdiparams not found in {model_dir}")

        config = paddle_infer.Config()
        logger.info("Loading PIR format model (inference.json)")
        config.set_model(json_path, params_path)

        if use_gpu:
            config.enable_use_gpu(memory_pool_init_size_mb=512, device_id=0)
            logger.info("Inference backend: GPU")
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_threads)
            if enable_mkldnn:
                config.enable_mkldnn()
                logger.info("Inference backend: CPU + MKL-DNN (%d threads)", cpu_threads)
            else:
                logger.info("Inference backend: CPU (%d threads)", cpu_threads)

        config.disable_glog_info()
        config.switch_ir_optim(True)

        return paddle_infer.create_predictor(config)

    def run(self, batch: np.ndarray) -> np.ndarray:
        """Executes forward pass on batch (N, C, H, W) and returns logits (N, T, vocab_size)."""
        inp_handle = self._predictor.get_input_handle(self._input_names[0])
        inp_handle.reshape(list(batch.shape))
        inp_handle.copy_from_cpu(batch)

        self._predictor.run()

        out_handle = self._predictor.get_output_handle(self._output_names[0])
        return out_handle.copy_to_cpu()
