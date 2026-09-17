"""
Image loading, resizing, and normalisation for the Tamil PP-OCRv5 recognition model.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray, Image.Image]


class ImageLoader:
    """Loads an image from a file path, PIL Image, or NumPy array into a BGR uint8 array."""

    def __call__(self, img: ImageInput) -> np.ndarray:
        if isinstance(img, (str, Path)):
            pil = Image.open(img).convert("RGB")
            arr = np.array(pil, dtype=np.uint8)[:, :, ::-1]
        elif isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
        elif isinstance(img, np.ndarray):
            arr = img.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        return arr


class RecResizeImg:
    """Resizes an image preserving aspect ratio and pads to target shape (C, H, W)."""

    def __init__(self, image_shape: Tuple[int, int, int] = (3, 48, 320)):
        self.imgC, self.imgH, self.imgW = image_shape

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        ratio = w / float(h)
        new_w = min(math.ceil(self.imgH * ratio), self.imgW)

        resized = np.array(
            Image.fromarray(img[:, :, ::-1]).resize((new_w, self.imgH), Image.LANCZOS)
        )[:, :, ::-1]

        canvas = np.ones((self.imgH, self.imgW, self.imgC), dtype=np.uint8) * 255
        canvas[:, :new_w, :] = resized

        valid_ratio = min(1.0, new_w / self.imgW)
        return canvas, valid_ratio


class Normalise:
    """Normalises uint8 HWC array to float32 CHW array: (x / 255 - mean) / std."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean) / self.std
        return img.astype(np.float32)


class Preprocessor:
    """Batched preprocessing pipeline: load -> resize -> normalise -> stack."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (3, 48, 320),
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.loader = ImageLoader()
        self.resizer = RecResizeImg(image_shape)
        self.normalise = Normalise(mean, std)

    def process_single(
        self, img: ImageInput
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        arr = self.loader(img)
        orig_shape = arr.shape[:2]
        resized, valid_ratio = self.resizer(arr)
        normed = self.normalise(resized)
        return normed, valid_ratio, orig_shape

    def __call__(
        self, images: Sequence[ImageInput]
    ) -> Tuple[np.ndarray, List[float], List[Tuple[int, int]]]:
        tensors, ratios, shapes = [], [], []
        for img in images:
            t, r, s = self.process_single(img)
            tensors.append(t)
            ratios.append(r)
            shapes.append(s)
        batch = np.stack(tensors, axis=0)
        logger.debug("Preprocessed batch shape: %s", batch.shape)
        return batch, ratios, shapes
