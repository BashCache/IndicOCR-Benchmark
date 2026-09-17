"""
CTC greedy decoding for the Tamil PP-OCRv5 recognition model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over the last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class CTCDecoder:
    """Greedy CTC decoder for model logits."""

    def __init__(
        self,
        character_dict: List[str],
        score_threshold: float = 0.5,
    ):
        self.character_list = [" "] + character_dict
        self.blank_idx = 0
        self.score_threshold = score_threshold
        logger.info(
            "CTCDecoder initialised. Vocab size (incl. blank): %d",
            len(self.character_list),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.character_list)

    def decode_single(
        self, logits: np.ndarray, valid_ratio: float = 1.0
    ) -> Tuple[str, float]:
        """Decodes one sequence of logits into text and a mean confidence score."""
        probs = _softmax(logits)

        T = logits.shape[0]
        valid_T = max(1, int(T * valid_ratio))
        probs = probs[:valid_T]

        indices = probs.argmax(axis=-1)
        max_probs = probs.max(axis=-1)

        chars, scores = [], []
        prev = -1
        for idx, prob in zip(indices, max_probs):
            if idx != prev:
                if idx != self.blank_idx and 0 <= idx < len(self.character_list):
                    chars.append(self.character_list[idx])
                    scores.append(float(prob))
            prev = idx

        text = "".join(chars)
        score = float(np.mean(scores)) if scores else 0.0
        return text, score

    def __call__(
        self,
        batch_logits: np.ndarray,
        valid_ratios: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Decodes a batch of logits into text, confidence, and low_confidence flags."""
        n = batch_logits.shape[0]
        if valid_ratios is None:
            valid_ratios = [1.0] * n

        results = []
        for i in range(n):
            text, score = self.decode_single(batch_logits[i], valid_ratios[i])
            results.append(
                {
                    "text": text,
                    "confidence": round(score, 4),
                    "low_confidence": score < self.score_threshold,
                }
            )
            logger.debug("Decoded [%d/%d]: %r  conf=%.4f", i + 1, n, text, score)
        return results
