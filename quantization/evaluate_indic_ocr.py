#!/usr/bin/env python3
"""
evaluate_indic_ocr.py
---------------------
Comprehensive OCR Evaluation Benchmark for Indian Languages
based on Sarvam AI's Indic OCR Bench (https://huggingface.co/datasets/sarvamai/indic-ocr-bench).

Evaluates OCR models across 23 languages (English + 22 scheduled Indian languages)
with a full suite of standard, academic, and practical OCR metrics:

Metrics:
  1. Character Error Rate (CER):
     - Standard CER: sum(Levenshtein(gt, pred)) / sum(len(gt))
     - Macro CER: average per-sample CER
     - Normalized CER: with Unicode NFKC + whitespace normalization
  2. Word Error Rate (WER):
     - Standard WER: sum(WordLevenshtein(gt, pred)) / sum(word_count(gt))
     - Macro WER: average per-sample WER
     - Normalized WER: with case-folding + punctuation removal
  3. Exact Match (EM / Accuracy %):
     - Raw EM: pred == gt
     - Normalized EM: norm(pred) == norm(gt)
  4. Edit Similarity / Character Accuracy:
     - Levenshtein ratio (%): (len_gt + len_pred - dist) / (len_gt + len_pred)
     - Clamped Char Accuracy (%): max(0, 1 - CER) * 100
  5. BLEU Scores:
     - Sentence BLEU (BLEU-1, BLEU-2, BLEU-4) via nltk with smoothing
  6. Generation Quality:
     - Length Ratio: len(pred) / len(gt) (detects repetition or early truncation)
  7. Speed / Latency:
     - Average inference time per image (ms) and throughput (samples/sec)

Usage:
------
# 1. Quick benchmark (5 samples per language) on local parquet with Bodhan OCR
python evaluate_indic_ocr.py --limit-per-lang 5 --device cuda:1

# 2. Evaluate full 'test' split from Hugging Face
python evaluate_indic_ocr.py --dataset sarvamai/indic-ocr-bench --split test --device cuda:1

# 3. Evaluate specific languages
python evaluate_indic_ocr.py --languages Hindi,Tamil,Telugu,English --limit-per-lang 10

# 4. Evaluate an AutoRound quantized model checkpoint
python evaluate_indic_ocr.py --model quantization/bodhan-ai-indic-ocr-AutoRound-W4A16-G128 --device cuda:1

# 5. Offline evaluation of existing predictions JSONL (no GPU needed)
python evaluate_indic_ocr.py --predictions outputs/eval_preds.jsonl
"""

import argparse
from collections import defaultdict
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image

# Third-party metric libraries
try:
    import Levenshtein
except ImportError:
    sys.exit("Please install python-Levenshtein: pip install python-Levenshtein")

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    SMOOTHING = SmoothingFunction().method1
except ImportError:
    sentence_bleu = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_indic_ocr")

DEFAULT_PROMPT = "Transcribe the text in this image."


# ============================================================================
# Text Normalization & Metric Utilities
# ============================================================================

def normalize_text_for_cer(text: str) -> str:
    """Unicode NFKC normalization and whitespace collapsing for Character Error Rate."""
    if not text:
        return ""
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", str(text))
    # Replace all unicode whitespaces with standard space and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_for_wer(text: str) -> str:
    """Whitespace, lowercase, and punctuation stripping for Word Error Rate."""
    text = normalize_text_for_cer(text).lower()
    # Strip common punctuation but keep word characters and spaces
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_sample_metrics(ref: str, pred: str) -> Dict[str, float]:
    """
    Computes all sample-level metrics between reference (ground truth) and prediction.
    """
    ref_raw = ref or ""
    pred_raw = pred or ""

    ref_norm_c = normalize_text_for_cer(ref_raw)
    pred_norm_c = normalize_text_for_cer(pred_raw)

    ref_norm_w = normalize_text_for_wer(ref_raw)
    pred_norm_w = normalize_text_for_wer(pred_raw)

    # 1. Character distances & lengths
    char_len_ref = len(ref_raw)
    char_len_pred = len(pred_raw)
    char_dist_raw = Levenshtein.distance(ref_raw, pred_raw)

    char_len_norm_ref = len(ref_norm_c)
    char_dist_norm = Levenshtein.distance(ref_norm_c, pred_norm_c)

    # Raw CER & Norm CER
    cer_raw = char_dist_raw / max(1, char_len_ref)
    cer_norm = char_dist_norm / max(1, char_len_norm_ref)

    # 2. Word distances & lengths
    ref_words = ref_raw.split()
    pred_words = pred_raw.split()
    word_len_ref = len(ref_words)
    word_len_pred = len(pred_words)
    word_dist_raw = Levenshtein.distance(ref_words, pred_words)

    ref_words_norm = ref_norm_w.split()
    pred_words_norm = pred_norm_w.split()
    word_len_norm_ref = len(ref_words_norm)
    word_dist_norm = Levenshtein.distance(ref_words_norm, pred_words_norm)

    # Raw WER & Norm WER
    wer_raw = word_dist_raw / max(1, word_len_ref)
    wer_norm = word_dist_norm / max(1, word_len_norm_ref)

    # 3. Exact Match
    exact_match = 1.0 if ref_raw == pred_raw else 0.0
    norm_exact_match = 1.0 if ref_norm_c == pred_norm_c else 0.0

    # 4. Levenshtein ratio / Similarity %
    lev_ratio = Levenshtein.ratio(ref_raw, pred_raw)
    char_acc = max(0.0, 1.0 - cer_raw)
    word_acc = max(0.0, 1.0 - wer_raw)

    # 5. Length ratio
    length_ratio = char_len_pred / max(1, char_len_ref)

    # 6. BLEU score
    bleu_1 = 0.0
    bleu_2 = 0.0
    bleu_4 = 0.0
    if sentence_bleu and ref_words:
        try:
            bleu_1 = sentence_bleu([ref_words], pred_words, weights=(1, 0, 0, 0), smoothing_function=SMOOTHING)
            bleu_2 = sentence_bleu([ref_words], pred_words, weights=(0.5, 0.5, 0, 0), smoothing_function=SMOOTHING)
            bleu_4 = sentence_bleu([ref_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SMOOTHING)
        except Exception:
            pass

    return {
        "char_len_ref": char_len_ref,
        "char_len_pred": char_len_pred,
        "char_dist_raw": char_dist_raw,
        "char_len_norm_ref": char_len_norm_ref,
        "char_dist_norm": char_dist_norm,
        "cer_raw": cer_raw,
        "cer_norm": cer_norm,
        "word_len_ref": word_len_ref,
        "word_len_pred": word_len_pred,
        "word_dist_raw": word_dist_raw,
        "word_len_norm_ref": word_len_norm_ref,
        "word_dist_norm": word_dist_norm,
        "wer_raw": wer_raw,
        "wer_norm": wer_norm,
        "exact_match": exact_match,
        "norm_exact_match": norm_exact_match,
        "lev_ratio": lev_ratio,
        "char_acc": char_acc,
        "word_acc": word_acc,
        "length_ratio": length_ratio,
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4,
    }


# ============================================================================
# Dataset Loading
# ============================================================================

def resolve_model_path(model_name_or_path: str) -> Path:
    """
    Resolves model from local disk (handling relative paths whether run from workspace root,
    quantization/ folder, or current working directory), or downloads snapshot from Hugging Face Hub.
    """
    raw_path = Path(model_name_or_path)
    script_dir = Path(__file__).resolve().parent

    candidates = [
        raw_path,
        raw_path.resolve(),
        Path.cwd() / raw_path,
        script_dir / raw_path,
        script_dir.parent / raw_path,
        Path("/home/shruthim/OCR") / raw_path,
        Path("/home/shruthim/OCR/quantization") / raw_path,
    ]

    # Handle cases where path includes a leading folder like 'quantization/...' but CWD is already inside quantization/
    if raw_path.parts and raw_path.parts[0] in ("quantization", script_dir.name):
        sub_path = Path(*raw_path.parts[1:])
        candidates.extend([
            sub_path,
            script_dir / sub_path,
            Path.cwd() / sub_path,
            Path("/home/shruthim/OCR/quantization") / sub_path,
        ])

    for cand in candidates:
        if cand.exists():
            logger.info("Resolved local model path: %s", cand.resolve())
            return cand.resolve()

    # If it is not a local directory, try downloading snapshot from Hugging Face Hub
    token = os.environ.get("HF_TOKEN")
    logger.info("Local path '%s' not found. Checking Hugging Face Hub snapshot...", model_name_or_path)
    from huggingface_hub import snapshot_download
    try:
        downloaded = snapshot_download(repo_id=model_name_or_path, token=token)
        return Path(downloaded)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model '{model_name_or_path}'. It was not found locally in any of:\n"
            + "\n".join(f"  - {str(c)}" for c in candidates[:6])
            + f"\nand could not be downloaded from Hugging Face Hub ({exc})."
        ) from exc


def load_evaluation_data(
    dataset_name: Optional[str] = None,
    split: str = "test",
    parquet_path: Optional[str] = None,
    languages: Optional[List[str]] = None,
    limit: Optional[int] = None,
    limit_per_lang: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads dataset rows from Hugging Face or a local parquet file.
    Returns list of dicts with: image, image_name, gt, language.
    """
    rows = []
    script_dir = Path(__file__).resolve().parent

    # Priority 1: explicitly given parquet file
    if parquet_path:
        cand_p = [Path(parquet_path), script_dir / parquet_path, Path.cwd() / parquet_path, script_dir.parent / parquet_path]
        matched = next((p for p in cand_p if p.exists()), None)
        if not matched:
            raise FileNotFoundError(f"Specified parquet file not found: {parquet_path}")
        logger.info("Loading evaluation dataset from parquet: %s", matched)
        df = pd.read_parquet(matched)
    else:
        # Check if local representative parquet matches or if user requested full HF test split
        local_cand = None
        if split == "small_representative":
            candidates = [
                script_dir / "small_representative" / "small_representative-00000-of-00001.parquet",
                Path("small_representative/small_representative-00000-of-00001.parquet"),
                Path("quantization/small_representative/small_representative-00000-of-00001.parquet"),
            ]
            local_cand = next((p for p in candidates if p.exists()), None)

        if local_cand and dataset_name is None:
            logger.info("Found local dataset parquet: %s. Using it.", local_cand)
            df = pd.read_parquet(local_cand)
        else:
            repo_id = dataset_name or "sarvamai/indic-ocr-bench"
            logger.info("Loading dataset '%s' (split: '%s') from Hugging Face Hub...", repo_id, split)
            from datasets import load_dataset
            token = os.environ.get("HF_TOKEN")
            ds = load_dataset(repo_id, split=split, token=token)
            df = ds.to_pandas()

    logger.info("Loaded dataset with %d rows across %d languages.", len(df), df["language"].nunique())

    # Filter languages if requested
    if languages:
        lang_set = {l.strip().lower() for l in languages}
        df = df[df["language"].str.lower().isin(lang_set)]
        logger.info("Filtered to languages %s: %d rows remain.", languages, len(df))

    # Apply per-language limits
    if limit_per_lang:
        df = df.groupby("language", as_index=False).head(limit_per_lang)
        logger.info("Applied limit_per_lang=%d: %d rows selected.", limit_per_lang, len(df))

    if limit and len(df) > limit:
        df = df.head(limit)
        logger.info("Applied global limit=%d: %d rows selected.", limit, len(df))

    for _, row in df.iterrows():
        img_val = row["image"]
        # Skip samples where the image payload is missing or empty
        if isinstance(img_val, dict) and not img_val.get("bytes") and not (img_val.get("path") and os.path.exists(str(img_val.get("path")))):
            logger.warning("Skipping sample %s (%s): missing image payload in dataset", row.get("image_name"), row.get("language"))
            continue

        lang_val = row.get("language")
        if pd.isna(lang_val) or not str(lang_val).strip() or str(lang_val).strip().lower() == "none":
            lang_val = "Unknown"

        rows.append({
            "image": row["image"],
            "image_name": str(row.get("image_name", "")),
            "gt": str(row.get("gt", "")),
            "language": str(lang_val),
        })

    return rows


def extract_pil_image(img_data: Any) -> Optional[Image.Image]:
    """Extracts a PIL RGB Image from HF dict, bytes, or file path. Returns None if invalid/unreadable."""
    try:
        if isinstance(img_data, Image.Image):
            return img_data.convert("RGB")
        if isinstance(img_data, dict):
            b = img_data.get("bytes")
            if b is not None and len(b) > 0:
                img = Image.open(io.BytesIO(b))
                img.load()
                return img.convert("RGB")
            p = img_data.get("path")
            if p and os.path.exists(str(p)):
                img = Image.open(str(p))
                img.load()
                return img.convert("RGB")
            return None
        if isinstance(img_data, (bytes, bytearray)):
            if len(img_data) > 0:
                img = Image.open(io.BytesIO(img_data))
                img.load()
                return img.convert("RGB")
            return None
        if isinstance(img_data, (str, Path)) and os.path.exists(str(img_data)):
            img = Image.open(str(img_data))
            img.load()
            return img.convert("RGB")
    except Exception as e:
        logger.warning("Failed to decode image (%s): %s", type(e).__name__, e)
        return None
    return None


# ============================================================================
# Model Wrapper
# ============================================================================

class ModelEvaluator:
    """
    Handles model loading and batched/single OCR inference.
    Supports:
      - Bodhan AI IndicOCR (native repository weights)
      - AutoRound quantized checkpoints (local or Hub)
      - Standard HuggingFace VLMs (Qwen2-VL, etc.)
    """

    def __init__(
        self,
        model_name_or_path: str = "bodhan-ai/indic-ocr",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 1024,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

        # Resolve model path (local directory or download from Hugging Face Hub)
        model_path = resolve_model_path(model_name_or_path)

        # Inject repository into Python sys.path so any vendored modules are importable
        repo_str = str(model_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # Check for weights/ocr subfolder (Bodhan pipeline) vs direct checkpoint
        if (model_path / "weights" / "ocr").exists():
            ocr_ckpt = model_path / "weights" / "ocr"
        elif (model_path / "config.json").exists():
            ocr_ckpt = model_path
        else:
            raise FileNotFoundError(f"Could not locate model weights or config.json in {model_path}")

        logger.info("Loading processor from %s...", ocr_ckpt)
        self.processor = AutoProcessor.from_pretrained(str(ocr_ckpt), trust_remote_code=True)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
            self.eos_token_id = self.processor.tokenizer.eos_token_id
        else:
            self.eos_token_id = getattr(self.processor, "eos_token_id", None)

        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
        logger.info("Loading model from %s on %s (%s)...", ocr_ckpt, device, dtype)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(ocr_ckpt),
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def predict_batch(self, images: List[Image.Image]) -> List[str]:
        """Runs inference on a batch of images and returns transcription texts."""
        if not images:
            return []

        import torch

        messages_batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            for _ in images
        ]
        text_prompts = [
            self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            for msgs in messages_batch
        ]
        inputs = self.processor(
            text=text_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        if self.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.eos_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_ids = output_ids[:, input_len:]
        decoded = self.processor.batch_decode(new_ids, skip_special_tokens=True)
        return [p.strip() for p in decoded]

    def predict(self, image: Image.Image) -> str:
        """Runs inference on a single image and returns transcription text."""
        return self.predict_batch([image])[0]


# ============================================================================
# Aggregation & Reporting
# ============================================================================

def aggregate_metrics(sample_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes per-language, micro-average, and macro-average metrics.
    """
    by_lang = defaultdict(list)
    for r in sample_records:
        by_lang[r["language"]].append(r)

    def calc_group_stats(records: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(records)
        if n == 0:
            return {}

        total_ref_chars = sum(r["char_len_ref"] for r in records)
        total_char_dist = sum(r["char_dist_raw"] for r in records)

        total_norm_ref_chars = sum(r["char_len_norm_ref"] for r in records)
        total_norm_char_dist = sum(r["char_dist_norm"] for r in records)

        total_ref_words = sum(r["word_len_ref"] for r in records)
        total_word_dist = sum(r["word_dist_raw"] for r in records)

        total_norm_ref_words = sum(r["word_len_norm_ref"] for r in records)
        total_norm_word_dist = sum(r["word_dist_norm"] for r in records)

        # Micro (corpus-level) error rates: sum(errors) / sum(length)
        micro_cer = (total_char_dist / max(1, total_ref_chars)) * 100.0
        micro_norm_cer = (total_norm_char_dist / max(1, total_norm_ref_chars)) * 100.0

        micro_wer = (total_word_dist / max(1, total_ref_words)) * 100.0
        micro_norm_wer = (total_norm_word_dist / max(1, total_norm_ref_words)) * 100.0

        # Macro (sample-averaged) error rates
        macro_cer = (sum(r["cer_raw"] for r in records) / n) * 100.0
        macro_wer = (sum(r["wer_raw"] for r in records) / n) * 100.0

        exact_match = (sum(r["exact_match"] for r in records) / n) * 100.0
        norm_exact_match = (sum(r["norm_exact_match"] for r in records) / n) * 100.0

        char_acc = (sum(r["char_acc"] for r in records) / n) * 100.0
        lev_similarity = (sum(r["lev_ratio"] for r in records) / n) * 100.0
        word_acc = (sum(r["word_acc"] for r in records) / n) * 100.0

        bleu_1 = (sum(r["bleu_1"] for r in records) / n) * 100.0
        bleu_2 = (sum(r["bleu_2"] for r in records) / n) * 100.0
        bleu_4 = (sum(r["bleu_4"] for r in records) / n) * 100.0
        len_ratio = sum(r["length_ratio"] for r in records) / n

        return {
            "samples": n,
            "micro_cer": round(micro_cer, 2),
            "micro_norm_cer": round(micro_norm_cer, 2),
            "micro_wer": round(micro_wer, 2),
            "micro_norm_wer": round(micro_norm_wer, 2),
            "macro_cer": round(macro_cer, 2),
            "macro_wer": round(macro_wer, 2),
            "exact_match": round(exact_match, 2),
            "norm_exact_match": round(norm_exact_match, 2),
            "char_acc": round(char_acc, 2),
            "lev_similarity": round(lev_similarity, 2),
            "word_acc": round(word_acc, 2),
            "bleu_1": round(bleu_1, 2),
            "bleu_2": round(bleu_2, 2),
            "bleu_4": round(bleu_4, 2),
            "length_ratio": round(len_ratio, 3),
        }

    per_language = {}
    for lang, recs in sorted(by_lang.items()):
        per_language[lang] = calc_group_stats(recs)

    micro_overall = calc_group_stats(sample_records)

    # Macro average across languages
    macro_keys = [
        "micro_cer", "micro_norm_cer", "micro_wer", "micro_norm_wer",
        "exact_match", "norm_exact_match", "char_acc", "lev_similarity",
        "word_acc", "bleu_1", "bleu_2", "bleu_4", "length_ratio"
    ]
    num_langs = max(1, len(per_language))
    macro_overall = {
        k: round(sum(stats[k] for stats in per_language.values()) / num_langs, 2)
        for k in macro_keys
    }
    macro_overall["languages_count"] = num_langs
    macro_overall["total_samples"] = len(sample_records)

    return {
        "per_language": per_language,
        "micro_overall": micro_overall,
        "macro_overall": macro_overall,
    }


def format_markdown_table(summary: Dict[str, Any]) -> str:
    """Renders a clean GitHub-flavored markdown table of the evaluation results."""
    header = (
        "| Language | Samples | Micro CER (%) | Norm CER (%) | Micro WER (%) | Norm WER (%) | Exact Match (%) | Edit Sim (%) | BLEU-4 | Len Ratio |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    rows = []
    for lang, stats in sorted(summary["per_language"].items()):
        rows.append(
            f"| **{lang}** | {stats['samples']} | {stats['micro_cer']:.2f} | {stats['micro_norm_cer']:.2f} | "
            f"{stats['micro_wer']:.2f} | {stats['micro_norm_wer']:.2f} | {stats['exact_match']:.2f} | "
            f"{stats['lev_similarity']:.2f} | {stats['bleu_4']:.2f} | {stats['length_ratio']:.3f} |"
        )

    # Separator
    rows.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    # Overall Micro
    m = summary["micro_overall"]
    rows.append(
        f"| **OVERALL (Micro Avg)** | {m['samples']} | **{m['micro_cer']:.2f}** | **{m['micro_norm_cer']:.2f}** | "
        f"**{m['micro_wer']:.2f}** | **{m['micro_norm_wer']:.2f}** | **{m['exact_match']:.2f}** | "
        f"**{m['lev_similarity']:.2f}** | **{m['bleu_4']:.2f}** | **{m['length_ratio']:.3f}** |"
    )

    # Overall Macro
    mac = summary["macro_overall"]
    rows.append(
        f"| **OVERALL (Macro Avg)** | {mac['total_samples']} | **{mac['micro_cer']:.2f}** | **{mac['micro_norm_cer']:.2f}** | "
        f"**{mac['micro_wer']:.2f}** | **{mac['micro_norm_wer']:.2f}** | **{mac['exact_match']:.2f}** | "
        f"**{mac['lev_similarity']:.2f}** | **{mac['bleu_4']:.2f}** | **{mac['length_ratio']:.3f}** |"
    )

    return header + "\n".join(rows) + "\n"


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR models on Indic-OCR-Bench")
    # Dataset Options
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name (default: sarvamai/indic-ocr-bench)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split: 'test' or 'small_representative'")
    parser.add_argument("--parquet", type=str, default=None, help="Path to local parquet file")
    parser.add_argument("--predictions", type=str, default=None, help="Path to precomputed predictions JSONL for offline evaluation")
    parser.add_argument("--languages", "-l", type=str, default=None, help="Comma-separated list of languages to evaluate (e.g. 'Hindi,Tamil,English')")
    parser.add_argument("--limit", type=int, default=None, help="Maximum total samples to evaluate")
    parser.add_argument("--limit-per-lang", type=int, default=None, help="Maximum samples per language to evaluate")

    # Model Options
    parser.add_argument("--model", type=str, default="bodhan-ai/indic-ocr", help="Model name or weights directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device (cuda:0, cuda:1, cpu, etc.)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt passed to OCR model")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum generated tokens per image")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size for model inference (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Resume evaluation by appending to existing predictions JSONL")

    # Output Options
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save evaluation artifacts")
    parser.add_argument("--output-predictions", type=str, default=None, help="Output path for predictions JSONL")
    parser.add_argument("--output-summary", type=str, default=None, help="Output path for summary metrics JSON")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = args.output_predictions or str(out_dir / "predictions.jsonl")
    summary_path = args.output_summary or str(out_dir / "summary_metrics.json")
    report_md_path = str(out_dir / "evaluation_report.md")

    sample_records: List[Dict[str, Any]] = []

    # Case A: Evaluate precomputed predictions file
    if args.predictions:
        logger.info("Evaluating existing predictions from: %s", args.predictions)
        with open(args.predictions, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                ref = item.get("gt", "")
                pred = item.get("pred", "")
                metrics = compute_sample_metrics(ref, pred)
                record = {**item, **metrics}
                sample_records.append(record)

    # Case B: Run model inference on dataset
    else:
        langs = [x.strip() for x in args.languages.split(",")] if args.languages else None
        data_rows = load_evaluation_data(
            dataset_name=args.dataset,
            split=args.split,
            parquet_path=args.parquet,
            languages=langs,
            limit=args.limit,
            limit_per_lang=args.limit_per_lang,
        )

        if not data_rows:
            sys.exit("No samples matched the selection criteria.")

        evaluator = ModelEvaluator(
            model_name_or_path=args.model,
            device=args.device,
            dtype=args.dtype,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

        seen_images = set()
        if args.resume and os.path.exists(preds_path):
            logger.info("Found existing predictions file at %s. Loading completed samples to resume...", preds_path)
            with open(preds_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        sample_records.append(rec)
                        if "image_name" in rec and rec["image_name"]:
                            seen_images.add(rec["image_name"])
                    except Exception:
                        pass
            initial_count = len(sample_records)
            data_rows = [r for r in data_rows if r["image_name"] not in seen_images]
            logger.info("Resumed from predictions: %d samples already completed. %d remaining to evaluate.",
                        initial_count, len(data_rows))

        logger.info("Starting inference on %d samples...", len(data_rows))
        start_time = time.time()

        batch_size = max(1, args.batch_size)
        total_samples = len(data_rows)
        processed_count = 0
        file_mode = "a" if (args.resume and seen_images) else "w"

        with open(preds_path, file_mode, encoding="utf-8") as pf:
            for b_idx in range(0, total_samples, batch_size):
                batch_rows = data_rows[b_idx : b_idx + batch_size]

                # Safely extract images, filtering out any corrupt images
                valid_items = []
                for r in batch_rows:
                    img = extract_pil_image(r["image"])
                    if img is not None:
                        valid_items.append((r, img))
                    else:
                        logger.warning("Skipping corrupted/unreadable image for sample %s (%s)",
                                       r.get("image_name"), r.get("language"))

                if not valid_items:
                    continue

                curr_rows, batch_imgs = zip(*valid_items)
                curr_rows = list(curr_rows)
                batch_imgs = list(batch_imgs)

                t0 = time.time()
                preds = evaluator.predict_batch(batch_imgs)
                batch_latency_ms = (time.time() - t0) * 1000.0
                per_sample_latency = batch_latency_ms / len(curr_rows)

                for offset, (row, pred) in enumerate(zip(curr_rows, preds)):
                    sample_idx = len(sample_records)
                    ref = row["gt"]
                    lang = row["language"]
                    img_name = row["image_name"]

                    metrics = compute_sample_metrics(ref, pred)
                    record = {
                        "sample_idx": sample_idx,
                        "image_name": img_name,
                        "language": lang,
                        "latency_ms": round(per_sample_latency, 2),
                        "gt": ref,
                        "pred": pred,
                        **metrics,
                    }
                    sample_records.append(record)
                    pf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pf.flush()

                processed_count += len(curr_rows)
                elapsed = time.time() - start_time
                fps = processed_count / max(1e-5, elapsed)
                last_cer = sample_records[-1]["cer_raw"] * 100 if sample_records else 0.0
                logger.info("Processed %d/%d remaining samples (%.2f samples/sec) | Current CER: %.2f%%",
                            processed_count, total_samples, fps, last_cer)

        total_elapsed = time.time() - start_time
        logger.info("Inference completed in %.2fs (%.2f samples/sec)", total_elapsed, processed_count / max(1e-5, total_elapsed))

    # Compute aggregate metrics
    logger.info("Aggregating metrics across %d samples...", len(sample_records))
    summary = aggregate_metrics(sample_records)

    # Format Markdown Table & Console Output
    table_md = format_markdown_table(summary)
    print("\n" + "=" * 80)
    print("INDIC OCR BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(table_md)

    # Save summary JSON
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved summary metrics to: %s", summary_path)

    # Save Markdown report
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Indic OCR Benchmark Evaluation Report\n\n")
        f.write(f"- **Evaluated Samples**: {len(sample_records)}\n")
        f.write(f"- **Languages Evaluated**: {summary['macro_overall']['languages_count']}\n")
        f.write(f"- **Micro CER**: {summary['micro_overall']['micro_cer']}%\n")
        f.write(f"- **Micro Norm-CER**: {summary['micro_overall']['micro_norm_cer']}%\n")
        f.write(f"- **Micro WER**: {summary['micro_overall']['micro_wer']}%\n")
        f.write(f"- **Norm Exact Match**: {summary['micro_overall']['norm_exact_match']}%\n\n")
        f.write("## Per-Language Breakdown\n\n")
        f.write(table_md)
    logger.info("Saved markdown report to: %s", report_md_path)
    logger.info("Saved predictions JSONL to: %s", preds_path)


if __name__ == "__main__":
    main()
