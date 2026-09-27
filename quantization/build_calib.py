"""
Build an AutoRound calibration dataset from indic-ocr-bench small_representative.

Steps:
  1. Download the parquet (huggingface.co/datasets/sarvamai/indic-ocr-bench,
     file: small_representative/small_representative-00000-of-00001.parquet).
  2. Group by `language`, sample N per language.
  3. Save images to disk as PNG.
  4. Emit calib.jsonl in AutoRound multimodal format:
       {"image": <path>, "messages": [{"role": "user", "content": [...]}]}

Usage:
  pip install pandas pyarrow pillow
  python build_calib.py
"""

import io
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

# ---------- config ----------
SCRIPT_DIR = Path(__file__).resolve().parent

CANDIDATE_PARQUET_PATHS = [
    SCRIPT_DIR / "small_representative" / "small_representative-00000-of-00001.parquet",
    SCRIPT_DIR / "small_representative-00000-of-00001.parquet",
    SCRIPT_DIR.parent / "small_representative" / "small_representative-00000-of-00001.parquet",
    Path("small_representative/small_representative-00000-of-00001.parquet"),
    Path("small_representative-00000-of-00001.parquet"),
]

PARQUET_PATH = next((p for p in CANDIDATE_PARQUET_PATHS if p.exists()), CANDIDATE_PARQUET_PATHS[0])
OUT_DIR = SCRIPT_DIR / "calib_images"
OUT_JSONL = SCRIPT_DIR / "calib.jsonl"
OUT_JSON = SCRIPT_DIR / "calib.json"
SAMPLES_PER_LANG = 10
PROMPT = "Transcribe the text in this image."   # match your real OCR prompt
SEED = 42
# -----------------------------

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} rows, {df['language'].nunique()} languages")

OUT_DIR.mkdir(exist_ok=True)
random.seed(SEED)

records = []
lang_counter = Counter()
for lang, grp in df.groupby("language"):
    n = min(SAMPLES_PER_LANG, len(grp))
    chosen = grp.sample(n=n, random_state=SEED)
    lang_counter[str(lang)] = n
    for _, row in chosen.iterrows():
        img = row["image"]
        # parquet image column may be PIL, dict-of-bytes, or raw bytes
        if isinstance(img, dict):                      # HF datasets "image" feature
            img = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        assert isinstance(img, Image.Image), f"unexpected image type: {type(img)}"

        name = Path(str(row["image_name"])).stem
        safe_lang = str(lang).replace("/", "-")
        img_path = OUT_DIR / f"{safe_lang}_{name}.png"
        img.convert("RGB").save(img_path)

        records.append({
            "image": str(img_path),
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ]},
                # groundtruth kept as reference; AutoRound only needs inputs,
                # but keeping it documents what each sample is.
                {"role": "assistant", "content": [
                    {"type": "text", "text": str(row["gt"])},
                ]},
            ],
        })

random.shuffle(records)
with open(OUT_JSONL, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

llava_records = [
    {
        "image": r["image"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{PROMPT}"},
            {"from": "gpt", "value": r["messages"][1]["content"][0]["text"]},
        ],
    }
    for r in records
]
with open(OUT_JSON, "w") as f:
    json.dump(llava_records, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(records)} samples to {OUT_JSONL} and {OUT_JSON}")
print("Per-language counts:", dict(lang_counter))