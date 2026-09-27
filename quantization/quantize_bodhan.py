"""
Quantize bodhan-ai/indic-ocr (local) with AutoRound -> W4A16, group_size=128.

Mirrors the CLI:
  auto_round <model_path> --mllm --dataset calib.jsonl
      --scheme W4A16 --group_size 128 --iters 200 --model_dtype bf16
      --ignore_layers "lm_head,visual,vision_tower,mm_projector" ...

Prereqs:
  pip install auto-round transformers accelerate pandas pyarrow pillow
  (run build_calib.py first to produce calib.jsonl + calib_images/)

Output loads in vLLM / SGLang / transformers (auto_round format).
"""

import os
import sys
from pathlib import Path

import torch
from auto_round import AutoRound
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---------- config ----------
# Model repo or local directory. If pointing to the full repo, "weights/ocr" is used automatically.
MODEL_PATH = "bodhan-ai/indic-ocr"
HF_REPO_ID = "bodhan-ai/indic-ocr"

CALIB_DATA = "calib.jsonl"          # from build_calib.py
OUTPUT_DIR = "bodhan-ai-indic-ocr-AutoRound-W4A16-G128"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SCHEME = "W4A16"                    # 4-bit weights, fp16/bf16 activations
GROUP_SIZE = 128
ITERS = 200
SEED = 42
# vision + head layers stay bf16; add "linear_attn" here if 4-bit hurts OCR CER
IGNORE_LAYERS = "lm_head,visual,vision_tower,mm_projector"
EXPORT_FORMAT = "auto_round"        # standard AutoRound format for vLLM / transformers
# -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Locate or convert calibration file (AutoRound MLLM expects JSON format)
def prepare_calib_data() -> Path:
    candidates = [
        SCRIPT_DIR / "calib.json",
        Path("calib.json"),
        SCRIPT_DIR / "calib.jsonl",
        Path("calib.jsonl"),
    ]
    existing = next((p for p in candidates if p.is_file()), None)
    if existing is None:
        raise FileNotFoundError("calib.json or calib.jsonl not found - run build_calib.py first")

    # If only jsonl exists, convert to json (AutoRound LlavaDataset format)
    if existing.suffix == ".jsonl":
        json_path = existing.with_suffix(".json")
        if not json_path.exists():
            import json
            data = []
            with open(existing) as f:
                for line in f:
                    item = json.loads(line)
                    user_prompt = ""
                    gpt_response = ""
                    for msg in item.get("messages", []):
                        if msg.get("role") == "user":
                            for c in msg.get("content", []):
                                if c.get("type") == "text":
                                    user_prompt = c["text"]
                        elif msg.get("role") == "assistant":
                            for c in msg.get("content", []):
                                if c.get("type") == "text":
                                    gpt_response = c["text"]
                    data.append({
                        "image": item["image"],
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{user_prompt}"},
                            {"from": "gpt", "value": gpt_response},
                        ],
                    })
            with open(json_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Converted {existing.name} -> {json_path.name} ({len(data)} samples)")
        return json_path
    return existing

calib_path = prepare_calib_data()

# Register custom dataset with AutoRound's MLLM registry to prevent KeyError
from auto_round.compressors.mllm.dataset import MLLM_DATASET
MLLM_DATASET[str(calib_path)] = MLLM_DATASET["liuhaotian/llava"]
MLLM_DATASET[calib_path.name] = MLLM_DATASET["liuhaotian/llava"]

# Resolve model directory (local dir or download snapshot via huggingface_hub like bodhan_indic_ocr.py)
def resolve_model_dir(model_path: str | Path | None = None) -> Path:
    if model_path is not None and Path(model_path).exists():
        return Path(model_path)

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or DEFAULT_HF_TOKEN
    )
    print(f"Downloading/verifying Bodhan AI IndicOCR snapshot ({HF_REPO_ID})...")
    downloaded_path = snapshot_download(
        repo_id=HF_REPO_ID,
        token=token,
        local_files_only=False,
    )
    return Path(downloaded_path)

model_dir = resolve_model_dir(MODEL_PATH)

# Inject repository into Python sys.path so any vendored modules are importable
repo_str = str(model_dir.resolve())
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

# IndicOCR text recognition model lives under weights/ocr
if (model_dir / "weights" / "ocr").exists():
    ocr_ckpt = model_dir / "weights" / "ocr"
elif (model_dir / "config.json").exists():
    ocr_ckpt = model_dir
else:
    raise FileNotFoundError(f"OCR checkpoint not found in {model_dir}")

print(f"Loading IndicBlockOCR model from {ocr_ckpt} on {DEVICE}...")

# Load processor and configure padding side flush to generation boundary
processor = AutoProcessor.from_pretrained(str(ocr_ckpt), trust_remote_code=True)
if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
    processor.tokenizer.padding_side = "left"

# Load model matching bodhan_indic_ocr.py / idp_recognizer.py
dtype = torch.bfloat16 if (torch.cuda.is_available() and DEVICE != "cpu") else torch.float32
model = AutoModelForImageTextToText.from_pretrained(
    str(ocr_ckpt),
    dtype=dtype,
    device_map=DEVICE,
    attn_implementation="sdpa" if DEVICE != "cpu" else None,
    trust_remote_code=True,
)
model.eval()
model.config.use_cache = False   # avoid KV-cache warnings during calibration

import torch._dynamo
torch._dynamo.config.suppress_errors = True

output_dir = SCRIPT_DIR / OUTPUT_DIR

autoround = AutoRound(
    model,
    tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
    processor=processor,
    dataset=str(calib_path),
    batch_size=1,                 # VLM images have variable resolutions/token counts
    enable_torch_compile=False,   # prevents dynamo incompatibility with linear_attn kernel
    scheme=SCHEME,
    group_size=GROUP_SIZE,
    iters=ITERS,
    seed=SEED,
    ignore_layers=IGNORE_LAYERS,
    # low GPU memory? uncomment:
    # gradient_accumulate_steps=4, enable_quanted_input=True,
)

print("Starting quantization (this runs per-layer optimization over the calib set)...")
autoround.quantize()

print(f"Saving to {output_dir} ({EXPORT_FORMAT})...")
autoround.save_quantized(str(output_dir), format=EXPORT_FORMAT)

# copy processor/tokenizer files so the quantized repo is self-contained
processor.save_pretrained(str(output_dir))
print("Done.")
