# Indic Document OCR & Multi-Column Reading Order Pipeline

A modular Python repository for **Document Layout Detection**, **Multi-Column Reading-Order Recovery**, and **Multilingual Text Recognition** (English, Tamil, and 22+ Indic languages).

The repository integrates two complementary OCR engines:
1. **Bodhan AI IndicOCR** (`bodhan_indic_ocr.py`): Neural layout detection (**PP-DocLayoutV3** with intrinsic pairwise reading-order head) + Vision-Language Model (**Qwen3.5-0.8B**) text recognition.
2. **PaddleOCR Pipeline** (`run_ocr.py`, `layout_detect.py`): Lightweight layout detection (**PP-DocLayout / PicoDet-LCNet**) + fast mobile recognition (**PP-OCRv5**) with adaptive whitespace-projection reading-order sorting.

---

## 🌟 Key Features

* **Intrinsic & Projection Reading-Order Recovery**: Correctly orders text blocks across 1, 2, 3, 4, or 5-column documents, multi-column newspapers, and research papers.
* **Layout Detection**: Detects titles, paragraphs, section titles, headers, footers, tables, figures, equations, and captions.
* **Block Text Recognition with Visual Crops**: Generates color-coded visual overlays and per-block crops with text recognition bounding boxes (`<block_id>_<label>_bbox.png`).
* **Structured JSON & Markdown Output**: Synthesizes structured document representations and clean Markdown formatted text.

---

## 📂 Repository Structure

```text
├── bodhan_indic_ocr.py        # Bodhan AI IndicOCR engine (PP-DocLayoutV3 + Qwen3.5-0.8B VLM)
├── layout_detect.py           # Standalone PaddleOCR Layout Detection (PP-DocLayout / PicoDet)
├── run_ocr.py                 # Full end-to-end PaddleOCR pipeline CLI
├── ocr_pipeline/              # Modular OCR pipeline components
│   ├── full_pipeline.py       # End-to-end multi-column pipeline orchestrator
│   └── ...
├── fix_layout_reordering.py   # Multi-column reading order reordering utilities
├── single_col_fix_reading_order.py # Single/Multi-column projection profile sorter
├── requirements.txt           # Python dependencies
└── data/                      # Input image directory
```

---

## 🛠️ Installation

### 1. Prerequisites
* Python 3.9+ (Python 3.10 – 3.12 recommended)
* PyTorch (CPU or CUDA 11.8/12.4)

### 2. Environment Setup

```bash
# Clone the repository and navigate into it
cd OCR

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/python

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow huggingface_hub accelerate
```

*For GPU support (recommended for VLM text recognition), install PyTorch with CUDA:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 🚀 Quick Start & Usage

### 1. Bodhan AI IndicOCR (`bodhan_indic_ocr.py`)

Runs **PP-DocLayoutV3** layout detection and **Qwen3.5-0.8B** text recognition in intrinsic reading order.

#### Full Pipeline (Layout + Text Recognition + Visual Overlay + Crops)
```bash
python bodhan_indic_ocr.py \
    --images data/ \
    --visualize \
    --save-crops \
    --output-dir outputs/bodhan_ocr \
    --conf-threshold 0.4
```

#### Fast Layout-Only Mode (Bounding Boxes & Reading Order in ~2s/page)
```bash
python bodhan_indic_ocr.py \
    --images data/tamil-2column.png \
    --layout-only \
    --visualize \
    --output-dir outputs/bodhan_layout
```

#### Key Arguments for `bodhan_indic_ocr.py`:
* `--images`, `-i`: File paths, directories, or glob patterns.
* `--layout-only`: Skips text recognition and only runs layout detection & reading order.
* `--visualize`: Saves color-coded overlay images showing bounding boxes and reading order ranks (`#0`, `#1`, ...).
* `--save-crops`: Saves raw crops (`<block_id>_<label>.png`) and annotated crops with text recognition bounding boxes (`<block_id>_<label>_bbox.png`).
* `--device {cpu,cuda}`: Specifies compute device (defaults to `cuda` if GPU available, otherwise `cpu`).
* `--max-blocks N`: Limits the maximum number of blocks to transcribe per page (useful for testing).

---

### 2. Standalone PaddleOCR Layout Detection (`layout_detect.py`)

Uses PaddleOCR's **PP-DocLayout (PicoDet-LCNet)** for lightweight layout region extraction.

```bash
# Detect layout and generate visual overlays for all images in data/
python layout_detect.py --images data/ --visualize --save-crops --output-dir outputs/layout_results
```

---

### 3. Full Multi-Column OCR Pipeline (`run_ocr.py`)

Runs layout detection, adaptive whitespace-projection reading-order sorting, and line-level OCR text recognition:

```bash
python run_ocr.py --images data/eng-news-multicolumn.png --full-pipeline --output-dir outputs/pipeline_results
```

---

## 📊 Reading Order Algorithms

1. **Intrinsic Neural Reading Order (Bodhan AI)**: Uses PP-DocLayoutV3's pairwise query logit head. It evaluates reading order sequence directly inside the model's neural network without geometric rules or column thresholds.
2. **Whitespace Projection Column Sorting (`single_col_fix_reading_order.py` / `ocr_pipeline`)**:
   * Merges bounding box x-intervals to project column boundaries.
   * Dynamically excludes multi-column headlines (`--max-span-pct 0.40`).
   * Detects gutters using an adaptive threshold (`max(10px, page_width * 3%)`).

---

## 📈 Outputs & Artifacts

For each processed document (`<filename>.<ext>`), output files are saved in `--output-dir`:
* `<filename>_bodhan_ocr.json` / `<filename>_layout.json`: Full JSON report containing image metadata, bounding box coordinates (`[x0, y0, x1, y1]`), confidence scores, labels, reading order, and transcribed text.
* `<filename>_bodhan_text.md`: Synthesized document text structured in reading order.
* `<filename>_overlay.png`: High-contrast annotated visual image overlay.
* `<filename>_crops/`: Folder containing raw block crops and annotated text-recognition bounding box crops.
