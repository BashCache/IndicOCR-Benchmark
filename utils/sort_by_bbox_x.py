#!/usr/bin/env python3
"""
sort_by_bbox_x.py
-----------------
Dynamically detect columns in OCR output and sort lines into reading order:
  left column (top→bottom) → next column (top→bottom) → ...

Column detection — whitespace projection approach
-------------------------------------------------
The algorithm sweeps the x-axis and finds contiguous ranges where NO bbox
occupies any pixel.  These "x-whitespace corridors" are the true column gaps.

  Step 1  Build the coverage set: for each bbox, mark all x pixels from
          xleft to xright as "covered".

  Step 2  Find uncovered (whitespace) runs. A run qualifies as a column gap
          if its width ≥ min_gap_px.

  Step 3  The boundary between two columns is the MIDPOINT of that whitespace
          run.  No bbox can straddle this midpoint by construction — the entire
          run is empty.

  Step 4  (Optional) Merge runs that are narrower than min_gap_px — these are
          noise from minor indentation or sub-pixel rounding.

  Step 5  Assign each line to a column by its x-centre relative to boundaries.

  Step 6  Sort lines by (column_index, y_top).

Why projection, not left-edge gap or x-centre clustering?
  • Left-edge gap: the gap between left edges (239 → 741 = 502px) places the
    candidate boundary at x=490, but left-col bboxes extend to x=674 —
    they straddle x=490 and falsely "confirm" a rejection.
  • x-centre clustering: wide titles/footers sit in the middle x-range and
    form a false third cluster.
  • Whitespace projection is the classical document-analysis approach and is
    immune to both problems: it only sees where bboxes actually end.

Usage
-----
    python sort_by_bbox_x.py outputs/result.json
    python sort_by_bbox_x.py outputs/result.json --text
    python sort_by_bbox_x.py outputs/result.json --full
    python sort_by_bbox_x.py outputs/result.json --output sorted.json
    python sort_by_bbox_x.py outputs/result.json --min-gap 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Bbox helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xleft(bbox: List[List[int]]) -> int:
    return min(p[0] for p in bbox)

def _xright(bbox: List[List[int]]) -> int:
    return max(p[0] for p in bbox)

def _ytop(bbox: List[List[int]]) -> int:
    return min(p[1] for p in bbox)

def _xcentre(bbox: List[List[int]]) -> float:
    return (_xleft(bbox) + _xright(bbox)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Whitespace projection column detector
# ─────────────────────────────────────────────────────────────────────────────

def _find_column_boundaries(
    lines: List[Dict[str, Any]],
    min_gap_px: int = 0,
    min_gap_pct: float = 0.03,
    floor_px: int = 10,
    max_span_pct: float = 0.40,
) -> Tuple[List[float], int]:
    """
    Find column boundaries by sweeping the x-axis for whitespace corridors.

    Spanning elements (headlines, figures) whose width exceeds
    max_span_pct × page_width are EXCLUDED from the coverage calculation
    so they don't bridge and erase real column gaps.  After boundaries are
    found, every line (including wide ones) is assigned to a column by its
    x-centre in detect_and_sort.

    The effective gap threshold is adaptive:
        effective_min = max(floor_px, page_width * min_gap_pct)
    Override with min_gap_px > 0 to use an absolute pixel value.

    Returns:
        (boundaries, effective_min_used)
    """
    if not lines:
        return [], 0

    # Build all x-intervals
    all_intervals: List[Tuple[int, int]] = [
        (_xleft(ln["bbox"]), _xright(ln["bbox"])) for ln in lines
    ]

    # Infer page width from the rightmost bbox edge
    page_width = max(xr for _, xr in all_intervals)

    # Effective threshold
    if min_gap_px > 0:
        effective_min = min_gap_px
    else:
        effective_min = max(floor_px, int(page_width * min_gap_pct))

    # Exclude wide spanning elements (headlines, figures) from gap detection
    max_span_px = page_width * max_span_pct
    body_intervals = [
        (xl, xr) for xl, xr in all_intervals
        if (xr - xl) <= max_span_px
    ]
    n_excluded = len(all_intervals) - len(body_intervals)
    # (wide elements excluded: n_excluded — logged in detect_and_sort if needed)

    # Use body-only intervals to find gaps
    intervals = sorted(body_intervals)
    merged: List[Tuple[int, int]] = []
    for xl, xr in intervals:
        if merged and xl <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], xr))
        else:
            merged.append((xl, xr))

    # Whitespace runs between merged covered intervals
    boundaries: List[float] = []
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end   = merged[i + 1][0]
        gap_width = gap_end - gap_start
        if gap_width >= effective_min:
            boundaries.append((gap_start + gap_end) / 2.0)

    return boundaries, effective_min



# ─────────────────────────────────────────────────────────────────────────────
# Column assignment
# ─────────────────────────────────────────────────────────────────────────────

def _assign_column(bbox: List[List[int]], boundaries: List[float]) -> int:
    """
    Return 0-based column index for a bbox.
    Assignment uses x-centre so wide spanning elements go to the nearest column.
    """
    cx = _xcentre(bbox)
    for col_idx, b in enumerate(boundaries):
        if cx < b:
            return col_idx
    return len(boundaries)


# ─────────────────────────────────────────────────────────────────────────────
# Main detect + sort
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_sort(
    lines: List[Dict[str, Any]],
    min_gap_px: int = 0,
    min_gap_pct: float = 0.03,
    floor_px: int = 10,
    max_span_pct: float = 0.40,
) -> Tuple[List[Dict[str, Any]], List[int], List[float], int]:
    """
    Detect column boundaries via whitespace projection and sort lines into
    reading order: left column top→bottom, then next column, etc.

    Wide spanning elements (headlines, figures whose width > max_span_pct ×
    page_width) are excluded from gap detection but still assigned to a column
    by their x-centre after boundaries are found.

    Threshold logic (adaptive by default):
        effective_min = max(floor_px, page_width * min_gap_pct)
    Override with min_gap_px > 0 to use an absolute pixel value.

    Returns:
        (sorted_lines, col_ids, boundaries, effective_min)
    """
    if not lines:
        return [], [], [], 0

    boundaries, effective_min = _find_column_boundaries(
        lines, min_gap_px, min_gap_pct, floor_px, max_span_pct
    )

    annotated = [
        (_assign_column(ln["bbox"], boundaries), _ytop(ln["bbox"]), ln)
        for ln in lines
    ]
    annotated.sort(key=lambda t: (t[0], t[1]))

    sorted_lines = [t[2] for t in annotated]
    col_ids      = [t[0] for t in annotated]

    return sorted_lines, col_ids, boundaries, effective_min


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coverage_summary(
    lines: List[Dict[str, Any]],
    boundaries: List[float],
    effective_min: int,
    min_gap_pct: float,
    page_width: int,
    max_span_pct: float = 0.40,
) -> None:
    """Print the whitespace corridor analysis."""
    intervals = sorted((_xleft(ln["bbox"]), _xright(ln["bbox"])) for ln in lines)
    max_span_px = page_width * max_span_pct
    body_intervals = [(xl, xr) for xl, xr in intervals if (xr - xl) <= max_span_px]
    n_wide = len(intervals) - len(body_intervals)
    if n_wide:
        print(f"  Wide elements skipped : {n_wide} (spanning >{max_span_pct:.0%} = {round(max_span_px)}px)")
    intervals = sorted(body_intervals)
    merged: List[Tuple[int, int]] = []
    for xl, xr in intervals:
        if merged and xl <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], xr))
        else:
            merged.append((xl, xr))

    print(f"\n{'─'*64}")
    print(f"  Column detection — whitespace projection")
    print(f"{'─'*64}")
    print(f"  Columns detected  : {len(boundaries) + 1}")
    print(f"  Boundaries at x≈  : {[round(b) for b in boundaries]}")
    print(f"  Page width (inferred) : {page_width}px")
    print(f"  Adaptive threshold    : max(10px, {page_width}×{min_gap_pct:.0%}) = {effective_min}px")
    print(f"  Span filter           : excluding bboxes wider than {max_span_pct:.0%} of page")
    print(f"  Covered x-blocks (body text only):")
    for i, (xl, xr) in enumerate(merged):
        print(f"    block {i+1}: x={xl} → {xr}  (width={xr-xl}px)")
    if len(merged) > 1:
        print(f"  Whitespace gaps   :")
        for i in range(len(merged) - 1):
            gs = merged[i][1]
            ge = merged[i+1][0]
            gw = ge - gs
            flag = " ← COLUMN GAP" if gw >= effective_min else f" (too narrow: {gw}px < {effective_min}px threshold)"
            print(f"    x={gs} → {ge}  (width={gw}px){flag}")
    print(f"{'─'*64}")


def print_table(sorted_lines: List[Dict[str, Any]], col_ids: List[int]) -> None:
    print(f"\n{'#':>3}  {'col':>4}  {'x_left':>7}  {'y_top':>6}  {'conf':>6}  text")
    print("─" * 108)
    for i, (ln, col) in enumerate(zip(sorted_lines, col_ids)):
        xl   = _xleft(ln["bbox"])
        yt   = _ytop(ln["bbox"])
        text = ln["text"]
        disp = text if len(text) <= 60 else text[:57] + "..."
        print(f"{i:>3}  {col+1:>4}  {xl:>7}  {yt:>6}  {ln['confidence']:>6.4f}  {disp}")


def print_full(sorted_lines: List[Dict[str, Any]], col_ids: List[int]) -> None:
    print("\n" + "=" * 90)
    prev = col_ids[0] if col_ids else 0
    for i, (ln, col) in enumerate(zip(sorted_lines, col_ids)):
        if col != prev:
            print(f"\n{'━'*36}  [ Column {col+1} ]  {'━'*36}\n")
        print(f"\n[{i}]  col={col+1}  x_left={_xleft(ln['bbox'])}  y_top={_ytop(ln['bbox'])}")
        print(f"     bbox : {ln['bbox']}")
        print(f"     text : {ln['text']}")
        print(f"     conf : {ln['confidence']}")
        prev = col


def print_text(sorted_lines: List[Dict[str, Any]], col_ids: List[int]) -> None:
    print("\n" + "═" * 70)
    print("  READING ORDER TEXT")
    print("═" * 70 + "\n")
    prev = col_ids[0] if col_ids else 0
    for ln, col in zip(sorted_lines, col_ids):
        if col != prev:
            print(f"\n{'─'*26}  [ Column {col+1} ]  {'─'*26}\n")
        print(ln["text"])
        prev = col


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sort_by_bbox_x.py",
        description=(
            "Detect columns via x-axis whitespace projection and sort OCR lines\n"
            "into reading order (left col top→bottom, then next col, etc.).\n\n"
            "Works automatically for 2, 3, or 4 column layouts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", metavar="JSON_FILE",
                        help="OCR results JSON file.")
    parser.add_argument("--image-index", "-i", type=int, default=0, metavar="N",
                        help="Image entry index (0-based). Default: 0.")
    parser.add_argument("--min-gap", "-g", type=int, default=0, metavar="PX",
                        help=(
                            "Absolute minimum whitespace gap (px) to count as a column "
                            "boundary. Overrides adaptive threshold when set. Default: 0 "
                            "(use adaptive threshold instead)."
                        ))
    parser.add_argument("--min-gap-pct", "-p", type=float, default=0.03, metavar="PCT",
                        help=(
                            "Gap threshold as a fraction of page width (adaptive). "
                            "e.g. 0.03 = 3%% of page width. Used when --min-gap is 0. "
                            "Default: 0.03."
                        ))
    parser.add_argument("--max-span-pct", type=float, default=0.40, metavar="PCT",
                        help=(
                            "Exclude bboxes wider than this fraction of page width from "
                            "gap detection (e.g. headlines spanning multiple columns). "
                            "Default: 0.40 (40%%). Decrease if narrow multi-col headlines "
                            "are erasing column gaps."
                        ))
    parser.add_argument("--full", "-f", action="store_true",
                        help="Print full bbox + text details per line.")
    parser.add_argument("--text", "-t", action="store_true",
                        help="Print plain reading-order text with column separators.")
    parser.add_argument("--output", "-o", default=None, metavar="FILE",
                        help="Save sorted results to a new JSON file.")

    args = parser.parse_args()

    # ── Load ──
    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 1

    data   = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images", [])

    if args.image_index >= len(images):
        print(f"ERROR: --image-index {args.image_index} out of range "
              f"(file has {len(images)} image(s)).", file=sys.stderr)
        return 1

    img_entry = images[args.image_index]
    lines     = img_entry.get("lines", [])

    print(f"\nFile        : {path}")
    print(f"Image       : {img_entry.get('image_id', '?')}  (index {args.image_index})")
    print(f"Lines       : {len(lines)}")
    page_width = max((_xright(ln["bbox"]) for ln in lines), default=0)
    print(f"Page width  : {page_width}px (inferred)")

    # ── Detect columns and sort ──
    sorted_lines, col_ids, boundaries, effective_min = detect_and_sort(
        lines,
        min_gap_px=args.min_gap,
        min_gap_pct=args.min_gap_pct,
        max_span_pct=args.max_span_pct,
    )
    n_cols = len(boundaries) + 1

    # ── Print analysis + table ──
    _coverage_summary(lines, boundaries, effective_min, args.min_gap_pct, page_width, args.max_span_pct)
    print_table(sorted_lines, col_ids)

    if args.full:
        print_full(sorted_lines, col_ids)

    if args.text:
        print_text(sorted_lines, col_ids)

    # ── Save JSON ──
    if args.output:
        out_data   = dict(data)
        out_images = list(images)
        entry      = dict(img_entry)
        entry["lines"]             = sorted_lines
        entry["columns_detected"]  = n_cols
        entry["column_boundaries"] = [round(b) for b in boundaries]
        entry["full_text"] = "\n\n".join(
            "\n".join(
                ln["text"] for ln, c in zip(sorted_lines, col_ids) if c == col
            )
            for col in range(n_cols)
        )
        out_images[args.image_index] = entry
        out_data["images"] = out_images

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
