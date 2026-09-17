#!/usr/bin/env python3
"""
fix_layout_reading_order.py

Reading-order fix for LAYOUT-LEVEL regions (paragraphs, titles, images,
figure captions) from a document layout detector (e.g. PP-DocLayout,
LayoutParser, Detectron2 layout models) -- as opposed to fix_reading_order.py,
which orders individual OCR text LINES within a single region.

Problem it solves
------------------
Layout detectors correctly localize regions but emit them in detection
order, not reading order. On a multi-column page (newspapers, journals,
magazines), a plain top-to-bottom / left-to-right sort interleaves
columns row-by-row, because two unrelated columns often sit at similar
y-heights on the page.

Core idea
---------
This reuses the SAME gap-detection + Otsu-threshold technique already
built for ordering OCR lines within a region (fix_reading_order.py's
otsu_threshold_1d), just applied to a different axis:

  - Line ordering:   cluster boxes by Y-center gaps  -> groups = lines
  - Column ordering: cluster boxes by X-center gaps  -> groups = columns

Algorithm
---------
1. Pull out "full-width" elements (headers, footers, bylines that span
   most of the page width) -- these aren't part of the column grid and
   are read as their own horizontal band, ordered by y.
2. Cluster the remaining ("body") elements' x-centers via gap-based
   clustering with an Otsu-derived threshold (self-calibrated from this
   page's own column-gap distribution, same as the line-ordering script).
   This adapts automatically to however many columns the page has, and
   to any local sub-column splits (e.g. a column narrowing around an
   inset photo or pull-quote), since those sub-splits still show up as
   larger-than-normal x-gaps in the same underlying distribution.
3. Sort column clusters left to right by mean x-center.
4. Within each column cluster, sort elements top to bottom by y-center.
5. Concatenate: full-width band(s) first (by y), then columns left to
   right, each internally top-to-bottom.

Known limitation -- floating/spanning elements
------------------------------------------------
An element that visually straddles two columns (a pull-quote centered
in the gutter between two narrowed columns, or a photo spanning two
sub-columns beneath it) has an x-center that sits ambiguously between
two clusters. The gap-clustering algorithm will resolve it one of two
ways depending on how the surrounding gaps calibrate: merge it into
whichever neighboring column it's closer to, or -- if it's roughly
equidistant -- split it into its own singleton "column" that reads
between its neighbors. Both are defensible geometric conventions for
call-out content, but pull-quotes are inherently ambiguous even for a
human reader (they're deliberately designed to be read independently of
the linear flow). This script flags such elements in its output
(`"_flag": "narrow_or_spanning"`) so a human reviewer -- or, in a larger
pipeline, a model like LayoutReader trained on human-annotated reading
order -- can make the final call rather than silently trusting geometry.

Usage
-----
    python3 fix_layout_reading_order.py layout.json -o corrected_layout.json \
        [--fullwidth-ratio 0.6] [--rtl]
"""

import argparse
import json
import sys


# ---- shared primitives (same technique as fix_reading_order.py, generalized to an axis) ----

def otsu_threshold_1d(values):
    """1D Otsu's method. See fix_reading_order.py for full derivation notes."""
    if len(values) < 2:
        return None, 0.0
    vals = sorted(values)
    n = len(vals)
    total_sum = sum(vals)
    total_mean = total_sum / n
    total_var = sum((v - total_mean) ** 2 for v in vals) / n
    if total_var == 0:
        return None, 0.0
    best_thresh = None
    best_between_var = -1.0
    sum1 = 0.0
    for i in range(n - 1):
        sum1 += vals[i]
        n1 = i + 1
        n2 = n - n1
        sum2 = total_sum - sum1
        mean1 = sum1 / n1
        mean2 = sum2 / n2
        between_var = (n1 / n) * (n2 / n) * (mean1 - mean2) ** 2
        if between_var > best_between_var:
            best_between_var = between_var
            best_thresh = (vals[i] + vals[i + 1]) / 2.0
    return best_thresh, best_between_var / total_var


def gap_cluster_1d(items, get_center, fallback_threshold, min_separation_score=0.3):
    """
    Generic 1-axis gap clustering, self-calibrated via Otsu on the
    center-to-center gaps of `items` (sorted by get_center). Falls back
    to `fallback_threshold` if the gap distribution isn't cleanly
    bimodal (e.g. everything really is one column/line).

    Returns: list of clusters, each a list of original indices.
    """
    n = len(items)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    centers = [get_center(it) for it in items]
    order = sorted(range(n), key=lambda i: centers[i])
    sorted_centers = [centers[i] for i in order]

    gaps = [sorted_centers[i + 1] - sorted_centers[i] for i in range(n - 1)]
    thresh, score = otsu_threshold_1d(gaps)
    if thresh is None or score < min_separation_score:
        thresh = fallback_threshold

    clusters = []
    current = [order[0]]
    running_sum = sorted_centers[0]
    for k in range(1, n):
        idx = order[k]
        c = sorted_centers[k]
        running_mean = running_sum / len(current)
        if abs(c - running_mean) <= thresh:
            current.append(idx)
            running_sum += c
        else:
            clusters.append(current)
            current = [idx]
            running_sum = c
    clusters.append(current)
    return clusters


# ---- layout-specific helpers ----

def x_center(bbox):
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0


def y_center(bbox):
    x0, y0, x1, y1 = bbox
    return (y0 + y1) / 2.0


def width(bbox):
    return bbox[2] - bbox[0]


def dedupe(elements):
    """Layout detectors sometimes emit the same region twice (e.g. a
    preview pass + a final pass). Drop exact-duplicate (label, bbox)
    pairs, keeping the first occurrence's order as a tiebreak."""
    seen = set()
    out = []
    for el in elements:
        key = (el["label"], tuple(round(v, 1) for v in el["bbox"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(el)
    return out


def fix_layout_reading_order(elements, page_width, fullwidth_ratio=0.6, rtl=False,
                              spanning_width_ratio=1.5):
    elements = dedupe(elements)
    if not elements:
        return []

    page_width = max(float(page_width or 1.0), 1.0)

    fullwidth, body = [], []
    for el in elements:
        if width(el["bbox"]) / page_width >= fullwidth_ratio:
            fullwidth.append(el)
        else:
            body.append(el)

    fullwidth.sort(key=lambda el: y_center(el["bbox"]))

    # ---- Pass 1: establish clean column boundaries from "normal-width"
    # elements only. A wide element (e.g. a photo spanning two narrower
    # sub-columns beneath it) has a center that can coincidentally land
    # close to a neighboring column's members, causing it to get
    # sequentially merged into a column it doesn't really belong to --
    # pure center-distance clustering has no notion of "this element is
    # wide enough to span two columns," it just sees nearby points. So
    # we exclude outlier-wide elements from the boundary-finding pass.
    widths = sorted(width(el["bbox"]) for el in body) or [100.0]
    median_width = widths[len(widths) // 2]
    spanning_cutoff = spanning_width_ratio * median_width

    seed = [el for el in body if width(el["bbox"]) <= spanning_cutoff]
    spanning = [el for el in body if width(el["bbox"]) > spanning_cutoff]

    fallback_threshold = 0.5 * median_width
    col_clusters = gap_cluster_1d(
        seed,
        get_center=lambda el: x_center(el["bbox"]),
        fallback_threshold=fallback_threshold,
    )
    # column x-range = min/max extent of its seed members (used for
    # overlap-based assignment of spanning elements in pass 2)
    columns = []
    for idxs in col_clusters:
        x0 = min(seed[i]["bbox"][0] for i in idxs)
        x1 = max(seed[i]["bbox"][2] for i in idxs)
        columns.append({"x0": x0, "x1": x1, "members": [seed[i] for i in idxs]})
    columns.sort(key=lambda c: (c["x0"] + c["x1"]) / 2, reverse=rtl)

    # A column formed from just 1-2 seed elements, sitting among several
    # wider columns, is itself a red flag -- likely a small floating
    # caption/credit line or an isolated aside that geometry alone
    # can't confidently place, rather than a genuine printed column.
    if len(columns) > 2:
        for col in columns:
            if len(col["members"]) <= 2:
                for el in col["members"]:
                    el["_flag"] = "narrow_or_spanning"

    # ---- Pass 2: assign each spanning/wide element to the column it
    # overlaps most. If no column dominates (overlap is split roughly
    # evenly across 2+ columns -- a genuine pull-quote/graphic straddling
    # a gutter), flag it for human review instead of guessing.
    for el in spanning:
        x0, y0, x1, y1 = el["bbox"]
        el_width = max(x1 - x0, 1.0)
        overlaps = []
        for col in columns:
            ov = max(0.0, min(x1, col["x1"]) - max(x0, col["x0"]))
            overlaps.append(ov)
        total_ov = sum(overlaps)
        if total_ov == 0:
            if columns:
                best_col = min(columns, key=lambda c: abs((c["x0"] + c["x1"]) / 2 - x_center(el["bbox"])))
                best_col["members"].append(el)
            else:
                columns.append({"x0": x0, "x1": x1, "members": [el]})
            el["_flag"] = "narrow_or_spanning"
            continue
        best_idx = max(range(len(columns)), key=lambda i: overlaps[i])
        best_frac = overlaps[best_idx] / el_width
        columns[best_idx]["members"].append(el)
        if best_frac < 0.75:  # no single column clearly dominates
            el["_flag"] = "narrow_or_spanning"

    ordered_body = []
    for col in columns:
        members_sorted = sorted(col["members"], key=lambda el: y_center(el["bbox"]))
        ordered_body.extend(members_sorted)

    ordered = fullwidth + ordered_body
    for n, el in enumerate(ordered, start=1):
        el["_corrected_order"] = n
    return ordered


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_json", help="Path to layout detection results JSON file.")
    ap.add_argument("-o", "--output", default="corrected_layout.json", help="Output JSON path.")
    ap.add_argument("--fullwidth-ratio", type=float, default=0.6,
                     help="Elements whose width/page_width exceeds this are treated as "
                          "full-width bands (headers/footers), not part of the column grid.")
    ap.add_argument("--rtl", action="store_true", help="Right-to-left column reading order.")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support batch layout results ("results": [...]), list of results, or single image result dict
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        target_items = data["results"]
    elif isinstance(data, list):
        target_items = data
    else:
        target_items = [data]

    for item in target_items:
        if not isinstance(item, dict) or "elements" not in item:
            continue

        page_width = None
        if "image_shape" in item and isinstance(item["image_shape"], dict):
            page_width = item["image_shape"].get("width")

        if not page_width:
            coords = [el["bbox"][2] for el in item["elements"] if "bbox" in el and len(el["bbox"]) >= 3]
            page_width = max(coords) if coords else 1000.0

        ordered = fix_layout_reading_order(
            item["elements"], page_width,
            fullwidth_ratio=args.fullwidth_ratio, rtl=args.rtl,
        )
        item["elements"] = ordered

        img_name = item.get("image_name", "Image")
        print(f"\n--- {img_name} ({len(ordered)} elements) ---")
        for el in ordered:
            flag = f"  [{el['_flag']}]" if "_flag" in el else ""
            print(f"{el['_corrected_order']:>3}  {el['label']:<16} id={el.get('layout_id')}{flag}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nCorrected layout JSON written to: {args.output}")


if __name__ == "__main__":
    sys.exit(main())