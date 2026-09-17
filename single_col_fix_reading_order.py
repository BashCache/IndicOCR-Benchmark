#!/usr/bin/env python3
"""
fix_reading_order.py

General-purpose fix for OCR bounding-box reading-order errors.

Problem it solves
------------------
OCR / layout pipelines often emit boxes in *detection order* (whatever order
the model happened to find them in), not true left-to-right, top-to-bottom
*reading order*. On skewed photos, or lines with tall/short glyphs, a box
can get grouped into the wrong "line" and end up numbered out of sequence
(e.g. box 9 appearing after box 10 on the same line).

Approach
--------
1. Estimate page skew from box top-edge slopes; deskew all coordinates.
2. Compute the gap between consecutive box y-centers (top to bottom) and
   run 1D Otsu's method on those gaps to find the natural split between
   "same line" gaps (small) and "line break" gaps (large) -- calibrated
   fresh from each page's own data, no fixed constant tuned on one document.
3. Walk top to bottom, starting a new line cluster whenever a gap exceeds
   that threshold.
4. Order lines top-to-bottom by mean y-center.
5. Order boxes within each line left-to-right by x-center (or right-to-left
   if you pass --rtl, for Arabic/Urdu/Hebrew-style scripts).
6. Re-number everything 1..N in that corrected order.

Usage
-----
    python3 fix_reading_order.py input.json -o corrected.json \
        [--overlay original.jpg -O overlay_fixed.jpg] \
        [--gap-ratio 0.6] [--rtl]

Only `input.json` (the OCR output containing a "lines" list of
{bbox, text, confidence, ...}) is required. Everything else is optional,
including --gap-ratio, which is only needed to override auto-calibration.
"""

import argparse
import json
import math
import sys


def estimate_skew(boxes):
    """
    Estimate the page's rotation angle (radians) from the slope of each
    box's top edge (corner 0 -> corner 1). Photographed pages are rarely
    perfectly level, and that tilt is exactly what makes naive y-overlap
    clustering chain unrelated lines together (a box's y-range creeps into
    the next line's range as you move across a wide, slanted line).
    We use the median angle (robust to a few rotated/short outlier boxes)
    weighted toward wider boxes, which give a more reliable slope estimate.
    """
    angles = []
    for b in boxes:
        (x0, y0), (x1, y1) = b["bbox"][0], b["bbox"][1]
        width = x1 - x0
        if width < 20:  # too narrow to give a reliable angle estimate
            continue
        angles.append(math.atan2(y1 - y0, x1 - x0))
    if not angles:
        return 0.0
    angles.sort()
    return angles[len(angles) // 2]  # median


def rotate_point(x, y, theta):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return x * cos_t + y * sin_t, -x * sin_t + y * cos_t


def deskewed_bbox(bbox, theta):
    return [rotate_point(x, y, theta) for x, y in bbox]


def y_range(bbox):
    ys = [pt[1] for pt in bbox]
    return min(ys), max(ys)


def x_range(bbox):
    xs = [pt[0] for pt in bbox]
    return min(xs), max(xs)


def horizontal_overlap_ratio(bbox1, bbox2):
    """Calculates horizontal overlap ratio relative to the narrower box."""
    x1_min, x1_max = x_range(bbox1)
    x2_min, x2_max = x_range(bbox2)
    overlap = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    min_w = max(1e-3, min(x1_max - x1_min, x2_max - x2_min))
    return overlap / min_w


def x_center(bbox):
    xs = [pt[0] for pt in bbox]
    return sum(xs) / len(xs)


def y_center(bbox):
    ys = [pt[1] for pt in bbox]
    return sum(ys) / len(ys)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def otsu_threshold_1d(values):
    """
    Real-valued Otsu's method. Given a list of non-negative numbers
    (here: gaps between consecutive box y-centers on a page, after
    sorting), finds the split point that maximizes the between-class
    variance of the resulting two groups -- i.e. the natural boundary
    between a cluster of small values and a cluster of large values.

    This is the same idea as image-binarization Otsu, just applied to a
    1D list of gap sizes instead of a pixel-intensity histogram. No
    binning is needed since we only ever have on the order of tens to
    low-hundreds of gaps per page.

    Returns (threshold, separation_score). separation_score is the
    between-class variance at the optimal split, normalized by the total
    variance of the data -- close to 1.0 means a clean bimodal split
    (real line breaks vs. real within-line gaps); close to 0 means the
    data doesn't actually split into two groups (e.g. a single-line page,
    or uniformly-spaced text with no line breaks at all).
    """
    if len(values) < 2:
        return None, 0.0

    vals = sorted(values)
    n = len(vals)
    total_sum = sum(vals)
    total_mean = total_sum / n
    total_var = sum((v - total_mean) ** 2 for v in vals) / n
    if total_var == 0:
        return None, 0.0  # every gap identical -> nothing to split

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

    separation_score = best_between_var / total_var
    return best_thresh, separation_score


def estimate_line_gap_threshold(boxes, theta=0.0, fallback_ratio=0.5,
                                 min_separation_score=0.3, max_ratio=0.6):
    """
    Self-calibrating line gap threshold. Finds the natural split between
    "words on the same line" (small gaps) and "line breaks" (large gaps).

    Important physical constraint:
    Two words on the same line can never have their vertical centers separated
    by more than ~0.5-0.6x median box height. If Otsu suggests a threshold
    greater than max_ratio * median_height, it was triggered by large paragraph,
    section, or illustration gaps on pages where boxes are already whole lines.
    In such cases, we clamp/fallback to fallback_ratio * median_height.
    """
    working_bboxes = [deskewed_bbox(b["bbox"], theta) for b in boxes]
    centers = sorted(y_center(bb) for bb in working_bboxes)
    heights = [max(y_range(bb)[1] - y_range(bb)[0], 1) for bb in working_bboxes]
    sorted_heights = sorted(heights)
    median_height = sorted_heights[len(sorted_heights) // 2]

    if len(centers) < 3:
        return fallback_ratio * median_height

    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    thresh, score = otsu_threshold_1d(gaps)

    max_allowed = max_ratio * median_height

    # If Otsu found no split, low score, or a threshold larger than max_allowed (e.g. paragraph/image gap)
    if thresh is None or score < min_separation_score or thresh > max_allowed:
        return fallback_ratio * median_height

    return thresh


def cluster_into_lines(boxes, threshold, theta=0.0):
    """
    Groups boxes into lines based on vertical proximity and horizontal non-overlap.

    1. Sort boxes by y-center (top to bottom).
    2. Add box to the current line if:
       - Its y-center is within `threshold` of the running mean of the current line, AND
       - It does NOT collide horizontally with any box already in the current line.
         (Two words on the same line in a single column never occupy the same X range).
    """
    n = len(boxes)
    if n == 0:
        return []

    working_bboxes = [deskewed_bbox(b["bbox"], theta) for b in boxes]
    centers = [y_center(bb) for bb in working_bboxes]

    order = sorted(range(n), key=lambda i: centers[i])

    clusters = []
    current = [order[0]]
    running_sum = centers[order[0]]
    for idx in order[1:]:
        running_mean = running_sum / len(current)
        y_dist = abs(centers[idx] - running_mean)

        has_horizontal_collision = any(
            horizontal_overlap_ratio(working_bboxes[idx], working_bboxes[c_idx]) > 0.25
            for c_idx in current
        )

        if y_dist <= threshold and not has_horizontal_collision:
            current.append(idx)
            running_sum += centers[idx]
        else:
            clusters.append(current)
            current = [idx]
            running_sum = centers[idx]
    clusters.append(current)
    return clusters



def fix_reading_order(boxes, gap_ratio=None, rtl=False, auto_deskew=True):
    """
    boxes: list of dicts with 'bbox' (and any other fields, e.g. 'text').
    gap_ratio: if given, overrides self-calibration with a fixed
               fraction of median box height (escape hatch for pages
               where auto-calibration misbehaves). Default None means
               "derive the threshold from this page's own gap
               distribution via Otsu" -- the production-safe default.
    Returns a new list of the same dicts, reordered into corrected
    reading order, each with an added '_corrected_order' (1-indexed).
    """
    if not boxes:
        return []
    theta = estimate_skew(boxes) if auto_deskew else 0.0

    if gap_ratio is not None:
        working_bboxes = [deskewed_bbox(b["bbox"], theta) for b in boxes]
        heights = [max(y_range(bb)[1] - y_range(bb)[0], 1) for bb in working_bboxes]
        median_height = sorted(heights)[len(heights) // 2]
        threshold = gap_ratio * median_height
    else:
        threshold = estimate_line_gap_threshold(boxes, theta=theta)

    clusters = cluster_into_lines(boxes, threshold, theta=theta)

    working = {i: deskewed_bbox(boxes[i]["bbox"], theta) for i in range(len(boxes))}

    # order lines top -> bottom by mean y-center of all boxes in the line
    # (using deskewed coordinates, so tilt doesn't distort the ordering)
    def cluster_y(cluster_idxs):
        ys = [y_center(working[i]) for i in cluster_idxs]
        return sum(ys) / len(ys)

    clusters.sort(key=cluster_y)

    ordered = []
    for cluster_idxs in clusters:
        cluster_idxs.sort(
            key=lambda i: x_center(working[i]),
            reverse=rtl,
        )
        for i in cluster_idxs:
            ordered.append(boxes[i])

    for n, box in enumerate(ordered, start=1):
        box["_corrected_order"] = n

    return ordered


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_json", help="OCR output JSON (with an 'images' list of 'lines')")
    ap.add_argument("-o", "--output", default="corrected.json",
                     help="Where to write the corrected JSON")
    ap.add_argument("--gap-ratio", type=float, default=None,
                     help="Manual override: fixed gap threshold as a fraction of median box height. "
                          "By default (omit this flag) the threshold is self-calibrated per page "
                          "via Otsu's method on the page's own gap distribution -- use this flag "
                          "only as an escape hatch if auto-calibration misbehaves on a given page.")
    ap.add_argument("--rtl", action="store_true",
                     help="Sort each line right-to-left instead of left-to-right")
    ap.add_argument("--no-deskew", action="store_true",
                     help="Disable automatic skew correction before clustering")
    ap.add_argument("--overlay", help="Optional base image to draw a corrected overlay on")
    ap.add_argument("-O", "--overlay-output", default="overlay_fixed.jpg",
                     help="Where to save the corrected overlay image")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for image in data.get("images", []):
        lines = image.get("lines", [])
        corrected = fix_reading_order(
            lines,
            gap_ratio=args.gap_ratio,
            rtl=args.rtl,
            auto_deskew=not args.no_deskew,
        )
        image["lines"] = corrected
        image["full_text"] = "\n".join(b["text"] for b in corrected)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Corrected JSON written to: {args.output}")

    if args.overlay:
        draw_overlay(data, args.overlay, args.overlay_output)
        print(f"Corrected overlay image written to: {args.overlay_output}")


def draw_overlay(data, base_image_path, out_path):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    for image in data.get("images", []):
        for box in image["lines"]:
            pts = [tuple(p) for p in box["bbox"]]
            n = box["_corrected_order"]
            draw.polygon(pts, outline=(255, 0, 0, 255), width=2)
            x0, y0 = pts[0]
            draw.rectangle([x0 - 2, y0 - 20, x0 + 22, y0], fill=(0, 0, 0, 220))
            draw.text((x0, y0 - 20), str(n), fill=(255, 255, 255, 255), font=font)

    img.save(out_path)


if __name__ == "__main__":
    sys.exit(main())