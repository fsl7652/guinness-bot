"""
segment.py

Two-stage glass detection pipeline:

  Stage 1 — OpenCV candidate finder (fast, ~50ms)
    Runs three complementary strategies and unions results:
    - Colour filter: dark body + bright head (Guinness HSV signature)
    - Vertical edge detection: tall parallel edges (glass sides)
    - Contour detection: large tall-narrow blobs

  Stage 2 — MobileSAM refinement (accurate, ~2-3s per glass)
    For each OpenCV candidate, runs SamPredictor with 3 prompt points.
    Produces accurate mask → clean crop for scoring modules.

  Fallback — if OpenCV finds no candidates, falls back to a 2x3 grid
    of SAM prompts (6 calls) rather than full auto-generation (256 calls).

Usage:
    from segment import get_glass_crops
    crops = get_glass_crops(image_rgb)

Test locally:
    python segment.py path/to/pint.jpg [output.jpg] [--debug]
"""

import cv2
import numpy as np
import sys
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────

MIN_ASPECT      = 1.5     # height/width — glasses are taller than wide
MIN_AREA_FRAC   = 0.03    # at least 3% of frame
MAX_AREA_FRAC   = 0.75    # not the whole image
MAX_GLASSES     = 4
PADDING         = 16
SAM_IOU_THRESH  = 0.75    # minimum SAM IoU score to accept a mask


# ── MobileSAM lazy load ───────────────────────────────────────

_sam_predictor  = None
_sam_model      = None

def _get_sam():
    global _sam_predictor, _sam_model
    if _sam_predictor is not None:
        return _sam_predictor

    try:
        from mobile_sam import sam_model_registry, SamPredictor
        import torch

        weights = Path(__file__).parent / "models" / "mobile_sam.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"MobileSAM weights not found at {weights}\n"
                "Download: https://github.com/ChaoningZhang/MobileSAM"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()

        _sam_model = sam_model_registry["vit_t"](checkpoint=str(weights))
        _sam_model.to(device).eval()
        _sam_predictor = SamPredictor(_sam_model)

        print(f"[segment] MobileSAM loaded on {device}")
        return _sam_predictor

    except ImportError:
        raise ImportError(
            "MobileSAM not installed.\n"
            "Run: pip install git+https://github.com/ChaoningZhang/MobileSAM.git timm==0.6.13"
        )


# ── Stage 1: OpenCV candidate finder ─────────────────────────

def _candidates_from_colour(image_rgb, h, w):
    """
    Find regions with Guinness colour signature:
    dark (near-black) body in lower portion, bright (white/cream) head on top.
    Uses HSV colour filtering and looks for vertically stacked dark/bright regions.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Dark body: low saturation, low value (dark brown/black)
    dark_mask = cv2.inRange(hsv,
        np.array([0,   0,   0]),
        np.array([180, 80, 80])
    )

    # Bright head: low saturation, high value (white/cream)
    bright_mask = cv2.inRange(hsv,
        np.array([0,   0,   160]),
        np.array([40,  80,  255])
    )

    candidates = []
    # Slide a vertical window across the image looking for dark-over-bright stacking
    step  = w // 8
    win_w = w // 5

    for cx in range(win_w // 2, w - win_w // 2, step):
        x1 = max(0, cx - win_w // 2)
        x2 = min(w, cx + win_w // 2)

        dark_col   = dark_mask[:, x1:x2]
        bright_col = bright_mask[:, x1:x2]

        # Find topmost bright region (head)
        bright_rows = np.where(bright_col.any(axis=1))[0]
        dark_rows   = np.where(dark_col.any(axis=1))[0]

        if len(bright_rows) < 5 or len(dark_rows) < 10:
            continue

        head_top = int(bright_rows.min())
        head_bot = int(bright_rows.max())
        body_bot = int(dark_rows.max())

        # Head must be above body
        if head_bot >= body_bot:
            continue

        total_h = body_bot - head_top
        if total_h < h * 0.15:
            continue

        candidates.append((x1, head_top, x2, body_bot))

    return candidates


def _candidates_from_edges(image_rgb, h, w):
    """
    Find tall vertical edge pairs — the sides of a glass.
    """
    gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 30, 100)

    # Dilate edges horizontally to connect nearby verticals
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    dilated = cv2.dilate(edges, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect    = ch / max(cw, 1)
        area_frac = (cw * ch) / (h * w)

        if aspect < MIN_ASPECT:
            continue
        if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
            continue

        candidates.append((x, y, x + cw, y + ch))

    return candidates


def _candidates_from_contours(image_rgb, h, w):
    """
    Find large tall-narrow blobs using adaptive thresholding.
    Works well for glasses against varied backgrounds.
    """
    gray    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thresh  = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )

    kernel  = np.ones((5, 5), np.uint8)
    closed  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect    = ch / max(cw, 1)
        area_frac = (cw * ch) / (h * w)

        if aspect < MIN_ASPECT:
            continue
        if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
            continue

        candidates.append((x, y, x + cw, y + ch))

    return candidates


def _merge_boxes(boxes, iou_thresh=0.3):
    """
    Merge overlapping bounding boxes from different detection strategies.
    Uses IoU-based greedy merging.
    """
    if not boxes:
        return []

    boxes  = list(set(boxes))  # deduplicate exact matches
    merged = []

    while boxes:
        base = list(boxes.pop(0))
        to_merge = []

        remaining = []
        for b in boxes:
            ix1 = max(base[0], b[0])
            iy1 = max(base[1], b[1])
            ix2 = min(base[2], b[2])
            iy2 = min(base[3], b[3])

            if ix2 <= ix1 or iy2 <= iy1:
                remaining.append(b)
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = (base[2]-base[0]) * (base[3]-base[1])
            area2 = (b[2]-b[0]) * (b[3]-b[1])
            union = area1 + area2 - inter

            if inter / union > iou_thresh:
                to_merge.append(b)
            else:
                remaining.append(b)

        # Expand base to cover all merged boxes
        for b in to_merge:
            base[0] = min(base[0], b[0])
            base[1] = min(base[1], b[1])
            base[2] = max(base[2], b[2])
            base[3] = max(base[3], b[3])

        merged.append(tuple(base))
        boxes = remaining

    return merged


def _find_opencv_candidates(image_rgb, debug=False):
    """
    Run all three OpenCV strategies, merge results, filter by aspect ratio.
    Returns list of (x1, y1, x2, y2) candidate bounding boxes.
    """
    h, w = image_rgb.shape[:2]

    all_boxes = []
    all_boxes.extend(_candidates_from_colour(image_rgb, h, w))
    all_boxes.extend(_candidates_from_edges(image_rgb, h, w))
    all_boxes.extend(_candidates_from_contours(image_rgb, h, w))

    merged = _merge_boxes(all_boxes)

    # Final aspect ratio + area filter
    filtered = []
    for (x1, y1, x2, y2) in merged:
        bw = x2 - x1
        bh = y2 - y1
        if bh / max(bw, 1) < MIN_ASPECT:
            continue
        area_frac = (bw * bh) / (h * w)
        if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
            continue
        filtered.append((x1, y1, x2, y2))

    # Sort left to right, cap at MAX_GLASSES
    filtered.sort(key=lambda b: b[0])
    filtered = filtered[:MAX_GLASSES]

    if debug:
        print(f"[segment] OpenCV: {len(all_boxes)} raw → {len(merged)} merged → {len(filtered)} filtered")

    return filtered


# ── Stage 2: SAM refinement ───────────────────────────────────

def _sam_refine(predictor, image_rgb, bbox, debug=False):
    """
    Run SamPredictor on a single candidate bounding box.
    Prompts SAM with 3 points inside the box (top-third, centre, bottom-third).
    Returns the best mask as a bounding box, or None if no good mask found.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    h_third = (y2 - y1) // 3

    points = np.array([
        [cx, y1 + h_third],        # upper third (head region)
        [cx, (y1 + y2) // 2],      # centre
        [cx, y2 - h_third],        # lower third (body region)
    ], dtype=np.float32)

    labels = np.ones(len(points), dtype=np.int32)

    # Also pass the bounding box as a SAM box prompt for better accuracy
    sam_box = np.array([x1, y1, x2, y2], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=sam_box,
        multimask_output=True,
    )

    # Take highest scoring mask that passes IoU threshold
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_mask  = masks[best_idx]

    if debug:
        print(f"[segment]   SAM scores: {scores}  best={best_score:.3f}")

    if best_score < SAM_IOU_THRESH:
        if debug:
            print(f"[segment]   Rejected — SAM score {best_score:.3f} < {SAM_IOU_THRESH}")
        return None, best_score

    # Convert mask to bounding box
    rows = np.where(best_mask.any(axis=1))[0]
    cols = np.where(best_mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None, best_score

    mx1, my1 = int(cols.min()), int(rows.min())
    mx2, my2 = int(cols.max()), int(rows.max())

    return (mx1, my1, mx2, my2), best_score


def _fallback_grid_candidates(h, w):
    """
    Fallback: 2x3 grid of candidate regions when OpenCV finds nothing.
    Covers left/centre/right × top-half/bottom-half.
    Far fewer SAM calls than full auto-generation (6 vs 256).
    """
    candidates = []
    for col_frac in [0.2, 0.5, 0.8]:
        for row_frac in [0.3, 0.7]:
            cx = int(w * col_frac)
            cy = int(h * row_frac)
            bw = w // 4
            bh = h // 2
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, cx + bw // 2)
            y2 = min(h, cy + bh // 2)
            candidates.append((x1, y1, x2, y2))
    return candidates


# ── Guinness colour check ─────────────────────────────────────

def _is_likely_guinness(crop_rgb, debug=False):
    """
    Verify crop has bright top (head) over dark bottom (body).
    """
    THRESHOLD = 25

    gray              = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h                 = gray.shape[0]
    top_brightness    = float(gray[:h // 4].mean())
    bottom_brightness = float(gray[h // 2:].mean())
    diff              = top_brightness - bottom_brightness

    if debug:
        print(f"[segment]   Guinness check: top={top_brightness:.1f} "
              f"bot={bottom_brightness:.1f} diff={diff:.1f} pass={diff > THRESHOLD}")

    return diff > THRESHOLD


# ── Main entry point ──────────────────────────────────────────

def get_glass_crops(image_rgb, debug=False):
    """
    Detect all pint glasses in an RGB image and return cropped regions.

    Args:
        image_rgb: numpy HxWx3 RGB array
        debug:     verbose logging

    Returns list of dicts:
        {
            "crop":  numpy RGB array,
            "bbox":  (x1, y1, x2, y2),
            "score": SAM IoU confidence,
            "index": 0-based left-to-right position
        }
    """
    h, w      = image_rgb.shape[:2]
    predictor = _get_sam()

    # Set image once — shared across all SAM calls for this frame
    predictor.set_image(image_rgb)

    # Stage 1: OpenCV candidates
    candidates = _find_opencv_candidates(image_rgb, debug=debug)
    fallback   = False

    if not candidates:
        print("[segment] No OpenCV candidates — using fallback grid")
        candidates = _fallback_grid_candidates(h, w)
        fallback   = True

    if debug:
        print(f"[segment] {len(candidates)} candidate(s) → SAM refinement")

    # Stage 2: SAM refine each candidate
    results     = []
    seen_bboxes = []

    for i, bbox in enumerate(candidates):
        if debug:
            print(f"[segment] Candidate {i}: bbox={bbox}")

        refined_bbox, score = _sam_refine(predictor, image_rgb, bbox, debug=debug)

        if refined_bbox is None:
            if debug:
                print(f"[segment] Candidate {i} rejected by SAM")
            continue

        rx1, ry1, rx2, ry2 = refined_bbox

        # Deduplicate against already accepted results
        duplicate = False
        for sb in seen_bboxes:
            sx1, sy1, sx2, sy2 = sb
            ix1, iy1 = max(rx1, sx1), max(ry1, sy1)
            ix2, iy2 = min(rx2, sx2), min(ry2, sy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                area1 = (rx2-rx1)*(ry2-ry1)
                area2 = (sx2-sx1)*(sy2-sy1)
                if inter / (area1 + area2 - inter) > 0.4:
                    duplicate = True
                    break

        if duplicate:
            if debug:
                print(f"[segment] Candidate {i} duplicate — skipped")
            continue

        # Aspect ratio check on refined bbox
        bw = rx2 - rx1
        bh = ry2 - ry1
        if bh / max(bw, 1) < MIN_ASPECT:
            if debug:
                print(f"[segment] Candidate {i} aspect ratio too low ({bh/max(bw,1):.2f})")
            continue

        # Pad crop
        px1 = max(0, rx1 - PADDING)
        py1 = max(0, ry1 - PADDING)
        px2 = min(w, rx2 + PADDING)
        py2 = min(h, ry2 + PADDING)
        crop = image_rgb[py1:py2, px1:px2]

        if not _is_likely_guinness(crop, debug=debug):
            print(f"[segment] Candidate {i} rejected — failed Guinness colour check")
            continue

        seen_bboxes.append(refined_bbox)
        results.append({
            "crop":  crop,
            "bbox":  (px1, py1, px2, py2),
            "score": score,
            "index": len(results),
        })

        if len(results) >= MAX_GLASSES:
            break

    # Sort left to right and reindex
    results.sort(key=lambda r: r["bbox"][0])
    for i, r in enumerate(results):
        r["index"] = i

    mode = "fallback-grid" if fallback else "opencv+sam"
    print(f"[segment] {len(results)} glass(es) detected ({mode})")

    return results


# ── Visualisation ─────────────────────────────────────────────

def visualise(image_path, output_path=None, debug=False):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    crops   = get_glass_crops(img_rgb, debug=debug)

    if not crops:
        print("No glasses detected")
        return

    colours   = [(0,255,0), (0,165,255), (255,0,0), (0,255,255)]
    annotated = img_bgr.copy()

    for g in crops:
        x1, y1, x2, y2 = g["bbox"]
        col = colours[g["index"] % len(colours)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 3)
        cv2.putText(
            annotated,
            f"Glass {g['index']+1}  {g['score']:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2
        )
        print(f"  Glass {g['index']+1}: bbox={g['bbox']}  "
              f"score={g['score']:.3f}  crop={g['crop'].shape[:2]}")

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved → {output_path}")
        stem = Path(output_path).stem
        for g in crops:
            crop_file = f"{stem}_glass{g['index']+1}.jpg"
            cv2.imwrite(crop_file, cv2.cvtColor(g["crop"], cv2.COLOR_RGB2BGR))
            print(f"Saved crop → {crop_file}")
    else:
        cv2.imshow("Detected glasses", annotated)
        for g in crops:
            cv2.imshow(f"Glass {g['index']+1}", cv2.cvtColor(g["crop"], cv2.COLOR_RGB2BGR))
        print("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segment.py <image> [output] [--debug]")
        sys.exit(1)

    _debug      = "--debug" in sys.argv
    _image_path = sys.argv[1]
    _output     = next((a for a in sys.argv[2:] if not a.startswith("--")), None)

    visualise(_image_path, _output, debug=_debug)