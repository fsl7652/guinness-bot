"""
segment.py

Detects and crops all pint glasses in an image using MobileSAM.
Filters results by aspect ratio, area, and Guinness colour signature.

Usage:
    from segment import get_glass_crops
    crops = get_glass_crops(image_rgb)  # list of dicts

Test locally:
    python segment.py path/to/pint.jpg
    python segment.py path/to/pint.jpg out.jpg
"""

import cv2
import numpy as np
import sys
from pathlib import Path


# ── MobileSAM lazy load ───────────────────────────────────────

_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    try:
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        import torch

        weights = Path(__file__).parent / "models" / "mobile_sam.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"MobileSAM weights not found at {weights}\n"
                "Download from: https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = sam_model_registry["vit_t"](checkpoint=str(weights))
        model.to(device).eval()

        _predictor = SamAutomaticMaskGenerator(
            model,
            points_per_side=16,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=5000,
        )
        print(f"[segment] MobileSAM loaded on {device}")
        return _predictor

    except ImportError:
        raise ImportError(
            "MobileSAM not installed.\n"
            "Run: pip install git+https://github.com/ChaoningZhang/MobileSAM.git timm"
        )


# ── Filtering helpers ─────────────────────────────────────────

def _is_likely_guinness(crop_rgb, debug=False):
    """
    Check crop has a bright top region (head) over a dark bottom region (body).
    Rejects false positives like tables, arms, background objects.
    Tune BRIGHTNESS_DIFF_THRESHOLD if rejecting real pints or passing false positives.
    """
    BRIGHTNESS_DIFF_THRESHOLD = 30

    gray              = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h                 = gray.shape[0]
    top_brightness    = gray[:h // 4].mean()
    bottom_brightness = gray[h // 2:].mean()
    diff              = top_brightness - bottom_brightness

    if debug:
        print(f"[segment]   brightness diff={diff:.1f}  "
              f"top={top_brightness:.1f}  bottom={bottom_brightness:.1f}  "
              f"pass={diff > BRIGHTNESS_DIFF_THRESHOLD}")

    return diff > BRIGHTNESS_DIFF_THRESHOLD


def _masks_to_glass_bboxes(masks, image_h, image_w):
    """
    Filter SAM masks to likely pint glass candidates by shape heuristics.
    Aspect ratio and area fraction are the main gates.
    """
    MIN_ASPECT      = 1.5    # height/width — glasses are taller than wide
    MIN_AREA_FRAC   = 0.03   # at least 3% of frame
    MAX_AREA_FRAC   = 0.70   # not the whole image
    MAX_GLASSES     = 4      # score at most 4 at once

    candidates = []
    for m in masks:
        x, y, w, h = m["bbox"]  # SAM returns xywh
        aspect     = h / max(w, 1)
        area_frac  = (w * h) / (image_h * image_w)

        if aspect < MIN_ASPECT:
            continue
        if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
            continue

        candidates.append({
            "bbox":      (x, y, x + w, y + h),
            "area_frac": area_frac,
            "iou_score": m["predicted_iou"],
        })

    candidates.sort(key=lambda c: c["iou_score"], reverse=True)
    return candidates[:MAX_GLASSES]


def _deduplicate(candidates, iou_thresh=0.4):
    """
    Remove overlapping detections — SAM often returns multiple masks
    for the same object at different granularities.
    """
    kept = []
    for c in candidates:
        x1, y1, x2, y2 = c["bbox"]
        duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            if inter / union > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
    return kept


# ── Main entry point ──────────────────────────────────────────

def get_glass_crops(image_rgb, padding=16, debug=False):
    """
    Detect all pint glasses in an RGB image and return cropped regions.

    Args:
        image_rgb:  numpy array HxWx3 in RGB
        padding:    pixels to add around each detected bbox
        debug:      print brightness values for threshold tuning

    Returns:
        List of dicts (sorted left to right):
        {
            "crop":  numpy array (RGB),
            "bbox":  (x1, y1, x2, y2) in original image coords,
            "score": SAM IoU confidence,
            "index": 0-based position left to right
        }
        Empty list if no glasses found.
    """
    h, w  = image_rgb.shape[:2]
    masks = _get_predictor().generate(image_rgb)

    candidates = _masks_to_glass_bboxes(masks, h, w)
    candidates = _deduplicate(candidates)
    candidates.sort(key=lambda c: c["bbox"][0])  # left to right

    results = []
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = c["bbox"]
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(w, x2 + padding)
        y2p = min(h, y2 + padding)

        crop = image_rgb[y1p:y2p, x1p:x2p]

        if debug:
            print(f"[segment] Candidate {i}  bbox={c['bbox']}  iou={c['iou_score']:.3f}")

        if not _is_likely_guinness(crop, debug=debug):
            print(f"[segment] Rejected candidate {i} — failed colour check")
            continue

        results.append({
            "crop":  crop,
            "bbox":  (x1p, y1p, x2p, y2p),
            "score": c["iou_score"],
            "index": len(results),   # reindex after rejections
        })

    print(f"[segment] {len(results)} glass(es) detected from {len(candidates)} candidates")
    return results


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
        print(f"Saved annotated image → {output_path}")
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