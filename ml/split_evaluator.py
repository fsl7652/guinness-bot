"""
splitg.py

Detects whether the drinker has correctly split the G — i.e. the first sip
bisects the Guinness harp logo on the glass, leaving half above the liquid line
and half below.

This is evaluated on a SEPARATE mid-sip photo, not the full pint photo.

Stub mode: finds the logo region by colour (the gold harp) and checks
whether the liquid line intersects it.

Real mode: loads a binary ONNX classifier trained on labelled mid-sip photos.

Usage:
    from splitg import analyse
    result = analyse(mid_sip_rgb)
    # {"detected": True, "confidence": 0.91, "mode": "stub"}

Test locally:
    python splitg.py <image_path> [output_path]
"""

import cv2
import numpy as np
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "splitg.onnx"
USE_STUB   = not MODEL_PATH.exists()


# ── Stub: heuristic logo bisect detection ────────────────────

def _find_logo_region(crop_rgb):
    """
    Find the harp logo region using colour.
    The Guinness harp is gold/yellow — distinct from the dark body.
    Returns bounding box (x1, y1, x2, y2) or None.
    """
    # Convert to HSV for colour filtering
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    # Gold/yellow hue range
    lower_gold = np.array([15,  60,  60])
    upper_gold = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # Also catch the white GUINNESS text which can help locate the logo area
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask  = cv2.inRange(hsv, lower_white, upper_white)

    combined = cv2.bitwise_or(mask, white_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour in the middle vertical band
    h, w = crop_rgb.shape[:2]
    mid_contours = [c for c in contours
                    if w * 0.2 < cv2.boundingRect(c)[0] < w * 0.8]
    if not mid_contours:
        mid_contours = contours

    largest = max(mid_contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    return (x, y, x + cw, y + ch)


def _find_liquid_line(crop_rgb, logo_bbox):
    """
    Find the liquid surface line in the vicinity of the logo.
    For a mid-sip photo, the liquid line should bisect the logo.
    Uses edge detection focused on the logo region.
    """
    if logo_bbox is None:
        return None

    x1, y1, x2, y2 = logo_bbox
    h, w = crop_rgb.shape[:2]

    # Search in a vertical band around the logo x-range
    search_x1 = max(0,  x1 - 20)
    search_x2 = min(w,  x2 + 20)
    search_y1 = max(0,  y1 - 40)
    search_y2 = min(h,  y2 + 40)

    region = crop_rgb[search_y1:search_y2, search_x1:search_x2]
    gray   = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Horizontal edges = liquid surface
    sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    col_means = np.abs(sobel_y).mean(axis=1)  # average horizontal edge per row

    if col_means.max() == 0:
        return None

    liquid_row_local = int(np.argmax(col_means))
    return search_y1 + liquid_row_local  # convert back to full image coords


def _stub_analyse(crop_rgb):
    """
    Heuristic split-the-G detection.
    Finds logo region, finds liquid line, checks if line bisects logo.
    """
    logo_bbox   = _find_logo_region(crop_rgb)
    liquid_line = _find_liquid_line(crop_rgb, logo_bbox)

    if logo_bbox is None:
        return {
            "detected":   False,
            "confidence": 0.3,
            "mode":       "stub",
            "reason":     "logo not found",
            "logo_bbox":  None,
            "liquid_y":   None,
        }

    x1, y1, x2, y2 = logo_bbox
    logo_centre_y   = (y1 + y2) / 2
    logo_height     = y2 - y1

    if liquid_line is None:
        return {
            "detected":   False,
            "confidence": 0.3,
            "mode":       "stub",
            "reason":     "liquid line not found",
            "logo_bbox":  logo_bbox,
            "liquid_y":   None,
        }

    # Check if liquid line passes through the logo vertically
    # Allow some tolerance — within 30% of logo height from centre
    tolerance    = logo_height * 0.3
    bisected     = abs(liquid_line - logo_centre_y) < tolerance

    # Confidence based on how close to dead centre
    dist_from_centre = abs(liquid_line - logo_centre_y)
    confidence   = max(0.3, 1.0 - dist_from_centre / max(logo_height, 1))

    return {
        "detected":        bool(bisected),
        "confidence":      round(float(confidence), 3),
        "mode":            "stub",
        "logo_bbox":       logo_bbox,
        "liquid_y":        int(liquid_line),
        "logo_centre_y":   int(logo_centre_y),
        "dist_from_centre": round(float(dist_from_centre), 1),
    }


# ── Real model ────────────────────────────────────────────────

def _model_analyse(crop_rgb):
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH))
    img     = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
    img     = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    inp     = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    logits  = session.run(None, {"image": inp})[0][0]
    probs   = np.exp(logits) / np.exp(logits).sum()
    detected    = bool(probs[1] > probs[0])
    confidence  = float(probs[1] if detected else probs[0])

    return {
        "detected":   detected,
        "confidence": round(confidence, 3),
        "mode":       "model",
        "logo_bbox":  None,
        "liquid_y":   None,
    }


# ── Main entry point ──────────────────────────────────────────

def analyse(mid_sip_rgb):
    """
    Args:
        mid_sip_rgb: numpy RGB array of a mid-sip photo

    Returns dict:
        detected    — bool, True if G was split correctly
        confidence  — 0–1
        mode        — "stub" or "model"
        logo_bbox   — (x1,y1,x2,y2) of detected logo or None
        liquid_y    — detected liquid line y position or None
    """
    return _stub_analyse(mid_sip_rgb) if USE_STUB else _model_analyse(mid_sip_rgb)


# ── Visualisation ─────────────────────────────────────────────

def visualise(image_path, output_path=None):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = analyse(img_rgb)

    print(f"Detected:    {result['detected']}")
    print(f"Confidence:  {result['confidence']}")
    print(f"Mode:        {result['mode']}")
    if result.get("logo_bbox"):
        print(f"Logo bbox:   {result['logo_bbox']}")
    if result.get("liquid_y"):
        print(f"Liquid line: y={result['liquid_y']}")

    annotated = img_bgr.copy()
    col       = (0, 255, 0) if result["detected"] else (0, 0, 255)

    if result.get("logo_bbox"):
        x1, y1, x2, y2 = result["logo_bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(annotated, "logo", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    if result.get("liquid_y"):
        ly = result["liquid_y"]
        cv2.line(annotated, (0, ly), (annotated.shape[1], ly), (255, 255, 0), 2)

    label = f"{'✓ Split' if result['detected'] else '✗ Not split'}  {result['confidence']:.0%}"
    cv2.putText(annotated, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved → {output_path}")
    else:
        cv2.imshow("Split the G", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python splitg.py <image> [output]")
        sys.exit(1)
    visualise(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)