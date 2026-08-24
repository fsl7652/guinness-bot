"""
split_evaluator.py (splitg.py)

Scores the G split using TRT engine (or ONNX fallback).
Ordinal classes: perfect, close, partial, near_miss, missed

Falls back to heuristic logo/liquid-line detection if model unavailable.
"""

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
TRT_PATH  = MODEL_DIR / "splitg.trt"
ONNX_PATH = MODEL_DIR / "splitg.onnx"
JSON_PATH = MODEL_DIR / "splitg.json"

CLASSES   = ["perfect", "close", "partial", "near_miss", "missed"]
SPLIT_CLASSES = {"perfect", "close", "partial"}  # count as detected

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        from trt_infer import load_classifier
        classes = CLASSES
        if JSON_PATH.exists():
            import json
            with open(JSON_PATH) as f:
                data = json.load(f)
                classes = data.get("classes", CLASSES)
        _classifier = load_classifier(TRT_PATH, ONNX_PATH, classes)
    return _classifier


def analyse(mid_sip_rgb):
    """
    Args:
        mid_sip_rgb: numpy RGB array of mid-sip photo

    Returns dict:
        detected    — bool
        confidence  — 0-1
        label       — ordinal class
        mode        — 'trt' | 'onnx' | 'stub'
    """
    try:
        clf         = _get_classifier()
        label, conf = clf.predict(mid_sip_rgb)
        detected    = label in SPLIT_CLASSES
        mode        = 'trt' if TRT_PATH.exists() else 'onnx'
    except Exception as e:
        print(f"[splitg] Classifier error: {e} — using stub", file=sys.stderr)
        return _stub_analyse(mid_sip_rgb)

    return {
        "detected":   detected,
        "confidence": round(conf, 3),
        "label":      label,
        "mode":       mode,
    }


def _stub_analyse(mid_sip_rgb):
    """Heuristic fallback — logo colour + liquid line detection."""
    import cv2
    import numpy as np

    hsv          = cv2.cvtColor(mid_sip_rgb, cv2.COLOR_RGB2HSV)
    gold_mask    = cv2.inRange(hsv, np.array([15,60,60]), np.array([40,255,255]))
    white_mask   = cv2.inRange(hsv, np.array([0,0,180]),  np.array([180,40,255]))
    combined     = cv2.morphologyEx(
        cv2.bitwise_or(gold_mask, white_mask),
        cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)
    )

    contours, _  = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w         = mid_sip_rgb.shape[:2]
    mid_contours = [c for c in contours
                    if w*0.2 < cv2.boundingRect(c)[0] < w*0.8]

    if not mid_contours:
        return {"detected":False,"confidence":0.3,"label":"missed","mode":"stub"}

    largest     = max(mid_contours, key=cv2.contourArea)
    x,y,cw,ch  = cv2.boundingRect(largest)
    logo_cy     = y + ch/2

    # Find liquid line
    gray        = cv2.cvtColor(mid_sip_rgb, cv2.COLOR_RGB2GRAY)
    sobel       = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    liquid_y    = int(np.argmax(sobel.mean(axis=1)))

    dist        = abs(liquid_y - logo_cy)
    tolerance   = ch * 0.3
    detected    = dist < tolerance
    confidence  = max(0.3, 1.0 - dist/max(ch,1))

    if detected:
        label = "perfect" if dist < ch*0.1 else "close" if dist < ch*0.2 else "partial"
    else:
        label = "near_miss" if dist < ch*0.6 else "missed"

    return {
        "detected":   bool(detected),
        "confidence": round(float(confidence),3),
        "label":      label,
        "mode":       "stub",
    }