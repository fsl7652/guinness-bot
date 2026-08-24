"""
glass_check.py

Classifies glass type using TRT engine (or ONNX fallback).
Classes: guinness_midsip, guinness_tulip, not_glass, wrong_glass

Returns score: 10.0 for tulip, 4.0 otherwise.
"""

import sys
from pathlib import Path

MODEL_DIR  = Path(__file__).parent / "models"
TRT_PATH   = MODEL_DIR / "glass_check.trt"
ONNX_PATH  = MODEL_DIR / "glass_check.onnx"
JSON_PATH  = MODEL_DIR / "glass_check.json"

CLASSES    = ["guinness_midsip", "guinness_tulip", "not_glass", "wrong_glass"]
TULIP_CLASS = "guinness_tulip"

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        from trt_infer import load_classifier
        # Load classes from json if available
        classes = CLASSES
        if JSON_PATH.exists():
            import json
            with open(JSON_PATH) as f:
                data = json.load(f)
                classes = data.get("classes", CLASSES)
        _classifier = load_classifier(TRT_PATH, ONNX_PATH, classes)
    return _classifier


def analyse(crop_rgb):
    """
    Args:
        crop_rgb: numpy RGB array

    Returns dict:
        is_tulip    — bool
        confidence  — 0-1
        score       — 10.0 if tulip, 4.0 otherwise
        label       — raw class label
        mode        — 'trt' or 'onnx'
    """
    try:
        clf          = _get_classifier()
        label, conf  = clf.predict(crop_rgb)
        is_tulip     = label == TULIP_CLASS
        mode         = 'trt' if TRT_PATH.exists() else 'onnx'
    except Exception as e:
        print(f"[glass_check] Classifier error: {e} — using stub", file=sys.stderr)
        return _stub_analyse(crop_rgb)

    return {
        "is_tulip":   is_tulip,
        "confidence": round(conf, 3),
        "score":      10.0 if is_tulip else 4.0,
        "label":      label,
        "mode":       mode,
    }


def _stub_analyse(crop_rgb):
    """Heuristic fallback — profile width analysis."""
    import cv2
    import numpy as np

    gray   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h, w   = gray.shape
    rows   = np.linspace(int(h*0.1), int(h*0.9), 40).astype(int)
    widths = []

    for y in rows:
        row    = gray[y]
        thresh = cv2.threshold(row.reshape(1,-1), 0, 255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1][0]
        nz     = np.where(thresh > 0)[0]
        widths.append(int(nz[-1]-nz[0]) if len(nz)>=2 else 0)

    widths   = np.array(widths, dtype=float)
    if widths.max() == 0:
        return {"is_tulip":True,"confidence":0.5,"score":10.0,"label":"unknown","mode":"stub"}

    n   = len(widths)
    top = widths[:n//3].mean()
    mid = widths[n//3:2*n//3].mean()
    bot = widths[2*n//3:].mean()

    narrowing  = (top-mid)/max(top,1)
    flare      = (top-bot)/max(top,1)
    is_tulip   = narrowing > 0.05 and flare > 0.05
    confidence = min(0.95, 0.5+narrowing+flare)

    return {
        "is_tulip":   bool(is_tulip),
        "confidence": round(float(confidence),3),
        "score":      10.0 if is_tulip else 4.0,
        "label":      "guinness_tulip" if is_tulip else "wrong_glass",
        "mode":       "stub",
    }