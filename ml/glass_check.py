"""
glass_check.py

Classifies whether the glass is a proper Guinness tulip glass.
Currently uses a shape heuristic stub — replace with trained model later.

The tulip glass has a distinctive profile:
- Narrows in the lower third
- Widens toward the top
- Has a curved lip

Stub mode uses column-width profile analysis.
Real mode loads a MobileNetV3 binary classifier (tulip / not-tulip).

Usage:
    from glass_check import analyse
    result = analyse(crop_rgb)
    # {"is_tulip": True, "confidence": 0.81, "score": 10.0, "mode": "stub"}

Test locally:
    python glass_check.py <image_path> [output_path]
"""

import cv2
import numpy as np
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "glass_check.onnx"
USE_STUB   = not MODEL_PATH.exists()


# ── Stub: heuristic shape analysis ───────────────────────────

def _profile_widths(crop_rgb, n_rows=40):
    """
    Measure the glass width at evenly spaced rows using edge detection.
    Returns an array of widths from top to bottom.
    """
    gray    = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h, w    = gray.shape
    rows    = np.linspace(int(h * 0.1), int(h * 0.9), n_rows).astype(int)
    widths  = []

    for y in rows:
        row     = gray[y]
        # Find leftmost and rightmost dark→light transition
        thresh  = cv2.threshold(row.reshape(1, -1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1][0]
        nonzero = np.where(thresh > 0)[0]
        if len(nonzero) < 2:
            widths.append(0)
        else:
            widths.append(int(nonzero[-1] - nonzero[0]))

    return np.array(widths, dtype=float)


def _stub_analyse(crop_rgb):
    """
    Heuristic tulip detection based on glass profile shape.

    A tulip glass narrows in the bottom third then widens toward the top.
    We measure width at top, middle, and bottom and check the ratio.
    """
    widths = _profile_widths(crop_rgb)
    if widths.max() == 0:
        return {"is_tulip": True, "confidence": 0.5, "mode": "stub",
                "reason": "could not measure profile"}

    n    = len(widths)
    top  = widths[:n // 3].mean()
    mid  = widths[n // 3: 2 * n // 3].mean()
    bot  = widths[2 * n // 3:].mean()

    # Tulip: top > mid (narrows then flares), bottom narrower than top
    # Straight pint: roughly uniform width
    narrowing = (top - mid) / max(top, 1)   # positive = narrows toward middle
    flare     = (top - bot) / max(top, 1)   # positive = wider at top than bottom

    is_tulip   = narrowing > 0.05 and flare > 0.05
    confidence = min(0.95, 0.5 + narrowing + flare)

    return {
        "is_tulip":   bool(is_tulip),
        "confidence": round(float(confidence), 3),
        "mode":       "stub",
        "narrowing":  round(float(narrowing), 3),
        "flare":      round(float(flare), 3),
    }


# ── Real model ────────────────────────────────────────────────

def _model_analyse(crop_rgb):
    """
    Run ONNX MobileNetV3 classifier.
    Output class 0 = not tulip, class 1 = tulip.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH))

    img = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    inp = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    logits  = session.run(None, {"image": inp})[0][0]
    probs   = np.exp(logits) / np.exp(logits).sum()
    is_tulip    = bool(probs[1] > probs[0])
    confidence  = float(probs[1] if is_tulip else probs[0])

    return {
        "is_tulip":   is_tulip,
        "confidence": round(confidence, 3),
        "mode":       "model",
    }


# ── Main entry point ──────────────────────────────────────────

def analyse(crop_rgb):
    """
    Args:
        crop_rgb: numpy RGB array — full glass crop

    Returns dict:
        is_tulip    — bool
        confidence  — 0–1
        score       — 10.0 if tulip, 4.0 if not (fed into aggregator)
        mode        — "stub" or "model"
    """
    result = _stub_analyse(crop_rgb) if USE_STUB else _model_analyse(crop_rgb)
    result["score"] = 10.0 if result["is_tulip"] else 4.0
    return result


# ── Visualisation ─────────────────────────────────────────────

def visualise(image_path, output_path=None):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = analyse(img_rgb)

    print(f"Is tulip:    {result['is_tulip']}")
    print(f"Confidence:  {result['confidence']}")
    print(f"Score:       {result['score']}/10")
    print(f"Mode:        {result['mode']}")

    # Draw profile widths
    annotated = img_bgr.copy()
    widths    = _profile_widths(img_rgb)
    h, w      = annotated.shape[:2]
    rows      = np.linspace(int(h * 0.1), int(h * 0.9), len(widths)).astype(int)
    col       = (0, 255, 0) if result["is_tulip"] else (0, 0, 255)

    for i, (y, wd) in enumerate(zip(rows, widths)):
        cx = w // 2
        cv2.line(annotated, (cx - int(wd/2), y), (cx + int(wd/2), y), col, 1)

    cv2.putText(annotated,
                f"{'Tulip' if result['is_tulip'] else 'Not tulip'}  "
                f"{result['confidence']:.0%}  [{result['mode']}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved → {output_path}")
    else:
        cv2.imshow("Glass check", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python glass_check.py <image> [output]")
        sys.exit(1)
    visualise(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)