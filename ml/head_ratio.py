"""
head_ratio.py

Measures the ratio of head height to total liquid height in a Guinness crop.
Uses a gradient-based approach to find the sharpest luminance transition
from bright head to dark body.

Usage:
    from head_ratio import analyse
    result = analyse(crop_rgb)
    # {"ratio": 0.24, "score": 9.6, "boundary_y": 112, "confident": True}

Test locally:
    python head_ratio.py <image_path> [output_path]
"""

import cv2
import numpy as np
import sys
from pathlib import Path


# Target head ratio and scoring curve steepness
TARGET_RATIO   = 0.25
CURVE_STEEPNESS = 40   # score = 10 - abs(ratio - TARGET) * STEEPNESS


def extract_luminance(crop_rgb, kernel_size=21):
    """
    Average luminance across the centre third of the crop,
    then smooth to reduce noise from glass label, logo, condensation.
    """
    h, w = crop_rgb.shape[:2]
    strip     = crop_rgb[:, w // 3 : 2 * w // 3]
    luminance = strip.mean(axis=(1, 2))  # shape: (h,)

    kernel    = np.ones(kernel_size) / kernel_size
    smoothed  = np.convolve(luminance, kernel, mode='same')
    return luminance, smoothed


def find_boundary(smoothed, search_top_frac=0.6):
    """
    Find the head/body boundary as the sharpest downward gradient
    in the top portion of the image.

    We only search the top 60% — a head ratio above 0.6 is physically
    implausible so we avoid false matches from the base of the glass.

    Returns (boundary_y, confidence) where confidence is the gradient
    magnitude at the boundary (higher = sharper = more confident).
    """
    limit     = int(len(smoothed) * search_top_frac)
    gradient  = np.diff(smoothed[:limit])
    boundary  = int(np.argmin(gradient))          # sharpest drop
    confidence = float(-gradient[boundary])       # magnitude of drop

    return boundary, confidence


def score_from_ratio(ratio):
    """
    Non-linear scoring curve centred on TARGET_RATIO (0.25).
    A 25% head scores 10. Scores drop symmetrically either side.
    Clamped to 0–10.
    """
    raw = 10 - abs(ratio - TARGET_RATIO) * CURVE_STEEPNESS
    return round(max(0.0, min(10.0, raw)), 1)


def analyse(crop_rgb):
    """
    Main entry point.

    Args:
        crop_rgb: numpy array HxWx3 in RGB — should be a single glass crop
                  from segment.get_glass_crops()

    Returns dict:
        ratio       — head height / total height (0.0–1.0)
        score       — 0–10 based on scoring curve
        boundary_y  — pixel row of detected boundary
        confident   — False if gradient was weak (blurry photo, flat pint)
    """
    CONFIDENCE_THRESHOLD = 4.0   # gradient magnitude below this = low confidence

    h             = crop_rgb.shape[0]
    raw, smoothed = extract_luminance(crop_rgb)
    boundary_y, confidence = find_boundary(smoothed)

    ratio     = boundary_y / h
    score     = score_from_ratio(ratio)
    confident = confidence > CONFIDENCE_THRESHOLD

    return {
        "ratio":      round(ratio, 3),
        "score":      score,
        "boundary_y": boundary_y,
        "confident":  confident,
        "confidence": round(confidence, 2),
    }


# ── Visualisation ─────────────────────────────────────────────

def visualise(image_path, output_path=None):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = analyse(img_rgb)

    print(f"Ratio:      {result['ratio']:.3f}  (target: {TARGET_RATIO})")
    print(f"Score:      {result['score']}/10")
    print(f"Boundary:   y={result['boundary_y']}px")
    print(f"Confident:  {result['confident']}  (gradient={result['confidence']})")

    # Draw boundary line on a copy
    annotated = img_bgr.copy()
    colour    = (0, 255, 0) if result["confident"] else (0, 165, 255)
    y         = result["boundary_y"]
    cv2.line(annotated, (0, y), (annotated.shape[1], y), colour, 2)
    cv2.putText(annotated,
                f"ratio={result['ratio']:.2f}  score={result['score']}/10",
                (10, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    # Also draw the smoothed luminance curve as a sidebar
    h, w      = annotated.shape[:2]
    _, smoothed = extract_luminance(img_rgb)
    norm      = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)
    sidebar_w = 80
    sidebar   = np.full((h, sidebar_w, 3), 30, dtype=np.uint8)
    for row in range(h - 1):
        x1 = int(norm[row]     * (sidebar_w - 4)) + 2
        x2 = int(norm[row + 1] * (sidebar_w - 4)) + 2
        cv2.line(sidebar, (x1, row), (x2, row + 1), (100, 220, 100), 1)
    cv2.line(sidebar, (0, y), (sidebar_w, y), colour, 1)
    annotated = np.hstack([annotated, sidebar])

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved → {output_path}")
    else:
        cv2.imshow("Head ratio", annotated)
        print("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python head_ratio.py <image> [output]")
        sys.exit(1)

    visualise(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)