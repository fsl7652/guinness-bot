"""
colour_sep.py

Scores the sharpness of the colour separation between head and body.
A well-poured Guinness has a crisp, clean line between white head and dark body.
A bad pour has a murky gradient — grey, streaky, or uneven.

Usage:
    from colour_sep import analyse
    result = analyse(crop_rgb, boundary_y)
    # {"score": 8.2, "gradient_magnitude": 42.1, "uniformity": 0.87}

Test locally:
    python colour_sep.py <image_path> <boundary_y> [output_path]
"""

import cv2
import numpy as np
import sys
from pathlib import Path


# Number of pixel rows above and below boundary to sample
SAMPLE_WINDOW = 20


def sample_boundary_region(crop_rgb, boundary_y):
    """
    Extract a horizontal band around the boundary.
    Returns the region above (head side) and below (body side).
    """
    h = crop_rgb.shape[0]
    y_top = max(0, boundary_y - SAMPLE_WINDOW)
    y_bot = min(h, boundary_y + SAMPLE_WINDOW)

    above = crop_rgb[y_top:boundary_y]
    below = crop_rgb[boundary_y:y_bot]
    band  = crop_rgb[y_top:y_bot]

    return above, below, band


def gradient_magnitude(band_rgb):
    """
    Measure sharpness of the transition across the band.
    Sharp line = high gradient. Murky gradient = low value.
    """
    gray     = cv2.cvtColor(band_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Sobel in Y direction captures horizontal transitions
    sobel_y  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = float(np.abs(sobel_y).mean())
    return magnitude


def colour_contrast(above_rgb, below_rgb):
    """
    Measure the mean colour difference between head (above) and body (below).
    High contrast = clearly separated colours.
    Returns per-channel differences and overall contrast.
    """
    above_mean = above_rgb.mean(axis=(0, 1))  # mean RGB of head side
    below_mean = below_rgb.mean(axis=(0, 1))  # mean RGB of body side

    diff     = np.abs(above_mean - below_mean)
    contrast = float(diff.mean())

    return {
        "contrast":    round(contrast, 2),
        "diff_r":      round(float(diff[0]), 2),
        "diff_g":      round(float(diff[1]), 2),
        "diff_b":      round(float(diff[2]), 2),
        "above_mean":  above_mean.tolist(),
        "below_mean":  below_mean.tolist(),
    }


def boundary_uniformity(crop_rgb, boundary_y, n_columns=10):
    """
    Check how consistent the boundary is horizontally.
    A clean pour has a level boundary across the glass.
    A messy pour has the boundary at different heights in different columns.

    Samples n_columns vertical strips and finds the boundary in each.
    Returns the std dev of boundary positions — lower = more uniform.
    """
    from head_ratio import find_boundary, extract_luminance

    h, w = crop_rgb.shape[:2]
    strip_w = w // n_columns
    boundaries = []

    for i in range(n_columns):
        x1 = i * strip_w
        x2 = min(w, x1 + strip_w)
        strip = crop_rgb[:, x1:x2]

        # Simple luminance scan per strip
        lum  = strip.mean(axis=(1, 2))
        kern = np.ones(11) / 11
        smoothed = np.convolve(lum, kern, mode='same')
        grad = np.diff(smoothed[:int(h * 0.6)])
        if len(grad) > 0:
            boundaries.append(int(np.argmin(grad)))

    if len(boundaries) < 3:
        return 0.5  # not enough data — neutral

    std = float(np.std(boundaries))
    # Normalise: 0 std = score 1.0, 50px std = score 0.0
    uniformity = max(0.0, 1.0 - std / 50.0)
    return round(uniformity, 3)


def score_from_features(gradient_mag, contrast, uniformity):
    """
    Combine gradient magnitude, colour contrast, and boundary uniformity
    into a 0–10 colour separation score.
    """
    GRAD_WEIGHT    = 0.4
    CONTRAST_WEIGHT = 0.35
    UNIFORM_WEIGHT = 0.25

    # Gradient: scale 0–80 → 0–10
    grad_score     = min(10.0, gradient_mag / 8.0)
    # Contrast: scale 0–100 → 0–10
    contrast_score = min(10.0, contrast / 10.0)
    # Uniformity already 0–1 → 0–10
    uniform_score  = uniformity * 10.0

    combined = (
        GRAD_WEIGHT     * grad_score     +
        CONTRAST_WEIGHT * contrast_score +
        UNIFORM_WEIGHT  * uniform_score
    )
    return round(max(0.0, min(10.0, combined)), 1)


def analyse(crop_rgb, boundary_y):
    """
    Main entry point.

    Args:
        crop_rgb:   numpy RGB array — full glass crop
        boundary_y: pixel row of head/body boundary from head_ratio.py

    Returns dict:
        score              — 0–10
        gradient_magnitude — Sobel gradient at boundary (higher = sharper)
        contrast           — mean RGB diff between head and body regions
        uniformity         — how level the boundary is (0–1)
    """
    above, below, band = sample_boundary_region(crop_rgb, boundary_y)

    if above.shape[0] == 0 or below.shape[0] == 0:
        return {"score": 5.0, "error": "boundary too close to edge",
                "gradient_magnitude": 0.0, "contrast": 0.0, "uniformity": 0.5}

    grad        = gradient_magnitude(band)
    colour      = colour_contrast(above, below)
    uniformity  = boundary_uniformity(crop_rgb, boundary_y)
    score       = score_from_features(grad, colour["contrast"], uniformity)

    return {
        "score":              score,
        "gradient_magnitude": round(grad, 2),
        "uniformity":         uniformity,
        **colour,
    }


# ── Visualisation ─────────────────────────────────────────────

def visualise(image_path, boundary_y, output_path=None):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = analyse(img_rgb, boundary_y)

    print(f"Score:       {result['score']}/10")
    print(f"Gradient:    {result['gradient_magnitude']}")
    print(f"Contrast:    {result['contrast']}")
    print(f"Uniformity:  {result['uniformity']}")

    annotated = img_bgr.copy()
    # Draw boundary + sample window
    cv2.line(annotated, (0, boundary_y), (annotated.shape[1], boundary_y), (0, 255, 0), 2)
    cv2.line(annotated, (0, max(0, boundary_y - SAMPLE_WINDOW)),
             (annotated.shape[1], max(0, boundary_y - SAMPLE_WINDOW)), (0, 200, 200), 1)
    cv2.line(annotated, (0, min(annotated.shape[0], boundary_y + SAMPLE_WINDOW)),
             (annotated.shape[1], min(annotated.shape[0], boundary_y + SAMPLE_WINDOW)),
             (0, 200, 200), 1)
    cv2.putText(annotated,
                f"colour sep={result['score']}/10  uniformity={result['uniformity']}",
                (10, max(20, boundary_y - SAMPLE_WINDOW - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved → {output_path}")
    else:
        cv2.imshow("Colour separation", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python colour_sep.py <image> <boundary_y> [output]")
        sys.exit(1)

    visualise(sys.argv[1], int(sys.argv[2]),
              sys.argv[3] if len(sys.argv) > 3 else None)