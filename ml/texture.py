"""
texture.py

Analyses the head region of a Guinness crop for bubble texture quality.
Uses GLCM (Grey Level Co-occurrence Matrix) for texture regularity
and blob detection for bubble size distribution.

Usage:
    from texture import analyse
    result = analyse(crop_rgb, boundary_y)
    # {"score": 7.4, "avg_bubble_radius": 3.2, "bubble_count": 48,
    #  "homogeneity": 0.82, "energy": 0.14, "contrast": 12.1}

Test locally:
    python texture.py <image_path> <boundary_y> [output_path]
"""

import cv2
import numpy as np
import sys
from pathlib import Path


def extract_head_region(crop_rgb, boundary_y, min_height=20):
    """
    Crop just the head region — top of image down to boundary_y.
    Returns None if head region is too small to analyse.
    """
    head = crop_rgb[:boundary_y]
    if head.shape[0] < min_height:
        return None
    return head


def glcm_features(gray_region):
    """
    Compute Grey Level Co-occurrence Matrix features.
    High homogeneity + high energy + low contrast = fine creamy texture.

    Requires scikit-image. Falls back to basic stats if not available.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops

        # Reduce bit depth for manageable GLCM size
        reduced = (gray_region // 16).astype(np.uint8)  # 256 → 16 levels

        glcm = graycomatrix(
            reduced,
            distances=[2, 4],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16,
            symmetric=True,
            normed=True
        )

        return {
            "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
            "energy":      float(graycoprops(glcm, "energy").mean()),
            "contrast":    float(graycoprops(glcm, "contrast").mean()),
        }

    except ImportError:
        # Fallback: basic texture stats from raw pixel variance
        std  = float(gray_region.std())
        mean = float(gray_region.mean())
        return {
            "homogeneity": max(0.0, 1.0 - std / 128.0),
            "energy":      max(0.0, 1.0 - std / 255.0),
            "contrast":    std,
        }


def blob_features(gray_region):
    """
    Detect individual bubbles using blob detection.
    Small uniform bubbles = good creamy head.
    Large airy bubbles = bad frothy head.
    """
    try:
        from skimage.feature import blob_log

        # Normalise to 0-1
        norm = gray_region.astype(np.float32) / 255.0

        blobs = blob_log(
            norm,
            min_sigma=1,
            max_sigma=8,
            num_sigma=6,
            threshold=0.05
        )

        if len(blobs) == 0:
            return {"avg_bubble_radius": 0.0, "bubble_count": 0}

        radii = blobs[:, 2] * np.sqrt(2)
        return {
            "avg_bubble_radius": float(radii.mean()),
            "bubble_count":      int(len(blobs)),
        }

    except ImportError:
        # Fallback using OpenCV SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea  = True
        params.minArea       = 5
        params.maxArea       = 500
        params.filterByCircularity = True
        params.minCircularity      = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray_region)

        if not keypoints:
            return {"avg_bubble_radius": 0.0, "bubble_count": 0}

        radii = [kp.size / 2 for kp in keypoints]
        return {
            "avg_bubble_radius": float(np.mean(radii)),
            "bubble_count":      len(keypoints),
        }


def score_from_features(glcm, blobs):
    """
    Combine GLCM and blob features into a 0-10 texture score.

    Weights tuned empirically — adjust GLCM_WEIGHT / BLOB_WEIGHT
    once you have labelled examples to calibrate against.
    """
    GLCM_WEIGHT = 0.6
    BLOB_WEIGHT = 0.4

    # GLCM score: high homogeneity and energy, low contrast
    glcm_score = (
        glcm["homogeneity"] * 5.0 +
        glcm["energy"]      * 30.0 -
        glcm["contrast"]    * 0.05
    )
    glcm_score = max(0.0, min(10.0, glcm_score))

    # Blob score: small bubbles score higher
    avg_r = blobs["avg_bubble_radius"]
    if avg_r == 0:
        blob_score = 5.0  # can't tell — neutral
    else:
        blob_score = max(0.0, min(10.0, 10.0 - avg_r * 1.5))

    return round(GLCM_WEIGHT * glcm_score + BLOB_WEIGHT * blob_score, 1)


def analyse(crop_rgb, boundary_y):
    """
    Main entry point.

    Args:
        crop_rgb:   numpy RGB array — full glass crop from segment.py
        boundary_y: pixel row of head/body boundary from head_ratio.py

    Returns dict:
        score             — 0–10
        avg_bubble_radius — mean detected bubble radius in pixels
        bubble_count      — number of detected bubbles
        homogeneity       — GLCM homogeneity (higher = more uniform)
        energy            — GLCM energy (higher = more regular)
        contrast          — GLCM contrast (lower = smoother)
    """
    head = extract_head_region(crop_rgb, boundary_y)
    if head is None:
        return {"score": 5.0, "error": "head region too small",
                "avg_bubble_radius": 0.0, "bubble_count": 0,
                "homogeneity": 0.0, "energy": 0.0, "contrast": 0.0}

    gray  = cv2.cvtColor(head, cv2.COLOR_RGB2GRAY)
    glcm  = glcm_features(gray)
    blobs = blob_features(gray)
    score = score_from_features(glcm, blobs)

    return {
        "score":             score,
        "avg_bubble_radius": blobs["avg_bubble_radius"],
        "bubble_count":      blobs["bubble_count"],
        **glcm,
    }


# ── Visualisation ─────────────────────────────────────────────

def visualise(image_path, boundary_y, output_path=None):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = analyse(img_rgb, boundary_y)

    print(f"Score:          {result['score']}/10")
    print(f"Bubble radius:  {result['avg_bubble_radius']:.1f}px  "
          f"count={result['bubble_count']}")
    print(f"GLCM homog:     {result['homogeneity']:.3f}")
    print(f"GLCM energy:    {result['energy']:.3f}")
    print(f"GLCM contrast:  {result['contrast']:.3f}")

    annotated = img_bgr.copy()
    cv2.line(annotated, (0, boundary_y), (annotated.shape[1], boundary_y), (0, 255, 0), 2)
    cv2.putText(annotated,
                f"texture score={result['score']}/10  bubbles={result['bubble_count']}",
                (10, max(20, boundary_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved → {output_path}")
    else:
        cv2.imshow("Texture analysis", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python texture.py <image> <boundary_y> [output]")
        sys.exit(1)

    visualise(sys.argv[1], int(sys.argv[2]),
              sys.argv[3] if len(sys.argv) > 3 else None)