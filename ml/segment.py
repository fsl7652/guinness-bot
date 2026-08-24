"""
segment.py

Two-stage glass detection pipeline using ONNX Runtime GPU for MobileSAM.

Stage 1 — OpenCV candidate finder (~50ms)
Stage 2 — MobileSAM ONNX Runtime inference (encoder + decoder via CUDA EP)

Falls back to a 2x3 grid if OpenCV finds nothing.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────

MIN_ASPECT    = 1.5
MIN_AREA_FRAC = 0.03
MAX_AREA_FRAC = 0.75
MAX_GLASSES   = 4
PADDING       = 16
SAM_IOU_THRESH = 0.75

ENCODER_PATH = Path(__file__).parent / "models" / "mobile_sam_encoder.quant.onnx"
DECODER_PATH = Path(__file__).parent / "models" / "mobile_sam_decoder.quant.onnx"

IMAGE_SIZE = 1024   # MobileSAM expected input size

# ── ONNX Runtime sessions (lazy load) ────────────────────────

_encoder_session = None
_decoder_session = None

def _get_sessions():
    global _encoder_session, _decoder_session
    if _encoder_session is not None:
        return _encoder_session, _decoder_session

    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in available:
        print("[segment] WARNING: CUDAExecutionProvider not available, falling back to CPU", file=sys.stderr)
        providers = ['CPUExecutionProvider']

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _encoder_session = ort.InferenceSession(
        str(ENCODER_PATH), sess_options=opts, providers=providers
    )
    _decoder_session = ort.InferenceSession(
        str(DECODER_PATH), sess_options=opts, providers=providers
    )

    provider_used = _encoder_session.get_providers()[0]
    print(f"[segment] MobileSAM ONNX loaded ({provider_used})", file=sys.stderr)

    return _encoder_session, _decoder_session


# ── Image preprocessing ───────────────────────────────────────

def _preprocess_image(image_rgb):
    """
    Resize and normalise image for MobileSAM encoder.
    Returns preprocessed tensor and scale factors for coordinate mapping.
    """
    h, w = image_rgb.shape[:2]

    # Resize to 1024x1024 (samexporter --use-preprocess bakes this in)
    resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalise (ImageNet mean/std)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395, 57.12, 57.375],   dtype=np.float32)
    img  = (resized.astype(np.float32) - mean) / std

    # HWC → NCHW
    img = img.transpose(2, 0, 1)[np.newaxis]

    scale_x = IMAGE_SIZE / w
    scale_y = IMAGE_SIZE / h

    return img, scale_x, scale_y, h, w


# ── SAM inference ─────────────────────────────────────────────

def _encode_image(encoder, img_tensor):
    """Run encoder, return image embedding."""
    input_name = encoder.get_inputs()[0].name
    embedding  = encoder.run(None, {input_name: img_tensor})[0]
    return embedding


def _decode_mask(decoder, embedding, points, labels, orig_h, orig_w):
    """
    Run decoder with point prompts.
    Returns (mask, iou_score).
    """
    # Points must be float32, shape (1, N, 2)
    point_coords = points.astype(np.float32)[np.newaxis]
    point_labels = labels.astype(np.float32)[np.newaxis]

    orig_im_size = np.array([orig_h, orig_w], dtype=np.float32)

    inputs = {
        'image_embeddings': embedding,
        'point_coords':     point_coords,
        'point_labels':     point_labels,
        'orig_im_size':     orig_im_size,
    }

    # Handle decoders that also want mask_input and has_mask_input
    input_names = [i.name for i in decoder.get_inputs()]
    if 'mask_input' in input_names:
        inputs['mask_input']     = np.zeros((1, 1, 256, 256), dtype=np.float32)
        inputs['has_mask_input'] = np.zeros(1, dtype=np.float32)

    outputs = decoder.run(None, inputs)

    # outputs: [masks, iou_predictions, low_res_masks]
    masks           = outputs[0][0]   # (num_masks, H, W)
    iou_predictions = outputs[1][0]   # (num_masks,)

    best_idx   = int(np.argmax(iou_predictions))
    best_mask  = masks[best_idx] > 0  # threshold at 0
    best_score = float(iou_predictions[best_idx])

    return best_mask, best_score


def _sam_refine(encoder, decoder, embedding, scale_x, scale_y,
                orig_h, orig_w, bbox, debug=False):
    """
    Prompt SAM decoder with 3 points inside the candidate bbox.
    Returns (refined_bbox, score) or (None, score).
    """
    x1, y1, x2, y2 = bbox
    cx    = (x1 + x2) / 2
    h_third = (y2 - y1) / 3

    # Scale points to encoder input space
    raw_points = np.array([
        [cx,           y1 + h_third],
        [cx,           (y1 + y2) / 2],
        [cx,           y2 - h_third],
    ], dtype=np.float32)

    scaled_points = raw_points * np.array([[scale_x, scale_y]])
    labels        = np.ones(len(scaled_points), dtype=np.float32)

    mask, score = _decode_mask(decoder, embedding, scaled_points, labels, orig_h, orig_w)

    if debug:
        print(f"[segment]   SAM score={score:.3f}", file=sys.stderr)

    if score < SAM_IOU_THRESH:
        if debug:
            print(f"[segment]   Rejected — score {score:.3f} < {SAM_IOU_THRESH}", file=sys.stderr)
        return None, score

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None, score

    return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())), score


# ── OpenCV candidate finder ───────────────────────────────────

def _candidates_from_colour(image_rgb, h, w):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    dark_mask   = cv2.inRange(hsv, np.array([0,0,0]),   np.array([180,80,80]))
    bright_mask = cv2.inRange(hsv, np.array([0,0,160]), np.array([40,80,255]))

    candidates = []
    step  = w // 8
    win_w = w // 5

    for cx in range(win_w // 2, w - win_w // 2, step):
        x1 = max(0, cx - win_w // 2)
        x2 = min(w, cx + win_w // 2)

        bright_rows = np.where(bright_mask[:, x1:x2].any(axis=1))[0]
        dark_rows   = np.where(dark_mask[:, x1:x2].any(axis=1))[0]

        if len(bright_rows) < 5 or len(dark_rows) < 10:
            continue

        head_top = int(bright_rows.min())
        body_bot = int(dark_rows.max())

        if int(bright_rows.max()) >= body_bot:
            continue
        if body_bot - head_top < h * 0.15:
            continue

        candidates.append((x1, head_top, x2, body_bot))

    return candidates


def _candidates_from_edges(image_rgb, h, w):
    gray    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 100)
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,15)))

    candidates = []
    for c in cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch / max(cw,1) < MIN_ASPECT: continue
        if not (MIN_AREA_FRAC <= (cw*ch)/(h*w) <= MAX_AREA_FRAC): continue
        candidates.append((x, y, x+cw, y+ch))

    return candidates


def _candidates_from_contours(image_rgb, h, w):
    gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    candidates = []
    for c in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch / max(cw,1) < MIN_ASPECT: continue
        if not (MIN_AREA_FRAC <= (cw*ch)/(h*w) <= MAX_AREA_FRAC): continue
        candidates.append((x, y, x+cw, y+ch))

    return candidates


def _merge_boxes(boxes, iou_thresh=0.3):
    if not boxes: return []
    boxes  = list(set(boxes))
    merged = []

    while boxes:
        base = list(boxes.pop(0))
        remaining = []

        for b in boxes:
            ix1, iy1 = max(base[0],b[0]), max(base[1],b[1])
            ix2, iy2 = min(base[2],b[2]), min(base[3],b[3])
            if ix2 <= ix1 or iy2 <= iy1:
                remaining.append(b); continue

            inter = (ix2-ix1)*(iy2-iy1)
            union = (base[2]-base[0])*(base[3]-base[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter

            if inter/union > iou_thresh:
                base[0]=min(base[0],b[0]); base[1]=min(base[1],b[1])
                base[2]=max(base[2],b[2]); base[3]=max(base[3],b[3])
            else:
                remaining.append(b)

        merged.append(tuple(base))
        boxes = remaining

    return merged


def _find_opencv_candidates(image_rgb, debug=False):
    h, w = image_rgb.shape[:2]

    all_boxes = []
    all_boxes.extend(_candidates_from_colour(image_rgb, h, w))
    all_boxes.extend(_candidates_from_edges(image_rgb, h, w))
    all_boxes.extend(_candidates_from_contours(image_rgb, h, w))

    # Always include centre-of-frame for close-up shots
    margin = min(h, w) // 6
    all_boxes.append((margin, margin, w - margin, h - margin))

    merged   = _merge_boxes(all_boxes)
    filtered = []

    for (x1,y1,x2,y2) in merged:
        bw, bh = x2-x1, y2-y1
        if bh / max(bw,1) < MIN_ASPECT: continue
        if not (MIN_AREA_FRAC <= (bw*bh)/(h*w) <= MAX_AREA_FRAC): continue
        filtered.append((x1,y1,x2,y2))

    filtered.sort(key=lambda b: b[0])
    filtered = filtered[:MAX_GLASSES]

    if debug:
        print(f"[segment] OpenCV: {len(all_boxes)} raw → {len(merged)} merged → {len(filtered)} filtered", file=sys.stderr)

    return filtered


def _fallback_grid_candidates(h, w):
    candidates = []
    for col_frac in [0.2, 0.5, 0.8]:
        for row_frac in [0.3, 0.7]:
            cx, cy = int(w*col_frac), int(h*row_frac)
            bw, bh = w//4, h//2
            candidates.append((
                max(0,cx-bw//2), max(0,cy-bh//2),
                min(w,cx+bw//2), min(h,cy+bh//2)
            ))
    return candidates


def _is_likely_guinness(crop_rgb, debug=False):
    gray   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h      = gray.shape[0]
    top    = float(gray[:h//4].mean())
    bottom = float(gray[h//2:].mean())
    diff   = top - bottom

    # Body must be dark (near-black Guinness body)
    body_dark = bottom < 100

    if debug:
        print(f"[segment]   Guinness check: top={top:.1f} bot={bottom:.1f} diff={diff:.1f} dark={body_dark}", file=sys.stderr)

    return diff > 20 and body_dark


# ── Main entry point ──────────────────────────────────────────

def get_glass_crops(image_rgb, debug=False):
    """
    Detect all pint glasses in an RGB image and return cropped regions.

    Args:
        image_rgb: numpy HxWx3 RGB array
        debug:     verbose logging

    Returns list of dicts:
        { "crop", "bbox", "score", "index" }
    """
    h, w = image_rgb.shape[:2]

    encoder, decoder = _get_sessions()

    # Encode image once
    img_tensor, scale_x, scale_y, orig_h, orig_w = _preprocess_image(image_rgb)
    embedding = _encode_image(encoder, img_tensor)

    # OpenCV candidates
    candidates = _find_opencv_candidates(image_rgb, debug=debug)
    fallback   = False

    if not candidates:
        print("[segment] No OpenCV candidates — using fallback grid", file=sys.stderr)
        candidates = _fallback_grid_candidates(h, w)
        fallback   = True

    if debug:
        print(f"[segment] {len(candidates)} candidate(s) → SAM refinement", file=sys.stderr)

    results     = []
    seen_bboxes = []

    for i, bbox in enumerate(candidates):
        if debug:
            print(f"[segment] Candidate {i}: bbox={bbox}", file=sys.stderr)

        refined_bbox, score = _sam_refine(
            encoder, decoder, embedding,
            scale_x, scale_y, orig_h, orig_w,
            bbox, debug=debug
        )

        if refined_bbox is None:
            continue

        rx1, ry1, rx2, ry2 = refined_bbox

        # Deduplicate
        duplicate = False
        for sb in seen_bboxes:
            sx1,sy1,sx2,sy2 = sb
            ix1,iy1 = max(rx1,sx1), max(ry1,sy1)
            ix2,iy2 = min(rx2,sx2), min(ry2,sy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                a1    = (rx2-rx1)*(ry2-ry1)
                a2    = (sx2-sx1)*(sy2-sy1)
                if inter/(a1+a2-inter) > 0.4:
                    duplicate = True; break

        if duplicate:
            continue

        bw, bh = rx2-rx1, ry2-ry1
        if bh / max(bw,1) < MIN_ASPECT:
            continue

        px1 = max(0, rx1-PADDING)
        py1 = max(0, ry1-PADDING)
        px2 = min(w, rx2+PADDING)
        py2 = min(h, ry2+PADDING)
        crop = image_rgb[py1:py2, px1:px2]

        if not _is_likely_guinness(crop, debug=debug):
            print(f"[segment] Candidate {i} rejected — failed Guinness colour check", file=sys.stderr)
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

    results.sort(key=lambda r: r["bbox"][0])
    for i, r in enumerate(results):
        r["index"] = i

    mode = "fallback-grid" if fallback else "opencv+sam"
    print(f"[segment] {len(results)} glass(es) detected ({mode})", file=sys.stderr)

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

    colours   = [(0,255,0),(0,165,255),(255,0,0),(0,255,255)]
    annotated = img_bgr.copy()

    for g in crops:
        x1,y1,x2,y2 = g["bbox"]
        col = colours[g["index"] % len(colours)]
        cv2.rectangle(annotated, (x1,y1), (x2,y2), col, 3)
        cv2.putText(annotated, f"Glass {g['index']+1}  {g['score']:.2f}",
                    (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        print(f"  Glass {g['index']+1}: bbox={g['bbox']}  score={g['score']:.3f}  crop={g['crop'].shape[:2]}")

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
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segment.py <image> [output] [--debug]")
        sys.exit(1)

    _debug  = "--debug" in sys.argv
    _image  = sys.argv[1]
    _output = next((a for a in sys.argv[2:] if not a.startswith("--")), None)
    visualise(_image, _output, debug=_debug)