"""
segment.py

Two-stage glass detection pipeline using ONNX Runtime GPU for MobileSAM.

Stage 1 — OpenCV candidate finder (~50ms)
Stage 2 — MobileSAM ONNX Runtime GPU refinement (~1-2s per glass)

No PyTorch dependency at inference time.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────

MIN_ASPECT     = 1.5
MIN_AREA_FRAC  = 0.03
MAX_AREA_FRAC  = 0.75
MAX_GLASSES    = 4
PADDING        = 16
SAM_IOU_THRESH = 0.75
ENCODER_PATH   = Path(__file__).parent / "models" / "mobile_sam_encoder.quant.onnx"
DECODER_PATH   = Path(__file__).parent / "models" / "mobile_sam_decoder.quant.onnx"
IMAGE_SIZE     = 1024


def _log(*args):
    print(*args, file=sys.stderr, flush=True)


# ── ONNX Runtime sessions ─────────────────────────────────────

_encoder_session = None
_decoder_session = None


def _get_sessions():
    global _encoder_session, _decoder_session
    if _encoder_session is not None:
        return _encoder_session, _decoder_session

    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _log(f"[segment] Available providers: {ort.get_available_providers()}")

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")
    if not DECODER_PATH.exists():
        raise FileNotFoundError(f"Decoder not found: {DECODER_PATH}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _encoder_session = ort.InferenceSession(str(ENCODER_PATH), sess_options=opts, providers=providers)
    _decoder_session = ort.InferenceSession(str(DECODER_PATH), sess_options=opts, providers=providers)

    _log(f"[segment] Encoder on: {_encoder_session.get_providers()[0]}")
    _log(f"[segment] Decoder on: {_decoder_session.get_providers()[0]}")

    return _encoder_session, _decoder_session


# ── Preprocessing ─────────────────────────────────────────────

def _preprocess(image_rgb):
    """
    Resize image for SAM encoder.
    samexporter --use-preprocess encoder expects HWC float32, no batch dim.
    Preprocessing (normalisation, padding) is done inside the ONNX graph.
    """
    h, w  = image_rgb.shape[:2]
    scale = IMAGE_SIZE / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(image_rgb, (new_w, new_h))

    # Pad to IMAGE_SIZE x IMAGE_SIZE
    padded = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized.astype(np.float32)

    return padded, scale, new_h, new_w


def _get_image_embedding(encoder, image_rgb):
    tensor, scale, new_h, new_w = _preprocess(image_rgb)
    # Input: HWC float32, no batch dimension
    embedding = encoder.run(None, {"input_image": tensor})[0]
    return embedding, scale, new_h, new_w


def _decode_mask(decoder, embedding, points, labels, orig_h, orig_w):
    inputs = {
        "image_embeddings": embedding,
        "point_coords":     points[np.newaxis].astype(np.float32),
        "point_labels":     labels[np.newaxis].astype(np.float32),
        "mask_input":       np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input":   np.array([0], dtype=np.float32),
        "orig_im_size":     np.array([orig_h, orig_w], dtype=np.float32),
    }

    masks, iou_scores, _ = decoder.run(None, inputs)

    best_idx   = int(np.argmax(iou_scores[0]))
    best_score = float(iou_scores[0][best_idx])
    best_mask  = masks[0][best_idx] > 0.0

    return best_mask, best_score


# ── OpenCV candidate finder ───────────────────────────────────

def _candidates_from_colour(image_rgb, h, w):
    hsv         = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    dark_mask   = cv2.inRange(hsv, np.array([0,0,0]),   np.array([180,80,80]))
    bright_mask = cv2.inRange(hsv, np.array([0,0,160]), np.array([40,80,255]))

    candidates = []
    step, win_w = w // 8, w // 5

    for cx in range(win_w//2, w-win_w//2, step):
        x1, x2       = max(0,cx-win_w//2), min(w,cx+win_w//2)
        bright_rows  = np.where(bright_mask[:,x1:x2].any(axis=1))[0]
        dark_rows    = np.where(dark_mask[:,x1:x2].any(axis=1))[0]

        if len(bright_rows) < 5 or len(dark_rows) < 10: continue
        head_top = int(bright_rows.min())
        body_bot = int(dark_rows.max())
        if body_bot <= bright_rows.max(): continue
        if body_bot - head_top < h * 0.15: continue
        candidates.append((x1, head_top, x2, body_bot))

    return candidates


def _candidates_from_edges(image_rgb, h, w):
    gray    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 30, 100)
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,15)))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x,y,cw,ch = cv2.boundingRect(c)
        if ch/max(cw,1) < MIN_ASPECT: continue
        if not (MIN_AREA_FRAC <= (cw*ch)/(h*w) <= MAX_AREA_FRAC): continue
        candidates.append((x,y,x+cw,y+ch))
    return candidates


def _candidates_from_contours(image_rgb, h, w):
    gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,4)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x,y,cw,ch = cv2.boundingRect(c)
        if ch/max(cw,1) < MIN_ASPECT: continue
        if not (MIN_AREA_FRAC <= (cw*ch)/(h*w) <= MAX_AREA_FRAC): continue
        candidates.append((x,y,x+cw,y+ch))
    return candidates


def _merge_boxes(boxes, iou_thresh=0.3):
    if not boxes: return []
    boxes  = list(set(boxes))
    merged = []

    while boxes:
        base = list(boxes.pop(0))
        remaining = []
        for b in boxes:
            ix1,iy1 = max(base[0],b[0]), max(base[1],b[1])
            ix2,iy2 = min(base[2],b[2]), min(base[3],b[3])
            if ix2<=ix1 or iy2<=iy1:
                remaining.append(b); continue
            inter = (ix2-ix1)*(iy2-iy1)
            union = (base[2]-base[0])*(base[3]-base[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            if inter/union > iou_thresh:
                base[0]=min(base[0],b[0]); base[1]=min(base[1],b[1])
                base[2]=max(base[2],b[2]); base[3]=max(base[3],b[3])
            else:
                remaining.append(b)
        merged.append(tuple(base))
        boxes = remaining

    return merged


def _find_opencv_candidates(image_rgb, debug=False):
    h, w      = image_rgb.shape[:2]
    all_boxes = (  _candidates_from_colour(image_rgb,h,w)
                 + _candidates_from_edges(image_rgb,h,w)
                 + _candidates_from_contours(image_rgb,h,w))

    merged   = _merge_boxes(all_boxes)
    filtered = [(x1,y1,x2,y2) for x1,y1,x2,y2 in merged
                if (y2-y1)/max(x2-x1,1) >= MIN_ASPECT
                and MIN_AREA_FRAC <= ((x2-x1)*(y2-y1))/(h*w) <= MAX_AREA_FRAC]

    # Always include centre-of-frame for close-up shots
    margin   = min(h,w)//6
    filtered = _merge_boxes(filtered + [(margin,margin,w-margin,h-margin)])
    filtered.sort(key=lambda b: b[0])
    filtered = filtered[:MAX_GLASSES]

    if debug:
        _log(f"[segment] OpenCV: {len(all_boxes)} raw → {len(merged)} merged → {len(filtered)} filtered")

    return filtered


def _fallback_grid_candidates(h, w):
    return [
        (max(0,int(w*cf)-w//8), max(0,int(h*rf)-h//4),
         min(w,int(w*cf)+w//8), min(h,int(h*rf)+h//4))
        for cf in [0.2,0.5,0.8] for rf in [0.3,0.7]
    ]


def _is_likely_guinness(crop_rgb, mask=None, debug=False):
    """
    Check crop has bright head over dark body.
    If a SAM mask is provided, zero out background pixels before checking
    so surrounding scene doesn't corrupt the brightness calculation.
    """
    THRESHOLD = 20
    DARK_MAX  = 130  # relaxed — pub lighting varies

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if mask is not None:
        # Crop mask to same region
        mh, mw = mask.shape
        ch, cw = gray.shape
        # Resize mask to crop dimensions
        mask_crop = cv2.resize(
            mask.astype(np.uint8),
            (cw, mh),
            interpolation=cv2.INTER_NEAREST
        )[:ch, :cw]
        # Replace background with NaN equivalent — use mean of masked region
        bg_val = float(gray[mask_crop > 0].mean()) if mask_crop.any() else gray.mean()
        gray_masked = gray.copy()
        gray_masked[mask_crop == 0] = bg_val
        gray = gray_masked

    h      = gray.shape[0]
    top    = float(gray[:h//4].mean())
    bottom = float(gray[h//2:].mean())
    diff   = top - bottom

    if debug:
        _log(f"[segment]   Guinness check: top={top:.1f} bot={bottom:.1f} "
             f"diff={diff:.1f} dark={bottom<DARK_MAX} pass={diff>THRESHOLD and bottom<DARK_MAX}")

    return diff > THRESHOLD and bottom < DARK_MAX


# ── SAM refinement ────────────────────────────────────────────

def _sam_refine(decoder, embedding, scale, orig_h, orig_w, bbox, debug=False):
    x1,y1,x2,y2 = bbox
    cx  = (x1+x2)/2
    h3  = (y2-y1)/3

    # Points in SCALED encoder space (not original image coords)
    raw_points = np.array([
        [cx*scale, y1*scale+h3*scale],
        [cx*scale, ((y1+y2)/2)*scale],
        [cx*scale, y2*scale-h3*scale],
        [0, 0],  # background
    ], dtype=np.float32)

    labels = np.array([1,1,1,0], dtype=np.float32)

    mask, score = _decode_mask(decoder, embedding, raw_points, labels, orig_h, orig_w)

    if debug:
        _log(f"[segment]   SAM score: {score:.3f}  mask coverage: {mask.mean()*100:.1f}%")

    if score < SAM_IOU_THRESH:
        return None, None, score

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows)==0 or len(cols)==0:
        return None, None, score

    return (int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())), mask, score


# ── Main entry point ──────────────────────────────────────────

def get_glass_crops(image_rgb, debug=False):
    h, w = image_rgb.shape[:2]
    encoder, decoder = _get_sessions()

    _log("[segment] Encoding image...")
    embedding, scale, new_h, new_w = _get_image_embedding(encoder, image_rgb)
    _log("[segment] Image encoded")

    candidates = _find_opencv_candidates(image_rgb, debug=debug)
    fallback   = False

    if not candidates:
        _log("[segment] No OpenCV candidates — using fallback grid")
        candidates = _fallback_grid_candidates(h, w)
        fallback   = True

    results     = []
    seen_bboxes = []

    for i, bbox in enumerate(candidates):
        if debug:
            _log(f"[segment] Candidate {i}: {bbox}")

        refined_bbox, sam_mask, score = _sam_refine(decoder, embedding, scale, h, w, bbox, debug)

        if refined_bbox is None:
            continue

        rx1,ry1,rx2,ry2 = refined_bbox

        # Deduplication
        dup = False
        for sb in seen_bboxes:
            sx1,sy1,sx2,sy2 = sb
            ix1,iy1 = max(rx1,sx1), max(ry1,sy1)
            ix2,iy2 = min(rx2,sx2), min(ry2,sy2)
            if ix2>ix1 and iy2>iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                a1=(rx2-rx1)*(ry2-ry1); a2=(sx2-sx1)*(sy2-sy1)
                if inter/(a1+a2-inter) > 0.4:
                    dup=True; break
        if dup:
            continue

        if (ry2-ry1)/max(rx2-rx1,1) < MIN_ASPECT:
            continue

        px1=max(0,rx1-PADDING); py1=max(0,ry1-PADDING)
        px2=min(w,rx2+PADDING); py2=min(h,ry2+PADDING)
        crop = image_rgb[py1:py2, px1:px2]

        # Crop mask to same padded region for colour check
        mask_crop = sam_mask[py1:py2, px1:px2] if sam_mask is not None else None

        if not _is_likely_guinness(crop, mask=mask_crop, debug=debug):
            _log(f"[segment] Candidate {i} rejected — failed Guinness colour check")
            continue

        seen_bboxes.append(refined_bbox)
        results.append({"crop":crop,"bbox":(px1,py1,px2,py2),"score":score,"index":len(results)})

        if len(results) >= MAX_GLASSES:
            break

    results.sort(key=lambda r: r["bbox"][0])
    for i,r in enumerate(results):
        r["index"] = i

    _log(f"[segment] {len(results)} glass(es) detected ({'fallback-grid' if fallback else 'opencv+sam'})")
    return results


# ── Visualisation ─────────────────────────────────────────────

def visualise(image_path, output_path=None, debug=False):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load: {image_path}"); return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    crops   = get_glass_crops(img_rgb, debug=debug)

    if not crops:
        print("No glasses detected"); return

    colours   = [(0,255,0),(0,165,255),(255,0,0),(0,255,255)]
    annotated = img_bgr.copy()

    for g in crops:
        x1,y1,x2,y2 = g["bbox"]
        col = colours[g["index"] % len(colours)]
        cv2.rectangle(annotated,(x1,y1),(x2,y2),col,3)
        cv2.putText(annotated,f"Glass {g['index']+1}  {g['score']:.2f}",
                    (x1,max(20,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.9,col,2)
        print(f"  Glass {g['index']+1}: bbox={g['bbox']} score={g['score']:.3f}")

    if output_path:
        cv2.imwrite(str(output_path), annotated)
        stem = Path(output_path).stem
        for g in crops:
            cv2.imwrite(f"{stem}_glass{g['index']+1}.jpg",
                        cv2.cvtColor(g["crop"],cv2.COLOR_RGB2BGR))
        print(f"Saved → {output_path}")
    else:
        print("Headless — pass output path to save")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segment.py <image> [output] [--debug]")
        sys.exit(1)
    _debug  = "--debug" in sys.argv
    _image  = sys.argv[1]
    _output = next((a for a in sys.argv[2:] if not a.startswith("--")), None)
    visualise(_image, _output, debug=_debug)