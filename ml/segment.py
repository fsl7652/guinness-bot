"""
segment.py

Two-stage glass detection:
  Stage 1 — OpenCV candidate finder (~50ms)
  Stage 2 — MobileSAM SamPredictor with PyTorch (~2-4s per glass)

Image is resized to 768x1024 before SAM to fit within Nano GPU memory.
Crops are returned in original image coordinates.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

MIN_ASPECT     = 1.5
MIN_AREA_FRAC  = 0.03
MAX_AREA_FRAC  = 0.80
MAX_GLASSES    = 4
PADDING        = 16
SAM_IOU_THRESH = 0.75
SAM_W, SAM_H   = 768, 1024   # resize target for GPU memory


def _log(*args):
    print(*args, file=sys.stderr, flush=True)


# ── MobileSAM lazy load ───────────────────────────────────────

_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    import torch
    from mobile_sam import sam_model_registry, SamPredictor

    weights = Path(__file__).parent / "models" / "mobile_sam.pt"
    if not weights.exists():
        raise FileNotFoundError(f"MobileSAM weights not found: {weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = sam_model_registry["vit_t"](checkpoint=str(weights))
    model.to(device).eval().float()

    _predictor = SamPredictor(model)
    _log(f"[segment] MobileSAM loaded on {device}")
    return _predictor


# ── OpenCV candidate finder ───────────────────────────────────

def _candidates_from_colour(image_rgb, h, w):
    hsv         = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    dark_mask   = cv2.inRange(hsv, np.array([0,0,0]),   np.array([180,80,80]))
    bright_mask = cv2.inRange(hsv, np.array([0,0,160]), np.array([40,80,255]))

    candidates = []
    step, win_w = w // 8, w // 5

    for cx in range(win_w//2, w-win_w//2, step):
        x1, x2      = max(0,cx-win_w//2), min(w,cx+win_w//2)
        bright_rows = np.where(bright_mask[:,x1:x2].any(axis=1))[0]
        dark_rows   = np.where(dark_mask[:,x1:x2].any(axis=1))[0]

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
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,21,4)
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
    margin   = min(h,w) // 6
    filtered = _merge_boxes(filtered + [(margin, margin, w-margin, h-margin)])
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


# ── Guinness colour check ─────────────────────────────────────

def _is_likely_guinness(crop_rgb, mask=None, debug=False):
    THRESHOLD = 20
    DARK_MAX  = 130

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if mask is not None and mask.any():
        mh, mw = mask.shape
        ch, cw = gray.shape
        m = cv2.resize(mask.astype(np.uint8),(cw,mh),
                       interpolation=cv2.INTER_NEAREST)[:ch,:cw].astype(bool)
        if m.any():
            bg = float(gray[m].mean())
            gray_m = gray.copy()
            gray_m[~m] = bg
            gray = gray_m

    h      = gray.shape[0]
    top    = float(gray[:h//4].mean())
    bottom = float(gray[h//2:].mean())
    diff   = top - bottom

    if debug:
        _log(f"[segment]   Guinness check: top={top:.1f} bot={bottom:.1f} "
             f"diff={diff:.1f} pass={diff>THRESHOLD and bottom<DARK_MAX}")

    return diff > THRESHOLD and bottom < DARK_MAX


# ── SAM refinement ────────────────────────────────────────────

def _sam_refine(predictor, image_small, sx, sy, bbox_orig, scale_x, scale_y, debug=False):
    """
    Run SamPredictor on a candidate bbox.
    bbox_orig is in original image coords.
    Points are scaled to SAM image (image_small) coords.
    Returns (mask_orig, score) where mask_orig is in original image coords.
    """
    import numpy as np

    x1,y1,x2,y2 = bbox_orig

    # Scale bbox to SAM image coords
    sx1 = int(x1 * scale_x); sy1 = int(y1 * scale_y)
    sx2 = int(x2 * scale_x); sy2 = int(y2 * scale_y)
    scx = (sx1+sx2)//2
    sh3 = (sy2-sy1)//4

    # 5 foreground points spanning full glass height + background point
    point_coords = np.array([
        [scx, sy1+sh3],          # top quarter
        [scx, sy1+sh3*2],        # upper middle
        [scx, sy1+sh3*3],        # lower middle
        [scx, sy2-sh3],          # bottom quarter
        [scx, (sy1+sy2)//2],     # centre
        [0, 0],                  # background
    ], dtype=np.float32)

    point_labels = np.array([1,1,1,1,1,0], dtype=np.int32)

    # SAM box prompt in SAM coords
    box = np.array([sx1, sy1, sx2, sy2], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_mask  = masks[best_idx]  # in SAM image (small) coords

    if debug:
        _log(f"[segment]   SAM scores: {scores}  best={best_score:.3f}  "
             f"coverage={best_mask.mean()*100:.1f}%")

    if best_score < SAM_IOU_THRESH:
        return None, None, best_score

    # Scale mask back to original image coords
    orig_h = int(image_small.shape[0] / scale_y)
    orig_w = int(image_small.shape[1] / scale_x)
    mask_orig = cv2.resize(
        best_mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Bbox from mask in original coords
    rows = np.where(mask_orig.any(axis=1))[0]
    cols = np.where(mask_orig.any(axis=0))[0]
    if len(rows)==0 or len(cols)==0:
        return None, None, best_score

    return (int(cols.min()),int(rows.min()),
            int(cols.max()),int(rows.max())), mask_orig, best_score


# ── Main entry point ──────────────────────────────────────────

def get_glass_crops(image_rgb, debug=False):
    """
    Detect all pint glasses and return cropped regions.
    """
    import torch

    orig_h, orig_w = image_rgb.shape[:2]

    # Resize for SAM GPU memory budget
    image_small = cv2.resize(image_rgb, (SAM_W, SAM_H))
    scale_x     = SAM_W / orig_w
    scale_y     = SAM_H / orig_h

    predictor = _get_predictor()

    # Set image once
    _log("[segment] Encoding image...")
    predictor.set_image(image_small)
    _log("[segment] Image encoded")

    # Stage 1: OpenCV on original image for accurate candidate coords
    candidates = _find_opencv_candidates(image_rgb, debug=debug)
    fallback   = False

    if not candidates:
        _log("[segment] No OpenCV candidates — using fallback grid")
        candidates = _fallback_grid_candidates(orig_h, orig_w)
        fallback   = True

    if debug:
        _log(f"[segment] {len(candidates)} candidate(s) → SAM refinement")

    results     = []
    seen_bboxes = []

    for i, bbox in enumerate(candidates):
        if debug:
            _log(f"[segment] Candidate {i}: {bbox}")

        refined_bbox, mask_orig, score = _sam_refine(
            predictor, image_small,
            SAM_W, SAM_H, bbox,
            scale_x, scale_y, debug
        )

        if refined_bbox is None:
            if debug:
                _log(f"[segment] Candidate {i} rejected by SAM")
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
            if debug:
                _log(f"[segment] Candidate {i} duplicate — skipped")
            continue

        if (ry2-ry1)/max(rx2-rx1,1) < MIN_ASPECT:
            if debug:
                _log(f"[segment] Candidate {i} aspect ratio too low")
            continue

        # Crop with padding
        px1=max(0,rx1-PADDING); py1=max(0,ry1-PADDING)
        px2=min(orig_w,rx2+PADDING); py2=min(orig_h,ry2+PADDING)
        crop = image_rgb[py1:py2, px1:px2]

        # Mask crop for colour check
        mask_crop = mask_orig[py1:py2, px1:px2] if mask_orig is not None else None

        if not _is_likely_guinness(crop, mask=mask_crop, debug=debug):
            _log(f"[segment] Candidate {i} rejected — failed Guinness colour check")
            continue

        seen_bboxes.append(refined_bbox)
        results.append({
            "crop":  crop,
            "bbox":  (px1,py1,px2,py2),
            "score": score,
            "index": len(results),
        })

        if len(results) >= MAX_GLASSES:
            break

        # Free GPU cache between candidates
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["bbox"][0])
    for i,r in enumerate(results):
        r["index"] = i

    _log(f"[segment] {len(results)} glass(es) detected "
         f"({'fallback-grid' if fallback else 'opencv+sam'})")

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
        print(f"  Glass {g['index']+1}: bbox={g['bbox']} "
              f"score={g['score']:.3f} crop={g['crop'].shape[:2]}")

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