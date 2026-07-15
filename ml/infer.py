"""
infer.py

Main entry point called by scorer.js via stdin/stdout.
Receives a JSON payload, runs the full pipeline, returns JSON.

Input (stdin):
    {
        "pint_image":   "<base64 encoded image>",
        "splitg_image": "<base64 encoded image>",  // optional
        "display_name": "Finn"                      // optional
    }

Output (stdout):
    {
        "glasses": [
            {
                "index":      1,
                "pint_score": 8.2,
                "final":      8.2,
                "splitg":     { "detected": true, "confidence": 0.91, ... },
                "verdict":    "😤 Serious pint. Respect.",
                "breakdown":  { ... },
                "warnings":   []
            },
            ...
        ],
        "message": "<formatted WhatsApp reply>"
    }

On error:
    { "error": "description" }

Usage:
    echo '{"pint_image": "<b64>"}' | python infer.py
"""

import sys
import json
import base64
import traceback
import io
import numpy as np
import cv2


def _b64_to_rgb(b64_string):
    """Decode base64 image string to numpy RGB array."""
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _run_pipeline(glass_crop_rgb):
    """
    Run the full scoring pipeline on a single glass crop.
    Returns the aggregated result dict.
    """
    from head_ratio  import analyse as hr_analyse
    from texture     import analyse as tx_analyse
    from colour_sep  import analyse as cs_analyse
    from glass_check import analyse as gc_analyse

    hr = hr_analyse(glass_crop_rgb)
    tx = tx_analyse(glass_crop_rgb, hr["boundary_y"])
    cs = cs_analyse(glass_crop_rgb, hr["boundary_y"])
    gc = gc_analyse(glass_crop_rgb)

    return hr, tx, cs, gc


def main():
    try:
        raw   = sys.stdin.read().strip()
        payload = json.loads(raw)
    except Exception as e:
        print(json.dumps({"error": f"Invalid input JSON: {e}"}))
        sys.exit(1)

    try:
        pint_b64    = payload.get("pint_image")
        splitg_b64  = payload.get("splitg_image")
        display_name = payload.get("display_name")

        if not pint_b64:
            print(json.dumps({"error": "pint_image required"}))
            sys.exit(1)

        pint_rgb = _b64_to_rgb(pint_b64)

        # ── Segment glasses ──────────────────────────────────
        from segment    import get_glass_crops
        from aggregator import aggregate, format_whatsapp

        crops = get_glass_crops(pint_rgb)

        if not crops:
            print(json.dumps({"error": "No glasses detected in image"}))
            sys.exit(1)

        splitg_result = None
        if splitg_b64:
            from splitg import analyse as sg_analyse
            splitg_rgb    = _b64_to_rgb(splitg_b64)
            splitg_result = sg_analyse(splitg_rgb)

        glasses     = []
        msg_parts   = []

        for g in crops:
            hr, tx, cs, gc = _run_pipeline(g["crop"])
            result = aggregate(hr, tx, cs, gc, splitg_result)
            result["index"] = g["index"] + 1  # 1-based for display

            glasses.append(result)

            msg_parts.append(
                format_whatsapp(result,
                                glass_index=g["index"] + 1 if len(crops) > 1 else None,
                                display_name=display_name)
            )

        message = ("\n\n" + "─" * 20 + "\n\n").join(msg_parts)

        output = {
            "glasses": glasses,
            "message": message,
            "pint_score": glasses[0]["pint_score"],
            "final":      glasses[0]["final"],
            "splitg":     glasses[0]["splitg"],
        }

        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({
            "error":     str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()