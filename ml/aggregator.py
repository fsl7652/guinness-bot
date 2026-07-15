# weighted formal score aggregator
"""
aggregator.py

Combines individual module scores into a final pint score using a weighted formula.
Split-the-G is kept entirely separate and returned alongside the pint score.

Pint score weights (must sum to 1.0):
    head_ratio    37.5%  — most important, geometry is king
    texture       31.25% — creamy vs frothy
    colour_sep    25.0%  — sharpness of the line
    glass_check   6.25%  — correct glass shape

Split-the-G is a separate binary result, not folded into the pint score.

Usage:
    from aggregator import aggregate
    result = aggregate(head_ratio_result, texture_result,
                       colour_sep_result, glass_result, splitg_result)
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Weights ───────────────────────────────────────────────────
# Adjust these once you have real labelled data and run calibrate.py

WEIGHTS = {
    "head_ratio":  0.375,
    "texture":     0.3125,
    "colour_sep":  0.25,
    "glass_check": 0.0625,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ── Verdict strings ───────────────────────────────────────────

def _verdict(score):
    if score >= 9.5: return "🏆 Perfection. Buy that barman a drink."
    if score >= 8.5: return "😤 Serious pint. Respect."
    if score >= 7.0: return "👍 Solid. No complaints."
    if score >= 5.5: return "😐 Drinkable. Just about."
    if score >= 4.0: return "😬 That's rough. Who poured this?"
    return               "🚨 Criminal. Send it back."

def _splitg_comment(detected, confidence):
    if detected:
        if confidence > 0.9: return "✅ Textbook split. Legend."
        if confidence > 0.7: return "✅ Clean split."
        return                       "✅ Split — just about."
    else:
        if confidence > 0.9: return "❌ Nowhere near."
        return                       "❌ Didn't split the G."


# ── Low confidence flag ───────────────────────────────────────

def _confidence_warnings(head_ratio_result, texture_result,
                          colour_sep_result, glass_result):
    warnings = []
    if not head_ratio_result.get("confident", True):
        warnings.append("head boundary unclear")
    if head_ratio_result.get("ratio", 0.25) < 0.05:
        warnings.append("very thin head — check photo angle")
    if texture_result.get("bubble_count", 1) == 0:
        warnings.append("no bubbles detected — head may be flat")
    if glass_result.get("mode") == "stub":
        warnings.append("glass check is estimated")
    return warnings


# ── Main entry point ──────────────────────────────────────────

def aggregate(head_ratio_result, texture_result,
              colour_sep_result, glass_result,
              splitg_result=None):
    """
    Args:
        head_ratio_result:  dict from head_ratio.analyse()
        texture_result:     dict from texture.analyse()
        colour_sep_result:  dict from colour_sep.analyse()
        glass_result:       dict from glass_check.analyse()
        splitg_result:      dict from splitg.analyse() or None if no mid-sip photo

    Returns dict:
        pint_score      — 0–10 weighted pint quality score
        splitg_score    — "split" / "not_split" / "not_evaluated"
        verdict         — cheeky text verdict
        breakdown       — per-factor scores for the reply message
        warnings        — list of low-confidence flags
        final           — alias for pint_score (used by infer.py)
    """
    scores = {
        "head_ratio":  head_ratio_result.get("score", 5.0),
        "texture":     texture_result.get("score",    5.0),
        "colour_sep":  colour_sep_result.get("score", 5.0),
        "glass_check": glass_result.get("score",      5.0),
    }

    pint_score = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    pint_score = round(max(0.0, min(10.0, pint_score)), 1)

    # Split-the-G — separate result
    if splitg_result is not None:
        splitg_status  = "split" if splitg_result["detected"] else "not_split"
        splitg_comment = _splitg_comment(splitg_result["detected"],
                                         splitg_result["confidence"])
    else:
        splitg_status  = "not_evaluated"
        splitg_comment = "💧 No mid-sip photo — send one to check the G split"

    warnings = _confidence_warnings(head_ratio_result, texture_result,
                                    colour_sep_result, glass_result)

    return {
        "pint_score":   pint_score,
        "final":        pint_score,   # alias used by infer.py / scorer.js
        "splitg":       {
            "status":     splitg_status,
            "detected":   splitg_result["detected"] if splitg_result else None,
            "confidence": splitg_result["confidence"] if splitg_result else None,
            "comment":    splitg_comment,
        },
        "verdict":      _verdict(pint_score),
        "breakdown":    {
            "head_ratio":  scores["head_ratio"],
            "texture":     scores["texture"],
            "colour_sep":  scores["colour_sep"],
            "glass_check": scores["glass_check"],
            "head_ratio_raw": head_ratio_result.get("ratio"),
            "bubble_count":   texture_result.get("bubble_count"),
            "is_tulip":       glass_result.get("is_tulip"),
        },
        "warnings":     warnings,
    }


def format_whatsapp(result, glass_index=None, display_name=None):
    """
    Format aggregator result as a WhatsApp message string.

    Args:
        result:       output of aggregate()
        glass_index:  1-based index if multiple glasses in one photo
        display_name: person's name
    """
    b     = result["breakdown"]
    g     = result["splitg"]
    score = result["pint_score"]

    header = f"🍺 *{display_name + \"'s \" if display_name else ''}"
    if glass_index:
        header += f"Glass {glass_index} — "
    header += f"{score}/10*"

    lines = [
        header,
        result["verdict"],
        "",
        f"Head ratio:    {b['head_ratio']}/10  ({b['head_ratio_raw']:.0%} head)",
        f"Texture:       {b['texture']}/10  ({b['bubble_count']} bubbles)",
        f"Colour sep:    {b['colour_sep']}/10",
        f"Glass:         {b['glass_check']}/10  "
            f"({'tulip ✓' if b['is_tulip'] else 'wrong glass ✗'})",
        "",
        g["comment"],
    ]

    if result["warnings"]:
        lines += ["", f"⚠️ _{', '.join(result['warnings'])}_"]

    return "\n".join(lines)