"""Stage C — raw confidence from evidence strength.

conf_h = score_h / (sum of all hypothesis scores + null_evidence). The null mass keeps
confidence low when the evidence is weak overall; the margin is top - runner-up.
"""

from __future__ import annotations


def raw_confidence(ranked: list[dict], null_evidence: float) -> list[dict]:
    total = sum(h["score"] for h in ranked) + null_evidence
    for h in ranked:
        h["conf"] = h["score"] / total if total > 0 else 0.0
    return ranked
