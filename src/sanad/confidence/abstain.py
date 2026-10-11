"""Abstention: output UNCERTAIN instead of a verdict when SANAD should not decide.

Rules (CLAUDE.md §9), each reported by name:
* LOW_CONFIDENCE      calibrated top confidence < abstain_conf (0.60)
* LOW_MARGIN          top - runner-up raw confidence < abstain_margin (0.15)
* OUT_OF_DISTRIBUTION Isolation-Forest score beyond the training range AND relations are
                      violated that the leading hypothesis does not explain
* FEW_REDUNDANT_PATHS the verdict relies on independent electrical paths but fewer than
                      ``min_redundant_paths`` healthy independent paths exist
"""

from __future__ import annotations

from sanad.physics.relations import RELATIONS


def unexplained(features: dict, hyp) -> list[str]:
    if hyp.main_class in ("COMMUNICATION_PROBLEM",) or hyp.subtype == "STUCK":
        return []
    scope = features.get("family_relations") or RELATIONS
    return [r for r in scope if features[f"viol_{r}"] >= 0.3 and r not in hyp.expects]


def independent_paths(features: dict, hyp=None) -> int:
    """Healthy independent witnesses of a feeder's electrical quantities (excl. the meter).

    Current/CT hypotheses: relay CT and bus-implied current (2-of-3 voting).
    Energy hypotheses: bus energy balance (R5B) and the relay CT.
    """
    if hyp is not None and hyp.main_class == "CORRUPTION_FALSIFICATION":
        n = int(features.get("bus_avail_R5B", 0.0) > 0.3)
        n += int(features["equipment_id"] in ("PUMP1", "COMP1") and features.get("avail_R3", 0.0) > 0.0)
        return n
    paths = [p for p in ("RELAY_CT", "BUS_IMPLIED") if p in features.get("vote_paths", [])]
    fp = features.get("vote")
    return len([p for p in paths if p != fp])


def abstain_reasons(features: dict, top: dict, runner: dict | None, conf_cal: float, if_threshold: float,
                    cfg: dict) -> list[str]:
    c = cfg["confidence"]
    reasons = []
    if conf_cal < c["abstain_conf"]:
        reasons.append("LOW_CONFIDENCE")
    if runner is not None and top["conf"] - runner["conf"] < c["abstain_margin"]:
        reasons.append("LOW_MARGIN")
    if features["if_min"] < if_threshold and unexplained(features, top["hyp"]):
        reasons.append("OUT_OF_DISTRIBUTION")
    if top["hyp"].needs_redundancy and independent_paths(features, top["hyp"]) < c["min_redundant_paths"]:
        reasons.append("FEW_REDUNDANT_PATHS")
    return reasons
