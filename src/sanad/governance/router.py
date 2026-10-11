"""Route a verdict to governed actions, an action tier and provenance labels."""

from __future__ import annotations

from sanad.governance.policy import tighten

SENSOR_CLASSES = {"SENSOR_FAULT", "BIAS_MISCALIBRATION"}


def route(case: dict, cfg: dict, tightened: set[str] | None = None) -> dict:
    """Return {'tier', 'actions', 'provenance', 'sensor_state'} for one case."""
    g = cfg["governance"]
    verdict, sub, conf = case["verdict"], case["subtype"], case["confidence"]
    lead_class = case["leading_class"]
    dur = case["duration_min"]
    actions: list[dict] = []
    prov, sensor_state = "VERIFIED", "OK"

    def act(text, tier):
        actions.append({"action": text, "tier": tier})

    if verdict == "UNCERTAIN":
        act("log + queue for operator review", "AUTO")
        act("human decides the verdict", "APPROVAL")
        prov = "SUSPECT" if lead_class not in ("PHYSICAL_EQUIPMENT_FAULT", "GRID_EVENT", "NORMAL") else "VERIFIED"
        sensor_state = "SUSPECT" if prov == "SUSPECT" else "OK"
    elif verdict == "COMMUNICATION_PROBLEM":
        if dur <= g["short_gap_max_min"]:
            act(f"impute gap (<= {g['short_gap_max_min']} min), label ESTIMATED", "AUTO")
        else:
            act(f"impute first {g['short_gap_max_min']} min, leave rest empty (QUARANTINED)", "AUTO")
            act("open communication ticket", "AUTO")
        prov = "ESTIMATED"
    elif verdict in SENSOR_CLASSES:
        if conf >= g["auto_quarantine_conf"]:
            act(f"quarantine {case['sensor_path'] or 'sensor'} and substitute estimate", "AUTO")
            act("open maintenance ticket", "AUTO")
            act("recalibration commit", "APPROVAL")
            prov, sensor_state = "QUARANTINED", "QUARANTINED"
        else:
            act("label SUSPECT", "AUTO")
            act("quarantine sensor", "APPROVAL")
            prov, sensor_state = "SUSPECT", "SUSPECT"
    elif verdict == "CORRUPTION_FALSIFICATION":
        act("hold affected data as SUSPECT", "AUTO")
        act("any conclusion about tampering (AI never accuses)", "APPROVAL")
        prov, sensor_state = "SUSPECT", "SUSPECT"
    elif verdict == "PHYSICAL_EQUIPMENT_FAULT" and sub == "BEARING_DEGRADATION":
        act("maintenance advisory: bearing degradation", "AUTO")
        act("shutdown scheduling", "APPROVAL")
    elif verdict in ("PHYSICAL_EQUIPMENT_FAULT", "GRID_EVENT"):
        act("notify operator, no data correction", "AUTO")
        severe = dur >= 30 or case.get("i_pu_max", 0) >= 1.25 or sub == "INTERRUPTION" and dur >= 5
        if severe:
            act("sustained / beyond thermal limit: immediate attention", "IMMEDIATE")
    else:  # UNKNOWN
        act("log + queue for operator review", "AUTO")
        act("human decides", "APPROVAL")
        prov = "SUSPECT"
    tier = max((a["tier"] for a in actions), key=["AUTO", "APPROVAL", "IMMEDIATE"].index)
    if tightened and verdict in tightened and tier == "AUTO":
        tier = tighten(tier)
        actions.append({"action": "adaptive autonomy: class tightened, operator approval required", "tier": tier})
    return {"tier": tier, "actions": actions, "provenance": prov, "sensor_state": sensor_state}


def escalate_concurrent(cases: list[dict]) -> None:
    """>= 2 concurrent critical (S3) cases or comm lost on >= 2 equipment -> IMMEDIATE."""
    def is_lost(o):
        return o["verdict"] == "COMMUNICATION_PROBLEM" and o["subtype"] == "DROPOUT"

    for c in cases:
        if c["severity"] != "S3" and not is_lost(c):
            continue
        conc = [o for o in cases if o is not c and o["k0"] < c["k1"] and c["k0"] < o["k1"]]
        crit = [o for o in conc + [c] if o["severity"] == "S3"]
        lost = {o["equipment_id"] for o in conc + [c] if is_lost(o)}
        if (c["severity"] == "S3" and len(crit) >= 2) or (is_lost(c) and len(lost) >= 2):
            if c["tier"] != "IMMEDIATE":
                c["tier"] = "IMMEDIATE"
                c["actions"].append({"action": ">= 2 critical events or multiple sensors lost", "tier": "IMMEDIATE"})
