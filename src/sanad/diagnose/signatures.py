"""Stage B — Diagnose: episode features + signature matrix (violated/held relations).

Every hypothesis is a list of named evidence rules. A rule returns a strength in [0, 1]
computed from named relations / validity features, and has a weight. The hypothesis
score is the weighted sum (negative weights are contradictions), floored at zero.
All scores are traceable to rule names (golden rule 6); no black box decides.

Principle: genuine events -> physics holds and independent sensors move coherently;
sensor faults -> one path disagrees with the independent ones; falsification -> energy
relations (R5) break while sensor-health cues look fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from sanad.diagnose.episodes import Episode
from sanad.diagnose.voting import vote
from sanad.physics.relations import APPLIES, RELATIONS
from sanad.schema import MOTORS

FLAT = ["flat_meter", "flat_relay", "flat_rtd", "flat_speed", "flat_acc_de", "flat_acc_nde"]
FLAT_FAMILY = {"flat_meter": "electrical", "flat_relay": "electrical", "flat_rtd": "thermal", "flat_speed": "speed",
               "flat_acc_de": "mech", "flat_acc_nde": "mech"}
FAMILY_RELATIONS = {"electrical": ["R1", "R2", "R3", "R4", "R8", "R9"], "energy": ["R5", "R5B"], "thermal": ["R6"],
                    "speed": ["R7"], "mech": ["R10", "R10E_DE", "R10E_NDE"], "data": [], "other": [],
                    "grid": ["R1", "R2", "R3", "R4", "R5", "R5B", "R6", "R7", "R8", "R9", "R10", "R10E_DE", "R10E_NDE"]}
FLAT_PATH = {"flat_meter": "METER_CT", "flat_relay": "RELAY_CT", "flat_rtd": "RTD", "flat_speed": "SPEED",
             "flat_acc_de": "ACC_DE", "flat_acc_nde": "ACC_NDE"}


# a supply event disturbs every electrical/energy/thermal/speed relation at once
GRID_EXPECTS = ("R1", "R2", "R3", "R4", "R5", "R5B", "R6", "R7", "R8", "R9")


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def episode_features(ep: Episode, res: dict[str, pd.DataFrame], paths: dict, vote_base: dict, cfg: dict,
                     eqt: pd.DataFrame) -> dict:
    """Summarise an episode into named features (relation strengths, validity, voting)."""
    eq = ep.equipment_id
    r = res[eq].iloc[ep.k0:ep.k1]
    fams = set(ep.families) or {"grid"}
    f: dict = {"equipment_id": eq, "kind": ep.kind, "dur_min": ep.k1 - ep.k0, "k0": ep.k0, "k1": ep.k1,
               "families": sorted(fams), "family_relations": sorted({x for fm in fams for x in FAMILY_RELATIONS[fm]}),
               "flat_allowed": [c for c in FLAT if FLAT_FAMILY[c] in fams or "grid" in fams]}
    for rel in RELATIONS:
        z = r[f"z_{rel}"]
        avail = z.notna().mean() if rel in APPLIES[eq] else 0.0
        f[f"avail_{rel}"] = float(avail)
        if avail > 0:
            fl = r[f"flag_{rel}"] if f"flag_{rel}" in r else pd.Series(False, index=r.index)
            f[f"viol_{rel}"] = float(fl[z.notna()].mean())
            f[f"zmed_{rel}"] = float(z.median())
            f[f"zmax_{rel}"] = float(z.abs().max())
        else:
            f[f"viol_{rel}"], f[f"zmed_{rel}"], f[f"zmax_{rel}"] = 0.0, 0.0, 0.0
    for c in ["lost", "dq_gap", "dq_rate", "undervoltage", "interruption", "overcurrent", "kwh_decrease",
              "time_skew", "pf_invalid"] + FLAT:
        f[f"{c}_frac"] = float(r[f"v_{c}"].mean()) if f"v_{c}" in r else 0.0
    f["temp_step_any"] = float(r["temp_step"].any())
    f["nan_frac"] = float(r["nan_frac"].mean())
    f["i_pu_max"] = float(r["i_pu"].max()) if r["i_pu"].notna().any() else 0.0
    f["i_pu_med"] = float(r["i_pu"].median()) if r["i_pu"].notna().any() else 0.0
    f["v_pu_min"] = float(r["v_pu"].min()) if r["v_pu"].notna().any() else np.nan
    f["if_min"] = float(r["if_score"].min())
    f["if_flag_frac"] = float(r["if_flag"].mean())
    # trend of the current disagreement (drift grows, bias steps)
    raw3 = r["raw_R3"].dropna()
    if len(raw3) >= 9:
        n3 = len(raw3) // 3
        early, late = raw3.iloc[:n3].median(), raw3.iloc[-n3:].median()
        f["r3_early"], f["r3_late"] = float(early), float(late)
        f["r3_growth"] = float(abs(late) / max(abs(early), 1e-3))
        # step at onset: first 10 min of the episode vs 30..5 min before it (a bias jumps, a drift creeps)
        pre = res[eq]["raw_R3"].iloc[max(0, ep.k0 - 30):max(0, ep.k0 - 5)].dropna()
        head = res[eq]["raw_R3"].iloc[ep.k0:ep.k0 + 10].dropna()
        f["r3_step"] = float(abs((head.median() if len(head) else raw3.iloc[0]) - (pre.median() if len(pre) else 0.0)))
    else:
        f["r3_early"] = f["r3_late"] = f["r3_step"] = 0.0
        f["r3_growth"] = 1.0
    de = r["raw_R10E_DE"].dropna()
    f["vib_growth"] = float(de.iloc[-max(1, len(de) // 3):].median() - de.iloc[: max(1, len(de) // 3)].median()) if len(de) > 6 else 0.0
    # bus-level witnesses (for feeders): R4 / R5B on BUS1 during the episode
    b = res["BUS1"].iloc[ep.k0:ep.k1]
    f["bus_viol_R4"] = float(b["flag_R4"].mean())
    f["bus_viol_R5B"] = float(b["flag_R5B"].mean())
    f["bus_zmed_R5B"] = float(b["z_R5B"].median()) if b["z_R5B"].notna().any() else 0.0
    f["bus_avail_R5B"] = float(b["z_R5B"].notna().mean())
    # 2-of-3 current voting
    if eq in paths:
        run = res[eq]["context_ok"].iloc[ep.k0:ep.k1].to_numpy()
        p = paths[eq].iloc[ep.k0:ep.k1][run]
        v = vote(p, vote_base.get(eq, {}), cfg["diagnosis"]["vote_tolerance"], 0.1 * eqt.loc[eq, "i_rated_a"])
    else:
        v = {"paths": [], "disagreement": {}, "faulty_path": "INSUFFICIENT", "n_paths": 0}
    f["vote"] = v["faulty_path"]
    f["vote_n_paths"] = v["n_paths"]
    f["vote_paths"] = v["paths"]
    f["vote_dis"] = v["disagreement"]
    f["absorbed"] = list(ep.absorbed)
    return f


# --------------------------------------------------------------------------- hypotheses
@dataclass
class Rule:
    name: str
    weight: float
    fn: Callable[[dict], float]
    primary: bool = False  # a hypothesis needs >= 1 primary rule with strength >= 0.3


@dataclass
class Hypothesis:
    main_class: str
    subtype: str
    rules: list
    expects: tuple = ()  # relations expected to be violated (others should hold)
    path: str = ""  # implicated measurement path (for governance / provenance)
    needs_redundancy: bool = False  # verdict relies on independent electrical paths
    families: tuple = ("electrical", "energy", "thermal", "speed", "mech", "data", "other")

    @property
    def key(self) -> str:
        return f"{self.main_class}/{self.subtype}"


def v(rel: str) -> Callable[[dict], float]:
    return lambda f: _clip(f[f"viol_{rel}"] / 0.3)


def held(rel: str) -> Callable[[dict], float]:
    return lambda f: (1.0 - _clip(f[f"viol_{rel}"] / 0.3)) if f[f"avail_{rel}"] > 0.3 else 0.0


def frac(name: str, scale: float = 0.3) -> Callable[[dict], float]:
    return lambda f: _clip(f.get(name, 0.0) / scale)


def is_vote(*paths: str) -> Callable[[dict], float]:
    return lambda f: 1.0 if f["vote"] in paths else 0.0


def max_flat(f: dict) -> float:
    allowed = f.get("flat_allowed", FLAT)
    return _clip(max([f[f"{c}_frac"] for c in allowed] or [0.0]) / 0.3)


def hypotheses_for(eq: str, kind: str, families=None) -> list[Hypothesis]:
    hs = _hypotheses(eq, kind)
    if kind != "equipment" or not families:
        return hs
    fams = set(families)
    return [h for h in hs if fams & set(h.families)]


def _hypotheses(eq: str, kind: str) -> list[Hypothesis]:
    comm = [
        Hypothesis("COMMUNICATION_PROBLEM", "DROPOUT", [Rule("comm_status LOST", 3.0, frac("lost_frac", 0.5), primary=True)],
                   families=("data",)),
        Hypothesis("COMMUNICATION_PROBLEM", "DATA_GAP", [
            Rule("values missing with comm OK (burst)", 2.5, frac("dq_gap_frac", 0.2), primary=True),
            Rule("elevated missing-value rate", 2.0, frac("dq_rate_frac", 0.4), primary=True),
            Rule("contradiction: comm LOST", -2.0, frac("lost_frac", 0.5)),
        ], families=("data",)),
    ]
    if kind == "grid":
        return [
            Hypothesis("GRID_EVENT", "INTERRUPTION", [
                Rule("voltage ~0 on >= 3 meters", 3.0, frac("interruption_frac", 0.3), primary=True),
                Rule("R8 cross-meter voltage agreement holds", 0.5, held("R8")),
            ], expects=GRID_EXPECTS),
            Hypothesis("GRID_EVENT", "SUSTAINED_UNDERVOLTAGE", [
                Rule("under-voltage on >= 3 meters", 3.0, frac("undervoltage_frac", 0.3), primary=True),
                Rule("R8 cross-meter voltage agreement holds", 0.5, held("R8")),
                Rule("contradiction: voltage ~0", -1.5, frac("interruption_frac", 0.3)),
            ], expects=GRID_EXPECTS),
            Hypothesis("UNKNOWN", "UNEXPLAINED", [Rule("isolation forest out-of-range", 0.5, frac("if_flag_frac", 0.3), primary=True)]),
        ]
    if kind == "bus_balance":
        return comm + [
            Hypothesis("CORRUPTION_FALSIFICATION", "BUS_REGISTER", [
                Rule("R5B bus vs feeders energy breaks", 1.5, v("R5B"), primary=True),
                Rule("R4 bus power balance holds", 0.5, held("R4")),
            ], expects=("R5B",), needs_redundancy=True),
            Hypothesis("UNKNOWN", "BUS_BALANCE_MISMATCH", [
                Rule("R4 bus power balance breaks", 1.0, v("R4"), primary=True),
                Rule("R5B energy balance breaks", 0.5, v("R5B")),
            ], expects=("R4", "R5B")),
        ]
    hs = list(comm)
    hs += [
        Hypothesis("SENSOR_FAULT", "STUCK", [
            Rule("flatline on a varying channel", 3.0, max_flat, primary=True),
            Rule("contradiction: comm LOST", -2.0, frac("lost_frac", 0.5)),
        ], expects=("R3", "R4", "R5", "R6", "R7", "R10", "R10E_DE", "R10E_NDE", "R2", "R1", "R8", "R9")),
        Hypothesis("BIAS_MISCALIBRATION", "VOLTAGE", [
            Rule("R8 voltage disagrees with the other meters", 2.5, v("R8"), primary=True),
            Rule("R9 frequency agreement holds", 0.3, held("R9")),
            Rule("R1 internal consistency holds", 0.3, held("R1")),
            Rule("contradiction: values missing", -1.0, frac("nan_frac", 0.3)),
            Rule("contradiction: meter flatline (frozen values)", -2.5, frac("flat_meter_frac", 0.2)),
        ], expects=("R8", "R4", "R5B"), path="METER_CT", families=("electrical",)),
        Hypothesis("CORRUPTION_FALSIFICATION", "ENERGY_UNDERREPORT", [
            Rule("R5 register increments below own integrated power", 2.0,
                 lambda f: v("R5")(f) * (1.0 if f["zmed_R5"] < 0 else 0.3), primary=True),
            Rule("R5B bus energy balance breaks", 1.0, lambda f: _clip(f["bus_viol_R5B"] / 0.3)),
            Rule("R1/R2 meter internal checks hold", 0.4, lambda f: 0.5 * held("R1")(f) + 0.5 * held("R2")(f)),
            Rule("contradiction: flatline (stuck meter)", -2.5, frac("flat_meter_frac", 0.2)),
            Rule("contradiction: comm LOST", -2.0, frac("lost_frac", 0.3)),
        ], expects=("R5", "R5B"), path="METER_CT", needs_redundancy=True, families=("energy",)),
    ]
    if eq in MOTORS:
        hs += [
            Hypothesis("SENSOR_FAULT", "DRIFT", [
                Rule("2-of-3 vote: meter CT is the odd one out", 1.5, is_vote("METER_CT"), primary=True),
                Rule("R3 meter vs relay current breaks", 1.0, v("R3"), primary=True),
                Rule("disagreement grows over time (drift)", 1.2,
                     lambda f: _clip((f["r3_growth"] - 1.0) / 1.5) if f["r3_step"] < 0.012 else 0.2),
                Rule("R2 meter internal consistency holds", 0.4, held("R2")),
                Rule("R6 thermal model (relay-driven) holds", 0.3, held("R6")),
                Rule("contradiction: abrupt step at onset", -1.0, lambda f: _clip((f["r3_step"] - 0.006) / 0.012)),
                Rule("contradiction: flatline", -2.0, max_flat),
            ], expects=("R3", "R4", "R5B", "R7"), path="METER_CT", needs_redundancy=True, families=("electrical",)),
            Hypothesis("BIAS_MISCALIBRATION", "CURRENT", [
                Rule("2-of-3 vote: one CT is the odd one out", 1.5, is_vote("METER_CT", "RELAY_CT"), primary=True),
                Rule("R3 meter vs relay current breaks", 1.0, v("R3"), primary=True),
                Rule("abrupt step at onset (bias)", 1.2, lambda f: _clip(f["r3_step"] / 0.012)),
                Rule("contradiction: disagreement keeps growing", -1.0,
                     lambda f: _clip((f["r3_growth"] - 1.3) / 1.0) if f["r3_step"] < 0.012 else 0.0),
                Rule("R6 breaks together with relay CT (relay path)", 0.5,
                     lambda f: v("R6")(f) if f["vote"] == "RELAY_CT" else 0.0),
                Rule("contradiction: flatline", -2.0, max_flat),
            ], expects=("R3", "R4", "R5B", "R6", "R7"), path="CT", needs_redundancy=True, families=("electrical", "thermal")),
            Hypothesis("BIAS_MISCALIBRATION", "TEMPERATURE", [
                Rule("R6 winding temperature vs thermal model breaks", 2.0, v("R6"), primary=True),
                Rule("R3 currents agree", 0.6, held("R3")),
                Rule("temperature step faster than tau allows", 1.0, frac("temp_step_any", 1.0)),
                Rule("no over-current", 0.4, lambda f: 1.0 - frac("overcurrent_frac")(f)),
                Rule("contradiction: RTD flatline", -2.0, frac("flat_rtd_frac", 0.2)),
            ], expects=("R6",), path="RTD", families=("thermal",)),
            Hypothesis("PHYSICAL_EQUIPMENT_FAULT", "OVERLOAD", [
                Rule("current above rating on both CTs", 2.5, frac("overcurrent_frac", 0.3), primary=True),
                Rule("R3 independent currents agree", 0.8, held("R3")),
                Rule("R6 temperature follows the thermal model", 0.8, held("R6")),
                Rule("2-of-3 vote: all current paths agree", 0.5, is_vote("NONE")),
            ], expects=("R7", "R6"), families=("electrical", "thermal")),
            Hypothesis("PHYSICAL_EQUIPMENT_FAULT", "BEARING_DEGRADATION", [
                Rule("DE and NDE vibration both above load envelope", 2.5,
                     lambda f: min(v("R10E_DE")(f), v("R10E_NDE")(f)), primary=True),
                Rule("vibration grows over the episode", 0.5, lambda f: _clip(f["vib_growth"] / 0.03)),
                Rule("electrical relations hold", 0.3, lambda f: held("R3")(f)),
                Rule("contradiction: accelerometer flatline", -2.0, lambda f: _clip((f["flat_acc_de_frac"] + f["flat_acc_nde_frac"]) / 0.2)),
            ], expects=("R10", "R10E_DE", "R10E_NDE"), families=("mech",)),
            Hypothesis("SENSOR_FAULT", "ACCELEROMETER", [
                Rule("only one accelerometer leaves the envelope", 1.5,
                     lambda f: _clip(abs(v("R10E_DE")(f) - v("R10E_NDE")(f))), primary=True),
                Rule("R10 DE/NDE ratio breaks", 0.5, v("R10")),
            ], expects=("R10", "R10E_DE", "R10E_NDE"), path="ACC", families=("mech",)),
            Hypothesis("SENSOR_FAULT", "SPEED", [
                Rule("R7 speed vs frequency/slip breaks", 1.5, v("R7"), primary=True),
                Rule("R9 frequency agreement holds", 0.3, held("R9")),
                Rule("R3 currents agree", 0.3, held("R3")),
                Rule("contradiction: over-current (real slip change)", -1.0, frac("overcurrent_frac")),
            ], expects=("R7",), path="SPEED", families=("speed",)),
        ]
    hs += [
        Hypothesis("SENSOR_FAULT", "FREQUENCY", [
            Rule("R9 frequency disagrees with other meters", 1.5, v("R9"), primary=True),
        ], expects=("R9", "R7"), path="METER_CT", families=("electrical",)),
        Hypothesis("UNKNOWN", "UNEXPLAINED", [
            Rule("isolation forest out-of-range", 0.8, frac("if_flag_frac", 0.3), primary=True),
            Rule("unattributed relation violations", 0.2,
                 lambda f: _clip(max(f[f"viol_{r}"] for r in RELATIONS) / 0.5)),
        ]),
    ]
    return hs


def score(f: dict, hyps: list[Hypothesis]) -> list[dict]:
    """Score every hypothesis; returns a list sorted by score (desc) with fired rules."""
    out = []
    for h in hyps:
        fired, s, primary = [], 0.0, 0.0
        for rule in h.rules:
            e = float(rule.fn(f))
            s += rule.weight * e
            if rule.primary:
                primary = max(primary, e)
            if e > 0.3 and rule.weight > 0:
                fired.append({"rule": rule.name, "strength": round(e, 3), "weight": rule.weight})
            if e > 0.3 and rule.weight < 0:
                fired.append({"rule": rule.name, "strength": round(e, 3), "weight": rule.weight})
        if primary < 0.3:  # supporting ("held") evidence alone never makes a hypothesis
            s = 0.0
        out.append({"key": h.key, "main_class": h.main_class, "subtype": h.subtype, "score": max(0.0, s),
                     "rules": fired, "hyp": h})
    out.sort(key=lambda x: -x["score"])
    if out and out[0]["score"] <= 0:
        unk = [h for h in out if h["main_class"] == "UNKNOWN"]
        if unk:
            out.remove(unk[0])
            out.insert(0, unk[0])
    return out
