"""Template explanations filled with rule/feature evidence (no LLM, golden rule 5)."""

from __future__ import annotations

from sanad.physics.relations import RELATIONS

REL_NAMES = {
    "R1": "S²=P²+Q²", "R2": "P=√3·V·I·PF", "R3": "meter vs relay current", "R4": "bus power balance",
    "R5": "kWh register vs ∫P", "R5B": "bus vs feeder energy", "R6": "winding temp vs thermal model",
    "R7": "speed vs frequency/slip", "R8": "voltage agreement across meters", "R9": "frequency agreement",
    "R10": "DE/NDE vibration ratio", "R10E_DE": "DE vibration vs load envelope", "R10E_NDE": "NDE vibration vs load envelope",
}


def _pct(x) -> str:
    return f"{100 * x:+.1f}%"


def top_factors(f: dict, n: int = 4) -> list[dict]:
    """Relations with the strongest evidence in the episode (violation fraction, |z|)."""
    rows = [{"relation": r, "name": REL_NAMES[r], "violated_frac": round(f[f"viol_{r}"], 3),
             "z_median": round(f[f"zmed_{r}"], 2)} for r in RELATIONS if f[f"avail_{r}"] > 0]
    rows.sort(key=lambda x: (-x["violated_frac"], -abs(x["z_median"])))
    return rows[:n]


def held_relations(f: dict) -> list[str]:
    return [r for r in RELATIONS if f[f"avail_{r}"] > 0.3 and f[f"viol_{r}"] < 0.05]


def explain(case: dict, f: dict) -> str:
    eq, cls, sub = case["equipment_id"], case["leading_class"], case["leading_subtype"]
    dis = f.get("vote_dis", {})
    mr = dis.get("METER_CT/RELAY_CT")
    mb = dis.get("METER_CT/BUS_IMPLIED")
    rb = dis.get("RELAY_CT/BUS_IMPLIED")
    held = ", ".join(held_relations(f)[:4]) or "none"
    dur = case["duration_min"]
    if cls == "GRID_EVENT" and sub == "INTERRUPTION":
        s = (f"Voltage collapsed to ~0 on {3 + (len(f.get('absorbed', [])) > 3)}+ meters for {dur} min; motors coasted down "
             f"and kWh registers stopped together, so this is a genuine supply interruption, not a sensor fault.")
    elif cls == "GRID_EVENT":
        s = (f"All meters agree on a sustained under-voltage (min {f['v_pu_min']:.2f} pu, {dur} min); cross-meter voltage "
             f"agreement (R8) holds, so the grid really sagged — no sensor is to blame.")
    elif sub == "DRIFT":
        s = (f"{eq} meter current reads {_pct(mr or 0)} vs the relay CT and {_pct(mb or 0)} vs the bus balance, and the gap "
             f"grew during the episode; meter internal checks ({held}) hold, so the motor is likely fine and the meter CT is drifting.")
    elif cls == "BIAS_MISCALIBRATION" and sub == "CURRENT":
        odd = f.get("vote", "?")
        s = (f"{eq}: the {odd.replace('_', ' ').lower()} disagrees with the two other current paths "
             f"(meter/relay {_pct(mr or 0)}, relay/bus {_pct(rb or 0)}) with an abrupt onset — a constant bias on that CT.")
    elif cls == "BIAS_MISCALIBRATION" and sub == "VOLTAGE":
        s = (f"{eq} voltage is offset from the other meters (R8 median z {f['zmed_R8']:+.1f}) while frequency and internal "
             f"checks hold — a voltage-sensing bias on this meter, not a real voltage change.")
    elif cls == "BIAS_MISCALIBRATION" and sub == "TEMPERATURE":
        s = (f"{eq} winding temperature departs from the thermal model driven by the relay current (R6 median z "
             f"{f['zmed_R6']:+.1f}) while both currents agree and there is no over-current — the RTD reading is biased.")
    elif sub == "STUCK":
        s = (f"{eq} {case['sensor_path'] or 'channel'} reports an identical value for {dur} min while the machine runs; "
             f"independent paths keep moving — the sensor is stuck (frozen value).")
    elif sub == "OVERLOAD":
        s = (f"{eq} draws up to {f['i_pu_max']:.2f}× rated current on BOTH meter and relay CTs; the winding temperature "
             f"follows the thermal model and the bus balance holds — a genuine overload, the measurements are trustworthy.")
    elif sub == "BEARING_DEGRADATION":
        s = (f"{eq} vibration rises above the load envelope on both DE and NDE accelerometers, growing over the episode, "
             f"while electrical relations hold — mechanical (bearing) degradation is likely.")
    elif cls == "CORRUPTION_FALSIFICATION":
        s = (f"{eq} kWh register advances slower than the meter's own integrated power (R5 median z {f['zmed_R5']:+.1f}) "
             f"and the bus energy balance breaks, while sensor-health checks ({held}) look fine — the energy register is "
             f"under-reporting. SANAD does not conclude on intent; an operator must review.")
    elif sub == "DROPOUT":
        s = f"{eq} sent no data for {dur} min (comm_status LOST); values were not measured, not zero."
    elif sub == "DATA_GAP":
        s = (f"{eq} has missing values while the link is up ({100 * f['nan_frac']:.0f}% of fields empty on average) — "
             f"a data-quality gap; short gaps are imputed and labelled ESTIMATED.")
    elif sub == "SPEED":
        s = f"{eq} speed disagrees with frequency and slip (R7) while frequency agreement holds — speed sensor suspect."
    elif sub == "ACCELEROMETER":
        s = f"{eq}: only one accelerometer leaves the load envelope while the other stays normal — single-sensor fault."
    else:
        s = f"{eq}: anomalous pattern not matching a known signature (relations held: {held})."
    if case["verdict"] == "UNCERTAIN":
        s = (f"UNCERTAIN ({', '.join(case['abstain_reasons'])}). Leading hypothesis {cls}/{sub} at "
             f"{case['confidence']:.2f}, runner-up {case['runner_up']}: " + s)
    return s
