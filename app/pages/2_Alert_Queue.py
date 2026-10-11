"""Alert Queue & Case Card with operator actions (Approve / Override / Defer / Annotate)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import TIER_ICON, feedback_path, load_cases, load_details, plotly_layout, record_operator_action, sidebar  # noqa: E402
from sanad.governance.trust import read_feedback  # noqa: E402

st.set_page_config(page_title="Alert Queue · SANAD", layout="wide")
run = sidebar()
st.title("Alert Queue & Case Card")

cases = load_cases(run)
details = load_details(run)
fb = read_feedback(feedback_path(run))
done = set(fb.case_id) if len(fb) else set()

f1, f2, f3, f4 = st.columns(4)
tiers = f1.multiselect("Tier", ["IMMEDIATE", "APPROVAL", "AUTO"], default=["IMMEDIATE", "APPROVAL"])
verd = f2.multiselect("Verdict", sorted(cases.verdict.unique()))
eqs = f3.multiselect("Equipment", sorted(cases.equipment_id.unique()))
hide_done = f4.checkbox("Hide reviewed cases", value=False)
q = cases[cases.tier.isin(tiers)] if tiers else cases
if verd:
    q = q[q.verdict.isin(verd)]
if eqs:
    q = q[q.equipment_id.isin(eqs)]
if hide_done:
    q = q[~q.case_id.isin(done)]
q = q.sort_values(["tier", "start_utc"], key=lambda s: s.map({"IMMEDIATE": 0, "APPROVAL": 1, "AUTO": 2}) if s.name == "tier" else s)
view = q[["case_id", "equipment_id", "start_utc", "duration_min", "verdict", "subtype", "confidence", "tier", "second_look"]].copy()
view["tier"] = view["tier"].map(TIER_ICON)
view["reviewed"] = view.case_id.isin(done)
st.caption(f"{len(q)} of {len(cases)} cases")
st.dataframe(view, hide_index=True, use_container_width=True, height=260)

if not len(q):
    st.stop()
cid = st.selectbox("Open case", q.case_id.tolist())
c = cases.set_index("case_id").loc[cid]
d = details[cid]

st.divider()
h1, h2 = st.columns([2, 1])
with h1:
    st.subheader(f"{cid} · {c.equipment_id} · {c.verdict}" + ("" if c.verdict == "UNCERTAIN" else f" / {c.subtype}"))
    st.markdown(f"**{TIER_ICON[c.tier]}** · severity {c.severity} · {c.start_utc:%Y-%m-%d %H:%M} → {c.end_utc:%H:%M} UTC "
                f"({c.duration_min} min) · families: {c.families}")
    if c.verdict == "UNCERTAIN":
        st.warning("SANAD abstained: " + ", ".join(c.abstain_reasons) + f". Leading hypothesis: {c.leading_class}/{c.subtype}.")
    st.info(c.explanation)
    st.markdown(f"**Runner-up:** {c.runner_up} ({c.runner_up_conf:.2f}) · **margin** {c.margin:.2f} · "
                f"**sensor:** {c.sensor_id if isinstance(c.sensor_id, str) and c.sensor_id else '—'} → state **{c.sensor_state}**, "
                f"data labelled **{c.provenance}**")
with h2:
    g = go.Figure(go.Indicator(mode="gauge+number", value=float(c.confidence) * 100, number={"suffix": "%"},
                               title={"text": "Calibrated confidence"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2a78d6"},
                                      "steps": [{"range": [0, 60], "color": "#f0efec"}, {"range": [60, 85], "color": "#e1e0d9"}],
                                      "threshold": {"line": {"color": "#0b0b0b", "width": 2}, "value": 60}}))
    g.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(g, use_container_width=True)
    st.caption("Abstain below 60 %; auto-quarantine of a sensor needs ≥ 85 %.")

e1, e2 = st.columns(2)
with e1:
    st.markdown("**Top factors (relations)**")
    tf = c.top_factors
    fig = go.Figure(go.Bar(x=[t["violated_frac"] * 100 for t in tf], y=[f"{t['relation']} {t['name']}" for t in tf],
                           orientation="h", marker=dict(color="#2a78d6"),
                           hovertemplate="%{y}: violated %{x:.0f}% of minutes<extra></extra>"))
    fig.update_xaxes(title="minutes violated in the episode (%)", range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(plotly_layout(fig, 220), use_container_width=True)
    st.caption("Relations that held: " + (", ".join(c.held_relations) or "none"))
with e2:
    st.markdown("**Rules fired (leading hypothesis)**")
    st.dataframe([{"rule": r["rule"], "strength": r["strength"], "weight": r["weight"]} for r in d["evidence"]],
                 hide_index=True, use_container_width=True)
    st.markdown("**Hypotheses considered**")
    st.dataframe([{"hypothesis": h["key"], "score": h["score"], "raw conf": h["conf"]} for h in d["hypotheses"]],
                 hide_index=True, use_container_width=True)
    vd = d["features"].get("vote_dis", {})
    if vd:
        st.caption("2-of-3 current vote: " + d["features"]["vote"] + " · " +
                   ", ".join(f"{k} {v * 100:+.1f}%" for k, v in vd.items()))

st.markdown("**Recommended actions**")
st.dataframe([{"action": a["action"], "tier": TIER_ICON[a["tier"]]} for a in c.actions], hide_index=True, use_container_width=True)

st.markdown("**Operator decision** (logged to the audit chain and to feedback.jsonl)")
o1, o2 = st.columns([1, 2])
operator = o1.text_input("Operator", value="operator-1")
reason = o2.text_input("Reason / note (mandatory for Override and Annotate)")
label = st.selectbox("Corrected verdict (for Override)", ["", "NORMAL", "PHYSICAL_EQUIPMENT_FAULT", "GRID_EVENT", "SENSOR_FAULT",
                                                         "BIAS_MISCALIBRATION", "CORRUPTION_FALSIFICATION", "COMMUNICATION_PROBLEM", "UNKNOWN"])
b1, b2, b3, b4 = st.columns(4)
act = None
if b1.button("✅ Approve", use_container_width=True):
    act = "APPROVE"
if b2.button("✋ Override", use_container_width=True):
    act = "OVERRIDE"
if b3.button("⏸ Defer", use_container_width=True):
    act = "DEFER"
if b4.button("📝 Annotate", use_container_width=True):
    act = "ANNOTATE"
if act:
    if act in ("OVERRIDE", "ANNOTATE") and not reason.strip():
        st.error(f"{act.title()} requires a reason.")
    elif act == "OVERRIDE" and not label:
        st.error("Override requires the corrected verdict.")
    else:
        e = record_operator_action(run, {"case_id": cid, "verdict": c.verdict}, act, reason.strip(), operator,
                                   label if act == "OVERRIDE" else "")
        st.success(f"{act} recorded as audit entry #{e['seq']} (hash {e['hash'][:12]}…).")
if cid in done:
    st.caption("Previous operator actions on this case:")
    st.dataframe(fb[fb.case_id == cid], hide_index=True, use_container_width=True)
