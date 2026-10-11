"""Governance: autonomy tiers, override rate, per-class track record, adaptive autonomy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import SERIES, TIER_ICON, feedback_path, load_cases, load_json, plotly_layout, sidebar  # noqa: E402
from sanad.config import load_config  # noqa: E402
from sanad.governance.policy import POLICY_TABLE  # noqa: E402
from sanad.governance.trust import read_feedback, tightened_classes, track_record  # noqa: E402

st.set_page_config(page_title="Governance · SANAD", layout="wide")
run = sidebar()
st.title("Governance")
st.markdown("Autonomy follows **reversibility and impact**, not confidence alone. SANAD never actuates the plant.")

cfg = load_config()
cases = load_cases(run)
fb = read_feedback(feedback_path(run))
gs = load_json(run, "governance_state.json")

st.subheader("Policy")
st.dataframe(pd.DataFrame(POLICY_TABLE, columns=["Situation", "Automatic", "Needs operator approval", "Immediate human"]),
             hide_index=True, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Cases by tier")
    tc = cases.tier.value_counts().reindex(["AUTO", "APPROVAL", "IMMEDIATE"], fill_value=0)
    fig = go.Figure(go.Bar(x=[TIER_ICON[t] for t in tc.index], y=tc.values, marker=dict(color=SERIES[0]),
                           text=tc.values, textposition="outside", hovertemplate="%{x}: %{y} cases<extra></extra>"))
    st.plotly_chart(plotly_layout(fig, 260), use_container_width=True)
with c2:
    st.subheader("Trust controls")
    rec = track_record(cases, fb)
    reviewed = int(rec.reviewed.sum()) if len(rec) else 0
    overr = int(rec.overridden.sum()) if len(rec) else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Reviewed cases", reviewed)
    m2.metric("Override rate", f"{(overr / reviewed * 100):.0f}%" if reviewed else "–",
              help=f"Limit {cfg['governance']['override_rate_limit'] * 100:.0f}% per class (min {cfg['governance']['min_cases_for_trust']} reviews)")
    m3.metric("Second-look queue", int(cases.second_look.sum()), help="Forced review sample of auto-resolved cases")
    cal = gs.get("calibration", {})
    tight = tightened_classes(rec, cfg, cal.get("ece_after")) if len(rec) else set()
    st.markdown("**Adaptive autonomy** — classes tightened by one tier: " + (", ".join(sorted(tight)) or "none"))
    st.caption("Takes effect on the next `python -m sanad.cli run` (the router reads feedback.jsonl).")

st.subheader("Per-class track record")
st.dataframe(rec, hide_index=True, use_container_width=True)
st.subheader("Second-look sample (auto-resolved cases to re-check)")
st.dataframe(cases[cases.second_look][["case_id", "equipment_id", "verdict", "subtype", "confidence", "start_utc"]],
             hide_index=True, use_container_width=True)
if cal:
    st.caption(f"Confidence calibration: {cal.get('method')} on {cal.get('n')} validation cases; "
               f"validation accuracy {cal.get('accuracy', float('nan')):.2f}; ECE before {cal.get('ece_before', float('nan')):.3f}.")
