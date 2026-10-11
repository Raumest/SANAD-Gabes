"""SANAD-Gabès dashboard — home. Run with: streamlit run app/Home.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from common import TIER_ICON, load_cases, load_json, sidebar  # noqa: E402

st.set_page_config(page_title="SANAD-Gabès", page_icon="🛡️", layout="wide")
run = sidebar()

st.title("🛡️ SANAD-Gabès")
st.markdown("**Sensor-Assured Network for Autonomous Decisions** — *Before the grid acts, SANAD verifies.*")
st.markdown(
    "An explainable trust layer for a fictional 400 V industrial line in the Gabès industrial zone. "
    "For every anomaly it answers: **is the plant misbehaving, or is the measurement lying?** "
    "SANAD is advisory: it never trips breakers or sheds load; it labels data and routes decisions to people."
)

cases = load_cases(run)
lat = load_json(run, "latency.json")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cases", len(cases))
c2.metric("Uncertain (abstained)", int((cases.verdict == "UNCERTAIN").sum()))
c3.metric("Need approval", int((cases.tier == "APPROVAL").sum()))
c4.metric("Immediate human", int((cases.tier == "IMMEDIATE").sum()))
c5.metric("Decision latency p95", f"{lat.get('decision_ms_p95', '–')} ms")

st.subheader("Verdicts in this run")
counts = cases.groupby(["verdict", "tier"]).size().rename("cases").reset_index()
counts["tier"] = counts["tier"].map(TIER_ICON)
st.dataframe(counts, hide_index=True, use_container_width=True)

st.subheader("Pages")
st.markdown(
    """
1. **Live Plant** — single-line diagram, sensors coloured by trust state, live trends.
2. **Alert Queue & Case Card** — verdict, confidence, runner-up, evidence, explanation; Approve / Override / Defer / Annotate.
3. **Trust Fingerprint** — R1–R10 residual heatmap per equipment.
4. **Audit Log** — searchable hash-chained log, *Verify chain*, tamper demo.
5. **Governance** — autonomy tiers, override rate, per-class track record, adaptive autonomy.
6. **Noosphere** — Pattern Card schema and sharing protocol (Phase 1, static).
7. **Simulator Control** — scripted demo scenario, replayed step by step.
"""
)
st.caption(f"Run `{run}` · outputs read from data/processed/{run}/ (precomputed by `python -m sanad.cli run`).")
