"""Simulator Control: replay the precomputed, deterministic demo scenario step by step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import PROCESSED, SERIES, TIER_ICON, plotly_layout, sidebar  # noqa: E402
from sanad import io  # noqa: E402
from sanad.audit.log import read_entries, verify_chain  # noqa: E402

st.set_page_config(page_title="Simulator Control · SANAD", layout="wide")
sidebar()
st.title("Simulator Control — scripted demo scenario")
SC = PROCESSED / "demo_scenario"
if not (SC / "scenario.json").exists():
    st.info("No scenario found. Build it with `python -m sanad.cli demo` (deterministic, < 1 min).")
    st.stop()


@st.cache_data(show_spinner="Loading scenario…")
def load():
    sc = io.read_json(SC / "scenario.json")
    stream = io.read_csv(SC / "trusted_stream.csv", low_memory=False)
    stream["timestamp_gw_utc"] = pd.to_datetime(stream["timestamp_gw_utc"], utc=True)
    meas = io.read_csv(SC / "measured_window.csv")
    meas["timestamp_gw_utc"] = pd.to_datetime(meas["timestamp_gw_utc"], utc=True)
    cases = io.read_csv(SC / "cases.csv")
    for c in ("start_utc", "end_utc"):
        cases[c] = pd.to_datetime(cases[c], utc=True)
    ctrl = io.read_csv(SC / "controllers.csv")
    ctrl["timestamp_gw_utc"] = pd.to_datetime(ctrl["timestamp_gw_utc"], utc=True)
    return sc, stream, meas, cases, ctrl


sc, stream, meas, cases, ctrl = load()
steps = sc["steps"]
st.markdown(sc["intro"])
if "step" not in st.session_state:
    st.session_state.step = 0
b1, b2, b3 = st.columns([1, 1, 6])
if b1.button("◀ Previous", disabled=st.session_state.step == 0):
    st.session_state.step -= 1
if b2.button("Next ▶", disabled=st.session_state.step >= len(steps) - 1):
    st.session_state.step += 1
i = b3.select_slider("Step", options=list(range(len(steps))), value=st.session_state.step,
                     format_func=lambda k: f"{k + 1}. {steps[k]['title']}")
st.session_state.step = i
s = steps[i]
st.subheader(f"Step {i + 1} — {s['title']}")
st.markdown(s["narrative"])

t0, t1 = pd.Timestamp(s["t0"]), pd.Timestamp(s["t1"])
if s.get("signals"):
    fig = go.Figure()
    k = 0
    for eq, col, source in s["signals"]:
        src = meas if source == "measured" else stream
        d = src[(src.equipment_id == eq) & src.timestamp_gw_utc.between(t0, t1)]
        fig.add_trace(go.Scatter(x=d.timestamp_gw_utc, y=d[col], mode="lines", name=f"{eq} {col} ({source})",
                                 line=dict(width=2, color=SERIES[k % 8])))
        k += 1
    for r in cases[(cases.end_utc > t0) & (cases.start_utc < t1) & cases.equipment_id.isin({e for e, _, _ in s["signals"]} | {"BUS1"})].itertuples():
        fig.add_vrect(x0=max(r.start_utc, t0), x1=min(r.end_utc, t1), fillcolor="#fab219", opacity=0.12, line_width=0,
                      annotation_text=f"{r.case_id}", annotation_position="top left")
    fig.update_yaxes(title=s.get("y_title", ""))
    st.plotly_chart(plotly_layout(fig, 340), use_container_width=True)

sel = cases[cases.case_id.isin(s.get("case_ids", []))]
if len(sel):
    st.markdown("**SANAD verdicts in this step**")
    for r in sel.itertuples():
        head = f"{r.case_id} · {r.equipment_id} · **{r.verdict}**" + ("" if r.verdict == "UNCERTAIN" else f" / {r.subtype}")
        st.markdown(f"{head} · confidence {r.confidence:.2f} · {TIER_ICON[r.tier]}")
        st.info(r.explanation)
        acts = json.loads(r.actions)
        st.caption("Actions: " + "; ".join(f"{a['action']} [{a['tier']}]" for a in acts))
elif s.get("expect_no_case"):
    st.success(s["expect_no_case"])

if s.get("controller"):
    st.markdown("**Side panel — naive vs SANAD-protected load-limit controller** (what-if, offline; SANAD itself never actuates)")
    c = ctrl[ctrl.timestamp_gw_utc.between(t0, t1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.timestamp_gw_utc, y=c.true_total_naive_a, name="True group current — naive controller",
                             line=dict(width=2, color=SERIES[1])))
    fig.add_trace(go.Scatter(x=c.timestamp_gw_utc, y=c.true_total_protected_a, name="True group current — SANAD-protected",
                             line=dict(width=2, color=SERIES[0])))
    fig.add_trace(go.Scatter(x=c.timestamp_gw_utc, y=c.limit_a, name="Group current limit", line=dict(width=2, color="#0b0b0b", dash="dash")))
    fig.update_yaxes(title="A (true, per phase)")
    st.plotly_chart(plotly_layout(fig, 320), use_container_width=True)
    st.markdown(s["controller"])

if s.get("audit"):
    entries = read_entries(SC / "audit_log.jsonl")
    ok, _, msg = verify_chain(entries)
    st.markdown(f"**Verify chain** on the scenario audit log: {'✅' if ok else '❌'} {msg}")
    forged = json.loads(json.dumps(entries))
    k = min(2, len(forged) - 1)
    forged[k]["payload"]["verdict"] = "NORMAL"
    ok2, bad, msg2 = verify_chain(forged)
    st.markdown(f"**After editing entry #{k}** (verdict → NORMAL, in memory): {'✅' if ok2 else '❌'} {msg2}")
