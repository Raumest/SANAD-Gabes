"""Shared helpers for the Streamlit dashboard. Reads precomputed outputs only (fast)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sanad import io  # noqa: E402
from sanad.audit.log import AuditLog, read_entries  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DISCLAIMER = io.DISCLAIMER

# Status palette (reference instance) — always shown together with a text label
TRUST_COLORS = {"VERIFIED": "#0ca30c", "ESTIMATED": "#2a78d6", "SUSPECT": "#fab219", "QUARANTINED": "#d03b3b", "": "#c3c2b7"}
TRUST_ICON = {"VERIFIED": "✔", "ESTIMATED": "≈", "SUSPECT": "⚠", "QUARANTINED": "⛔", "": "·"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TIER_ICON = {"AUTO": "🟢 AUTO", "APPROVAL": "🟠 APPROVAL", "IMMEDIATE": "🔴 IMMEDIATE"}
EQUIPMENT = ["BUS1", "PUMP1", "COMP1", "AUX1"]


def available_runs() -> list[str]:
    runs = [p.name for p in sorted(PROCESSED.iterdir()) if (p / "cases.csv").exists()] if PROCESSED.exists() else []
    return sorted(runs, key=lambda r: (r != "main", r))


def current_run() -> str | None:
    runs = available_runs()
    if not runs:
        return None
    run = st.session_state.get("run_id")
    if run not in runs:
        run = runs[0]
        st.session_state["run_id"] = run
    return run


def sidebar() -> str | None:
    runs = available_runs()
    st.sidebar.caption(DISCLAIMER)
    if not runs:
        st.error("No processed run found. Run `python -m sanad.cli generate` and `python -m sanad.cli run`.")
        st.stop()
    run = current_run()
    choice = st.sidebar.selectbox("Run", runs, index=runs.index(run), help="Processed outputs in data/processed/")
    st.session_state["run_id"] = choice
    return choice


def run_dir(run: str) -> Path:
    return PROCESSED / run


@st.cache_data(show_spinner=False)
def load_cases(run: str) -> pd.DataFrame:
    c = io.read_csv(run_dir(run) / "cases.csv")
    for col in ("start_utc", "end_utc"):
        c[col] = pd.to_datetime(c[col], utc=True)
    for col in ("abstain_reasons", "actions", "rules_fired", "top_factors", "held_relations"):
        c[col] = c[col].map(json.loads)
    return c


@st.cache_data(show_spinner=False)
def load_details(run: str) -> dict:
    out = {}
    with open(run_dir(run) / "case_details.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["case_id"]] = d
    return out


@st.cache_data(show_spinner="Loading trusted stream…")
def load_stream(run: str) -> pd.DataFrame:
    df = io.read_csv(run_dir(run) / "trusted_stream.csv", low_memory=False)
    df["timestamp_gw_utc"] = pd.to_datetime(df["timestamp_gw_utc"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_fingerprint(run: str) -> pd.DataFrame:
    df = io.read_csv(run_dir(run) / "fingerprint.csv")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_json(run: str, name: str) -> dict:
    p = run_dir(run) / name
    return io.read_json(p) if p.exists() else {}


def audit_path(run: str) -> Path:
    return run_dir(run) / "audit_log.jsonl"


def feedback_path(run: str) -> Path:
    return run_dir(run) / "feedback.jsonl"


def record_operator_action(run: str, case: dict, action: str, reason: str, operator: str, label: str = "") -> dict:
    """Append an operator action to the audit chain and to the feedback-label file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {"case_id": case["case_id"], "action": action, "reason": reason, "operator": operator,
               "verdict": case["verdict"], "label": label or case["verdict"]}
    entry = AuditLog(audit_path(run)).append("OPERATOR_ACTION", now, payload)
    with open(feedback_path(run), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts_utc": now, **payload}) + "\n")
    return entry


def read_audit(run: str) -> list[dict]:
    return read_entries(audit_path(run)) if audit_path(run).exists() else []


def trust_badge(label: str) -> str:
    return f"{TRUST_ICON.get(label, '·')} {label or 'n/a'}"


def plotly_layout(fig, height: int = 320):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9", linecolor="#c3c2b7")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e0d9", linecolor="#c3c2b7")
    return fig
