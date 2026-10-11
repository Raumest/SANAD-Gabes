"""Trust Fingerprint: heatmap of R1-R10 residuals (|z|, 5-min max) over time per equipment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import load_cases, load_fingerprint, sidebar  # noqa: E402
from sanad.explain.templates import REL_NAMES  # noqa: E402
from sanad.physics.relations import APPLIES  # noqa: E402

st.set_page_config(page_title="Trust Fingerprint · SANAD", layout="wide")
run = sidebar()
st.title("Trust Fingerprint")
st.markdown("Each row is a physics relation; colour = normalised residual |z| (5-min max, capped at 30). "
            "A **genuine event** lights few rows while independent paths keep agreeing; a **sensor fault** lights the "
            "rows that involve the faulty path only.")

fp = load_fingerprint(run)
cases = load_cases(run)
c1, c2, c3 = st.columns([1, 1, 2])
eq = c1.selectbox("Equipment", ["PUMP1", "COMP1", "AUX1", "BUS1"])
pick = c2.selectbox("Centre on case", ["—"] + cases[cases.equipment_id == eq].case_id.tolist())
t0, t1 = fp.timestamp_utc.min(), fp.timestamp_utc.max()
centre = cases.set_index("case_id").loc[pick, "start_utc"] if pick != "—" else t0 + (t1 - t0) / 2
days = c3.select_slider("Window (days)", [0.5, 1, 2, 4, 7, 14], value=2)
a, b = centre - pd.Timedelta(days=days / 2), centre + pd.Timedelta(days=days / 2)
d = fp[(fp.equipment_id == eq) & fp.timestamp_utc.between(a, b)]
rels = APPLIES[eq]
z = d[rels].T.to_numpy()
blues = [[0, "#fcfcfb"], [0.1, "#cde2fb"], [0.2, "#9ec5f4"], [0.35, "#6da7ec"], [0.5, "#3987e5"], [0.7, "#256abf"], [1, "#0d366b"]]
fig = go.Figure(go.Heatmap(z=z, x=d.timestamp_utc, y=[f"{r} {REL_NAMES[r]}" for r in rels], colorscale=blues, zmin=0, zmax=15,
                           colorbar=dict(title="|z|"), hovertemplate="%{y}<br>%{x}<br>|z| = %{z}<extra></extra>",
                           xgap=0, ygap=2))
for r in cases[(cases.equipment_id.isin([eq, "BUS1"])) & (cases.end_utc > a) & (cases.start_utc < b)].itertuples():
    if r.equipment_id == "BUS1" and r.kind != "grid" and eq != "BUS1":
        continue
    fig.add_vrect(x0=r.start_utc, x1=r.end_utc, line=dict(color="#d03b3b", width=1), fillcolor="rgba(0,0,0,0)",
                  annotation_text=f"{r.case_id} {r.verdict if r.verdict == 'UNCERTAIN' else r.subtype}",
                  annotation_position="top left", annotation_font_size=10)
fig.update_layout(height=120 + 34 * len(rels), margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
st.caption("Red outlines = SANAD cases. Thresholds: |z| > 6 per minute, or EWMA/CUSUM on z beyond limits learned on the normal-only block.")
with st.expander("Table view"):
    st.dataframe(d[["timestamp_utc"] + rels + ["alarm"]], hide_index=True, use_container_width=True)
