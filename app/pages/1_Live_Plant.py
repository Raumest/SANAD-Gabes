"""Live Plant: single-line diagram with sensors coloured by trust state + live trends."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import time as dtime, timedelta  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import SERIES, TRUST_COLORS, TRUST_ICON, load_cases, load_stream, plotly_layout, sidebar  # noqa: E402

st.set_page_config(page_title="Live Plant · SANAD", layout="wide")
run = sidebar()
st.title("Live Plant")

stream = load_stream(run)
cases = load_cases(run)
times = stream["timestamp_gw_utc"].drop_duplicates().sort_values().reset_index(drop=True)

# jump to a case or pick a time
pick = st.selectbox("Jump to a case (or use the time slider)", ["—"] + [
    f"{r.case_id} · {r.equipment_id} · {r.verdict}/{r.subtype} · {r.start_utc:%Y-%m-%d %H:%M}" for r in cases.itertuples()])
default_t = times.iloc[len(times) // 2]
if pick != "—":
    default_t = cases.set_index("case_id").loc[pick.split(" · ")[0], "start_utc"] + pd.Timedelta(minutes=10)
day = st.date_input("Day", value=default_t.date(), min_value=times.iloc[0].date(), max_value=times.iloc[-1].date())
tod = st.slider("Time of day (UTC)", value=dtime(default_t.hour, default_t.minute), step=timedelta(minutes=1), format="HH:mm")
t = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=tod.hour, minutes=tod.minute)
t = times.iloc[(times - t).abs().argmin()]
now = stream[stream.timestamp_gw_utc == t].set_index("equipment_id")
st.markdown(f"**Snapshot at {t:%Y-%m-%d %H:%M} UTC**")

# ---- single-line diagram -------------------------------------------------------
SENSORS = {"BUS1": [("meter", "BUS CT"), ("energy", "kWh")],
           "PUMP1": [("meter", "Meter CT"), ("relay", "Relay CT"), ("rtd", "RTD"), ("speed", "Speed"),
                     ("acc_de", "Acc DE"), ("acc_nde", "Acc NDE"), ("energy", "kWh")],
           "COMP1": [("meter", "Meter CT"), ("relay", "Relay CT"), ("rtd", "RTD"), ("speed", "Speed"),
                     ("acc_de", "Acc DE"), ("acc_nde", "Acc NDE"), ("energy", "kWh")],
           "AUX1": [("meter", "Meter CT"), ("energy", "kWh")]}
POS = {"BUS1": (2.0, 3.0), "PUMP1": (0.7, 1.0), "COMP1": (2.0, 1.0), "AUX1": (3.3, 1.0)}
fig = go.Figure()
fig.add_shape(type="line", x0=0.2, x1=3.8, y0=2.2, y1=2.2, line=dict(color="#0b0b0b", width=4))
fig.add_annotation(x=3.85, y=2.2, text="400 V bus", showarrow=False, xanchor="left", font=dict(color="#52514e"))
fig.add_shape(type="line", x0=2.0, x1=2.0, y0=2.2, y1=3.0, line=dict(color="#52514e", width=2))
for eq in ("PUMP1", "COMP1", "AUX1"):
    x, y = POS[eq]
    fig.add_shape(type="line", x0=x, x1=x, y0=1.0, y1=2.2, line=dict(color="#52514e", width=2))
pts = {"x": [], "y": [], "c": [], "t": [], "h": []}
for eq, (x, y) in POS.items():
    p = now.loc[eq] if eq in now.index else None
    kw = "" if p is None or pd.isna(p.get("p_kw")) else f"{p['p_kw']:.1f} kW"
    fig.add_annotation(x=x + (0.45 if eq == "BUS1" else 0), y=y + (0.05 if eq == "BUS1" else -0.35),
                       text=f"<b>{eq}</b><br>{kw}", showarrow=False, font=dict(size=13))
    for i, (grp, name) in enumerate(SENSORS[eq]):
        lab = "" if p is None else str(p.get(f"trust_{grp}", "") or "")
        dx = (i - (len(SENSORS[eq]) - 1) / 2) * 0.13
        pts["x"].append(x + dx)
        pts["y"].append(y + (0.0 if eq == "BUS1" else 0.25))
        pts["c"].append(TRUST_COLORS.get(lab, "#c3c2b7"))
        pts["t"].append(TRUST_ICON.get(lab, "·"))
        pts["h"].append(f"{eq} · {name}: {lab or 'n/a'}")
fig.add_trace(go.Scatter(x=pts["x"], y=pts["y"], mode="markers+text", text=pts["t"], textposition="top center",
                         marker=dict(size=16, color=pts["c"], line=dict(color="#fcfcfb", width=2)),
                         hovertext=pts["h"], hoverinfo="text", showlegend=False))
fig.add_annotation(x=2.0, y=3.5, text="30/0.4 kV · 1 MVA incomer", showarrow=False, font=dict(color="#52514e"))
fig.add_annotation(x=0.05, y=0.45, text="Hover a dot for the sensor name; ✓ VERIFIED · ≈ ESTIMATED · ⚠ SUSPECT · ⛔ QUARANTINED",
                   showarrow=False, xanchor="left", font=dict(color="#898781", size=11))
fig.update_xaxes(visible=False, range=[-0.1, 4.3])
fig.update_yaxes(visible=False, range=[0.3, 3.7])
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", hovermode="closest")
st.plotly_chart(fig, use_container_width=True)
st.markdown(" · ".join(f"<span style='color:{TRUST_COLORS[k]}'>●</span> {TRUST_ICON[k]} {k}"
                       for k in ("VERIFIED", "ESTIMATED", "SUSPECT", "QUARANTINED")), unsafe_allow_html=True)

active = cases[(cases.start_utc <= t) & (cases.end_utc > t)]
if len(active):
    st.warning("Active cases: " + "; ".join(f"{r.case_id} {r.equipment_id} {r.verdict}/{r.subtype}" for r in active.itertuples()))

# ---- trends -----------------------------------------------------------------------
st.subheader("Trends (trusted stream)")
c1, c2, c3 = st.columns([1, 2, 1])
eq = c1.selectbox("Equipment", ["PUMP1", "COMP1", "AUX1", "BUS1"])
opts = [c for c in ["i_l1_a", "i_relay_a", "p_kw", "v_l1_v", "winding_temp_c", "speed_rpm", "vib_de_g", "vib_nde_g",
                    "kwh_register", "freq_hz", "pf"] if stream.loc[stream.equipment_id == eq, c].notna().any()]
sig = c2.multiselect("Signals (same unit on one chart)", opts, default=[o for o in ("i_l1_a", "i_relay_a") if o in opts])
hours = c3.select_slider("Window (h)", [2, 6, 12, 24, 48], value=12)
win = stream[(stream.equipment_id == eq) & (stream.timestamp_gw_utc.between(t - pd.Timedelta(hours=hours / 2), t + pd.Timedelta(hours=hours / 2)))]
if sig:
    tf = go.Figure()
    for i, s in enumerate(sig):
        tf.add_trace(go.Scatter(x=win.timestamp_gw_utc, y=win[s], mode="lines", name=s, line=dict(width=2, color=SERIES[i % 8])))
    for r in cases[(cases.equipment_id == eq) & (cases.end_utc > win.timestamp_gw_utc.min()) & (cases.start_utc < win.timestamp_gw_utc.max())].itertuples():
        tf.add_vrect(x0=r.start_utc, x1=r.end_utc, fillcolor="#fab219", opacity=0.12, line_width=0,
                     annotation_text=f"{r.case_id} {r.subtype}", annotation_position="top left")
    tf.add_vline(x=t, line=dict(color="#898781", dash="dot"))
    st.plotly_chart(plotly_layout(tf, 340), use_container_width=True)
    grp = {"i_l1_a": "meter", "p_kw": "meter", "v_l1_v": "meter", "freq_hz": "meter", "pf": "meter", "i_relay_a": "relay",
           "winding_temp_c": "rtd", "speed_rpm": "speed", "vib_de_g": "acc_de", "vib_nde_g": "acc_nde", "kwh_register": "energy"}
    labs = win[[f"trust_{grp[s]}" for s in sig]].apply(lambda c: c.value_counts()).fillna(0).astype(int)
    st.caption("Provenance labels of the plotted signals in this window (minutes):")
    st.dataframe(labs, use_container_width=True)
