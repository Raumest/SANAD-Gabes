"""Audit Log: searchable, Verify chain, tamper demo (on an in-memory copy)."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import audit_path, read_audit, sidebar  # noqa: E402
from sanad.audit.log import verify_chain  # noqa: E402

st.set_page_config(page_title="Audit Log · SANAD", layout="wide")
run = sidebar()
st.title("Audit Log")
st.markdown("Append-only JSONL; each entry stores `prev_hash` and `hash` = SHA-256 of its canonical JSON. "
            f"File: `{audit_path(run).relative_to(audit_path(run).parents[3])}`")

entries = read_audit(run)
rows = [{"seq": e["seq"], "ts_utc": e["ts_utc"], "kind": e["kind"], "case_id": e["payload"].get("case_id", ""),
         "summary": (e["payload"].get("verdict", "") + " " + e["payload"].get("subtype", "") + " " +
                     e["payload"].get("action", "")).strip(),
         "hash": e["hash"][:16] + "…", "prev_hash": e["prev_hash"][:16] + "…"} for e in entries]
df = pd.DataFrame(rows)
qtext = st.text_input("Search (case id, verdict, kind, action …)")
if qtext:
    mask = df.apply(lambda r: qtext.lower() in " ".join(map(str, r.values)).lower(), axis=1)
    df = df[mask]
st.dataframe(df, hide_index=True, use_container_width=True, height=320)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Verify chain")
    if st.button("🔐 Verify chain"):
        ok, bad, msg = verify_chain(audit_path(run))
        (st.success if ok else st.error)(msg)
    sel = st.number_input("Show entry #", 0, max(0, len(entries) - 1), 0)
    if entries:
        st.json(entries[int(sel)])
with c2:
    st.subheader("Tamper demo")
    st.markdown("Edits one entry **in a copy held in memory** (the file is untouched), then re-verifies.")
    if entries:
        k = st.number_input("Entry to tamper with", 1 if len(entries) > 1 else 0, len(entries) - 1, min(1, len(entries) - 1))
        new_verdict = st.text_input("Replace verdict/action with", "NORMAL")
        if st.button("✏️ Edit entry and verify"):
            forged = copy.deepcopy(entries)
            p = forged[int(k)]["payload"]
            field = "verdict" if "verdict" in p else ("action" if "action" in p else "run_id")
            old = p.get(field)
            p[field] = new_verdict
            ok, bad, msg = verify_chain(forged)
            st.write(f"Changed `{field}` of entry #{k}: `{old}` → `{new_verdict}`")
            (st.success if ok else st.error)(msg)
            if not ok:
                st.caption("The stored hash no longer matches the content, and every later entry points to the old hash — "
                           "the edit is detectable.")
