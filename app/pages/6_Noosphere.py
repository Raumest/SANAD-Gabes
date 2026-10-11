"""Energy Noosphere (Phase 1): Pattern Card schema and what is shared vs kept local."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import sidebar  # noqa: E402
from sanad.noosphere.pattern_card import JSON_SCHEMA, SHARING_TABLE, example_card  # noqa: E402

st.set_page_config(page_title="Noosphere · SANAD", layout="wide")
sidebar()
st.title("Energy Noosphere — multi-node sharing (Phase 1)")
st.markdown("Nodes share **Pattern Cards** — anonymised relation signatures of diagnosed patterns — never raw data. "
            "Importing a card is a model change: it **always requires operator approval**. "
            "Phase 1 delivers the schema and protocol only; no network exchange is implemented.")
st.subheader("Shared vs local")
st.dataframe(pd.DataFrame(SHARING_TABLE, columns=["Item", "Shared / local", "Why"]), hide_index=True, use_container_width=True)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Pattern Card JSON schema")
    st.json(JSON_SCHEMA)
with c2:
    st.subheader("Example card (illustrative values)")
    st.code(example_card().to_json(), language="json")
st.subheader("Protocol (Phase 2)")
st.markdown("""
1. A node accumulates ≥ N operator-approved cases of one pattern and drafts a card (relation signature + magnitudes).
2. The card is signed (SHA-256 of its canonical body) and published to peers.
3. A receiving node validates the schema, checks the signature, and queues the card for **operator approval**.
4. Once approved, the card adds a hypothesis to the local signature matrix; its thresholds are re-learned on local normal data.
5. Per-card track record (overrides on matched cases) is fed back; cards with poor records are retired.
""")
