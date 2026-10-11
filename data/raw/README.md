# data/raw — read-only real datasets (Phase 2)

Phase 1 runs entirely on **synthetic** data (`data_source: synthetic`). This folder is
reserved for real datasets that will be wired in later through the adapter stubs in
`src/sanad/adapters/`. Files here are **read-only**: never modify, move or delete them.

| Dataset | Expected path | Resolution | Licence | Adapter |
|---|---|---|---|---|
| UCI Steel Industry Energy Consumption (2018) | `data/raw/steel/Steel_industry_data.csv` | 15 min | CC BY 4.0 | `adapters/steel.py` |
| PVGIS hourly, Gabès 2023 | `data/raw/pvgis/PVGIS_Gabes_2023.csv` | 1 h, UTC | PVGIS terms (EU JRC) | `adapters/pvgis.py` |
| Paderborn bearing dataset (K003, KA01, KA04, KI01) | `data/raw/paderborn/<set>/*.mat` | kHz | CC BY-NC 4.0 | `adapters/paderborn.py` |

`data/raw/paderborn/` and `*.mat` are git-ignored and must never be committed.
Only relative statistics (RMS, kurtosis, crest factor, healthy-vs-damaged ratios) will
ever be derived from Paderborn; raw kHz signals are never merged with minute data.
