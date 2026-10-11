# Energy Noosphere — protocol (Phase 1: documentation + schema only)

*Fictional process plant in the Gabès industrial zone. All parameters are simulation assumptions. No measurement originates from a real company.*

Nodes share **Pattern Cards**, never raw data. A card is an anonymised relation signature of a diagnosed pattern:
which relations broke and held, normalised magnitude hints, verdict class, recommended governance tier, evidence count
and calibrated accuracy, signed with SHA-256 of its canonical body. Schema and example: `src/sanad/noosphere/pattern_card.py`
(dashboard page *Noosphere*).

| Item | Shared / local | Why |
|---|---|---|
| Relation signature (violated / held) | shared | physics is site-independent |
| Normalised magnitudes (z ranges, % disagreement) | shared | relative values only |
| Verdict class, subtype, calibration summary | shared | lets peers weigh reliability |
| Recommended tier and actions | shared | spreads good governance practice |
| Coarse equipment class | shared | never an asset tag |
| Raw time series, kWh, timestamps | local | commercially sensitive |
| Site, company, node, sensor identity | local | privacy and security |
| Operator names and override reasons | local | personal data; only aggregate rates |
| Thresholds and calibration constants | local | site-specific; re-learned on local normal data |

**Exchange (Phase 2):** draft after ≥ N approved cases → sign → publish → receiver validates schema and signature →
**operator approval is always required to import** (governance: model change) → the card adds a hypothesis to the local
signature matrix with locally re-learned thresholds → per-card track record; poor cards are retired.
