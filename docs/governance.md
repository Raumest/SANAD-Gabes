# Governance

*Fictional process plant in the Gabès industrial zone. All parameters are simulation assumptions. No measurement originates from a real company.*

SANAD is an **advisory and data-gating** layer: it labels data (VERIFIED / ESTIMATED / SUSPECT / QUARANTINED),
substitutes estimates, opens tickets and routes decisions to people. It never trips breakers or sheds load.
Autonomy follows **reversibility and impact**, not confidence alone. Code: `src/sanad/governance/`.

| Situation | Automatic | Needs operator approval | Immediate human |
|---|---|---|---|
| Short data gap (≤ 15 min) | impute, label ESTIMATED | | |
| Sensor drift/bias, conf ≥ 0.85 | quarantine + substitute estimate + open ticket | recalibration commit | |
| Sensor drift/bias, conf 0.60-0.85 | label SUSPECT only | quarantine | |
| Suspected falsification | hold data as SUSPECT | any conclusion (AI never accuses) | |
| Genuine overload / grid event | notify, no data correction | | if sustained (≥ 30 min), ≥ 1.25 × rated, or interruption ≥ 5 min |
| Bearing degradation | maintenance advisory | shutdown scheduling | |
| UNCERTAIN | log + queue | human decides | |
| Threshold / model change, importing a Pattern Card | | always | |
| ≥ 2 concurrent critical (S3) events or comm lost on ≥ 2 equipment | | | always |

The case **tier** is the highest tier among its actions.

## Provenance labels (trusted stream)
* **VERIFIED** — raw value present and not implicated by any case.
* **ESTIMATED** — value imputed (gap ≤ 15 min, linear) or substituted from independent paths while the sensor is
  quarantined (meter current ← relay CT × learned ratio; relay ← meter; RTD ← thermal model; speed ← slip model;
  DE vibration ← NDE × learned ratio).
* **SUSPECT** — raw value kept but flagged (conf 0.60-0.85, falsification hold, UNCERTAIN on a measurement hypothesis).
* **QUARANTINED** — raw value withheld and no estimate available (value empty).

## Operator actions and trust controls
* Dashboard actions **Approve / Override (reason + corrected verdict mandatory) / Defer / Annotate (note mandatory)**
  are appended to the hash-chained audit log and to `feedback.jsonl` (the feedback-label file).
* **Override-rate monitor** and **per-class track record** (Governance page).
* **Forced second look:** a deterministic 5 % sample of AUTO cases is flagged for review.
* **Adaptive autonomy:** if a class's override rate exceeds 25 % (≥ 5 reviewed cases), or the calibration error exceeds
  0.15, that class is tightened by one tier (AUTO → APPROVAL) on the next `cli run`.
