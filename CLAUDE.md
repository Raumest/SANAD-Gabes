# SANAD-Gabès — Project Spec for Claude Code

> SANAD = Sensor-Assured Network for Autonomous Decisions. Slogan: "Before the grid acts, SANAD verifies."
> TSYP14 EnergyMind Tunisia challenge (IEEE PES / IES). Phase 1 deadline: 15/10/2026. Language of code, docs and UI: English.

## 1. What we are building

An explainable AI **trust layer** for a **fictional** industrial electrical line in the Gabès industrial zone, Tunisia.
For every anomaly it must answer: **is the plant misbehaving, or is the measurement lying?**
It outputs a verdict, a calibrated confidence (or "UNCERTAIN"), a plain-English explanation, a governed action, and a hash-chained audit entry.
It also outputs a **trusted data stream** with provenance labels: VERIFIED / ESTIMATED / SUSPECT / QUARANTINED.

The four challenge goals map to modules:
1. Detect + distinguish genuine events from data-quality problems -> `physics/`, `detect/`, `diagnose/`
2. Confidence + explanation + audit log -> `confidence/`, `explain/`, `audit/`
3. Human-AI governance (auto vs approval, over/under-trust) -> `governance/`, dashboard
4. Multi-node sharing ("Energy Noosphere") -> `noosphere/` (Phase 1: documented protocol + Pattern Card schema only)

## 2. Golden rules (never break these)

1. **Physical events modify the TRUE plant state** and propagate to every dependent variable. **Sensor/data/comm faults modify only the MEASURED channel**, after the true state is computed.
2. **Ground truth never leaks into model inputs.** `measurements.csv` has no labels, no clean values, no injection parameters. Those live in `truth_clean.csv`, `events.csv`, `generation_config.yaml`.
3. **A feeder meter computes V, I, P, Q, S, PF and kWh from its own CT.** So a CT gain drift does NOT break P = √3·V·I·PF inside that meter. CT drift is only visible through **independent paths**: relay CT, bus meter balance, thermal model, speed.
4. **SANAD never actuates.** It is an advisory and data-gating layer. It never trips breakers or sheds load. Protection relays are out of scope.
5. **No LLM makes any decision.** Explanations come from templates filled with rule/feature evidence.
6. **Every verdict is traceable** to named relations/rules and features. No black-box-only decision.
7. **Reproducible:** every random process is seeded from the config. Same seed -> identical output.
8. **No invented results.** Docs and README only quote numbers produced by the code. Example numbers must be labelled "illustrative".
9. **`data/raw/` is read-only.** Never modify, move or delete raw files. Never commit `data/raw/paderborn/` or `*.mat`.
10. **Keep it simple.** Prefer rules and statistics; use ML only where physics cannot decide. No Kafka, Kubernetes, blockchain, deep learning for Phase 1.

## 3. Data mode

The config has `data_source: synthetic | real`.

- **`synthetic` (build this first, default):** fully parametric.
  - Load envelope: shift-based industrial profile (night low, morning ramp, weekday high, weekend low), states LIGHT / MEDIUM / MAXIMUM, with AR(1) noise.
  - Climate: Gabès-like ambient temperature = annual sinusoid + daily sinusoid + noise (summer afternoons 38–45 °C, winter nights 6–10 °C). Irradiance: clear-sky-like bell curve.
  - Vibration health: parametric model with documented assumptions.
- **`real` (later — create clean adapter interfaces + TODO stubs now, do not implement fully):**
  - `adapters/steel.py`: UCI Steel Industry Energy Consumption, `data/raw/steel/Steel_industry_data.csv`, 15-min, 2018, CC BY 4.0. Provides kWh, lagging/leading kVarh, PF, load type, week status.
  - `adapters/pvgis.py`: `data/raw/pvgis/PVGIS_Gabes_2023.csv`, hourly, UTC, header metadata + footer legend lines; columns time, Gb(i), Gd(i), Gr(i), H_sun, T2m, WS10m, Int.
  - `adapters/paderborn.py`: `data/raw/paderborn/{K003,KA01,KA04,KI01}/*.mat`. Only relative statistics (RMS, kurtosis, crest factor, healthy-vs-damaged ratios) are ever used. Raw kHz signals are never merged with minute data. CC BY-NC 4.0.
  - Both modes must produce the exact same output schema so the rest of the platform does not care which mode was used.

Every generated file carries the disclaimer: *"Fictional process plant in the Gabès industrial zone. All parameters are simulation assumptions. No measurement originates from a real company."*

## 4. The plant (fixed for Phase 1)

400 V L-L / 230 V L-N, 50 Hz, fed by a 1 MVA 30/0.4 kV transformer.

| ID | Type | Rating (assumption) | Measurement paths |
|---|---|---|---|
| BUS1 | Incomer meter | 1 MVA | Own CT set: V×3, I×3, P, Q, S, PF, f, THD, kWh |
| PUMP1 | Seawater/process pump, induction motor, 4-pole | 90 kW | Feeder meter (CT A) + motor-relay current (CT B, independent) + winding RTD + DE & NDE accelerometers + speed |
| COMP1 | Process compressor, induction motor, 4-pole, load/unload cycling | 132 kW | Same as PUMP1 |
| AUX1 | Lighting, HVAC, small drives | ~60 kW | Feeder meter only |

PV is Phase 2. Rated currents, efficiency and PF curves are computed in code from the ratings and stored in `equipment.csv`.

## 5. Output files (`data/generated/<run_id>/`)

- `measurements.csv` — one row per (timestamp, equipment). Model input only.
- `truth_clean.csv` — uncorrupted signals, same keys. Evaluation/imputation scoring only.
- `events.csv` — ground truth intervals.
- `equipment.csv` — ratings, applicability mask, thermal τ, sensor classes.
- `sensors.csv` — sensor_id, equipment_id, path (METER_CT / RELAY_CT / RTD / ACC_DE / ACC_NDE / SPEED / BUS_CT), accuracy class, install & calibration dates.
- `generation_config.yaml` — frozen copy of the config incl. seeds.

### measurements.csv columns
`timestamp_gw_utc, timestamp_dev_utc, node_id, equipment_id, v_l1_v, v_l2_v, v_l3_v, i_l1_a, i_l2_a, i_l3_a, i_relay_a, p_kw, q_kvar, s_kva, pf, freq_hz, kwh_register, thd_v_pct, thd_i_pct, speed_rpm, winding_temp_c, vib_de_g, vib_nde_g, ambient_temp_c, op_state, run_cmd, comm_status, maint_mode`

- Non-applicable columns are empty (declared in `equipment.csv`). Truly missing values are also empty.
- Inject a small natural baseline of missing values, spikes and timestamp jitter into normal data too, so "NaN = anomaly" cannot be learned.
- `op_state`: OFF / STARTING / RUN_LOW / RUN_NORMAL / RUN_HIGH / STOPPING. `comm_status`: OK / DELAYED / LOST. `maint_mode`: NONE / PLANNED_SHUTDOWN / CALIBRATION / INSPECTION.

### events.csv columns
`event_id, main_class, subtype, origin (PHYSICAL|MEASUREMENT|COMMUNICATION|CYBER_FALSIFICATION), severity (S1|S2|S3), equipment_id, sensor_id, start_utc, end_utc, gt_detectability (HIGH|MEDIUM|LOW|NONE), params_json, compound_with`

## 6. Simulator models (all parameters in config)

Base resolution 1 min. Optional 1 s bursts around fast events (sag, interruption, start) — Phase 1 may keep everything at 1 min if time is short; document it.

1. Load fraction L(t) per machine = envelope × share × (1 + AR(1) noise), clipped to [0, 1.4].
2. P_in = L·P_rated / η(L); PF(L) rises with load (≈0.55 at 20% to ≈0.87 at 100%); Q = P·tan(acos PF); S = √(P²+Q²).
3. Per-phase current from P, V, PF with 0.5–1.5% baseline unbalance; balanced case I = P/(√3·V_LL·PF).
4. Speed n = (120·f/p)·(1 − s_rated·L), p = 4.
5. Thermal: T_ss = T_amb + ΔT_rated·(I/I_rated)²; T[k+1] = T[k] + (Δt/τ)(T_ss − T[k]); τ = 20–40 min, ΔT_rated = 60–80 K.
6. Vibration: v = v0 + k_L·L, × health multiplier h(t) ≥ 1, + noise; DE and NDE with separate noise.
7. kWh register integrates the meter's own measured P.
8. Bus: P_bus = ΣP_feeders + losses (1–2%) + small unmetered noise.
9. Start-up: inrush ≈ 5–7× I_rated decaying over a few seconds/minutes; this is NORMAL, not an anomaly.
10. Independent noise realization for every measurement path.

Pipeline: `true_state -> physical event library -> sensor path models (gain/offset/noise) -> data-quality & comm injectors -> write files -> relation unit tests on clean data`.

## 7. Phase 1 injection library

| Label | Kind | Injection (configurable) |
|---|---|---|
| NORMAL_START | normal transient | motor start inrush |
| PHYS_OVERLOAD | physical | L = 1.05–1.4 for 5–60 min |
| PHYS_BEARING_DEGRADATION | physical | h: 1 -> 3 over days (parametric, time-compressed, documented) |
| GRID_SAG | grid | V ×(0.4–0.9) on all meters, seconds–1 min |
| GRID_INTERRUPTION | grid | V≈0 on all meters, speed coast-down, kWh stops |
| SENS_DRIFT | sensor | meter CT gain 1 + r·(t−t0), r = 0.02–0.5 %/day |
| SENS_BIAS | bias | constant offset (V +2–8 V, T +3–10 K, I +0.5–3% rated) |
| SENS_STUCK | sensor | hold last value 10 min–6 h |
| DQ_MISSING | data quality | MCAR 0.1–5% + bursts 1–60 min |
| FALS_ENERGY | falsification | kWh under-reporting 10–40% for 1–14 days |
| COMM_DROPOUT | comm | rows missing 30 s–2 h, comm_status LOST, register catches up after |

Randomize onset times and parameters (no round timestamps). Schedule 1–2 compound events (e.g. real overload + CT drift) for the test block only.

## 8. Relations (the "Trust Fingerprint")

| ID | Relation | Independent paths? | Catches |
|---|---|---|---|
| R1 | S² ≈ P² + Q² within a meter | No | corruption, impossible values |
| R2 | P ≈ √3·V_LL·I·PF within a meter | No | corruption, unit errors (NOT CT drift) |
| R3 | I_meter ≈ I_relay | Yes | CT drift, bias, stuck on either current path |
| R4 | P_bus ≈ ΣP_feeders + losses | Yes | which feeder meter is wrong; falsification |
| R5 | ΔkWh ≈ ∫P dt (same meter) and ΔkWh_bus ≈ ΣΔkWh_feeders | Partly | register tampering, counter faults |
| R6 | winding temp ≈ thermal model driven by relay current & ambient | Yes | real overload vs current/temperature sensor faults |
| R7 | speed ≈ 120·f·(1−s(L))/p | Yes | speed/frequency faults, real frequency events |
| R8 | voltage agreement across meters and phases | Yes | voltage bias vs real sag |
| R9 | frequency agreement across meters | Yes | time coherence, replay |
| R10 | DE ≈ NDE vibration and within load envelope | Yes | mechanical fault vs single-sensor fault |

Plus: validity checks (range, rate of change, flatline, NaN, timestamps), temporal plausibility (temperature cannot move faster than τ allows), and state context (op_state, maint_mode suppress false alarms).

Residuals are normalized; EWMA and CUSUM run on residuals to catch slow drift.

## 9. Detection and diagnosis

- **Stage A — Detect:** validity checks + relation residual thresholds + EWMA/CUSUM + Isolation Forest on residual and state features (trained on the normal-only block) for unknown patterns.
- **Stage B — Diagnose:** signature matrix mapping the pattern of violated / held relations to hypotheses. **2-of-3 voting** on current: meter CT vs relay CT vs bus-implied current; the odd one out is the faulty path. If only one path is healthy, output UNCERTAIN.
- Genuine events: physics holds, many sensors move coherently. Sensor faults: one path disagrees with independent ones. Falsification: R4/R5/R9 break while sensor-health cues look fine.
- **Stage C — Confidence:** score per hypothesis from evidence strength; isotonic calibration on the validation block if time allows. **Abstain -> UNCERTAIN** when top confidence < 0.60, margin between top two < 0.15, input out-of-distribution (IF score beyond training range), or too few healthy redundant paths.
- Output taxonomy: NORMAL, PHYSICAL_EQUIPMENT_FAULT, GRID_EVENT, SENSOR_FAULT, BIAS_MISCALIBRATION, CORRUPTION_FALSIFICATION, COMMUNICATION_PROBLEM, UNKNOWN (+ subtype).

## 10. Governance

Autonomy follows **reversibility and impact**, not confidence alone.

| Situation | Automatic | Needs operator approval | Immediate human |
|---|---|---|---|
| Short data gap | impute, label ESTIMATED | | |
| Sensor drift/bias, conf ≥ 0.85 | quarantine + substitute estimate + open ticket | recalibration commit | |
| Sensor drift, conf 0.60–0.85 | label SUSPECT only | quarantine | |
| Suspected falsification | hold data as SUSPECT | any conclusion (AI never accuses) | |
| Genuine overload / grid event | notify, no data correction | | if sustained or beyond thermal limit |
| Bearing degradation | maintenance advisory | shutdown scheduling | |
| UNCERTAIN | log + queue | human decides | |
| Threshold / model change, importing a Pattern Card | | always | |
| ≥2 critical events or multiple sensors lost | | | always |

Operator actions: Approve / Override (reason mandatory) / Defer / Annotate. Every action is logged and goes to a feedback-label file.
Trust controls: override-rate monitor, forced second-look sampling of auto-resolved cases, per-class track record, **adaptive autonomy** (if override rate or calibration error crosses a limit, tighten one autonomy tier).

## 11. Explanations and audit

- Template explanation per verdict, citing relation residuals and values, e.g. *"PUMP1 meter current reads 8% below the relay current and the bus balance; voltage and power checks hold, so the motor is likely fine and the meter CT is drifting."*
- Each alert: event type, confidence, runner-up hypothesis, top factors (relations + features), rules fired, recommended action, action tier.
- Audit log: append-only JSONL, each entry has `prev_hash` and `hash` (SHA-256 of canonical JSON). Provide `verify_chain()` and a tamper test.

## 12. Evaluation

Time-block split with ≥24 h embargo: weeks 1–2 normal only, 3–4 train, 5–6 validation, 7–8 test (8 weeks total; config-driven).
Metrics: event-level precision/recall/F1 per class, false positives per day, detection delay, confusion matrix, abstention rate, p50/p95 latency per decision cycle. Results written to `reports/metrics.json` and `reports/metrics.md`.

## 13. Dashboard (Streamlit + Plotly)

Pages:
1. **Live Plant** — single-line diagram (BUS1 -> PUMP1, COMP1, AUX1) with each sensor colored by trust state; live trends.
2. **Alert Queue & Case Card** — verdict, confidence gauge, runner-up, top factors, explanation, action; Approve / Override (reason) / Defer / Annotate.
3. **Trust Fingerprint** — heatmap of R1–R10 residuals over time per equipment.
4. **Audit Log** — searchable; "Verify chain" button; tamper demo.
5. **Governance** — autonomy tier, override rate, per-class track record.
6. **Noosphere** — Phase 1: Pattern Card schema + shared/local/why table (static).
7. **Simulator Control** — pick a scripted scenario, inject, replay step by step.

Must run with: `streamlit run app/Home.py`. Works on the synthetic dataset with zero downloads.

### Demo scenario (scripted, ~3 min)
1. Normal operation + COMP1 motor start -> not flagged.
2. Real overload on COMP1 -> fingerprint clean, GENUINE overload, operator alerted.
3. CT drift on PUMP1 -> R3/R4/R6 light up, SENSOR_DRIFT, sensor QUARANTINED, estimate substituted. Side panel: a **naive load-limit controller** using raw PUMP1 current lets COMP1 ramp up wrongly; the **SANAD-protected** controller does not.
4. kWh falsification on AUX1 -> UNCERTAIN, escalated, operator overrides with reason.
5. Audit -> Verify chain -> edit one entry -> chain breaks.

## 14. Repo layout (target)

```
SANAD-Gabes/
  CLAUDE.md  README.md  pyproject.toml  requirements.txt  .gitignore
  configs/default.yaml
  data/raw/ (read-only)  data/generated/  data/processed/
  src/sanad/
    config.py  cli.py
    sim/        (load, climate, electrical, thermal, vibration, plant, generator)
    inject/     (physical, sensor, data_quality, comm, falsification, scheduler)
    adapters/   (steel.py, pvgis.py, paderborn.py — interfaces + TODO stubs)
    physics/    (relations.py, residuals.py)
    detect/     (validity.py, statistical.py, iforest.py)
    diagnose/   (signatures.py, voting.py)
    confidence/ (scoring.py, calibration.py, abstain.py)
    governance/ (policy.py, router.py, trust.py)
    explain/    (templates.py)
    audit/      (log.py)
    pipeline.py (stream -> detect -> diagnose -> confidence -> route -> explain -> log)
    evaluate/   (metrics.py, report.py)
    noosphere/  (pattern_card.py — schema only)
    demo/       (scenarios.py, controllers.py)
  app/ (Home.py + pages/)
  tests/
  docs/ (architecture.md, data_card.md, injection_table.md, governance.md, noosphere_protocol.md, compliance_matrix.md)
  reports/
```

CLI: `python -m sanad.cli generate`, `python -m sanad.cli run`, `python -m sanad.cli evaluate`, `python -m sanad.cli demo`.

## 15. Out of scope for now

Autoencoder, SHAP, One-Class SVM baseline, PV, multi-node simulation, real-data integration (adapters are stubs), Docker, auth. These are Phase 2.

## 16. Working rules for Claude Code

- Plan first, then build milestone by milestone. After each milestone: run `pytest`, fix failures, `git commit` with a clear message, give a 3-line summary.
- Small, readable modules with docstrings and type hints. No hard-coded absolute paths.
- Keep the default synthetic dataset small enough to generate in < 1 min and keep in git (Phase 1 requires the dataset in the repo).
- When an assumption is needed, put it in `configs/default.yaml` and document it in `docs/`.
- Do not claim performance that was not measured.
