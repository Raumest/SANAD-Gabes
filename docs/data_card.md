# Data card — synthetic SANAD-Gabès dataset

*Fictional process plant in the Gabès industrial zone. All parameters are simulation assumptions. No measurement originates from a real company.*

| Item | Value |
|---|---|
| Generator | `python -m sanad.cli generate [--config ...]` (deterministic, seed in config) |
| Default run | `configs/default.yaml` → `data/generated/main/` (8 weeks, ~15 s, committed as gzipped CSV, ~26 MB) |
| Committed run | `configs/demo.yaml` → `data/generated/demo/` (2 weeks, gzipped CSV) |
| Resolution | 1 min for everything (Phase 1). No 1-s bursts |
| Plant | 400 V / 50 Hz bus fed by a 1 MVA 30/0.4 kV transformer; BUS1 incomer, PUMP1 90 kW, COMP1 132 kW, AUX1 ~60 kW |
| Rows | one per (minute, equipment): 4 × 80 640 for 8 weeks |

## Files
`measurements.csv` (model input only), `truth_clean.csv` (uncorrupted true signals + load fraction, bearing health,
irradiance — evaluation only), `events.csv` (ground truth), `equipment.csv` (ratings computed from the motor model,
applicability mask, τ, ΔT), `sensors.csv` (paths, classes, dates), `generation_config.yaml` (frozen config + seed +
per-path static sensor errors, which are ground truth and never model inputs).

## Units and conventions
`v_l*_v` are phase-to-neutral volts (≈ 231 V). Currents in A, power in kW/kvar/kVA, temperature °C, vibration g RMS,
speed rpm, frequency Hz. Empty cell = not applicable (see `equipment.csv`) or truly missing. Every file starts with a
`# disclaimer` line (`pandas.read_csv(..., comment="#")`).

## Simulation assumptions (all in `configs/default.yaml`)
* **Load:** shift profile (night 0.45, morning ramp, day ~0.9, evening 0.7), weekend × 0.6, per-machine share,
  multiplicative AR(1) noise. PUMP1 runs continuously with one nightly process pause (a natural restart);
  COMP1 runs weekdays ~05:00-23:00 with load/unload cycling (loaded 8-25 min, unloaded at 25 % for 3-10 min);
  AUX1 always on with an HVAC term following ambient temperature. One planned shutdown per motor per 2 weeks.
* **Motor model:** P_in = L·P_r + P_loss,r·(0.3 + 0.7·L²); PF(L) = 0.88 − 0.33·e^(−4.5(L−0.2)) (0.55 at 20 %, 0.87 at 100 %);
  rated currents computed from these (PUMP1 ≈ 158 A, COMP1 ≈ 230 A, AUX1 ≈ 94 A). Per-phase unbalance 0.4-1.2 %.
* **Speed:** n = 120 f/p · (1 − s_r · L / v²), 4 poles. **Thermal:** first order, τ 30/35 min, ΔT_rated 70/75 K.
* **Vibration:** (v0 + k_L L) × h(t) on DE; NDE sees 60 % of a DE fault; background 0.004 g when stopped.
* **Start-up:** inrush 6-6.5 × I_rated decaying over ~6-7 s appears as a **1-min average** (≈1.5-2 × running current)
  with `op_state = STARTING`. This is normal behaviour, not an anomaly.
* **Grid:** voltage with a ±0.8 % daily swing + AR(1); frequency 50 Hz + AR(1) (σ ≈ 0.01 Hz).
* **Bus:** P_bus = 1.015 ΣP_feeders + 2 kW unmetered (+ AR noise). kWh registers integrate each meter's own P.
* **Climate:** Gabès-like ambient = annual + daily sinusoid + AR(1) (June afternoons ≈ 35-39 °C with defaults);
  clear-sky irradiance bell curve (truth file only; PV is Phase 2).
* **Sensors:** static gain errors per path (meter 0.15 %, relay 0.4 %, accelerometer 2 %, RTD offset 0.3 K),
  independent noise per path, rounding per register.
* **Natural imperfections in all data:** 0.05 % MCAR missing values, 0.02 % single-minute spikes (±15-50 %),
  device-clock jitter (σ 0.6 s) and 0.2 % DELAYED rows — so "NaN = anomaly" cannot be learned.

## Time compression (documented deviations)
* **Bearing degradation** is compressed to **3-3.6 days** (real wear takes weeks to months).
* **CT drift** uses 1-3 %/day over 0.75-1.5 days (spec range 0.02-0.5 %/day) so that drifts become visible within an
  8-week horizon; slow drifts are labelled LOW detectability by `gt_detectability`.
* **FALS_ENERGY** lasts 1-2 days (spec 1-14 days) so that ≥ 6 events fit the 2-week test block.
* **GRID_SAG** at 1-min resolution is a **sustained under-voltage (≥ 1 min)**, not a sub-second sag.
* **COMM_DROPOUT** of 30 s rounds up to one 1-min row; the gateway keeps placeholder rows with empty values and
  `comm_status = LOST`; the kWh register catches up afterwards.

## Splits
Weeks 1-2 normal only (no injected events) · 3-4 train · 5-6 validation · 7-8 test, each later block starting with a
24 h embargo (no events placed there, no cases scored). Event counts per label and block are reported in
`reports/metrics.md`.
