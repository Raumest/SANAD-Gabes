# Injection table (Phase 1)

*Fictional process plant in the Gabès industrial zone. All parameters are simulation assumptions. No measurement originates from a real company.*

Parameters are drawn uniformly from the ranges in `configs/default.yaml → injection`; onsets are random (never on a
quarter hour). Scheduling: `src/sanad/inject/scheduler.py` (3 per label in train, 3 in validation, 6 in test, plus one
compound pair in test: a real COMP1/PUMP1 overload inside a meter-CT drift on the same motor).

| Label | Output class / subtype | Acts on | Injection | Targets |
|---|---|---|---|---|
| NORMAL_START | NORMAL / START | true state | natural motor starts (inrush as 1-min average) | PUMP1, COMP1 |
| PHYS_OVERLOAD | PHYSICAL_EQUIPMENT_FAULT / OVERLOAD | true state | L → 1.10-1.40 for 10-60 min (3-min ramp) | PUMP1, COMP1 |
| PHYS_BEARING_DEGRADATION | PHYSICAL_EQUIPMENT_FAULT / BEARING_DEGRADATION | true state | h: 1 → 2.2-3.0 quadratically over 3-3.6 days (time-compressed) | PUMP1, COMP1 |
| GRID_SAG | GRID_EVENT / SUSTAINED_UNDERVOLTAGE | true state | V × 0.40-0.88 on all meters for 1-3 min | plant |
| GRID_INTERRUPTION | GRID_EVENT / INTERRUPTION | true state | V = 0, speed coast-down, kWh stops, restart after, 2-12 min | plant |
| SENS_DRIFT | SENSOR_FAULT / DRIFT | meter CT (raw) | gain 1 ± r·t, r = 1-3 %/day, 0.75-1.5 days; one drift at a time plant-wide | PUMP1, COMP1 |
| SENS_BIAS | BIAS_MISCALIBRATION / BIAS | raw signal | V ± 3-8 V (any meter) · winding T ± 4-10 K · I ± 1.5-3 % rated on meter or relay CT; 3-12 h | all |
| SENS_STUCK | SENSOR_FAULT / STUCK | measured | hold last value 20-360 min: whole meter, RTD, speed or DE accelerometer | PUMP1, COMP1 |
| DQ_MISSING | COMMUNICATION_PROBLEM / DATA_GAP | measured | MCAR 1-5 % per value over 2-6 h + 1-3 bursts of 5-60 min (link OK) | all |
| FALS_ENERGY | CORRUPTION_FALSIFICATION / ENERGY_UNDERREPORT | kWh register | increments × (1 − 0.10..0.40) for 1-2 days; deficit persists | PUMP1, COMP1, AUX1 |
| COMM_DROPOUT | COMMUNICATION_PROBLEM / DROPOUT | measured | all values empty, comm_status LOST, 1-120 min; register catches up | all |

`severity` and `gt_detectability` in `events.csv` are derived from the drawn magnitude (`scheduler._severity`).

## Expected signatures (what the diagnosis looks for)

| Label | Breaks | Holds |
|---|---|---|
| Overload | over-current on both CTs | R3, R4, R6 (temperature follows the thermal model) |
| Bearing | R10E DE and NDE above load envelope (R10 ratio shifts) | electrical relations |
| Grid | ≥ 3 meters under-voltage / ~0 V together | R8 (meters agree) |
| Meter CT drift | R3, R4/R5B (bus), growing; vote → METER_CT | R1, R2, R6 |
| Current bias | R3 with a step; vote → odd CT (relay bias also breaks R6) | R1, R2 |
| Voltage bias | R8 for one meter | R9, R1 |
| Temperature bias | R6 with a step faster than τ allows | R3, no over-current |
| Stuck | flatline on a channel while running | other paths keep moving |
| Falsification | R5 (register vs own ∫P) and R5B (bus energy) | R1, R2, R3, R4 |
| Data gap / dropout | missing values with link OK / comm_status LOST | — |
