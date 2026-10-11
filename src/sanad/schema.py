"""Output schemas shared by the synthetic generator and the (future) real-data adapters."""

from __future__ import annotations

EQUIPMENT_ORDER = ["BUS1", "PUMP1", "COMP1", "AUX1"]
FEEDERS = ["PUMP1", "COMP1", "AUX1"]
MOTORS = ["PUMP1", "COMP1"]

KEY_COLUMNS = ["timestamp_gw_utc", "timestamp_dev_utc", "node_id", "equipment_id"]
METER_COLUMNS = [
    "v_l1_v", "v_l2_v", "v_l3_v", "i_l1_a", "i_l2_a", "i_l3_a",
    "p_kw", "q_kvar", "s_kva", "pf", "freq_hz", "kwh_register", "thd_v_pct", "thd_i_pct",
]
MOTOR_COLUMNS = ["i_relay_a", "speed_rpm", "winding_temp_c", "vib_de_g", "vib_nde_g"]
VALUE_COLUMNS = [
    "v_l1_v", "v_l2_v", "v_l3_v", "i_l1_a", "i_l2_a", "i_l3_a", "i_relay_a",
    "p_kw", "q_kvar", "s_kva", "pf", "freq_hz", "kwh_register", "thd_v_pct", "thd_i_pct",
    "speed_rpm", "winding_temp_c", "vib_de_g", "vib_nde_g", "ambient_temp_c",
]
STATE_COLUMNS = ["op_state", "run_cmd", "comm_status", "maint_mode"]
MEASUREMENT_COLUMNS = KEY_COLUMNS + VALUE_COLUMNS + STATE_COLUMNS

OP_STATES = ["OFF", "STARTING", "RUN_LOW", "RUN_NORMAL", "RUN_HIGH", "STOPPING"]
COMM_STATUS = ["OK", "DELAYED", "LOST"]
MAINT_MODES = ["NONE", "PLANNED_SHUTDOWN", "CALIBRATION", "INSPECTION"]

EVENT_COLUMNS = [
    "event_id", "main_class", "subtype", "origin", "severity", "equipment_id", "sensor_id",
    "start_utc", "end_utc", "gt_detectability", "params_json", "compound_with",
]

SENSOR_PATHS = ["METER_CT", "RELAY_CT", "RTD", "ACC_DE", "ACC_NDE", "SPEED", "BUS_CT", "AMBIENT"]

CLASSES = [
    "NORMAL", "PHYSICAL_EQUIPMENT_FAULT", "GRID_EVENT", "SENSOR_FAULT", "BIAS_MISCALIBRATION",
    "CORRUPTION_FALSIFICATION", "COMMUNICATION_PROBLEM", "UNKNOWN",
]
PROVENANCE = ["VERIFIED", "ESTIMATED", "SUSPECT", "QUARANTINED"]

# Ground-truth-only column names that must never appear in measurements.csv
FORBIDDEN_IN_MEASUREMENTS = {
    "label", "event_id", "main_class", "subtype", "origin", "severity", "true", "truth",
    "clean", "gain", "bias", "drift", "params", "injected", "health", "load_fraction",
}


def applicable_columns(eq_type: str) -> list[str]:
    """Measurement value columns that exist for an equipment type."""
    cols = list(METER_COLUMNS) + ["ambient_temp_c"]
    if eq_type == "motor":
        cols += MOTOR_COLUMNS
    return [c for c in VALUE_COLUMNS if c in cols]
