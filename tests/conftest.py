import pytest

from sanad.config import load_config


def small_cfg(days: int = 3, clean: bool = True, run_id: str = "pytest"):
    cfg = load_config("configs/demo.yaml")
    cfg["run_id"] = run_id
    cfg["paths"]["compress"] = False
    cfg["time"]["duration_days"] = days
    if clean:
        cfg["injection"]["enabled"] = False
        for k in ("mcar_missing_rate", "spike_rate", "delayed_rate"):
            cfg["baseline_dq"][k] = 0.0
    return cfg


@pytest.fixture(scope="session")
def clean_run(tmp_path_factory):
    from sanad import io
    from sanad.sim.generator import generate

    out = generate(small_cfg(3), tmp_path_factory.mktemp("clean"))
    return {
        "dir": out,
        "m": io.read_measurements(out),
        "t": io.read_csv(out / "truth_clean.csv"),
        "eq": io.read_csv(out / "equipment.csv").set_index("equipment_id"),
    }
