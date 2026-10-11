import pytest

from sanad.adapters.paderborn import PaderbornAdapter
from sanad.adapters.pvgis import PVGISAdapter
from sanad.adapters.steel import SteelAdapter
from sanad.cli import main
from sanad.config import block_bounds, deep_merge, load_config, repo_path, time_index


def test_default_config_loads():
    cfg = load_config()
    assert cfg["data_source"] == "synthetic"
    assert set(cfg["plant"]["equipment"]) == {"PUMP1", "COMP1", "AUX1"}
    assert len(time_index(cfg)) == 56 * 24 * 60


def test_demo_config_extends_default():
    cfg = load_config("configs/demo.yaml")
    assert cfg["run_id"] == "demo" and cfg["time"]["duration_days"] == 14
    assert cfg["plant"]["v_ll_nominal"] == 400.0  # inherited
    b = block_bounds(cfg)
    assert b["test"][1] - b["normal"][0] == __import__("pandas").Timedelta(days=14)


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}}
    b = deep_merge(a, {"x": {"y": 5}})
    assert b == {"x": {"y": 5, "z": 2}} and a["x"]["y"] == 1


@pytest.mark.parametrize("cls", [SteelAdapter, PVGISAdapter, PaderbornAdapter])
def test_adapters_are_stubs(cls):
    ad = cls(repo_path("data/raw"))
    assert isinstance(ad.available(), bool)
    with pytest.raises(NotImplementedError):
        ad.load()


def test_cli_rejects_unknown_command():
    with pytest.raises(SystemExit):
        main(["nope"])
