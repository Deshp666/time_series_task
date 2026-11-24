import json
import numpy as np
import pandas as pd
import pytest

from bike_forecast.data import causal_fill, design_frames, training_frame, WEATHER
from bike_forecast.evaluation import metrics, calibration_radius, paired_block_bootstrap


def synthetic(n=800):
    dates = pd.date_range("2012-01-01", periods=n, freq="h")
    df = pd.DataFrame({"unique_id": "bike", "ds": dates, "y": np.arange(n, dtype=float), "observed": True})
    for col in WEATHER:
        df[col] = np.arange(n, dtype=float)
    return df


def test_fill_is_causal_and_weekly():
    s = pd.Series([np.nan, 3., np.nan, 5.] + list(range(4, 170)), dtype=float)
    s.iloc[169] = np.nan
    result = causal_fill(s)
    assert np.isnan(result.iloc[0])
    assert result.iloc[2] == 3
    assert result.iloc[169] == 3
    modified = s.copy()
    modified.iloc[100:] = 10000
    np.testing.assert_equal(result.iloc[:100].to_numpy(), causal_fill(modified).iloc[:100].to_numpy())


def test_cutoff_and_future_weather_are_independent_of_test():
    df = synthetic()
    origin = df.ds.iloc[500]
    train = training_frame(df, str(origin))
    altered = df.copy()
    altered.loc[altered.ds >= origin, ["y", *WEATHER]] = -99999
    other = training_frame(altered, str(origin))
    pd.testing.assert_frame_equal(train, other)
    for h in (24, 168):
        x, future, features = design_frames(train, h, weather=True)
        assert "casual" not in x and "registered" not in x
        assert future.ds.min() == origin
        np.testing.assert_equal(future.temp_lag168.to_numpy(), df.temp.iloc[500-168:500-168+h].to_numpy())
        assert "y" not in future


def test_masked_metrics_and_zero_denominators():
    r = metrics([2, 10000, np.nan], [3, -100, 9], [True, False, False], np.ones(200))
    assert r["MAE"] == 1 and r["n_observed"] == 1 and r["MASE"] is None
    assert metrics([0], [1], [True], np.arange(200))["WAPE"] is None
    with pytest.raises(ValueError, match="Non-finite"):
        metrics([1], [np.nan], [True], np.arange(200))


def test_intervals_and_bootstrap():
    a = calibration_radius(range(100), .8)
    b = calibration_radius(range(100), .95)
    assert 0 <= a <= b
    result = paired_block_bootstrap([np.ones(48)*-2, np.ones(24)*-2])
    assert result["blocks"] == 3
    assert result["ci_low"] == result["ci_high"] == -2


def test_recursive_ml_does_not_receive_future_targets():
    from bike_forecast.models import forecast
    df = synthetic()
    train = training_frame(df, str(df.ds.iloc[600]))
    pred, _ = forecast(train, "Ridge", {"alpha": 1}, 24, threads=1)
    df.loc[600:, "y"] = 1e8
    pred_other, _ = forecast(training_frame(df, str(df.ds.iloc[600])), "Ridge", {"alpha": 1}, 24, threads=1)
    np.testing.assert_allclose(pred, pred_other)
    assert len(pred) == 24 and np.isfinite(pred).all()


def test_full_protocol_identical_windows():
    import yaml
    from bike_forecast.data import ROOT
    cfg = yaml.safe_load((ROOT / "configs/study.yaml").read_text())
    assert cfg["horizons"] == [24, 168]
    for origins in cfg["splits"].values():
        assert len(origins) == 4
        assert len(set(origins)) == 4
    assert max(cfg["splits"]["validation"]) < min(cfg["splits"]["calibration"])
    assert max(cfg["splits"]["calibration"]) < min(cfg["splits"]["test"])


def test_successful_job_cache_and_failed_retry(tmp_path, monkeypatch):
    from bike_forecast.runner import Run
    import bike_forecast.runner as runner
    run = object.__new__(Run)
    run.folder = tmp_path
    run.config = {"seed": 42, "dl": {}, "threads": 1, "job_timeout_seconds": 10}
    run.profile = "smoke"
    run.records = []
    run.cache_hits = 0
    import time
    run.deadline = time.monotonic() + 10
    calls = []
    def fake_run(command, **kwargs):
        job_path = __import__('pathlib').Path(command[-1])
        job = json.loads(job_path.read_text())
        runner.write_json(job_path.parent / "result.json", {**job, "status": "ok", "MAE": 1})
        (job_path.parent / "predictions.csv").write_text("ds,prediction\n2012-09-01,1\n")
        calls.append(command)
        return type("Process", (), {"returncode": 0})()
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = ("SeasonalNaive", {"season_length": 168}, 24, "validation", "2012-09-01")
    first = run.job(*args)
    second = run.job(*args)
    assert len(calls) == 1 and run.cache_hits == 1
    assert first == second
    run.job("SeasonalNaive", {"season_length": 24}, 24, "validation", "2012-09-01")
    assert len(calls) == 2
    old_path = tmp_path / "jobs" / first["job_id"] / "result.json"
    runner.write_json(old_path, {**first, "status": "error"})
    retried = run.job(*args)
    assert len(calls) == 3 and retried["status"] == "ok"


def test_fingerprint_tracks_config_and_code(tmp_path, monkeypatch):
    import bike_forecast.runner as runner
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/hourly.csv").write_text("data")
    (tmp_path / "bike_forecast").mkdir()
    code = tmp_path / "bike_forecast/test.py"
    code.write_text("v1")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "environment", lambda: {"python": "test"})
    first = runner.fingerprint({"seed": 1})
    assert first == runner.fingerprint({"seed": 1})
    assert first != runner.fingerprint({"seed": 2})
    code.write_text("v2")
    assert first != runner.fingerprint({"seed": 1})
