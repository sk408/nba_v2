import json


def test_run_overnight_counts_only_fully_evaluated_passes(tmp_path, monkeypatch):
    from src.analytics import pipeline

    state_path = tmp_path / "pipeline_state.json"
    monkeypatch.setattr(pipeline, "PIPELINE_STATE_PATH", str(state_path))

    # Keep overnight no-save auto-stop disabled in this isolated scenario.
    def fake_get_setting(key, default=None):
        if key == "overnight_max_no_save_passes":
            return 0
        return default

    monkeypatch.setattr("src.config.get", fake_get_setting)

    def fake_run_pipeline(callback=None):
        return {
            "backtest": {"fundamentals": {"winner_pct": 66.2}},
            "optimize_fundamentals": {"improved": True},
            "optimize_sharp": {"improved": True},
        }

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        pipeline,
        "run_optimize_fundamentals",
        lambda callback=None, is_cancelled=None: {"improved": False},
    )
    monkeypatch.setattr(
        pipeline,
        "run_optimize_sharp",
        lambda callback=None, is_cancelled=None: {"improved": False},
    )

    backtest_calls = {"count": 0}

    def fake_backtest(callback=None, is_cancelled=None):
        backtest_calls["count"] += 1
        return {"fundamentals": {"winner_pct": 67.0}}

    monkeypatch.setattr(pipeline, "run_backtest_and_compare", fake_backtest)

    # Simulate budget exhaustion right after sharp optimization in pass 2.
    time_points = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 3599.0, 3605.0, 3605.0]

    def fake_time():
        if time_points:
            return time_points.pop(0)
        return 3605.0

    monkeypatch.setattr(pipeline.time, "time", fake_time)

    result = pipeline.run_overnight(max_hours=1.0)

    assert result["passes"] == 1
    assert result["attempted_passes"] == 2
    assert backtest_calls["count"] == 0

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    overnight = state["overnight_last_run"]
    assert overnight["passes"] == 1
    assert overnight["attempted_passes"] == 2
