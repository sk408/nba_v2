def test_run_backtest_and_compare_attaches_drift_and_report(monkeypatch):
    from src.analytics import pipeline
    from src.analytics import backtester
    from src.analytics import drift_monitor
    from src.analytics import phase_gates

    fake_backtest = {
        "fundamentals": {
            "winner_pct": 66.0,
            "upset_tier_metrics": {},
            "upset_roi_by_odds_band": {},
            "upset_quality_frontier": [],
        },
        "sharp": {
            "winner_pct": 65.0,
            "upset_tier_metrics": {},
            "upset_roi_by_odds_band": {},
            "upset_quality_frontier": [],
        },
        "comparison": {"sharp_net_value": -1.0},
    }

    monkeypatch.setattr(backtester, "invalidate_backtest_cache", lambda: None)
    monkeypatch.setattr(backtester, "run_backtest", lambda callback=None: dict(fake_backtest))
    monkeypatch.setattr(
        drift_monitor,
        "evaluate_underdog_drift",
        lambda metrics: {"triggered": False, "alert_count": 0, "alerts": []},
    )
    monkeypatch.setattr(
        drift_monitor,
        "write_weekly_frontier_report",
        lambda results: {
            "report_path": "data/reports/weekly_frontier_2026-03-13.json",
            "latest_path": "data/reports/weekly_frontier_latest.json",
            "drift_triggered": False,
            "fundamentals_alert_count": 0,
            "sharp_alert_count": 0,
        },
    )
    monkeypatch.setattr(
        phase_gates,
        "evaluate_phase_acceptance",
        lambda result: {"passed": True, "failed_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        phase_gates,
        "write_phase_acceptance_report",
        lambda report: {
            "report_path": "data/reports/phase_acceptance_2026-03-13.json",
            "latest_path": "data/reports/phase_acceptance_latest.json",
            "passed": True,
            "failed_count": 0,
        },
    )

    result = pipeline.run_backtest_and_compare()
    assert "drift" in result
    assert "weekly_frontier_report" in result
    assert "phase_acceptance" in result
    assert "phase_acceptance_report" in result
    assert result["weekly_frontier_report"]["report_path"].endswith(".json")


def test_run_recommendation_settlement_uses_backfill(monkeypatch):
    from src.analytics import pipeline
    from src.analytics import recommendation_outcomes

    monkeypatch.setattr(
        recommendation_outcomes,
        "backfill_recommendation_outcomes",
        lambda callback=None: {"pending": 5, "settled": 3},
    )
    result = pipeline.run_recommendation_settlement()
    assert result["pending"] == 5
    assert result["settled"] == 3


def test_run_weekly_retraining_evaluation_exposes_phase_gate_top_level(monkeypatch):
    from src.analytics import pipeline

    monkeypatch.setattr(
        pipeline,
        "run_overnight",
        lambda max_hours, reset_weights, callback=None: {"passes": 2},
    )
    monkeypatch.setattr(
        pipeline,
        "run_backtest_and_compare",
        lambda callback=None: {
            "drift": {"triggered": False},
            "weekly_frontier_report": {"report_path": "data/reports/weekly_frontier_x.json"},
            "phase_acceptance": {"passed": True, "failed_count": 0},
            "phase_acceptance_report": {"report_path": "data/reports/phase_acceptance_x.json"},
        },
    )

    result = pipeline.run_weekly_retraining_evaluation(max_hours=1.0, reset_weights=False)
    assert result["phase_acceptance"]["passed"] is True
    assert result["phase_acceptance_report"]["report_path"].endswith(".json")
