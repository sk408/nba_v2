import json

from src.analytics.drift_monitor import evaluate_underdog_drift, write_weekly_frontier_report


def _metrics_with_drift():
    return {
        "winner_pct": 66.0,
        "upset_rate": 30.0,
        "upset_accuracy": 49.0,
        "ml_roi": 1.2,
        "hit_rate_quality_observation": "obs",
        "upset_tier_metrics": {
            "A": {"count": 20, "hit_rate": 48.0},
            "B": {"count": 25, "hit_rate": 47.5},
            "C": {"count": 40, "hit_rate": 43.0},
        },
        "upset_roi_by_odds_band": {
            "2_00_2_99": {"label": "2.00x-2.99x", "bets": 30, "ml_roi": -6.0},
            "3_00_3_99": {"label": "3.00x-3.99x", "bets": 18, "ml_roi": -8.0},
            "4_plus": {"label": "4.00x+", "bets": 15, "ml_roi": -12.0},
        },
        "upset_quality_frontier": [],
    }


def test_evaluate_underdog_drift_triggers_alerts():
    report = evaluate_underdog_drift(_metrics_with_drift())
    assert report["triggered"] is True
    assert report["alert_count"] >= 3
    assert any(item["type"] == "tier_hit_rate_drift" for item in report["alerts"])
    assert any(item["type"] == "odds_band_roi_drift" for item in report["alerts"])


def test_write_weekly_frontier_report_outputs_files(tmp_path):
    backtest_results = {
        "fundamentals": _metrics_with_drift(),
        "sharp": _metrics_with_drift(),
        "comparison": {"sharp_net_value": 0.2},
    }
    result = write_weekly_frontier_report(
        backtest_results,
        output_dir=str(tmp_path),
        report_date="2026-03-13",
    )
    report_path = tmp_path / "weekly_frontier_2026-03-13.json"
    latest_path = tmp_path / "weekly_frontier_latest.json"
    assert str(report_path) == result["report_path"]
    assert str(latest_path) == result["latest_path"]
    assert report_path.exists()
    assert latest_path.exists()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["drift_triggered"] is True
    assert "fundamentals" in data
    assert "quality_frontier" in data["fundamentals"]
