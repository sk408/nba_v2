import src.analytics.backtester as backtester
from src.web.app import app


def test_accuracy_page_renders_quality_frontier(monkeypatch):
    monkeypatch.setattr(
        backtester,
        "run_backtest",
        lambda: {
            "fundamentals": {
                "total_games": 100,
                "winner_pct": 66.8,
                "favorites_pct": 64.2,
                "beats_favorites": True,
                "upset_rate": 22.0,
                "upset_count": 22,
                "upset_accuracy": 49.0,
                "upset_correct": 11,
                "ml_roi": 3.2,
                "ml_win_rate": 51.0,
                "spread_mae": 8.3,
                "tier_a_hit_rate": 61.0,
                "tier_b_hit_rate": 49.0,
                "tier_c_hit_rate": 42.0,
                "tier_a_coverage_pct": 6.0,
                "tier_b_coverage_pct": 10.0,
                "tier_c_coverage_pct": 6.0,
                "hit_rate_quality_observation": "tier hit A/B/C=61/49/42",
                "upset_quality_frontier": [
                    {
                        "upset_coverage_pct": 50.0,
                        "picks": 11.0,
                        "hit_rate": 58.0,
                        "ml_roi": 7.5,
                        "avg_confidence": 68.0,
                    }
                ],
                "upset_roi_by_odds_band": {
                    "2_00_2_99": {
                        "label": "2.00x-2.99x",
                        "bets": 10,
                        "hit_rate": 52.0,
                        "ml_roi": 5.1,
                    }
                },
            },
            "sharp": {
                "winner_pct": 67.0,
                "upset_rate": 21.0,
                "upset_count": 21,
                "upset_accuracy": 48.0,
                "upset_correct": 10,
                "ml_roi": 2.5,
                "ml_win_rate": 50.0,
                "spread_mae": 8.5,
            },
            "comparison": {
                "sharp_flipped_picks": 9,
                "sharp_flipped_correct": 5,
                "sharp_flipped_accuracy": 55.5,
                "sharp_net_value": 0.2,
            },
            "drift": {
                "fundamentals": {"alert_count": 1},
                "sharp": {"alert_count": 0},
            },
            "phase_acceptance": {
                "passed": True,
                "failed_count": 0,
                "failed_checks": [],
                "generated_at": "2026-03-13T04:00:00+00:00",
            },
            "phase_acceptance_report": {
                "report_path": "data/reports/phase_acceptance_2026-03-13.json",
                "latest_path": "data/reports/phase_acceptance_latest.json",
                "exists": True,
            },
        },
    )

    with app.test_client() as client:
        resp = client.get("/accuracy")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Phase Gate Status" in html
    assert "PASS" in html
    assert "Failed checks: 0" in html
    assert "Underdog Quality Frontier" in html
    assert "Top 50% upsets" in html
    assert "2.00x-2.99x" in html
