import json

from src.analytics.phase_gates import evaluate_phase_acceptance, write_phase_acceptance_report


def _passing_backtest_result():
    return {
        "fundamentals": {
            "winner_pct": 64.0,
            "upset_coverage_pct": 24.0,
            "ml_roi": 2.5,
            "upset_tier_metrics": {
                "A": {"hit_rate": 58.0},
            },
        },
        "drift": {
            "fundamentals": {"alert_count": 1},
        },
    }


def _failing_backtest_result():
    return {
        "fundamentals": {
            "winner_pct": 58.0,
            "upset_coverage_pct": 14.0,
            "ml_roi": -4.0,
            "upset_tier_metrics": {
                "A": {"hit_rate": 49.0},
            },
        },
        "drift": {
            "fundamentals": {"alert_count": 4},
        },
    }


def test_evaluate_phase_acceptance_passes_when_all_checks_clear():
    report = evaluate_phase_acceptance(_passing_backtest_result())
    assert report["passed"] is True
    assert report["failed_count"] == 0
    assert all(check["passed"] for check in report["checks"])


def test_evaluate_phase_acceptance_fails_on_regressions():
    report = evaluate_phase_acceptance(_failing_backtest_result())
    assert report["passed"] is False
    assert report["failed_count"] >= 3
    assert len(report["failed_checks"]) == report["failed_count"]


def test_write_phase_acceptance_report_outputs_files(tmp_path):
    report = evaluate_phase_acceptance(_passing_backtest_result())
    paths = write_phase_acceptance_report(
        report,
        output_dir=str(tmp_path),
        report_date="2026-03-13",
    )
    out_path = tmp_path / "phase_acceptance_2026-03-13.json"
    latest_path = tmp_path / "phase_acceptance_latest.json"
    assert out_path.exists()
    assert latest_path.exists()
    assert paths["report_path"] == str(out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["report_date"] == "2026-03-13"
    assert payload["passed"] is True
