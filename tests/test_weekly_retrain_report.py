import sys

import weekly_retrain_report as weekly_script


def test_weekly_retrain_report_reads_top_level_phase(monkeypatch, capsys):
    monkeypatch.setattr(weekly_script, "setup_logging", lambda: None)
    monkeypatch.setattr(weekly_script, "init_db", lambda: None)
    monkeypatch.setattr(
        weekly_script,
        "run_weekly_retraining_evaluation",
        lambda max_hours, reset_weights, callback=None: {
            "weekly_frontier_report": {
                "report_path": "data/reports/weekly_frontier_2026-03-13.json",
                "latest_path": "data/reports/weekly_frontier_latest.json",
                "drift_triggered": False,
                "fundamentals_alert_count": 0,
                "sharp_alert_count": 0,
            },
            "phase_acceptance": {
                "passed": True,
                "failed_count": 0,
            },
            "phase_acceptance_report": {
                "report_path": "data/reports/phase_acceptance_2026-03-13.json",
            },
        },
    )
    monkeypatch.setattr(sys, "argv", ["weekly_retrain_report.py", "--hours", "1"])

    weekly_script.main()
    out = capsys.readouterr().out
    assert "Weekly Frontier Report" in out
    assert "Passed: True" in out
    assert "phase_acceptance_2026-03-13.json" in out


def test_weekly_retrain_report_falls_back_to_backtest_phase(monkeypatch, capsys):
    monkeypatch.setattr(weekly_script, "setup_logging", lambda: None)
    monkeypatch.setattr(weekly_script, "init_db", lambda: None)
    monkeypatch.setattr(
        weekly_script,
        "run_weekly_retraining_evaluation",
        lambda max_hours, reset_weights, callback=None: {
            "weekly_frontier_report": {
                "report_path": "data/reports/weekly_frontier_2026-03-13.json",
                "latest_path": "data/reports/weekly_frontier_latest.json",
                "drift_triggered": True,
                "fundamentals_alert_count": 1,
                "sharp_alert_count": 2,
            },
            "backtest": {
                "phase_acceptance": {
                    "passed": False,
                    "failed_count": 2,
                },
                "phase_acceptance_report": {
                    "report_path": "data/reports/phase_acceptance_latest.json",
                },
            },
        },
    )
    monkeypatch.setattr(sys, "argv", ["weekly_retrain_report.py"])

    weekly_script.main()
    out = capsys.readouterr().out
    assert "Passed: False" in out
    assert "Failed checks: 2" in out
    assert "phase_acceptance_latest.json" in out
