"""Weekly retraining + frontier drift report runner."""

import argparse

from src.analytics.pipeline import run_weekly_retraining_evaluation
from src.bootstrap import setup_logging
from src.database.migrations import init_db


def main():
    parser = argparse.ArgumentParser(
        description="Run weekly retraining/evaluation and write frontier report."
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=6.0,
        help="Overnight retraining hours before evaluation (default: 6).",
    )
    parser.add_argument(
        "--reset-weights",
        action="store_true",
        help="Reset model weights before retraining.",
    )
    args = parser.parse_args()

    setup_logging()
    init_db()

    result = run_weekly_retraining_evaluation(
        max_hours=max(0.25, float(args.hours)),
        reset_weights=bool(args.reset_weights),
        callback=lambda msg: print(msg, flush=True),
    )

    backtest = result.get("backtest", {}) if isinstance(result, dict) else {}
    report = result.get("weekly_frontier_report", {}) if isinstance(result, dict) else {}
    phase = result.get("phase_acceptance", {}) if isinstance(result, dict) else {}
    if not phase and isinstance(backtest, dict):
        phase = backtest.get("phase_acceptance", {})
    phase_report = result.get("phase_acceptance_report", {}) if isinstance(result, dict) else {}
    if not phase_report and isinstance(backtest, dict):
        phase_report = backtest.get("phase_acceptance_report", {})
    print("")
    print("=== Weekly Frontier Report ===")
    print(f"Report: {report.get('report_path', 'n/a')}")
    print(f"Latest: {report.get('latest_path', 'n/a')}")
    print(f"Drift triggered: {bool(report.get('drift_triggered', False))}")
    print(
        "Alerts (fund/sharp): "
        f"{int(report.get('fundamentals_alert_count', 0))}/"
        f"{int(report.get('sharp_alert_count', 0))}"
    )
    print("")
    print("=== Phase Acceptance ===")
    print(f"Passed: {bool(phase.get('passed', False))}")
    print(f"Failed checks: {int(phase.get('failed_count', 0))}")
    print(f"Report: {phase_report.get('report_path', 'n/a')}")


if __name__ == "__main__":
    main()
