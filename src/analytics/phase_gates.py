"""Phase-acceptance guardrails for underdog model promotions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from src.config import get as get_setting


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _thresholds() -> Dict[str, float]:
    return {
        "min_winner_pct": _safe_float(get_setting("phase_gate_min_winner_pct", 60.0), 60.0),
        "min_upset_coverage_pct": _safe_float(
            get_setting("phase_gate_min_upset_coverage_pct", 20.0), 20.0
        ),
        "upset_coverage_tolerance_pp": _safe_float(
            get_setting("phase_gate_upset_coverage_tolerance_pp", 0.5),
            0.5,
        ),
        "min_tier_a_hit_rate": _safe_float(
            get_setting("phase_gate_min_tier_a_hit_rate", 56.0), 56.0
        ),
        "min_ml_roi": _safe_float(get_setting("phase_gate_min_ml_roi", 0.0), 0.0),
        "max_fund_drift_alerts": _safe_float(
            get_setting("phase_gate_max_fund_drift_alerts", 1), 1
        ),
    }


def _check(id_: str, label: str, actual: float, threshold: float, op: str) -> Dict[str, Any]:
    if op == ">=":
        passed = actual >= threshold
        detail = f"{actual:.2f} >= {threshold:.2f}"
    else:
        passed = actual <= threshold
        detail = f"{actual:.2f} <= {threshold:.2f}"
    return {
        "id": id_,
        "label": label,
        "passed": passed,
        "actual": round(actual, 4),
        "threshold": threshold,
        "operator": op,
        "detail": detail,
    }


def evaluate_phase_acceptance(backtest_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate promotion guardrails from latest backtest + drift outputs."""
    thresholds = _thresholds()
    fundamentals = (
        backtest_result.get("fundamentals", {})
        if isinstance(backtest_result, Mapping)
        else {}
    )
    drift = backtest_result.get("drift", {}) if isinstance(backtest_result, Mapping) else {}
    fundamentals_drift = drift.get("fundamentals", {}) if isinstance(drift, Mapping) else {}

    winner_pct = _safe_float(fundamentals.get("winner_pct"), 0.0)
    upset_coverage = _safe_float(fundamentals.get("upset_coverage_pct"), 0.0)
    tier_metrics = fundamentals.get("upset_tier_metrics", {}) if isinstance(fundamentals, Mapping) else {}
    tier_a = tier_metrics.get("A", {}) if isinstance(tier_metrics, Mapping) else {}
    tier_a_hit = _safe_float(tier_a.get("hit_rate"), 0.0)
    ml_roi = _safe_float(fundamentals.get("ml_roi"), 0.0)
    drift_alerts = _safe_float(fundamentals_drift.get("alert_count"), 0.0)

    coverage_target = thresholds["min_upset_coverage_pct"]
    coverage_tolerance = max(0.0, thresholds.get("upset_coverage_tolerance_pp", 0.0))
    coverage_effective_floor = max(0.0, coverage_target - coverage_tolerance)
    coverage_check = _check(
        "upset_coverage_floor",
        "Upset coverage floor",
        upset_coverage,
        coverage_effective_floor,
        ">=",
    )
    coverage_check["threshold_target"] = coverage_target
    coverage_check["tolerance_pp"] = coverage_tolerance
    coverage_check["effective_threshold"] = coverage_effective_floor
    if coverage_tolerance > 0.0:
        coverage_check["detail"] = (
            f"{upset_coverage:.2f} + {coverage_tolerance:.2f} >= {coverage_target:.2f}"
        )
        # Keep displayed threshold aligned with the user-facing target.
        coverage_check["threshold"] = coverage_target

    checks = [
        _check(
            "winner_pct_floor",
            "Winner accuracy floor",
            winner_pct,
            thresholds["min_winner_pct"],
            ">=",
        ),
        coverage_check,
        _check(
            "tier_a_hit_rate_floor",
            "Tier A hit-rate floor",
            tier_a_hit,
            thresholds["min_tier_a_hit_rate"],
            ">=",
        ),
        _check(
            "ml_roi_floor",
            "Moneyline ROI floor",
            ml_roi,
            thresholds["min_ml_roi"],
            ">=",
        ),
        _check(
            "fund_drift_alert_budget",
            "Fundamentals drift-alert budget",
            drift_alerts,
            thresholds["max_fund_drift_alerts"],
            "<=",
        ),
    ]

    failed = [c for c in checks if not c["passed"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": len(failed) == 0,
        "failed_count": len(failed),
        "checks": checks,
        "thresholds": thresholds,
        "failed_checks": failed,
    }


def write_phase_acceptance_report(
    report: Mapping[str, Any],
    output_dir: str | None = None,
    report_date: str | None = None,
) -> Dict[str, Any]:
    """Persist phase-acceptance report JSON under data/reports."""
    target_dir = str(output_dir or get_setting("weekly_frontier_report_dir", "data/reports"))
    os.makedirs(target_dir, exist_ok=True)
    day = report_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = os.path.join(target_dir, f"phase_acceptance_{day}.json")
    latest_path = os.path.join(target_dir, "phase_acceptance_latest.json")

    payload = dict(report)
    payload["report_date"] = day
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "report_path": out_path,
        "latest_path": latest_path,
        "passed": bool(report.get("passed")),
        "failed_count": int(report.get("failed_count", 0)),
    }
