"""Underdog quality drift monitoring + weekly frontier report writer."""

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
        "tier_a_min_hit_rate": _safe_float(get_setting("drift_tier_a_min_hit_rate", 56.0), 56.0),
        "tier_b_min_hit_rate": _safe_float(get_setting("drift_tier_b_min_hit_rate", 50.0), 50.0),
        "tier_c_min_hit_rate": _safe_float(get_setting("drift_tier_c_min_hit_rate", 44.0), 44.0),
        "tier_min_samples": _safe_float(get_setting("drift_tier_min_samples", 12), 12),
        "band_2_00_2_99_min_roi": _safe_float(get_setting("drift_band_2_00_2_99_min_roi", -2.0), -2.0),
        "band_3_00_3_99_min_roi": _safe_float(get_setting("drift_band_3_00_3_99_min_roi", -4.0), -4.0),
        "band_4_plus_min_roi": _safe_float(get_setting("drift_band_4_plus_min_roi", -8.0), -8.0),
        "band_min_bets": _safe_float(get_setting("drift_band_min_bets", 15), 15),
    }


def evaluate_underdog_drift(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate tier precision and odds-band ROI against drift thresholds."""
    thresholds = _thresholds()
    tier_metrics = metrics.get("upset_tier_metrics", {}) if isinstance(metrics, Mapping) else {}
    roi_by_band = metrics.get("upset_roi_by_odds_band", {}) if isinstance(metrics, Mapping) else {}
    alerts = []

    tier_specs = [
        ("A", "tier_a_min_hit_rate"),
        ("B", "tier_b_min_hit_rate"),
        ("C", "tier_c_min_hit_rate"),
    ]
    tier_min_samples = int(max(0.0, thresholds["tier_min_samples"]))
    for tier_name, threshold_key in tier_specs:
        tier = tier_metrics.get(tier_name, {}) if isinstance(tier_metrics, Mapping) else {}
        count = int(_safe_float(tier.get("count"), 0.0))
        hit_rate = _safe_float(tier.get("hit_rate"), 0.0)
        threshold = thresholds[threshold_key]
        if count < tier_min_samples:
            continue
        if hit_rate < threshold:
            alerts.append(
                {
                    "type": "tier_hit_rate_drift",
                    "tier": tier_name,
                    "severity": "high" if tier_name == "A" else "medium",
                    "metric": "hit_rate",
                    "value": round(hit_rate, 3),
                    "threshold": threshold,
                    "sample_size": count,
                    "message": (
                        f"Tier {tier_name} hit rate drift: {hit_rate:.1f}% below "
                        f"threshold {threshold:.1f}% ({count} picks)."
                    ),
                }
            )

    band_specs = [
        ("2_00_2_99", "band_2_00_2_99_min_roi"),
        ("3_00_3_99", "band_3_00_3_99_min_roi"),
        ("4_plus", "band_4_plus_min_roi"),
    ]
    band_min_bets = int(max(0.0, thresholds["band_min_bets"]))
    for band_key, threshold_key in band_specs:
        band = roi_by_band.get(band_key, {}) if isinstance(roi_by_band, Mapping) else {}
        bets = int(_safe_float(band.get("bets"), 0.0))
        roi = _safe_float(band.get("ml_roi"), 0.0)
        threshold = thresholds[threshold_key]
        if bets < band_min_bets:
            continue
        if roi < threshold:
            alerts.append(
                {
                    "type": "odds_band_roi_drift",
                    "band": band_key,
                    "label": str(band.get("label", band_key)),
                    "severity": "medium",
                    "metric": "ml_roi",
                    "value": round(roi, 3),
                    "threshold": threshold,
                    "sample_size": bets,
                    "message": (
                        f"Odds-band ROI drift ({band.get('label', band_key)}): "
                        f"{roi:+.1f}% below threshold {threshold:+.1f}% ({bets} bets)."
                    ),
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "triggered": bool(alerts),
        "alert_count": len(alerts),
        "alerts": alerts,
        "thresholds": thresholds,
    }


def _report_directory() -> str:
    raw = str(get_setting("weekly_frontier_report_dir", "data/reports") or "data/reports")
    return raw


def write_weekly_frontier_report(
    backtest_results: Mapping[str, Any],
    output_dir: str | None = None,
    report_date: str | None = None,
) -> Dict[str, Any]:
    """Write weekly frontier report JSON and return report metadata."""
    report_day = report_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fundamentals = backtest_results.get("fundamentals", {}) if isinstance(backtest_results, Mapping) else {}
    sharp = backtest_results.get("sharp", {}) if isinstance(backtest_results, Mapping) else {}
    comparison = backtest_results.get("comparison", {}) if isinstance(backtest_results, Mapping) else {}

    drift_fundamentals = evaluate_underdog_drift(fundamentals)
    drift_sharp = evaluate_underdog_drift(sharp)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report_date": report_day,
        "fundamentals": {
            "winner_pct": _safe_float(fundamentals.get("winner_pct"), 0.0),
            "upset_rate": _safe_float(fundamentals.get("upset_rate"), 0.0),
            "upset_accuracy": _safe_float(fundamentals.get("upset_accuracy"), 0.0),
            "ml_roi": _safe_float(fundamentals.get("ml_roi"), 0.0),
            "hit_rate_quality_observation": fundamentals.get("hit_rate_quality_observation", ""),
            "tier_metrics": fundamentals.get("upset_tier_metrics", {}),
            "roi_by_odds_band": fundamentals.get("upset_roi_by_odds_band", {}),
            "quality_frontier": fundamentals.get("upset_quality_frontier", []),
            "drift": drift_fundamentals,
        },
        "sharp": {
            "winner_pct": _safe_float(sharp.get("winner_pct"), 0.0),
            "upset_rate": _safe_float(sharp.get("upset_rate"), 0.0),
            "upset_accuracy": _safe_float(sharp.get("upset_accuracy"), 0.0),
            "ml_roi": _safe_float(sharp.get("ml_roi"), 0.0),
            "hit_rate_quality_observation": sharp.get("hit_rate_quality_observation", ""),
            "tier_metrics": sharp.get("upset_tier_metrics", {}),
            "roi_by_odds_band": sharp.get("upset_roi_by_odds_band", {}),
            "quality_frontier": sharp.get("upset_quality_frontier", []),
            "drift": drift_sharp,
        },
        "comparison": comparison,
        "drift_triggered": bool(drift_fundamentals.get("triggered") or drift_sharp.get("triggered")),
    }

    target_dir = output_dir or _report_directory()
    os.makedirs(target_dir, exist_ok=True)
    report_path = os.path.join(target_dir, f"weekly_frontier_{report_day}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    latest_path = os.path.join(target_dir, "weekly_frontier_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "report_path": report_path,
        "latest_path": latest_path,
        "drift_triggered": payload["drift_triggered"],
        "fundamentals_alert_count": int(drift_fundamentals.get("alert_count", 0)),
        "sharp_alert_count": int(drift_sharp.get("alert_count", 0)),
    }
