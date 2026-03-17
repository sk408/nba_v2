"""Shared underdog quality metrics for backtests and optimization runs.

This module focuses on the tradeoff between coverage (how many underdog picks
we surface) and quality (hit rate + ROI).
"""

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.config import get as get_setting

DEFAULT_TIER_A_MIN_CONFIDENCE = 70.0
DEFAULT_TIER_B_MIN_CONFIDENCE = 55.0
DEFAULT_FRONTIER_COVERAGE_PCTS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

ODDS_BANDS = (
    ("lt_1_50", "<1.50x", 0.0, 1.5),
    ("1_50_1_99", "1.50x-1.99x", 1.5, 2.0),
    ("2_00_2_99", "2.00x-2.99x", 2.0, 3.0),
    ("3_00_3_99", "3.00x-3.99x", 3.0, 4.0),
    ("4_plus", "4.00x+", 4.0, float("inf")),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    return default


def _normalize_confidence(raw_confidence: Any) -> float:
    return max(0.0, min(100.0, _safe_float(raw_confidence, 0.0)))


def _confidence_from_edge(edge_abs: float, edge_scale: float) -> float:
    scale = max(1.0, float(edge_scale))
    edge = max(0.0, float(edge_abs))
    # Logistic-style confidence curve that avoids early 100% saturation.
    return float(max(0.0, min(100.0, 100.0 * (1.0 - math.exp(-edge / scale)))))


def _normalize_sample(
    sample: Mapping[str, Any],
    *,
    use_edge_logistic: bool,
    edge_scale: float,
) -> Dict[str, Any]:
    raw_confidence = _normalize_confidence(sample.get("confidence", 0.0))
    edge_abs = abs(_safe_float(sample.get("edge_abs", 0.0), 0.0))
    confidence = (
        _confidence_from_edge(edge_abs, edge_scale)
        if use_edge_logistic and edge_abs > 0.0
        else raw_confidence
    )
    upset_correct = bool(sample.get("upset_correct", False))
    ml_profit = _safe_float(sample.get("ml_profit", 0.0), 0.0)
    ml_payout = max(0.0, _safe_float(sample.get("ml_payout", 0.0), 0.0))
    return {
        "confidence": confidence,
        "raw_confidence": raw_confidence,
        "edge_abs": edge_abs,
        "upset_correct": upset_correct,
        "ml_profit": ml_profit,
        "ml_payout": ml_payout,
    }


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def quality_tier_for_confidence(
    confidence: float,
    tier_a_min: float = DEFAULT_TIER_A_MIN_CONFIDENCE,
    tier_b_min: float = DEFAULT_TIER_B_MIN_CONFIDENCE,
) -> str:
    """Map confidence to a tier label."""
    if confidence >= tier_a_min:
        return "A"
    if confidence >= tier_b_min:
        return "B"
    return "C"


def _compute_tier_metrics(
    samples: Sequence[Mapping[str, Any]],
    total_games: int,
    tier_a_min: float,
    tier_b_min: float,
) -> Dict[str, Dict[str, float]]:
    tiers: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for sample in samples:
        tier = quality_tier_for_confidence(
            sample["confidence"],
            tier_a_min=tier_a_min,
            tier_b_min=tier_b_min,
        )
        tiers[tier].append(sample)

    metrics: Dict[str, Dict[str, float]] = {}
    denom_games = max(1, int(total_games))
    for tier in ("A", "B", "C"):
        rows = tiers[tier]
        count = len(rows)
        correct = sum(1 for row in rows if row["upset_correct"])
        total_profit = sum(row["ml_profit"] for row in rows)
        metrics[tier] = {
            "count": float(count),
            "hit_rate": _pct(correct, max(1, count)),
            "coverage_pct": _pct(count, denom_games),
            "ml_roi": _pct(total_profit, max(1, count)),
            "avg_confidence": (
                sum(row["confidence"] for row in rows) / max(1, count)
            ),
        }
    return metrics


def _compute_roi_by_odds_band(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    normalized = list(samples)
    for band_key, band_label, lo, hi in ODDS_BANDS:
        rows: List[Dict[str, Any]] = []
        for sample in normalized:
            payout = sample["ml_payout"]
            if payout < lo:
                continue
            if hi != float("inf") and payout >= hi:
                continue
            rows.append(sample)

        bets = len(rows)
        wins = sum(1 for row in rows if row["upset_correct"])
        total_profit = sum(row["ml_profit"] for row in rows)
        result[band_key] = {
            "label": band_label,
            "bets": bets,
            "hit_rate": _pct(wins, max(1, bets)),
            "ml_roi": _pct(total_profit, max(1, bets)),
            "avg_payout": (
                sum(row["ml_payout"] for row in rows) / max(1, bets)
            ),
        }
    return result


def _compute_quality_frontier(
    samples: Sequence[Mapping[str, Any]],
    total_games: int,
    coverage_pcts: Sequence[int],
) -> List[Dict[str, float]]:
    normalized = list(samples)
    if not normalized:
        return []

    sorted_rows = sorted(normalized, key=lambda row: row["confidence"], reverse=True)
    upset_total = len(sorted_rows)
    game_total = max(1, int(total_games))

    coverage_targets = {
        int(max(1, min(100, int(pct))))
        for pct in coverage_pcts
    }
    coverage_targets.add(100)

    frontier: List[Dict[str, float]] = []
    for target_pct in sorted(coverage_targets):
        picks = max(1, int(math.ceil(upset_total * (target_pct / 100.0))))
        subset = sorted_rows[:picks]
        correct = sum(1 for row in subset if row["upset_correct"])
        total_profit = sum(row["ml_profit"] for row in subset)
        frontier.append(
            {
                "upset_coverage_pct": _pct(picks, upset_total),
                "coverage_pct": _pct(picks, game_total),
                "picks": float(picks),
                "hit_rate": _pct(correct, picks),
                "ml_roi": _pct(total_profit, picks),
                "avg_confidence": (
                    sum(row["confidence"] for row in subset) / max(1, picks)
                ),
            }
        )
    return frontier


def _frontier_point_at_or_above(
    frontier: Sequence[Mapping[str, Any]],
    target_upset_coverage: float,
) -> Mapping[str, Any]:
    if not frontier:
        return {}
    for point in frontier:
        if _safe_float(point.get("upset_coverage_pct", 0.0), 0.0) >= target_upset_coverage:
            return point
    return frontier[-1]


def format_hit_rate_quality_observation(summary: Mapping[str, Any]) -> str:
    """Return concise line for logs/UI: quality tiers + frontier tradeoff."""
    upset_count = int(summary.get("upset_pick_count", 0))
    if upset_count <= 0:
        return "Hit/quality observation: no upset picks."

    tiers = summary.get("tier_metrics", {}) or {}
    tier_a = tiers.get("A", {}) or {}
    tier_b = tiers.get("B", {}) or {}
    tier_c = tiers.get("C", {}) or {}

    frontier = summary.get("quality_frontier", []) or []
    top30 = _frontier_point_at_or_above(frontier, 30.0)
    full = frontier[-1] if frontier else {}

    a_hit = _safe_float(tier_a.get("hit_rate", 0.0), 0.0)
    b_hit = _safe_float(tier_b.get("hit_rate", 0.0), 0.0)
    c_hit = _safe_float(tier_c.get("hit_rate", 0.0), 0.0)
    a_cov = _safe_float(tier_a.get("coverage_pct", 0.0), 0.0)
    b_cov = _safe_float(tier_b.get("coverage_pct", 0.0), 0.0)
    c_cov = _safe_float(tier_c.get("coverage_pct", 0.0), 0.0)

    top_cov = _safe_float(top30.get("upset_coverage_pct", 0.0), 0.0)
    top_hit = _safe_float(top30.get("hit_rate", 0.0), 0.0)
    full_hit = _safe_float(full.get("hit_rate", 0.0), 0.0)

    return (
        "Hit/quality observation: "
        f"tier hit A/B/C={a_hit:.1f}/{b_hit:.1f}/{c_hit:.1f}% "
        f"(coverage {a_cov:.1f}/{b_cov:.1f}/{c_cov:.1f}% of slate); "
        f"frontier top {top_cov:.0f}% upsets={top_hit:.1f}% vs full book={full_hit:.1f}%."
    )


def summarize_underdog_quality(
    upset_samples: Iterable[Mapping[str, Any]],
    total_games: int,
    tier_a_min_confidence: float = DEFAULT_TIER_A_MIN_CONFIDENCE,
    tier_b_min_confidence: float = DEFAULT_TIER_B_MIN_CONFIDENCE,
    frontier_coverage_pcts: Sequence[int] = DEFAULT_FRONTIER_COVERAGE_PCTS,
    confidence_use_edge_logistic: bool | None = None,
    confidence_edge_scale: float | None = None,
) -> Dict[str, Any]:
    """Aggregate underdog quality metrics from upset-pick samples.

    Each sample should contain:
    - confidence: 0..100
    - upset_correct: bool
    - ml_profit: profit for 1 unit stake (+x on win, -1 on loss)
    - ml_payout: decimal payout multiplier
    """
    if confidence_use_edge_logistic is None:
        confidence_use_edge_logistic = _safe_bool(
            get_setting("underdog_confidence_use_edge_logistic", True),
            True,
        )
    if confidence_edge_scale is None:
        confidence_edge_scale = max(
            1.0,
            _safe_float(get_setting("underdog_confidence_edge_scale", 70.0), 70.0),
        )

    samples = [
        _normalize_sample(
            sample,
            use_edge_logistic=bool(confidence_use_edge_logistic),
            edge_scale=float(confidence_edge_scale),
        )
        for sample in upset_samples
    ]
    total = max(1, int(total_games))
    upset_count = len(samples)
    upset_coverage_pct = _pct(upset_count, total)

    tier_metrics = _compute_tier_metrics(
        samples=samples,
        total_games=total,
        tier_a_min=tier_a_min_confidence,
        tier_b_min=tier_b_min_confidence,
    )
    roi_by_odds_band = _compute_roi_by_odds_band(samples)
    quality_frontier = _compute_quality_frontier(
        samples=samples,
        total_games=total,
        coverage_pcts=frontier_coverage_pcts,
    )

    summary: Dict[str, Any] = {
        "upset_pick_count": upset_count,
        "coverage_pct": upset_coverage_pct,
        "tier_thresholds": {
            "tier_a_min_confidence": float(tier_a_min_confidence),
            "tier_b_min_confidence": float(tier_b_min_confidence),
        },
        "confidence_calibration": {
            "use_edge_logistic": bool(confidence_use_edge_logistic),
            "edge_scale": float(confidence_edge_scale),
        },
        "tier_metrics": tier_metrics,
        "roi_by_odds_band": roi_by_odds_band,
        "quality_frontier": quality_frontier,
    }

    for tier_name in ("A", "B", "C"):
        tier = tier_metrics[tier_name]
        summary[f"tier_{tier_name.lower()}_count"] = int(tier["count"])
        summary[f"tier_{tier_name.lower()}_hit_rate"] = float(tier["hit_rate"])
        summary[f"tier_{tier_name.lower()}_coverage_pct"] = float(tier["coverage_pct"])
        summary[f"tier_{tier_name.lower()}_ml_roi"] = float(tier["ml_roi"])

    summary["hit_rate_quality_observation"] = format_hit_rate_quality_observation(summary)
    return summary
