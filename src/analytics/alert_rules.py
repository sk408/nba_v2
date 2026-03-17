"""Underdog alert rules and digest helpers."""

from typing import Any, Dict, Iterable, List, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _alert_priority(alert: Mapping[str, Any]) -> tuple:
    return (
        int(alert.get("priority", 0)),
        _safe_float(alert.get("rank_score"), 0.0),
        _safe_float(alert.get("confidence"), 0.0),
    )


def build_underdog_alert_candidates(
    candidates: Iterable[Mapping[str, Any]],
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    """Build rule-based alert candidates from screened underdogs."""
    alerts: List[Dict[str, Any]] = []
    for row in candidates:
        confidence = _safe_float(row.get("confidence"), 0.0)
        payout = _safe_float(row.get("dog_payout"), 0.0)
        tier = str(row.get("tier", "C"))
        rank_score = _safe_float(row.get("rank_score"), 0.0)
        is_value_zone = bool(row.get("is_value_zone"))
        sharp_agrees = row.get("sharp_agrees")
        caution_flags = row.get("caution_flags", [])

        code = ""
        label = ""
        priority = 0
        if is_value_zone and tier == "A":
            code = "value_upset_tier_a"
            label = "Value-zone Tier A underdog"
            priority = 95
        elif confidence >= 75.0 and rank_score >= 80.0:
            code = "high_confidence_edge"
            label = "High-confidence underdog edge"
            priority = 88
        elif payout >= 3.0 and confidence >= 60.0:
            code = "long_dog_opportunity"
            label = "Long-dog opportunity"
            priority = 80
        elif tier in {"A", "B"} and rank_score >= 72.0:
            code = "ranked_signal"
            label = "Ranked underdog signal"
            priority = 74
        else:
            continue

        caution_labels = []
        if isinstance(caution_flags, list):
            for flag in caution_flags:
                if isinstance(flag, dict):
                    text = str(flag.get("label", "")).strip()
                    if text:
                        caution_labels.append(text)

        alerts.append(
            {
                "code": code,
                "label": label,
                "priority": priority,
                "game_date": row.get("game_date"),
                "home_team_id": row.get("home_team_id"),
                "away_team_id": row.get("away_team_id"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "pick": row.get("pick"),
                "tier": tier,
                "confidence": confidence,
                "dog_payout": payout,
                "rank_score": rank_score,
                "is_value_zone": is_value_zone,
                "sharp_agrees": sharp_agrees,
                "cautions": caution_labels[:3],
                "matchup_url": (
                    f"/matchup/{row.get('home_team_id')}/{row.get('away_team_id')}/{row.get('game_date')}"
                ),
            }
        )

    alerts.sort(key=_alert_priority, reverse=True)
    return alerts[: max(1, int(max_items))]


def build_underdog_alert_digest(
    alerts: Iterable[Mapping[str, Any]],
    total_candidates: int,
) -> Dict[str, Any]:
    rows = list(alerts)
    code_counts: Dict[str, int] = {}
    for row in rows:
        code = str(row.get("code", "")).strip()
        if not code:
            continue
        code_counts[code] = code_counts.get(code, 0) + 1

    return {
        "alert_count": len(rows),
        "total_candidates": int(total_candidates),
        "coverage_pct": (
            (len(rows) / max(1, int(total_candidates))) * 100.0
            if total_candidates
            else 0.0
        ),
        "code_counts": code_counts,
    }
