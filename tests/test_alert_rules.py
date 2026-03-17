from src.analytics.alert_rules import (
    build_underdog_alert_candidates,
    build_underdog_alert_digest,
)


def test_build_underdog_alert_candidates_prioritizes_high_signal_rows():
    rows = [
        {
            "home_team_id": 1,
            "away_team_id": 2,
            "home_team": "LAL",
            "away_team": "BOS",
            "game_date": "2025-03-01",
            "pick": "AWAY",
            "tier": "A",
            "confidence": 77.0,
            "dog_payout": 2.3,
            "rank_score": 86.0,
            "is_value_zone": True,
            "sharp_agrees": False,
            "caution_flags": [{"label": "Sharp money disagrees"}],
        },
        {
            "home_team_id": 3,
            "away_team_id": 4,
            "home_team": "MIA",
            "away_team": "NYK",
            "game_date": "2025-03-01",
            "pick": "HOME",
            "tier": "B",
            "confidence": 62.0,
            "dog_payout": 3.4,
            "rank_score": 74.0,
            "is_value_zone": False,
            "sharp_agrees": True,
            "caution_flags": [],
        },
    ]

    alerts = build_underdog_alert_candidates(rows, max_items=10)
    assert len(alerts) == 2
    assert alerts[0]["code"] == "value_upset_tier_a"
    assert alerts[0]["priority"] >= alerts[1]["priority"]
    assert alerts[0]["matchup_url"].startswith("/matchup/")


def test_build_underdog_alert_digest_counts_codes():
    alerts = [
        {"code": "value_upset_tier_a"},
        {"code": "ranked_signal"},
        {"code": "ranked_signal"},
    ]
    digest = build_underdog_alert_digest(alerts, total_candidates=12)
    assert digest["alert_count"] == 3
    assert digest["total_candidates"] == 12
    assert digest["code_counts"]["ranked_signal"] == 2
