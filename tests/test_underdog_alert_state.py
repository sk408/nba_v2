from src.analytics.underdog_alert_state import update_underdog_alert_state


def _alert(signal_key, code, home="LAL", away="BOS", pick="AWAY"):
    return {
        "signal_key": signal_key,
        "code": code,
        "label": code.replace("_", " ").title(),
        "home_team": home,
        "away_team": away,
        "pick": pick,
        "confidence": 72.0,
        "dog_payout": 2.3,
    }


def test_update_underdog_alert_state_detects_new_and_resolved(isolated_db):
    scope_key = "2025-03-01:scope-a"

    run_one = update_underdog_alert_state(
        scope_key=scope_key,
        game_date="2025-03-01",
        alerts=[
            _alert("k1", "A_TIER_UPSET"),
            _alert("k2", "VALUE_ZONE_EDGE"),
        ],
        total_candidates=7,
        digest={"alert_count": 2},
    )
    assert run_one["new_count"] == 2
    assert run_one["resolved_count"] == 0
    assert run_one["active_count"] == 2

    run_two = update_underdog_alert_state(
        scope_key=scope_key,
        game_date="2025-03-01",
        alerts=[
            _alert("k2", "VALUE_ZONE_EDGE"),
            _alert("k3", "LONG_DOG_LIVE"),
        ],
        total_candidates=8,
        digest={"alert_count": 2},
    )
    assert run_two["new_count"] == 1
    assert run_two["resolved_count"] == 1
    assert run_two["persisting_count"] == 1
    assert run_two["active_count"] == 2
    assert run_two["new_alerts"][0]["signal_key"] == "k3"
    assert run_two["resolved_alerts"][0]["signal_key"] == "k1"


def test_update_underdog_alert_state_is_scope_isolated(isolated_db):
    update_underdog_alert_state(
        scope_key="2025-03-01:scope-a",
        game_date="2025-03-01",
        alerts=[_alert("same-key", "A_TIER_UPSET")],
        total_candidates=5,
        digest={"alert_count": 1},
    )
    run_other_scope = update_underdog_alert_state(
        scope_key="2025-03-01:scope-b",
        game_date="2025-03-01",
        alerts=[_alert("same-key", "A_TIER_UPSET")],
        total_candidates=5,
        digest={"alert_count": 1},
    )
    assert run_other_scope["new_count"] == 1
    assert run_other_scope["resolved_count"] == 0
