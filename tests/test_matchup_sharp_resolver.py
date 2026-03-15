from src.ui.views.matchup_sharp import (
    hydrate_scan_sharp_data,
    resolve_sharp_panel_values,
)


def test_uses_prediction_sharp_values_without_fetching():
    live_calls = {"count": 0}
    db_calls = {"count": 0}

    def _live(_home, _away):
        live_calls["count"] += 1
        return {}

    def _db(_date, _home_id, _away_id):
        db_calls["count"] += 1
        return {}

    result = {
        "pick": "HOME",
        "ml_sharp_home_public": 49,
        "ml_sharp_home_money": 61,
        "sharp_agrees": True,
    }
    game_data = {
        "game_date": "2026-03-14",
        "home_team": "LAL",
        "away_team": "DEN",
        "home_team_id": 1610612747,
        "away_team_id": 1610612743,
    }

    ml_pub, ml_mon, sharp_agrees = resolve_sharp_panel_values(
        result=result,
        game_data=game_data,
        fetch_live_odds=_live,
        fetch_db_odds=_db,
    )

    assert (ml_pub, ml_mon, sharp_agrees) == (49, 61, True)
    assert live_calls["count"] == 0
    assert db_calls["count"] == 0


def test_falls_back_to_live_when_prediction_has_no_sharp_data():
    def _live(_home, _away):
        return {"ml_home_public": 46, "ml_home_money": 58}

    def _db(_date, _home_id, _away_id):
        return {}

    result = {
        "pick": "HOME",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_agrees": None,
    }
    game_data = {
        "game_date": "2026-03-14",
        "home_team": "LAL",
        "away_team": "DEN",
        "home_team_id": 1610612747,
        "away_team_id": 1610612743,
    }

    ml_pub, ml_mon, sharp_agrees = resolve_sharp_panel_values(
        result=result,
        game_data=game_data,
        fetch_live_odds=_live,
        fetch_db_odds=_db,
    )

    assert ml_pub == 46
    assert ml_mon == 58
    assert sharp_agrees is True


def test_falls_back_to_db_when_live_missing():
    def _live(_home, _away):
        return {}

    def _db(_date, _home_id, _away_id):
        return {"ml_home_public": 44, "ml_home_money": 40}

    result = {
        "pick": "HOME",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_agrees": None,
    }
    game_data = {
        "game_date": "2026-03-14",
        "home_team": "LAL",
        "away_team": "DEN",
        "home_team_id": 1610612747,
        "away_team_id": 1610612743,
    }

    ml_pub, ml_mon, sharp_agrees = resolve_sharp_panel_values(
        result=result,
        game_data=game_data,
        fetch_live_odds=_live,
        fetch_db_odds=_db,
    )

    assert ml_pub == 44
    assert ml_mon == 40
    assert sharp_agrees is False


def test_returns_no_data_when_all_sources_empty():
    def _live(_home, _away):
        return {}

    def _db(_date, _home_id, _away_id):
        return {}

    result = {
        "pick": "AWAY",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_agrees": None,
    }
    game_data = {
        "game_date": "2026-03-14",
        "home_team": "LAL",
        "away_team": "DEN",
        "home_team_id": 1610612747,
        "away_team_id": 1610612743,
    }

    ml_pub, ml_mon, sharp_agrees = resolve_sharp_panel_values(
        result=result,
        game_data=game_data,
        fetch_live_odds=_live,
        fetch_db_odds=_db,
    )

    assert ml_pub == 0
    assert ml_mon == 0
    assert sharp_agrees is None


def test_scan_hydration_populates_both_modes_with_single_fallback_fetch():
    calls = {"live": 0, "db": 0}

    def _live(_home, _away):
        calls["live"] += 1
        return {"ml_home_public": 45, "ml_home_money": 57}

    def _db(_date, _home_id, _away_id):
        calls["db"] += 1
        return {}

    game_data = {
        "game_date": "2026-03-14",
        "home_team": "LAL",
        "away_team": "DEN",
        "home_team_id": 1610612747,
        "away_team_id": 1610612743,
    }
    fund = {
        "pick": "HOME",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_agrees": None,
    }
    sharp = {
        "pick": "AWAY",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_agrees": None,
    }

    out_fund, out_sharp = hydrate_scan_sharp_data(
        pred_fund=fund,
        pred_sharp=sharp,
        game_data=game_data,
        fetch_live_odds=_live,
        fetch_db_odds=_db,
    )

    assert out_fund["ml_sharp_home_public"] == 45
    assert out_fund["ml_sharp_home_money"] == 57
    assert out_fund["sharp_agrees"] is True
    assert out_fund["sharp_resolved"] is True
    assert out_sharp["ml_sharp_home_public"] == 45
    assert out_sharp["ml_sharp_home_money"] == 57
    assert out_sharp["sharp_agrees"] is False
    assert out_sharp["sharp_resolved"] is True
    assert calls["live"] == 1
    assert calls["db"] == 0


def test_scan_hydration_sets_resolved_flag_even_when_no_data():
    """When all sources return empty, sharp_resolved must still be True
    so the cache invalidation loop is broken."""
    def _live(_home, _away):
        return {}

    def _db(_date, _home_id, _away_id):
        return {}

    game_data = {
        "game_date": "2026-03-14",
        "home_team": "LAL",
        "away_team": "DEN",
        "home_team_id": 1610612747,
        "away_team_id": 1610612743,
    }
    fund = {
        "pick": "HOME",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_agrees": None,
    }
    sharp = {
        "pick": "AWAY",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_agrees": None,
    }

    out_fund, out_sharp = hydrate_scan_sharp_data(
        pred_fund=fund,
        pred_sharp=sharp,
        game_data=game_data,
        fetch_live_odds=_live,
        fetch_db_odds=_db,
    )

    # Values stay zero but resolved flag is set
    assert out_fund["ml_sharp_home_public"] == 0
    assert out_fund["ml_sharp_home_money"] == 0
    assert out_fund["sharp_resolved"] is True
    assert out_sharp["sharp_resolved"] is True


def test_live_fallback_returns_none_for_ml_uses_zero():
    """ActionNetwork may return None for ML fields while having spread data.
    The resolver must not crash and must return 0."""
    def _live(_home, _away):
        return {
            "spread_home_public": 48,
            "spread_home_money": 55,
            "ml_home_public": None,
            "ml_home_money": None,
        }

    def _db(_date, _home_id, _away_id):
        return {}

    result = {
        "pick": "HOME",
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
    }
    game_data = {
        "game_date": "2026-03-14",
        "home_team": "LAL",
        "away_team": "DEN",
        "home_team_id": 1610612747,
        "away_team_id": 1610612743,
    }

    ml_pub, ml_mon, sharp_agrees = resolve_sharp_panel_values(
        result=result,
        game_data=game_data,
        fetch_live_odds=_live,
        fetch_db_odds=_db,
    )

    assert ml_pub == 0
    assert ml_mon == 0
    assert sharp_agrees is None
