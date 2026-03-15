from src.ui.views.matchup_cache import (
    prediction_cache_key,
    sanitize_prediction_for_cache,
    should_use_cached_prediction,
)


def test_prediction_cache_key_includes_game_date():
    key_today = prediction_cache_key(1610612747, 1610612743, "2026-03-14")
    key_next = prediction_cache_key(1610612747, 1610612743, "2026-03-16")
    assert key_today != key_next


def test_should_not_use_today_cache_when_sharp_data_missing_and_unresolved():
    cached_pred = {
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
    }
    assert (
        should_use_cached_prediction(
            cached_pred,
            game_date="2026-03-14",
            today="2026-03-14",
        )
        is False
    )


def test_should_use_today_cache_when_sharp_resolved_even_if_zeros():
    """Once the fallback chain has been exhausted (sharp_resolved=True),
    the cache entry should be accepted so we don't re-predict forever."""
    cached_pred = {
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_resolved": True,
    }
    assert (
        should_use_cached_prediction(
            cached_pred,
            game_date="2026-03-14",
            today="2026-03-14",
        )
        is True
    )


def test_should_use_today_cache_when_sharp_needs_refresh_flagged():
    cached_pred = {
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
        "sharp_resolved": False,
        "sharp_needs_refresh": True,
    }
    assert (
        should_use_cached_prediction(
            cached_pred,
            game_date="2026-03-14",
            today="2026-03-14",
        )
        is True
    )


def test_should_use_today_cache_when_sharp_data_present():
    cached_pred = {
        "ml_sharp_home_public": 54,
        "ml_sharp_home_money": 63,
    }
    assert (
        should_use_cached_prediction(
            cached_pred,
            game_date="2026-03-14",
            today="2026-03-14",
        )
        is True
    )


def test_should_allow_historical_cache_without_sharp_data():
    cached_pred = {
        "ml_sharp_home_public": 0,
        "ml_sharp_home_money": 0,
    }
    assert (
        should_use_cached_prediction(
            cached_pred,
            game_date="2026-03-01",
            today="2026-03-14",
        )
        is True
    )


def test_sanitize_prediction_for_cache_marks_live_entries_for_refresh():
    pred = {
        "ml_sharp_home_public": 54,
        "ml_sharp_home_money": 63,
        "sharp_agrees": True,
        "sharp_resolved": True,
    }

    out = sanitize_prediction_for_cache(
        pred,
        game_date="2026-03-14",
        today="2026-03-14",
    )

    assert pred["ml_sharp_home_public"] == 54
    assert pred["ml_sharp_home_money"] == 63
    assert out["ml_sharp_home_public"] == 0
    assert out["ml_sharp_home_money"] == 0
    assert out["sharp_resolved"] is False
    assert out["sharp_needs_refresh"] is True
    assert "sharp_agrees" not in out


def test_sanitize_prediction_for_cache_keeps_historical_entries():
    pred = {
        "ml_sharp_home_public": 48,
        "ml_sharp_home_money": 56,
        "sharp_resolved": True,
    }

    out = sanitize_prediction_for_cache(
        pred,
        game_date="2026-03-01",
        today="2026-03-14",
    )

    assert out["ml_sharp_home_public"] == 48
    assert out["ml_sharp_home_money"] == 56
    assert out["sharp_resolved"] is True
    assert "sharp_needs_refresh" not in out
