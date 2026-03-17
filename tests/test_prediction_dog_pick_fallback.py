from src.analytics.prediction import _classify_dog_pick


def test_classify_dog_pick_prefers_moneyline_over_spread_when_both_present():
    # Conflict case: spread says home dog, but moneyline says home favorite.
    # Moneyline should drive the upset classification.
    is_dog, is_value, payout = _classify_dog_pick(
        game_score=2.2,
        vegas_spread=5.0,
        vegas_home_ml=-130,
        vegas_away_ml=110,
    )
    assert is_dog is False
    assert is_value is True
    assert payout == 0.0


def test_classify_dog_pick_uses_moneyline_when_spread_missing():
    # Spread unavailable; infer favorite from moneyline (away favored).
    is_dog, is_value, payout = _classify_dog_pick(
        game_score=1.8,
        vegas_spread=0.0,
        vegas_home_ml=130,
        vegas_away_ml=-150,
    )
    assert is_dog is True
    assert is_value is False
    assert round(payout, 3) == 2.3


def test_classify_dog_pick_falls_back_to_spread_without_moneyline():
    is_dog, is_value, payout = _classify_dog_pick(
        game_score=2.0,
        vegas_spread=5.0,
        vegas_home_ml=0,
        vegas_away_ml=0,
    )
    assert is_dog is True
    assert is_value is True
    assert payout == 0.0


def test_classify_dog_pick_returns_false_without_market_direction():
    is_dog, is_value, payout = _classify_dog_pick(
        game_score=3.1,
        vegas_spread=0.0,
        vegas_home_ml=0,
        vegas_away_ml=0,
    )
    assert is_dog is False
    assert is_value is False
    assert payout == 0.0
