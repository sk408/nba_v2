import pytest

from src.analytics.backtester import _aggregate_from_per_game, _build_per_game_result
from src.analytics.prediction import GameInput, Prediction


def test_build_per_game_result_correct_upset_pick_and_profit():
    game = GameInput(
        game_date="2025-01-10",
        home_team_id=1,
        away_team_id=2,
        actual_home_score=102,
        actual_away_score=108,
        vegas_spread=-5.5,   # home favorite
        vegas_home_ml=-220,
        vegas_away_ml=180,
    )
    pred = Prediction(
        game_date=game.game_date,
        pick="AWAY",
        game_score=-4.0,
        confidence=63.0,
        vegas_spread=game.vegas_spread,
        vegas_home_ml=game.vegas_home_ml,
        vegas_away_ml=game.vegas_away_ml,
    )

    row = _build_per_game_result(
        game=game,
        pred=pred,
        abbr={1: "HOM", 2: "AWY"},
        competitive_dog_margin=7.5,
        long_dog_min_payout=3.0,
        long_dog_onepos_margin=3.0,
    )

    assert row["is_upset_pick"] is True
    assert row["upset_correct"] is True
    assert row["competitive_dog"] is True
    assert row["one_possession_dog"] is True
    assert row["long_dog_pick"] is False
    assert row["ml_payout"] == pytest.approx(2.8)
    assert row["ml_profit"] == pytest.approx(1.8)


def test_build_per_game_result_wrong_upset_pick_loses_unit():
    game = GameInput(
        game_date="2025-01-11",
        home_team_id=3,
        away_team_id=4,
        actual_home_score=99,
        actual_away_score=110,
        vegas_spread=4.0,    # away favorite
        vegas_home_ml=160,
        vegas_away_ml=-190,
    )
    pred = Prediction(
        game_date=game.game_date,
        pick="HOME",
        game_score=3.2,
        confidence=57.0,
        vegas_spread=game.vegas_spread,
        vegas_home_ml=game.vegas_home_ml,
        vegas_away_ml=game.vegas_away_ml,
    )

    row = _build_per_game_result(
        game=game,
        pred=pred,
        abbr={3: "H3", 4: "A4"},
        competitive_dog_margin=7.5,
        long_dog_min_payout=3.0,
        long_dog_onepos_margin=3.0,
    )

    assert row["is_upset_pick"] is True
    assert row["upset_correct"] is False
    assert row["competitive_dog"] is False
    assert row["one_possession_dog"] is False
    assert row["ml_payout"] == pytest.approx(2.6)
    assert row["ml_profit"] == pytest.approx(-1.0)


def test_aggregate_from_per_game_exposes_quality_metrics():
    games = [
        {
            "actual_winner": "AWAY",
            "model_correct": True,
            "vegas_spread": -5.5,
            "is_upset_pick": True,
            "upset_correct": True,
            "competitive_dog": True,
            "one_possession_dog": True,
            "long_dog_pick": False,
            "long_dog_onepos": False,
            "ml_payout": 2.8,
            "ml_profit": 1.8,
            "game_score": -4.0,
            "actual_home_score": 102,
            "actual_away_score": 108,
            "confidence": 63.0,
        },
        {
            "actual_winner": "HOME",
            "model_correct": True,
            "vegas_spread": -2.0,
            "is_upset_pick": False,
            "upset_correct": False,
            "competitive_dog": False,
            "one_possession_dog": False,
            "long_dog_pick": False,
            "long_dog_onepos": False,
            "ml_payout": 1.5,
            "ml_profit": 0.5,
            "game_score": 1.5,
            "actual_home_score": 104,
            "actual_away_score": 100,
            "confidence": 52.0,
        },
    ]

    agg = _aggregate_from_per_game(games)

    assert agg["total_games"] == 2
    assert agg["upset_count"] == 1
    assert agg["upset_coverage_pct"] == pytest.approx(50.0)
    tier_metrics = agg["upset_tier_metrics"]
    total_tier_count = (
        tier_metrics["A"]["count"] + tier_metrics["B"]["count"] + tier_metrics["C"]["count"]
    )
    assert total_tier_count == pytest.approx(1.0)
    assert agg["upset_quality_frontier"][-1]["hit_rate"] == pytest.approx(100.0)
    assert isinstance(agg["upset_roi_by_odds_band"], dict)
    assert "tier hit A/B/C=" in agg["hit_rate_quality_observation"]
