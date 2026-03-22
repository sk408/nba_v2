"""Parity tests: optimizer VectorizedGames vs backtester _build_per_game_result.

Verifies that upset_correct semantics match between the vectorized (numpy)
optimizer path and the scalar backtester path, especially for push games.
"""

import pytest

from src.analytics.backtester import _build_per_game_result
from src.analytics.optimizer import VectorizedGames
from src.analytics.prediction import GameInput, Prediction
from src.analytics.thresholds import PUSH_MODEL_EDGE_MAX
from src.analytics.weight_config import WeightConfig


def _backtester_upset_correct(game: GameInput, game_score: float) -> bool:
    """Run the scalar backtester path and return upset_correct."""
    pick = "HOME" if game_score > 0 else "AWAY"
    pred = Prediction(
        game_date=game.game_date,
        pick=pick,
        game_score=game_score,
        confidence=min(100.0, abs(game_score) / 15.0 * 100.0),
        vegas_spread=game.vegas_spread,
        vegas_home_ml=game.vegas_home_ml,
        vegas_away_ml=game.vegas_away_ml,
    )
    row = _build_per_game_result(
        game=game,
        pred=pred,
        abbr={game.home_team_id: "HOM", game.away_team_id: "AWY"},
        competitive_dog_margin=7.5,
        long_dog_min_payout=3.0,
        long_dog_onepos_margin=3.0,
    )
    return row["upset_correct"]


def _optimizer_upset_correct(game: GameInput, weights: WeightConfig) -> bool:
    """Run the vectorized optimizer path and return upset_correct for the single game."""
    vg = VectorizedGames([game])
    metrics = vg.evaluate(weights, include_sharp=False, fast=True)
    return metrics["upset_correct_count"] == 1


def _game(home_court, vegas_spread, home_ml, away_ml, h_score, a_score):
    """Shorthand: build a GameInput with defaults zeroed except the params we care about."""
    return GameInput(
        game_date="2025-03-01",
        home_team_id=1,
        away_team_id=2,
        home_court=home_court,
        vegas_spread=vegas_spread,
        vegas_home_ml=home_ml,
        vegas_away_ml=away_ml,
        actual_home_score=h_score,
        actual_away_score=a_score,
    )


class TestUpsetPushParity:
    """Ensure optimizer and backtester agree on upset_correct for push games."""

    @pytest.fixture()
    def weights(self):
        return WeightConfig()

    def test_push_upset_correct_when_edge_within_bound(self, weights):
        """Push game, upset pick, |game_score| ≤ PUSH_MODEL_EDGE_MAX → True."""
        game = _game(
            home_court=2.5,       # game_score ≈ +2.5 → picks HOME
            vegas_spread=3.0,     # away favored → home pick is upset
            home_ml=150,
            away_ml=-180,
            h_score=100,
            a_score=100,          # push (spread = 0)
        )
        gs = 2.5
        assert abs(gs) <= PUSH_MODEL_EDGE_MAX

        bt = _backtester_upset_correct(game, gs)
        opt = _optimizer_upset_correct(game, weights)
        assert bt is True, "backtester should count push-correct upset"
        assert opt is True, "optimizer should count push-correct upset"
        assert bt == opt, "optimizer and backtester must agree"

    def test_push_upset_wrong_when_edge_exceeds_bound(self, weights):
        """Push game, upset pick, |game_score| > PUSH_MODEL_EDGE_MAX → False."""
        game = _game(
            home_court=4.0,       # game_score ≈ +4.0 → picks HOME
            vegas_spread=3.0,     # away favored → home pick is upset
            home_ml=200,
            away_ml=-250,
            h_score=100,
            a_score=100,          # push (spread = 0)
        )
        gs = 4.0
        assert abs(gs) > PUSH_MODEL_EDGE_MAX

        bt = _backtester_upset_correct(game, gs)
        opt = _optimizer_upset_correct(game, weights)
        assert bt is False, "backtester rejects push with large edge"
        assert opt is False, "optimizer rejects push with large edge"
        assert bt == opt

    def test_outright_win_upset_correct(self, weights):
        """Outright upset win → True in both paths (sanity check)."""
        game = _game(
            home_court=2.5,       # game_score ≈ +2.5 → picks HOME
            vegas_spread=3.0,     # away favored → home pick is upset
            home_ml=150,
            away_ml=-180,
            h_score=105,
            a_score=100,          # home wins by 5
        )
        gs = 2.5

        bt = _backtester_upset_correct(game, gs)
        opt = _optimizer_upset_correct(game, weights)
        assert bt is True
        assert opt is True
        assert bt == opt

    def test_outright_loss_upset_wrong(self, weights):
        """Outright upset loss → False in both paths."""
        game = _game(
            home_court=2.5,       # game_score ≈ +2.5 → picks HOME
            vegas_spread=3.0,     # away favored → home pick is upset
            home_ml=150,
            away_ml=-180,
            h_score=95,
            a_score=105,          # away wins by 10
        )
        gs = 2.5

        bt = _backtester_upset_correct(game, gs)
        opt = _optimizer_upset_correct(game, weights)
        assert bt is False
        assert opt is False
        assert bt == opt

    def test_non_upset_pick_always_false(self, weights):
        """Model agrees with Vegas → not an upset, upset_correct = False."""
        game = _game(
            home_court=2.5,       # game_score ≈ +2.5 → picks HOME
            vegas_spread=-3.0,    # home favored → home pick is NOT upset
            home_ml=-180,
            away_ml=150,
            h_score=105,
            a_score=100,          # home wins (correct but not upset)
        )
        gs = 2.5

        bt = _backtester_upset_correct(game, gs)
        opt = _optimizer_upset_correct(game, weights)
        assert bt is False
        assert opt is False

    def test_aggregate_upset_counts_match(self, weights):
        """Multi-game batch: optimizer aggregate upset_correct_count matches backtester sum."""
        games = [
            _game(2.5, 3.0, 150, -180, 100, 100),   # push upset correct (edge 2.5)
            _game(4.0, 3.0, 200, -250, 100, 100),    # push upset wrong (edge 4.0)
            _game(2.5, 3.0, 150, -180, 105, 100),    # outright upset correct
            _game(2.5, 3.0, 150, -180, 95, 105),     # outright upset wrong
            _game(2.5, -3.0, -180, 150, 105, 100),   # not upset (agrees with vegas)
        ]
        game_scores = [2.5, 4.0, 2.5, 2.5, 2.5]

        bt_count = sum(
            _backtester_upset_correct(g, gs) for g, gs in zip(games, game_scores)
        )
        vg = VectorizedGames(games)
        metrics = vg.evaluate(weights, include_sharp=False, fast=True)
        opt_count = metrics["upset_correct_count"]

        assert bt_count == 2, "backtester: push-correct + outright-correct"
        assert opt_count == 2, "optimizer: push-correct + outright-correct"
        assert bt_count == opt_count
