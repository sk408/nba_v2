import pytest

from src.analytics import optimizer as optimizer_module
from src.analytics import underdog_ml_scorer
from src.analytics.optimizer import (
    VectorizedGames,
    _build_rolling_time_folds,
    _compose_objective_loss,
    _normalize_objective_track,
    _passes_robust_save_gate,
    _rolling_cv_objective_loss,
    _weighted_onepos_credit_count,
)
from src.analytics.prediction import GameInput
from src.analytics.weight_config import WeightConfig


def test_build_rolling_time_folds_returns_chronological_windows():
    folds = _build_rolling_time_folds(
        n_games=1200,
        fold_count=4,
        min_train_games=360,
        val_games=160,
    )

    assert len(folds) == 4
    prev_train_end = 0
    for train_end, val_end in folds:
        assert train_end > prev_train_end
        assert val_end - train_end == 160
        assert train_end >= 360
        assert val_end <= 1200
        prev_train_end = train_end


def test_rolling_cv_objective_loss_penalizes_worst_fold():
    score, mean_loss, worst_loss = _rolling_cv_objective_loss(
        [2.0, 2.2, 3.1],
        worst_fold_mult=0.5,
    )

    assert mean_loss == pytest.approx(2.433333333333333)
    assert worst_loss == pytest.approx(3.1)
    # score = mean + (worst - mean) * mult
    assert score == pytest.approx(2.7666666666666666)


def test_normalize_objective_track_handles_aliases_and_invalid_values():
    assert _normalize_objective_track("live") == "live"
    assert _normalize_objective_track("oracle") == "oracle"
    assert _normalize_objective_track("dual_track") == "dual_track"
    assert _normalize_objective_track("dual") == "dual_track"
    assert _normalize_objective_track("both") == "dual_track"
    assert _normalize_objective_track("weird") == "dual_track"


def test_compose_objective_loss_uses_requested_track():
    live_loss = 2.0
    oracle_loss = 5.0

    assert _compose_objective_loss(live_loss, oracle_loss, "live", 0.7) == pytest.approx(2.0)
    assert _compose_objective_loss(live_loss, oracle_loss, "oracle", 0.7) == pytest.approx(5.0)
    assert _compose_objective_loss(live_loss, oracle_loss, "dual_track", 0.7) == pytest.approx(2.9)


def test_compare_walk_forward_underdog_scorer_passes_with_required_brier_lift(monkeypatch):
    baseline_w = WeightConfig()
    candidate_w = WeightConfig.from_dict(baseline_w.to_dict())

    def fake_evaluate(
        train_games,
        val_games,
        weights,
        *,
        include_sharp,
        min_train_samples,
        min_val_samples,
        learning_rate,
        l2,
        max_iter,
    ):
        if weights is baseline_w:
            return {
                "valid": True,
                "reason": "ok",
                "brier": 0.2200,
                "logloss": 0.6400,
            }
        return {
            "valid": True,
            "reason": "ok",
            "brier": 0.2050,
            "logloss": 0.6200,
        }

    monkeypatch.setattr(
        underdog_ml_scorer,
        "evaluate_walk_forward_underdog_scorer",
        fake_evaluate,
    )

    result = underdog_ml_scorer.compare_walk_forward_underdog_scorer(
        train_games=[],
        val_games=[],
        baseline_weights=baseline_w,
        candidate_weights=candidate_w,
        include_sharp=False,
        min_train_samples=100,
        min_val_samples=50,
        learning_rate=0.05,
        l2=1.0,
        max_iter=200,
        min_brier_improvement=0.0100,
    )

    assert result["applied"] is True
    assert result["passed"] is True
    assert result["brier_lift"] == pytest.approx(0.015)


def test_compare_walk_forward_underdog_scorer_fails_when_brier_lift_too_small(monkeypatch):
    baseline_w = WeightConfig()
    candidate_w = WeightConfig.from_dict(baseline_w.to_dict())

    def fake_evaluate(
        train_games,
        val_games,
        weights,
        *,
        include_sharp,
        min_train_samples,
        min_val_samples,
        learning_rate,
        l2,
        max_iter,
    ):
        if weights is baseline_w:
            return {
                "valid": True,
                "reason": "ok",
                "brier": 0.2120,
                "logloss": 0.6100,
            }
        return {
            "valid": True,
            "reason": "ok",
            "brier": 0.2085,
            "logloss": 0.6080,
        }

    monkeypatch.setattr(
        underdog_ml_scorer,
        "evaluate_walk_forward_underdog_scorer",
        fake_evaluate,
    )

    result = underdog_ml_scorer.compare_walk_forward_underdog_scorer(
        train_games=[],
        val_games=[],
        baseline_weights=baseline_w,
        candidate_weights=candidate_w,
        include_sharp=False,
        min_train_samples=100,
        min_val_samples=50,
        learning_rate=0.05,
        l2=1.0,
        max_iter=200,
        min_brier_improvement=0.0050,
    )

    assert result["applied"] is True
    assert result["passed"] is False
    assert "ml underdog gate failed" in result["reason"]


def test_weighted_onepos_credit_count_prefers_long_dog_weight():
    weighted = _weighted_onepos_credit_count(
        all_dog_near_miss_count=6,
        long_dog_near_miss_count=2,
        all_dogs_weight=0.5,
        long_dogs_weight=1.0,
    )
    # 4 non-long near misses @0.5 + 2 long near misses @1.0
    assert weighted == pytest.approx(4.0)

    clipped = _weighted_onepos_credit_count(
        all_dog_near_miss_count=3,
        long_dog_near_miss_count=9,
        all_dogs_weight=0.5,
        long_dogs_weight=1.0,
    )
    assert clipped == pytest.approx(3.0)


def test_onepos_credit_affects_winner_pct_toggle(monkeypatch):
    float_overrides = {
        "optimizer_onepos_credit_margin": 3.0,
        "optimizer_onepos_credit_all_dogs_weight": 0.5,
        "optimizer_onepos_credit_long_dogs_weight": 1.0,
        "optimizer_save_long_dog_min_payout": 3.0,
    }
    bool_overrides = {
        "optimizer_onepos_credit_enabled": True,
        "optimizer_onepos_credit_affects_winner_pct": True,
    }

    monkeypatch.setattr(
        optimizer_module,
        "_safe_float_setting",
        lambda key, default: float_overrides.get(key, default),
    )
    monkeypatch.setattr(
        optimizer_module,
        "_safe_bool_setting",
        lambda key, default: bool_overrides.get(key, default),
    )
    monkeypatch.setattr(
        optimizer_module,
        "_safe_int_setting",
        lambda key, default: int(default),
    )

    games = [
        GameInput(
            home_court=8.0,
            vegas_spread=6.0,  # away favored -> model home pick is underdog
            vegas_home_ml=250,
            vegas_away_ml=-300,
            actual_home_score=104,
            actual_away_score=100,  # upset win
        ),
        GameInput(
            home_court=8.0,
            vegas_spread=6.0,  # away favored -> model home pick is underdog
            vegas_home_ml=250,
            vegas_away_ml=-300,
            actual_home_score=100,
            actual_away_score=102,  # near-miss loss (2 points)
        ),
    ]
    vg = VectorizedGames(games)
    metrics = vg.evaluate(WeightConfig(), include_sharp=False, fast=True)

    assert metrics["upset_count"] == 2
    assert metrics["upset_correct_count"] == 1
    assert metrics["winner_pct_raw"] == pytest.approx(50.0)
    assert metrics["winner_pct_credit"] == pytest.approx(100.0)
    assert metrics["winner_pct"] == pytest.approx(100.0)
    assert metrics["upset_accuracy_raw"] == pytest.approx(50.0)
    assert metrics["upset_accuracy_credit"] == pytest.approx(100.0)
    assert metrics["onepos_credit_weighted_count"] == pytest.approx(1.0)

    bool_overrides["optimizer_onepos_credit_affects_winner_pct"] = False
    vg_no_winner = VectorizedGames(games)
    metrics_no_winner = vg_no_winner.evaluate(WeightConfig(), include_sharp=False, fast=True)

    assert metrics_no_winner["winner_pct_raw"] == pytest.approx(50.0)
    assert metrics_no_winner["winner_pct_credit"] == pytest.approx(100.0)
    assert metrics_no_winner["winner_pct"] == pytest.approx(50.0)
    assert metrics_no_winner["upset_accuracy"] == pytest.approx(100.0)


def test_save_gate_onepos_credit_bump_can_unlock_borderline_candidate(monkeypatch):
    float_overrides = {
        "optimizer_save_loss_margin": 0.01,
        "optimizer_save_max_winner_drop": 0.35,
        "optimizer_save_favorites_slack": 0.25,
        "optimizer_save_compression_floor": 0.55,
        "optimizer_save_onepos_credit_bump_mult": 0.0,
    }
    bool_overrides = {
        "optimizer_save_use_roi_gate": False,
        "optimizer_save_use_hybrid_loss_gate": False,
        "optimizer_save_use_long_dog_tiebreak_gate": False,
    }

    monkeypatch.setattr(
        optimizer_module,
        "_safe_float_setting",
        lambda key, default: float_overrides.get(key, default),
    )
    monkeypatch.setattr(
        optimizer_module,
        "_safe_bool_setting",
        lambda key, default: bool_overrides.get(key, default),
    )
    monkeypatch.setattr(
        optimizer_module,
        "_safe_int_setting",
        lambda key, default: int(default),
    )

    baseline = {
        "loss": 1.00,
        "winner_pct": 70.0,
        "winner_pct_raw": 70.0,
        "winner_pct_credit": 70.0,
        "winner_pct_credit_delta": 0.0,
        "ml_roi": 0.0,
        "long_dog_count": 60,
        "long_dog_onepos_count": 24,
        "long_dog_onepos_rate": 40.0,
    }
    candidate = {
        "loss": 0.98,  # clears loss improve margin
        "winner_pct": 68.75,  # borderline fail without bump (70 - 0.35 = 69.65)
        "winner_pct_raw": 68.75,
        "winner_pct_credit": 75.0,
        "winner_pct_credit_delta": 6.25,
        "favorites_pct": 69.0,
        "compression_ratio": 1.0,
        "upset_count": 5,  # keeps upset gate in "sample too small -> do not block" mode
        "upset_rate": 5.0,
        "upset_correct_count": 2,
        "ml_bet_count": 0,
        "ml_roi": 0.0,
        "ml_roi_lb95": 0.0,
        "long_dog_count": 60,
        "long_dog_onepos_count": 24,
        "long_dog_onepos_rate": 40.0,
    }

    save_ok_no_bump, _, details_no_bump = _passes_robust_save_gate(
        baseline=baseline,
        candidate=candidate,
        n_validation_games=500,
    )
    assert save_ok_no_bump is False
    assert details_no_bump["winner_guard"] is False
    assert details_no_bump["onepos_credit_bump"] == pytest.approx(0.0)

    float_overrides["optimizer_save_onepos_credit_bump_mult"] = 0.2
    save_ok_with_bump, _, details_with_bump = _passes_robust_save_gate(
        baseline=baseline,
        candidate=candidate,
        n_validation_games=500,
    )
    assert save_ok_with_bump is True
    assert details_with_bump["winner_guard"] is True
    assert details_with_bump["onepos_credit_lift"] == pytest.approx(6.25)
    assert details_with_bump["onepos_credit_bump"] > 0.0
