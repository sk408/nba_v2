import pytest

from src.analytics import underdog_ml_scorer
from src.analytics.optimizer import _build_rolling_time_folds, _rolling_cv_objective_loss
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
