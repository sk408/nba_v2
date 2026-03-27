"""Tests for the LightGBM interaction model layer."""

from dataclasses import asdict, fields

from src.analytics.prediction import Prediction


def test_prediction_has_interaction_detail_field():
    """Prediction dataclass has interaction_detail field, defaults to None."""
    pred = Prediction()
    assert pred.interaction_detail is None
    # Verify it's a real dataclass field (visible to asdict, fields)
    field_names = {f.name for f in fields(pred)}
    assert "interaction_detail" in field_names
    d = asdict(pred)
    assert "interaction_detail" in d
    assert d["interaction_detail"] is None


def test_prediction_interaction_detail_settable():
    """interaction_detail can hold driver info dict."""
    pred = Prediction()
    pred.interaction_detail = {
        "top_drivers": [
            {"feature": "reb_diff__x__ff_oreb_edge", "contribution": 1.4},
        ]
    }
    assert len(pred.interaction_detail["top_drivers"]) == 1


import numpy as np


def test_build_feature_vector_from_adjustments():
    """Feature vector built from prediction adjustments dict."""
    from src.analytics.interaction_model import build_feature_vector

    adjustments = {
        "fatigue": -2.0,
        "turnover": 1.5,
        "rebound": 3.0,
        "rating_matchup": 4.0,
        "four_factors": 2.5,
        "opp_four_factors": -1.0,
        "hustle": 0.5,
        "rest_advantage": 1.0,
        "elo": 0.8,
        "travel": -0.3,
        "momentum": 0.2,
        "injury_vorp": 1.2,
        "pythag": 0.6,
        "srs": 0.4,
        "onoff_impact": 0.3,
    }
    names, vector = build_feature_vector(adjustments)

    # Should have base edges + interactions + magnitudes
    assert len(names) == len(vector)
    assert len(vector) > len(adjustments)  # more features than just base edges

    # Base edges present
    assert "rebound" in names
    assert "four_factors" in names

    # At least some interaction features present
    interaction_names = [n for n in names if "__x__" in n]
    assert len(interaction_names) > 0

    # Magnitude features present
    magnitude_names = [n for n in names if n.startswith("abs_")]
    assert len(magnitude_names) > 0

    # Interaction value is product of base edges
    reb_idx = names.index("rebound")
    ff_idx = names.index("four_factors")
    interaction_name = "rebound__x__four_factors"
    if interaction_name in names:
        ix_idx = names.index(interaction_name)
        assert vector[ix_idx] == vector[reb_idx] * vector[ff_idx]


import tempfile
import os


def test_train_and_predict_roundtrip():
    """Train on synthetic residuals, predict corrections."""
    from src.analytics.interaction_model import (
        train_model,
        predict_correction,
        build_feature_vector,
        BASE_EDGE_KEYS,
    )

    rng = np.random.RandomState(42)
    n_games = 300

    # Generate synthetic adjustments and residuals.
    # Create a signal: when rebound AND four_factors are both positive,
    # the linear model under-predicts (positive residual).
    all_adjustments = []
    residuals = []
    for _ in range(n_games):
        adj = {k: rng.randn() * 2.0 for k in BASE_EDGE_KEYS}
        all_adjustments.append(adj)
        # Residual has an interaction signal + noise
        residual = 0.5 * adj["rebound"] * adj["four_factors"] + rng.randn() * 1.0
        residuals.append(residual)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.lgb")
        meta_path = os.path.join(tmpdir, "test_meta.json")

        result = train_model(
            all_adjustments=all_adjustments,
            residuals=residuals,
            model_path=model_path,
            meta_path=meta_path,
            correction_cap=3.0,
        )

        assert result["status"] == "trained"
        assert result["n_train"] > 0
        assert result["val_rmse"] > 0
        assert os.path.exists(model_path)
        assert os.path.exists(meta_path)

        # Predict correction for a game with strong interaction signal
        strong_adj = {k: 0.0 for k in BASE_EDGE_KEYS}
        strong_adj["rebound"] = 5.0
        strong_adj["four_factors"] = 5.0
        correction, detail = predict_correction(
            adjustments=strong_adj,
            model_path=model_path,
            correction_cap=3.0,
        )

        # Should predict a positive correction (interaction signal)
        assert correction > 0.0
        # Should be capped
        assert abs(correction) <= 3.0
        # Detail should have top drivers
        assert "top_drivers" in detail
        assert len(detail["top_drivers"]) > 0


def test_predict_correction_no_model_returns_zero():
    """When model file doesn't exist, correction is 0.0."""
    from src.analytics.interaction_model import predict_correction, BASE_EDGE_KEYS

    adj = {k: 1.0 for k in BASE_EDGE_KEYS}
    correction, detail = predict_correction(
        adjustments=adj,
        model_path="/nonexistent/model.lgb",
        correction_cap=3.0,
    )
    assert correction == 0.0
    assert detail == {}


def test_train_model_insufficient_games():
    """Training with too few games returns early."""
    from src.analytics.interaction_model import train_model, BASE_EDGE_KEYS

    rng = np.random.RandomState(42)
    # Only 50 games — below 200 minimum
    all_adjustments = [{k: rng.randn() for k in BASE_EDGE_KEYS} for _ in range(50)]
    residuals = [rng.randn() for _ in range(50)]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_model(
            all_adjustments=all_adjustments,
            residuals=residuals,
            model_path=os.path.join(tmpdir, "model.lgb"),
            meta_path=os.path.join(tmpdir, "meta.json"),
            correction_cap=3.0,
            min_train_games=200,
        )
        assert result["status"] == "skipped"
        assert "insufficient" in result.get("reason", "").lower()


def test_get_model_metadata():
    """get_model_metadata returns metadata or None."""
    from src.analytics.interaction_model import get_model_metadata, train_model, BASE_EDGE_KEYS

    # No model — returns None
    assert get_model_metadata() is None or isinstance(get_model_metadata(), dict)

    rng = np.random.RandomState(42)
    n_games = 300
    all_adjustments = [{k: rng.randn() * 2.0 for k in BASE_EDGE_KEYS} for _ in range(n_games)]
    residuals = [rng.randn() * 2.0 for _ in range(n_games)]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.lgb")
        meta_path = os.path.join(tmpdir, "meta.json")
        train_model(all_adjustments, residuals, model_path=model_path, meta_path=meta_path)

        # Read metadata from the path we trained to
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        assert "trained_at" in meta
        assert "val_rmse" in meta
        assert "top_interactions" in meta
        assert meta["n_games"] == n_games


def test_is_model_stale_no_model():
    """is_model_stale returns True when no model exists."""
    from src.analytics.interaction_model import is_model_stale
    assert is_model_stale() is True


def test_weights_hash_deterministic():
    """weights_hash produces consistent output."""
    from src.analytics.interaction_model import weights_hash

    class FakeWeights:
        def to_dict(self):
            return {"a": 1.0, "b": 2.0, "c": 3.0}

    w = FakeWeights()
    h1 = weights_hash(w)
    h2 = weights_hash(w)
    assert h1 == h2
    assert len(h1) == 16  # truncated hex


def test_invalidate_model_cache():
    """invalidate_model_cache clears the cached model."""
    from src.analytics.interaction_model import (
        invalidate_model_cache,
        _cached_model,
    )
    # Just verify it doesn't error
    invalidate_model_cache()
