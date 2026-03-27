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
