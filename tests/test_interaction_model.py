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
