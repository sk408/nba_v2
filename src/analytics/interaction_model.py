"""LightGBM interaction model — residual correction layer.

Learns feature interactions and non-linear scaling that the linear
prediction model misses. Trains on residuals (actual_margin - game_score_linear).
Correction is capped and applied after the linear score.

The model is excluded from the optimizer's evaluate() loop. It only runs
in live predict() and backtesting.
"""

import hashlib
import json
import logging
import os
import threading
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_MODEL_PATH = os.path.join(_MODEL_DIR, "interaction_model.lgb")
_META_PATH = os.path.join(_MODEL_DIR, "interaction_model_meta.json")

# Base edge keys used for interaction feature generation.
# These are the adjustment keys from predict() that represent meaningful
# matchup edges. Order matters — first N are used for pairwise interactions.
BASE_EDGE_KEYS = [
    "fatigue", "turnover", "rebound", "rating_matchup",
    "four_factors", "opp_four_factors", "hustle", "rest_advantage",
    "elo", "travel", "timezone", "cum_travel_7d",
    "momentum", "mov_trend", "injury_vorp", "ref_home_bias",
    "schedule_spots", "road_trip_game", "season_progress",
    "roster_shock", "srs", "pythag", "onoff_impact",
    "fg3_luck", "process_stats", "sharp_ml", "sharp_spread",
    "clutch", "altitude_b2b",
]

# Top edges used for pairwise interaction generation (subset of BASE_EDGE_KEYS).
# These are the features most likely to have meaningful interactions.
INTERACTION_EDGE_KEYS = [
    "rebound", "four_factors", "opp_four_factors", "rating_matchup",
    "fatigue", "turnover", "elo", "injury_vorp",
    "travel", "hustle", "momentum", "pythag",
    "srs", "onoff_impact", "fg3_luck",
]

# Edges that get magnitude (abs) features for non-linear scaling.
MAGNITUDE_EDGE_KEYS = [
    "rebound", "four_factors", "opp_four_factors", "rating_matchup",
    "fatigue", "turnover", "elo", "injury_vorp",
    "travel", "hustle", "momentum", "pythag",
    "srs", "onoff_impact", "fg3_luck",
]


def build_feature_vector(
    adjustments: Dict[str, float],
) -> Tuple[List[str], np.ndarray]:
    """Build the full feature vector from prediction adjustments.

    Returns (feature_names, feature_values) where feature_values is a 1-D
    NumPy array. Three categories of features:
      1. Base edges — raw adjustment values
      2. Interactions — pairwise products of top edges
      3. Magnitudes — abs() of key edges for non-linear scaling

    Args:
        adjustments: The pred.adjustments dict from predict().

    Returns:
        Tuple of (list of feature name strings, numpy array of values).
    """
    names: List[str] = []
    values: List[float] = []

    # 1. Base edges
    for key in BASE_EDGE_KEYS:
        names.append(key)
        values.append(adjustments.get(key, 0.0))

    # 2. Pairwise interaction features
    for a, b in combinations(INTERACTION_EDGE_KEYS, 2):
        name = f"{a}__x__{b}"
        val_a = adjustments.get(a, 0.0)
        val_b = adjustments.get(b, 0.0)
        names.append(name)
        values.append(val_a * val_b)

    # 3. Magnitude context features
    for key in MAGNITUDE_EDGE_KEYS:
        names.append(f"abs_{key}")
        values.append(abs(adjustments.get(key, 0.0)))

    return names, np.array(values, dtype=np.float64)
