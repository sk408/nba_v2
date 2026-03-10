from src.analytics.prediction import GameInput, predict
from src.analytics.weight_config import WeightConfig


def test_predict_home_edge_yields_home_pick():
    game = GameInput(
        home_team_id=1,
        away_team_id=2,
        game_date="2025-03-01",
        home_proj={"points": 118.0, "turnovers": 12.0, "rebounds": 45.0},
        away_proj={"points": 104.0, "turnovers": 15.0, "rebounds": 40.0},
        home_def_factor_raw=1.0,
        away_def_factor_raw=1.0,
        home_court=3.0,
    )

    pred = predict(game, WeightConfig(), include_sharp=False)
    assert pred.pick == "HOME"
    assert pred.game_score > 0
    assert pred.confidence > 0
