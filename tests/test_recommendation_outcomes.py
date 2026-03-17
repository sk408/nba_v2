from src.analytics.recommendation_outcomes import (
    backfill_recommendation_outcomes,
    persist_recommendation_snapshot,
)


def _seed_actual_game(isolated_db):
    isolated_db.execute(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?, ?, ?, ?)",
        (10, "Home Team", "HOM", "East"),
    )
    isolated_db.execute(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?, ?, ?, ?)",
        (20, "Away Team", "AWY", "West"),
    )
    for idx in range(1, 5):
        isolated_db.execute(
            "INSERT INTO players (player_id, name, team_id, position) VALUES (?, ?, ?, ?)",
            (100 + idx, f"Home {idx}", 10, "G"),
        )
        isolated_db.execute(
            "INSERT INTO players (player_id, name, team_id, position) VALUES (?, ?, ?, ?)",
            (200 + idx, f"Away {idx}", 20, "G"),
        )

    # Home side total = 104
    for idx in range(1, 5):
        isolated_db.execute(
            """
            INSERT INTO player_stats (
                player_id, opponent_team_id, is_home, game_date, game_id, season,
                points, rebounds, assists, minutes
            )
            VALUES (?, ?, 1, '2025-03-01', 'GAME-1', '2024-25', ?, 5, 5, 30)
            """,
            (100 + idx, 20, 26),
        )

    # Away side total = 88
    for idx in range(1, 5):
        isolated_db.execute(
            """
            INSERT INTO player_stats (
                player_id, opponent_team_id, is_home, game_date, game_id, season,
                points, rebounds, assists, minutes
            )
            VALUES (?, ?, 0, '2025-03-01', 'GAME-1', '2024-25', ?, 5, 5, 30)
            """,
            (200 + idx, 10, 22),
        )

    isolated_db.execute(
        """
        INSERT INTO game_odds (
            game_date, home_team_id, away_team_id, spread,
            home_moneyline, away_moneyline
        )
        VALUES ('2025-03-01', 10, 20, -4.5, -140, 120)
        """
    )


def test_persist_and_backfill_recommendation_outcomes(isolated_db):
    _seed_actual_game(isolated_db)

    persisted = persist_recommendation_snapshot(
        game_date="2025-03-01",
        scope_key="scope:abc",
        filters={"tier": "ALL"},
        rows=[
            {
                "signal_key": "sig-1",
                "game_date": "2025-03-01",
                "home_team_id": 10,
                "away_team_id": 20,
                "pick": "HOME",
                "tier": "A",
                "confidence": 72.0,
                "game_score": 5.0,
                "rank_score": 88.2,
                "dog_payout": 2.2,
                "vegas_spread": -4.5,
                "vegas_home_ml": -140,
                "vegas_away_ml": 120,
                "is_dog_pick": True,
                "is_value_zone": True,
            }
        ],
        total_candidates=1,
        summary={"count": 1},
        alert_digest={"alert_count": 1},
    )
    assert persisted["stored_count"] == 1

    settled = backfill_recommendation_outcomes(game_date="2025-03-01")
    assert settled["pending"] >= 1
    assert settled["settled"] == 1
    assert settled["wins"] == 1

    row = isolated_db.fetch_one(
        """
        SELECT is_settled, actual_winner, model_correct, profit_units, roi_pct,
               realized_margin_for_pick, realized_edge_delta
        FROM recommendation_snapshot_items
        WHERE signal_key = 'sig-1'
        """
    )
    assert row is not None
    assert row["is_settled"] == 1
    assert row["actual_winner"] == "HOME"
    assert row["model_correct"] == 1
    assert row["profit_units"] > 0
    assert row["roi_pct"] > 0
    assert row["realized_margin_for_pick"] == 16.0
    assert row["realized_edge_delta"] == 11.0


def test_backfill_returns_zero_when_nothing_pending(isolated_db):
    result = backfill_recommendation_outcomes()
    assert result["pending"] == 0
    assert result["settled"] == 0
