from src.database.migrations import _backfill_player_stats_team_id


def test_backfill_player_stats_team_id(isolated_db):
    db = isolated_db

    db.execute(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?,?,?,?)",
        (1, "Home Team", "HOM", "East"),
    )
    db.execute(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?,?,?,?)",
        (2, "Away Team", "AWY", "West"),
    )
    db.execute(
        "INSERT INTO players (player_id, name, team_id, position) VALUES (?,?,?,?)",
        (101, "Home Player", 1, "G"),
    )
    db.execute(
        "INSERT INTO players (player_id, name, team_id, position) VALUES (?,?,?,?)",
        (202, "Away Player", 2, "F"),
    )

    # Same game context: backfill should infer team_id from opponent mapping.
    db.execute(
        """
        INSERT INTO player_stats
        (player_id, opponent_team_id, is_home, game_date, game_id, season,
         points, rebounds, assists, minutes, team_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (101, 2, 1, "2025-01-01", "GAME1", "2024-25", 10, 4, 3, 25, None),
    )
    db.execute(
        """
        INSERT INTO player_stats
        (player_id, opponent_team_id, is_home, game_date, game_id, season,
         points, rebounds, assists, minutes, team_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (202, 1, 0, "2025-01-01", "GAME1", "2024-25", 12, 5, 2, 27, None),
    )

    # Missing game_id falls back to players.team_id.
    db.execute(
        """
        INSERT INTO player_stats
        (player_id, opponent_team_id, is_home, game_date, game_id, season,
         points, rebounds, assists, minutes, team_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (101, 2, 1, "2025-01-02", None, "2024-25", 9, 3, 4, 23, None),
    )

    _backfill_player_stats_team_id()

    home_row = db.fetch_one(
        "SELECT team_id FROM player_stats WHERE player_id = ? AND game_id = ?",
        (101, "GAME1"),
    )
    away_row = db.fetch_one(
        "SELECT team_id FROM player_stats WHERE player_id = ? AND game_id = ?",
        (202, "GAME1"),
    )
    fallback_row = db.fetch_one(
        "SELECT team_id FROM player_stats WHERE player_id = ? AND game_date = ?",
        (101, "2025-01-02"),
    )

    assert home_row["team_id"] == 1
    assert away_row["team_id"] == 2
    assert fallback_row["team_id"] == 1
