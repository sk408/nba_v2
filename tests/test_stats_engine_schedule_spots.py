def test_schedule_spots_next_game_query_includes_season(monkeypatch):
    from src.analytics import stats_engine

    calls = []

    def fake_fetch_all(sql, params=()):
        calls.append((sql, params))
        return []

    monkeypatch.setattr(stats_engine.db, "fetch_all", fake_fetch_all)
    stats_engine.compute_schedule_spots(
        team_id=1,
        game_date="2025-04-10",
        opponent_team_id=2,
        season="2024-25",
    )

    next_query = next(sql for sql, _ in calls if "ps.game_date > ?" in sql)
    next_params = next(params for sql, params in calls if "ps.game_date > ?" in sql)

    assert "ps.season = ?" in next_query
    assert next_params == ("2025-04-10", "2024-25", 1)
