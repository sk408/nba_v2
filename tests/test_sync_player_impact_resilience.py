def test_sync_player_impact_aborts_after_consecutive_on_off_failures(isolated_db, monkeypatch):
    from src.data import sync_service, nba_fetcher
    from src.database import db

    db.execute_many(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?, ?, ?, ?)",
        [
            (1610612737, "Atlanta Hawks", "ATL", "East"),
            (1610612738, "Boston Celtics", "BOS", "East"),
            (1610612739, "Cleveland Cavaliers", "CLE", "East"),
            (1610612740, "New Orleans Pelicans", "NOP", "West"),
            (1610612741, "Chicago Bulls", "CHI", "East"),
        ],
    )

    monkeypatch.setattr(nba_fetcher, "fetch_player_estimated_metrics", lambda: [])

    calls = {"count": 0}

    def fake_fetch_player_on_off(team_id):
        calls["count"] += 1
        return {"on": [], "off": [], "_ok": False}

    monkeypatch.setattr(nba_fetcher, "fetch_player_on_off", fake_fetch_player_on_off)
    monkeypatch.setattr(
        sync_service,
        "get_setting",
        lambda key, default=None: 3 if key == "nba_api_on_off_abort_after_failures" else default,
    )

    messages = []
    sync_service.sync_player_impact(callback=messages.append, force=True)

    assert calls["count"] == 3
    assert any(
        "stopping remaining team on/off fetches for this sync" in msg.lower()
        for msg in messages
    )


def test_sync_player_impact_retries_failed_teams_before_advancing(isolated_db, monkeypatch):
    from src.data import sync_service, nba_fetcher
    from src.database import db

    hawks = 1610612737
    celtics = 1610612738
    db.execute_many(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?, ?, ?, ?)",
        [
            (hawks, "Atlanta Hawks", "ATL", "East"),
            (celtics, "Boston Celtics", "BOS", "East"),
        ],
    )

    monkeypatch.setattr(nba_fetcher, "fetch_player_estimated_metrics", lambda: [])

    call_order = []
    outcomes = {
        hawks: [False, True],   # Fail once, then succeed on retry.
        celtics: [True],        # Succeeds immediately.
    }

    def fake_fetch_player_on_off(team_id):
        call_order.append(team_id)
        ok = outcomes[team_id].pop(0)
        return {"on": [], "off": [], "_ok": ok}

    monkeypatch.setattr(nba_fetcher, "fetch_player_on_off", fake_fetch_player_on_off)

    def fake_get_setting(key, default=None):
        if key == "nba_api_on_off_abort_after_failures":
            return 0
        if key == "nba_api_on_off_team_retry_attempts":
            return 1
        return default

    monkeypatch.setattr(sync_service, "get_setting", fake_get_setting)

    sleep_calls = []
    monkeypatch.setattr(sync_service.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    messages = []
    sync_service.sync_player_impact(callback=messages.append, force=True)

    assert call_order == [hawks, hawks, celtics]
    assert sleep_calls == [2.1]
    assert any("Retrying team_id=1610612737 on/off fetch in 2.1s" in msg for msg in messages)


def test_sync_player_impact_failure_backoff_increases_per_failure(isolated_db, monkeypatch):
    from src.data import sync_service, nba_fetcher
    from src.database import db

    lakers = 1610612747
    db.execute_many(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?, ?, ?, ?)",
        [(lakers, "Los Angeles Lakers", "LAL", "West")],
    )

    monkeypatch.setattr(nba_fetcher, "fetch_player_estimated_metrics", lambda: [])

    calls = {"count": 0}

    def fake_fetch_player_on_off(team_id):
        calls["count"] += 1
        return {"on": [], "off": [], "_ok": False}

    monkeypatch.setattr(nba_fetcher, "fetch_player_on_off", fake_fetch_player_on_off)

    def fake_get_setting(key, default=None):
        if key == "nba_api_on_off_abort_after_failures":
            return 0
        if key == "nba_api_on_off_team_retry_attempts":
            return 2
        return default

    monkeypatch.setattr(sync_service, "get_setting", fake_get_setting)

    sleep_calls = []
    monkeypatch.setattr(sync_service.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    sync_service.sync_player_impact(force=True)

    assert calls["count"] == 3
    assert sleep_calls == [2.1, 2.6]


def test_sync_player_impact_preserves_existing_onoff_when_team_fetch_fails(isolated_db, monkeypatch):
    from src.data import sync_service, nba_fetcher
    from src.database import db

    season = "2025-26"
    team_id = 1610612748
    player_id = 203952

    db.execute(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?, ?, ?, ?)",
        (team_id, "Miami Heat", "MIA", "East"),
    )
    db.execute(
        """INSERT INTO players
           (player_id, name, team_id, position, height, weight, age, experience)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (player_id, "Haywood Highsmith", team_id, "F", "6-5", "220", 27, 4),
    )
    db.execute(
        """INSERT INTO player_impact
           (player_id, team_id, season,
            on_court_off_rating, on_court_def_rating, on_court_net_rating,
            off_court_off_rating, off_court_def_rating, off_court_net_rating,
            net_rating_diff, on_court_minutes,
            e_usg_pct, e_off_rating, e_def_rating, e_net_rating,
            e_pace, e_ast_ratio, e_oreb_pct, e_dreb_pct, last_synced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            player_id, team_id, season,
            114.2, 111.1, 3.1,
            108.5, 112.7, -4.2,
            7.3, 942.0,
            10.0, 109.0, 110.0, -1.0,
            97.0, 12.0, 5.0, 20.0, "2026-03-14T00:00:00",
        ),
    )

    monkeypatch.setattr(sync_service, "get_season", lambda: season)
    monkeypatch.setattr(
        nba_fetcher,
        "fetch_player_estimated_metrics",
        lambda: [{"PLAYER_ID": player_id, "E_USG_PCT": 11.5}],
    )
    monkeypatch.setattr(
        nba_fetcher,
        "fetch_player_on_off",
        lambda _team_id: {"on": [], "off": [], "_ok": False},
    )
    monkeypatch.setattr(
        sync_service,
        "get_setting",
        lambda key, default=None: (
            0
            if key in ("nba_api_on_off_abort_after_failures", "nba_api_on_off_team_retry_attempts")
            else default
        ),
    )

    sync_service.sync_player_impact(force=True)

    row = db.fetch_one(
        """SELECT on_court_off_rating, on_court_def_rating, on_court_net_rating,
                  off_court_off_rating, off_court_def_rating, off_court_net_rating,
                  net_rating_diff, on_court_minutes, e_usg_pct
           FROM player_impact
           WHERE player_id=? AND season=?""",
        (player_id, season),
    )

    assert row["on_court_off_rating"] == 114.2
    assert row["on_court_def_rating"] == 111.1
    assert row["on_court_net_rating"] == 3.1
    assert row["off_court_off_rating"] == 108.5
    assert row["off_court_def_rating"] == 112.7
    assert row["off_court_net_rating"] == -4.2
    assert row["net_rating_diff"] == 7.3
    assert row["on_court_minutes"] == 942.0
    assert row["e_usg_pct"] == 11.5


def test_sync_player_impact_for_teams_repairs_only_requested_teams(isolated_db, monkeypatch):
    from src.data import sync_service, nba_fetcher
    from src.database import db

    season = "2025-26"
    dal = 1610612742
    mia = 1610612748
    dal_player = 203114
    mia_player = 203952

    db.execute_many(
        "INSERT INTO teams (team_id, name, abbreviation, conference) VALUES (?, ?, ?, ?)",
        [
            (dal, "Dallas Mavericks", "DAL", "West"),
            (mia, "Miami Heat", "MIA", "East"),
        ],
    )
    db.execute_many(
        """INSERT INTO players
           (player_id, name, team_id, position, height, weight, age, experience)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (dal_player, "Khris Middleton", dal, "F", "6-7", "222", 33, 12),
            (mia_player, "Haywood Highsmith", mia, "F", "6-5", "220", 27, 4),
        ],
    )
    db.execute_many(
        """INSERT INTO player_impact
           (player_id, team_id, season,
            on_court_off_rating, on_court_def_rating, on_court_net_rating,
            off_court_off_rating, off_court_def_rating, off_court_net_rating,
            net_rating_diff, on_court_minutes,
            e_usg_pct, e_off_rating, e_def_rating, e_net_rating,
            e_pace, e_ast_ratio, e_oreb_pct, e_dreb_pct, last_synced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                dal_player, dal, season,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0,
                8.0, 110.0, 111.0, -1.0,
                97.0, 10.0, 4.0, 22.0, "2026-03-14T00:00:00",
            ),
            (
                mia_player, mia, season,
                101.0, 103.0, -2.0,
                104.0, 102.0, 2.0,
                -4.0, 777.0,
                9.0, 108.0, 109.0, -1.0,
                96.0, 11.0, 5.0, 21.0, "2026-03-14T00:00:00",
            ),
        ],
    )

    monkeypatch.setattr(sync_service, "get_season", lambda: season)
    monkeypatch.setattr(
        nba_fetcher,
        "fetch_player_estimated_metrics",
        lambda: [
            {"PLAYER_ID": dal_player, "E_USG_PCT": 12.5},
            {"PLAYER_ID": mia_player, "E_USG_PCT": 99.0},
        ],
    )

    on_off_calls = []

    def fake_fetch_player_on_off(team_id):
        on_off_calls.append(team_id)
        assert team_id == dal
        return {
            "_ok": True,
            "on": [{"VS_PLAYER_ID": dal_player, "OFF_RATING": 112.1, "DEF_RATING": 110.4, "NET_RATING": 1.7, "MIN": 555}],
            "off": [{"VS_PLAYER_ID": dal_player, "OFF_RATING": 109.0, "DEF_RATING": 111.8, "NET_RATING": -2.8}],
        }

    monkeypatch.setattr(nba_fetcher, "fetch_player_on_off", fake_fetch_player_on_off)
    monkeypatch.setattr(
        sync_service,
        "get_setting",
        lambda key, default=None: (
            0
            if key in ("nba_api_on_off_abort_after_failures", "nba_api_on_off_team_retry_attempts")
            else default
        ),
    )

    sync_service.sync_player_impact_for_teams([dal], force=True)

    dal_row = db.fetch_one(
        "SELECT on_court_off_rating, off_court_net_rating, net_rating_diff, on_court_minutes, e_usg_pct FROM player_impact WHERE player_id=? AND season=?",
        (dal_player, season),
    )
    mia_row = db.fetch_one(
        "SELECT on_court_off_rating, off_court_net_rating, net_rating_diff, on_court_minutes, e_usg_pct FROM player_impact WHERE player_id=? AND season=?",
        (mia_player, season),
    )

    assert on_off_calls == [dal]
    assert dal_row["on_court_off_rating"] == 112.1
    assert dal_row["off_court_net_rating"] == -2.8
    assert dal_row["net_rating_diff"] == 4.5
    assert dal_row["on_court_minutes"] == 555.0
    assert dal_row["e_usg_pct"] == 12.5

    assert mia_row["on_court_off_rating"] == 101.0
    assert mia_row["off_court_net_rating"] == 2.0
    assert mia_row["net_rating_diff"] == -4.0
    assert mia_row["on_court_minutes"] == 777.0
    assert mia_row["e_usg_pct"] == 9.0
