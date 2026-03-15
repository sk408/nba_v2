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
