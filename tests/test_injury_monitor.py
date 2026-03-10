def test_injury_monitor_syncs_before_diff_and_notifies_once(monkeypatch):
    from src.notifications import injury_monitor

    sync_calls = {"n": 0}
    notifs = []

    def fake_sync():
        sync_calls["n"] += 1
        return 1

    monkeypatch.setattr(injury_monitor, "sync_injuries", fake_sync)
    monkeypatch.setattr(
        injury_monitor,
        "create_notification",
        lambda **kwargs: notifs.append(kwargs),
    )

    monkeypatch.setattr(
        injury_monitor.db,
        "fetch_all",
        lambda _sql: [
            {
                "player_id": 99,
                "player_name": "Test Player",
                "team_id": 1,
                "status": "Questionable",
                "reason": "Ankle",
                "mpg": 30.0,
                "ppg": 20.0,
                "abbreviation": "BOS",
            }
        ],
    )

    monitor = injury_monitor.InjuryMonitor()
    monitor._previous_state = {
        99: {
            "player_id": 99,
            "player_name": "Test Player",
            "team_id": 1,
            "status": "Out",
            "reason": "Ankle",
            "mpg": 30.0,
            "ppg": 20.0,
            "abbreviation": "BOS",
        }
    }

    monitor._check_changes()

    assert sync_calls["n"] == 1
    assert len(notifs) == 1
    assert "Status Change" in notifs[0]["title"]
