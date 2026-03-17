import pytest

from src.analytics import stats_engine


def test_compute_season_progress_normalizes_games_played(monkeypatch):
    monkeypatch.setattr(
        stats_engine,
        "_count_team_games_to_date",
        lambda team_id, game_date, season: 41,
    )

    progress = stats_engine.compute_season_progress(
        team_id=1610612737,
        game_date="2025-02-15",
        season="2024-25",
    )

    assert progress == pytest.approx(0.5)


def test_compute_tanking_signal_live_is_high_for_late_bad_team(monkeypatch):
    monkeypatch.setattr(
        stats_engine,
        "_count_team_games_to_date",
        lambda team_id, game_date, season: 55,
    )
    monkeypatch.setattr(
        stats_engine,
        "compute_season_progress",
        lambda team_id, game_date, season=None: 0.88,
    )
    monkeypatch.setattr(
        stats_engine,
        "_team_win_pct_to_date",
        lambda team_id, game_date, season: 0.24,
    )
    monkeypatch.setattr(
        stats_engine,
        "compute_momentum",
        lambda team_id, game_date, season=None: {"streak": -7},
    )
    monkeypatch.setattr(
        stats_engine,
        "compute_roster_shock",
        lambda team_id, game_date, season=None: 0.50,
    )

    signal = stats_engine.compute_tanking_signal(
        team_id=1610612765,
        game_date="2025-03-20",
        season="2024-25",
        mode="live",
    )

    assert signal > 0.35


def test_compute_tanking_signal_oracle_uses_final_record(monkeypatch):
    monkeypatch.setattr(
        stats_engine,
        "compute_season_progress",
        lambda team_id, game_date, season=None: 0.90,
    )
    monkeypatch.setattr(
        stats_engine.db,
        "fetch_one",
        lambda query, params: {"w_pct": 0.28},
    )

    signal = stats_engine.compute_tanking_signal(
        team_id=1610612766,
        game_date="2025-03-20",
        season="2024-25",
        mode="oracle",
    )

    assert signal > 0.2


def test_compute_roster_shock_detects_core_rotation_churn(monkeypatch):
    date_rows = [
        {"game_date": "2025-03-10"},
        {"game_date": "2025-03-08"},
        {"game_date": "2025-03-06"},
        {"game_date": "2025-03-04"},
        {"game_date": "2025-03-02"},
        {"game_date": "2025-02-28"},
        {"game_date": "2025-02-26"},
        {"game_date": "2025-02-24"},
        {"game_date": "2025-02-22"},
        {"game_date": "2025-02-20"},
        {"game_date": "2025-02-18"},
        {"game_date": "2025-02-16"},
        {"game_date": "2025-02-14"},
    ]
    monkeypatch.setattr(
        stats_engine.db,
        "fetch_all",
        lambda query, params: date_rows,
    )

    def _fake_window(team_id, season, dates):
        if len(dates) == 3:
            # Recent: one core player dropped, one new high-minute player added.
            return {1: 90.0, 2: 75.0, 9: 70.0}
        return {1: 220.0, 2: 210.0, 3: 190.0}

    monkeypatch.setattr(stats_engine, "_window_minutes_by_player", _fake_window)

    shock = stats_engine.compute_roster_shock(
        team_id=1610612744,
        game_date="2025-03-20",
        season="2024-25",
    )

    assert shock > 0.40
