"""Tests for odds monitor: num_bets column, upcoming odds sync, monitor integration."""

import pytest
from unittest.mock import patch, MagicMock
from src.database.migrations import init_db
from src.database import db

# Ensure migrations (including num_bets column) have run
init_db()


def test_game_odds_has_num_bets_column():
    """num_bets column exists in game_odds after init_db."""
    cols = db.fetch_all("PRAGMA table_info(game_odds)")
    col_names = [c["name"] for c in cols]
    assert "num_bets" in col_names


@patch("src.data.odds_sync.fetch_action_odds")
@patch("src.data.odds_sync.sync_betting_splits", return_value=0)
def test_sync_odds_saves_num_bets(mock_sbd, mock_fetch):
    """sync_odds_for_date saves num_bets from consensus odds."""
    from src.data.odds_sync import sync_odds_for_date
    from src.analytics.stats_engine import get_team_abbreviations

    id_to_abbr = get_team_abbreviations()
    abbr_to_id = {v: k for k, v in id_to_abbr.items()}

    # Pick two real teams from the DB
    abbrs = list(abbr_to_id.keys())[:2]
    home_abbr, away_abbr = abbrs[0], abbrs[1]
    home_id, away_id = abbr_to_id[home_abbr], abbr_to_id[away_abbr]

    mock_fetch.return_value = [{
        "id": 999,
        "home_team_id": 100,
        "away_team_id": 200,
        "teams": [
            {"id": 100, "abbr": home_abbr},
            {"id": 200, "abbr": away_abbr},
        ],
        "odds": [{
            "type": "game",
            "book_id": 15,
            "spread_home": -5.5,
            "total": 220.0,
            "ml_home": -220,
            "ml_away": 180,
            "spread_home_public": 60,
            "spread_away_public": 40,
            "spread_home_money": 55,
            "spread_away_money": 45,
            "ml_home_public": 70,
            "ml_away_public": 30,
            "ml_home_money": 65,
            "ml_away_money": 35,
            "num_bets": 12345,
        }],
    }]

    saved = sync_odds_for_date("2026-03-16", invalidate_cache=False)
    assert saved == 1

    row = db.fetch_one(
        "SELECT num_bets FROM game_odds WHERE game_date = ? AND home_team_id = ?",
        ("2026-03-16", home_id),
    )
    assert row is not None
    assert row["num_bets"] == 12345


def test_nba_tomorrow_returns_next_day():
    """nba_tomorrow() returns the day after nba_today()."""
    from src.utils.timezone_utils import nba_today, nba_tomorrow
    from datetime import datetime, timedelta

    today = datetime.strptime(nba_today(), "%Y-%m-%d")
    tomorrow = datetime.strptime(nba_tomorrow(), "%Y-%m-%d")
    assert tomorrow - today == timedelta(days=1)


def test_nba_game_date_from_utc_iso_handles_late_west_tipoffs():
    """UTC midnight tipoffs still map to the prior NBA slate date."""
    from src.utils.timezone_utils import nba_game_date_from_utc_iso

    # 2026-03-20 00:00Z == 2026-03-19 20:00 ET -> NBA date should be 2026-03-19
    assert nba_game_date_from_utc_iso("2026-03-20T00:00:00Z") == "2026-03-19"

    # 2026-03-20 13:00Z == 2026-03-20 09:00 ET -> NBA date should be 2026-03-20
    assert nba_game_date_from_utc_iso("2026-03-20T13:00:00Z") == "2026-03-20"


@patch("src.data.odds_sync.sync_odds_for_date")
def test_sync_upcoming_odds_calls_today_and_tomorrow(mock_sync):
    """sync_upcoming_odds calls sync_odds_for_date for both today and tomorrow."""
    from src.data.odds_sync import sync_upcoming_odds

    mock_sync.return_value = 3
    result = sync_upcoming_odds()

    assert mock_sync.call_count == 2
    dates_called = [call.args[0] for call in mock_sync.call_args_list]
    # Should contain two different dates (today and tomorrow)
    assert len(set(dates_called)) == 2
    assert result == 6  # 3 + 3


@patch("src.notifications.injury_monitor.sync_upcoming_odds")
@patch("src.notifications.injury_monitor.sync_injuries", return_value=0)
def test_injury_monitor_refreshes_odds_every_3rd_cycle(mock_inj, mock_odds):
    """InjuryMonitor calls sync_upcoming_odds every 3rd _check_changes cycle."""
    from src.notifications.injury_monitor import InjuryMonitor

    monitor = InjuryMonitor()
    monitor._previous_state = {}

    # Run 6 cycles
    for _ in range(6):
        monitor._check_changes()

    # Should have called odds sync exactly 2 times (cycles 3 and 6)
    assert mock_odds.call_count == 2


@patch("src.notifications.injury_monitor.sync_upcoming_odds")
@patch("src.notifications.injury_monitor.sync_injuries", return_value=0)
def test_injury_monitor_skips_odds_on_non_3rd_cycles(mock_inj, mock_odds):
    """InjuryMonitor does NOT call odds sync on cycles 1, 2."""
    from src.notifications.injury_monitor import InjuryMonitor

    monitor = InjuryMonitor()
    monitor._previous_state = {}

    # Run 2 cycles
    for _ in range(2):
        monitor._check_changes()

    assert mock_odds.call_count == 0


@patch("src.data.odds_sync.get_json")
@patch("src.analytics.stats_engine.get_team_abbreviations")
def test_sync_betting_splits_blank_values_do_not_block_later_fill(
    mock_team_map, mock_get_json
):
    """Blank split payloads stay NULL so later numeric payload can fill them."""
    from src.data.odds_sync import sync_betting_splits

    teams = db.fetch_all(
        "SELECT team_id, abbreviation FROM teams ORDER BY team_id ASC LIMIT 2"
    )
    assert len(teams) == 2

    home_id = teams[0]["team_id"]
    away_id = teams[1]["team_id"]
    home_abbr = teams[0]["abbreviation"]
    away_abbr = teams[1]["abbreviation"]
    game_date = "2099-01-01"

    mock_team_map.return_value = {home_id: home_abbr, away_id: away_abbr}

    db.execute(
        "DELETE FROM game_odds WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
        (game_date, home_id, away_id),
    )
    db.execute(
        """
        INSERT INTO game_odds (
            game_date, home_team_id, away_team_id, spread, over_under,
            home_moneyline, away_moneyline, provider, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'espn', ?)
        """,
        (game_date, home_id, away_id, -3.5, 218.5, -145, 125, "2099-01-01T00:00:00"),
    )

    blank_payload = {
        "data": [
            {
                "competitors": {
                    "home": {"abbreviation": home_abbr},
                    "away": {"abbreviation": away_abbr},
                },
                "bettingSplits": {
                    "spread": {
                        "home": {"betsPercentage": "", "stakePercentage": ""},
                        "away": {"betsPercentage": "", "stakePercentage": ""},
                    },
                    "moneyline": {
                        "home": {"betsPercentage": "", "stakePercentage": ""},
                        "away": {"betsPercentage": "", "stakePercentage": ""},
                    },
                },
            }
        ]
    }
    numeric_payload = {
        "data": [
            {
                "competitors": {
                    "home": {"abbreviation": home_abbr},
                    "away": {"abbreviation": away_abbr},
                },
                "bettingSplits": {
                    "spread": {
                        "home": {"betsPercentage": 62, "stakePercentage": 71},
                        "away": {"betsPercentage": 38, "stakePercentage": 29},
                    },
                    "moneyline": {
                        "home": {"betsPercentage": 64, "stakePercentage": 69},
                        "away": {"betsPercentage": 36, "stakePercentage": 31},
                    },
                },
            }
        ]
    }
    mock_get_json.side_effect = [blank_payload, numeric_payload]

    first = sync_betting_splits(game_date)
    second = sync_betting_splits(game_date)

    row = db.fetch_one(
        """
        SELECT spread_home_public, spread_home_money, ml_home_public, ml_home_money
        FROM game_odds
        WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?
        """,
        (game_date, home_id, away_id),
    )
    assert row is not None

    assert first == 0
    assert second == 1
    assert row["spread_home_public"] == 62
    assert row["spread_home_money"] == 71
    assert row["ml_home_public"] == 64
    assert row["ml_home_money"] == 69
