"""Elo rating computation for NBA teams.

Standard Elo with home-court advantage. Ratings are stored per-team
per-game-date in the ``elo_ratings`` table and can be queried for any
historical date.
"""

import logging
from typing import List, Tuple

from src.database import db

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
K = 20.0            # K-factor: how much a single game moves ratings
HOME_ELO_ADV = 70.0 # home court Elo advantage (~3.5 points)
INIT_ELO = 1500.0   # starting rating for every team


def _expected(rating_a: float, rating_b: float) -> float:
    """Logistic expected score for player A given both ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_all_elo(season: str = "2025-26") -> None:
    """Compute Elo ratings for every game in *season* and store them.

    Iterates chronologically through all games (identified from the home
    team's perspective in ``player_stats``), updates Elo after each game,
    and batch-inserts the post-game ratings into ``elo_ratings``.
    """
    # ── 1. Fetch all games for the season (one row per game) ─────
    games = db.fetch_all(
        """
        SELECT ps.game_date,
               ps.team_id   AS home_team_id,
               ps.opponent_team_id AS away_team_id,
               ps.win_loss
        FROM player_stats ps
        WHERE ps.season = ? AND ps.is_home = 1 AND ps.team_id IS NOT NULL
        GROUP BY ps.game_date, ps.team_id, ps.opponent_team_id
        ORDER BY ps.game_date
        """,
        (season,),
    )

    if not games:
        logger.warning("No games found for season %s — Elo not computed", season)
        return

    logger.info("Computing Elo for %d games in season %s", len(games), season)

    # ── 2. Clear existing ratings for the season ─────────────────
    # Determine the date range so we only delete this season's rows.
    first_date = games[0]["game_date"]
    last_date = games[-1]["game_date"]
    db.execute(
        "DELETE FROM elo_ratings WHERE game_date >= ? AND game_date <= ?",
        (first_date, last_date),
    )

    # ── 3. Walk through games chronologically ────────────────────
    # In-memory dict tracks the latest Elo for each team.
    current_elo: dict[int, float] = {}
    rows_to_insert: List[Tuple[int, str, float]] = []

    for g in games:
        home_id = g["home_team_id"]
        away_id = g["away_team_id"]
        game_date = g["game_date"]

        # Pre-game Elo (or initial)
        elo_home = current_elo.get(home_id, INIT_ELO)
        elo_away = current_elo.get(away_id, INIT_ELO)

        # Expected scores (home gets the court advantage)
        exp_home = _expected(elo_home + HOME_ELO_ADV, elo_away)
        exp_away = 1.0 - exp_home

        # Actual result: win_loss from home team player's perspective
        home_won = 1.0 if g["win_loss"] == "W" else 0.0
        away_won = 1.0 - home_won

        # Update Elo
        new_elo_home = elo_home + K * (home_won - exp_home)
        new_elo_away = elo_away + K * (away_won - exp_away)

        current_elo[home_id] = new_elo_home
        current_elo[away_id] = new_elo_away

        rows_to_insert.append((home_id, game_date, new_elo_home))
        rows_to_insert.append((away_id, game_date, new_elo_away))

    # ── 4. Batch insert ──────────────────────────────────────────
    db.execute_many(
        "INSERT OR REPLACE INTO elo_ratings (team_id, game_date, elo) VALUES (?, ?, ?)",
        rows_to_insert,
    )
    logger.info(
        "Stored %d Elo rating rows (%d games)", len(rows_to_insert), len(games)
    )


def get_team_elo(team_id: int, game_date: str, season: str = "2025-26") -> float:
    """Return the most recent Elo for *team_id* strictly before *game_date*.

    Falls back to ``INIT_ELO`` (1500.0) if the team has no prior rating.
    """
    row = db.fetch_one(
        """
        SELECT elo
        FROM elo_ratings
        WHERE team_id = ? AND game_date < ?
        ORDER BY game_date DESC
        LIMIT 1
        """,
        (team_id, game_date),
    )
    return row["elo"] if row else INIT_ELO
