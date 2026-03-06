"""Infer injury / missed-game history from player game-log gaps."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.database import db

logger = logging.getLogger(__name__)


def infer_injuries_from_logs(callback: Optional[Callable] = None) -> dict:
    """Build injury_history from gaps in player game logs.

    For each player, compares the team's game schedule (derived from all
    player_stats entries) against the player's own game dates to find
    missed games.  Inserts/updates records in injury_history.

    Returns dict with 'records' count of rows upserted.
    """
    # 1. Build team schedules: team_id -> set of game_dates
    rows = db.fetch_all(
        """SELECT p.team_id, ps.game_date
           FROM player_stats ps
           JOIN players p ON p.player_id = ps.player_id
           WHERE p.team_id IS NOT NULL
           GROUP BY p.team_id, ps.game_date"""
    )
    team_dates: dict[int, set[str]] = {}
    for r in rows:
        tid = r["team_id"]
        if tid not in team_dates:
            team_dates[tid] = set()
        team_dates[tid].add(r["game_date"])

    if callback:
        callback(f"Found schedules for {len(team_dates)} teams")

    # 2. Get all players with a team
    player_rows = db.fetch_all(
        "SELECT player_id, team_id FROM players WHERE team_id IS NOT NULL"
    )

    # 3. Current injury reasons (for context)
    injury_info: dict[int, str] = {}
    for ir in db.fetch_all("SELECT player_id, reason FROM injuries"):
        injury_info[ir["player_id"]] = ir["reason"] or ""

    # 4. For each player find games their team played but they didn't
    records = 0
    batch: list[tuple] = []

    for pr in player_rows:
        pid, tid = pr["player_id"], pr["team_id"]
        if tid not in team_dates:
            continue

        played = db.fetch_all(
            "SELECT DISTINCT game_date FROM player_stats WHERE player_id = ?",
            (pid,),
        )
        played_dates = {r["game_date"] for r in played}
        if not played_dates:
            continue

        # Only consider team games after this player's first appearance
        first_game = min(played_dates)
        missed = {d for d in team_dates[tid] if d >= first_game} - played_dates
        if not missed:
            continue

        avg_row = db.fetch_one(
            "SELECT AVG(minutes) as avg_min FROM player_stats WHERE player_id = ?",
            (pid,),
        )
        avg_min = avg_row["avg_min"] if avg_row and avg_row["avg_min"] else None
        reason = injury_info.get(pid, "")

        for gd in missed:
            batch.append((pid, tid, gd, 1, avg_min, reason))
            records += 1

    # 5. Upsert
    if batch:
        db.execute_many(
            """INSERT INTO injury_history
                   (player_id, team_id, game_date, was_out, avg_minutes, reason)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id, game_date) DO UPDATE SET
                   was_out   = excluded.was_out,
                   avg_minutes = excluded.avg_minutes,
                   reason    = excluded.reason""",
            batch,
        )

    if callback:
        callback(f"Processed {len(player_rows)} players, found {records} missed games")

    return {"records": records}
