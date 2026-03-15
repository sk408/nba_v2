"""Persist underdog recommendation snapshots and settle realized outcomes."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Mapping, Optional

from src.database import db
from src.analytics.prediction import get_actual_game_results
from src.utils.timezone_utils import nba_today

logger = logging.getLogger(__name__)

_RECOMMENDATION_OUTCOME_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recommendation_snapshot_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_key TEXT NOT NULL,
    game_date TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    filters TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '{}',
    alert_digest TEXT NOT NULL DEFAULT '{}',
    total_candidates INTEGER NOT NULL DEFAULT 0,
    screened_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recommendation_snapshot_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    signal_key TEXT NOT NULL,
    game_date TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    pick TEXT NOT NULL,
    tier TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    game_score REAL DEFAULT 0.0,
    rank_score REAL DEFAULT 0.0,
    dog_payout REAL DEFAULT 0.0,
    vegas_spread REAL DEFAULT 0.0,
    vegas_home_ml INTEGER DEFAULT 0,
    vegas_away_ml INTEGER DEFAULT 0,
    is_dog_pick INTEGER NOT NULL DEFAULT 0,
    is_value_zone INTEGER NOT NULL DEFAULT 0,
    ml_home_public REAL DEFAULT 0.0,
    ml_home_money REAL DEFAULT 0.0,
    filters TEXT NOT NULL DEFAULT '{}',
    why_pick TEXT NOT NULL DEFAULT '{}',
    feature_snapshot TEXT NOT NULL DEFAULT '{}',
    snapshot_at TEXT NOT NULL,
    is_settled INTEGER NOT NULL DEFAULT 0,
    settled_at TEXT,
    settlement_source TEXT,
    actual_home_score REAL,
    actual_away_score REAL,
    actual_winner TEXT,
    model_correct INTEGER,
    profit_units REAL,
    roi_pct REAL,
    realized_margin_for_pick REAL,
    realized_edge_delta REAL,
    FOREIGN KEY (run_id) REFERENCES recommendation_snapshot_runs(id),
    UNIQUE(run_id, signal_key)
);

CREATE INDEX IF NOT EXISTS idx_rec_items_unsettled
    ON recommendation_snapshot_items(is_settled, game_date);
CREATE INDEX IF NOT EXISTS idx_rec_items_matchup
    ON recommendation_snapshot_items(game_date, home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_rec_runs_scope_date
    ON recommendation_snapshot_runs(scope_key, game_date, snapshot_at DESC);
"""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_json(payload: Any) -> str:
    try:
        return json.dumps(payload or {}, sort_keys=True)
    except Exception:
        return "{}"


def _signal_key_for_row(row: Mapping[str, Any], fallback_date: str) -> str:
    key = str(row.get("signal_key", "")).strip()
    if key:
        return key
    game_date = str(row.get("game_date") or fallback_date)
    away_ref = str(row.get("away_team_id") or row.get("away_team") or "")
    home_ref = str(row.get("home_team_id") or row.get("home_team") or "")
    pick = str(row.get("pick", "")).upper()
    return f"{game_date}:{away_ref}:{home_ref}:{pick}"


def _moneyline_payout_multiplier(ml_line: int) -> float:
    if ml_line == 0:
        return 0.0
    if ml_line > 0:
        return 1.0 + ml_line / 100.0
    return 1.0 + 100.0 / abs(ml_line)


def ensure_recommendation_outcomes_schema():
    """Ensure recommendation snapshot/outcome tables exist."""
    db.execute_script(_RECOMMENDATION_OUTCOME_SCHEMA_SQL)


def persist_recommendation_snapshot(
    game_date: str,
    scope_key: str,
    filters: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    total_candidates: int,
    summary: Optional[Mapping[str, Any]] = None,
    alert_digest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist one recommendation snapshot run with per-row feature/odds payloads."""
    ensure_recommendation_outcomes_schema()
    safe_rows = [dict(row) for row in rows]
    run_id = db.execute_returning_id(
        """
        INSERT INTO recommendation_snapshot_runs (
            scope_key, game_date, snapshot_at, filters, summary, alert_digest,
            total_candidates, screened_count
        )
        VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?)
        """,
        (
            str(scope_key),
            str(game_date),
            _to_json(filters),
            _to_json(summary),
            _to_json(alert_digest),
            max(0, _safe_int(total_candidates, 0)),
            len(safe_rows),
        ),
    )

    stored = 0
    filters_json = _to_json(filters)
    for row in safe_rows:
        signal_key = _signal_key_for_row(row, fallback_date=game_date)
        why_pick = row.get("why_pick") if isinstance(row.get("why_pick"), dict) else {}
        feature_snapshot = {
            "adjustments": row.get("adjustments", {}),
            "drivers": row.get("drivers", []),
            "caution_flags": row.get("caution_flags", []),
            "start_utc": row.get("start_utc"),
            "sharp_agrees": row.get("sharp_agrees"),
        }
        db.execute(
            """
            INSERT INTO recommendation_snapshot_items (
                run_id, signal_key, game_date, home_team_id, away_team_id, pick, tier,
                confidence, game_score, rank_score, dog_payout, vegas_spread,
                vegas_home_ml, vegas_away_ml, is_dog_pick, is_value_zone,
                ml_home_public, ml_home_money, filters, why_pick, feature_snapshot,
                snapshot_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                run_id,
                signal_key,
                str(row.get("game_date") or game_date),
                _safe_int(row.get("home_team_id"), 0),
                _safe_int(row.get("away_team_id"), 0),
                str(row.get("pick", "")).upper(),
                str(row.get("tier", "")),
                _safe_float(row.get("confidence"), 0.0),
                _safe_float(row.get("game_score"), 0.0),
                _safe_float(row.get("rank_score"), 0.0),
                _safe_float(row.get("dog_payout"), 0.0),
                _safe_float(row.get("vegas_spread"), 0.0),
                _safe_int(row.get("vegas_home_ml"), 0),
                _safe_int(row.get("vegas_away_ml"), 0),
                1 if bool(row.get("is_dog_pick")) else 0,
                1 if bool(row.get("is_value_zone")) else 0,
                _safe_float(row.get("ml_sharp_home_public"), 0.0),
                _safe_float(row.get("ml_sharp_home_money"), 0.0),
                filters_json,
                _to_json(why_pick),
                _to_json(feature_snapshot),
            ),
        )
        stored += 1

    return {"run_id": run_id, "stored_count": stored}


def backfill_recommendation_outcomes(
    game_date: Optional[str] = None,
    callback=None,
) -> Dict[str, Any]:
    """Settle unresolved recommendation rows using historical game outcomes."""
    ensure_recommendation_outcomes_schema()
    today = nba_today()
    params: tuple[Any, ...]
    query = """
        SELECT id, game_date, home_team_id, away_team_id, pick,
               game_score, vegas_home_ml, vegas_away_ml
        FROM recommendation_snapshot_items
        WHERE is_settled = 0
          AND game_date <= ?
    """
    if game_date:
        query += " AND game_date = ?"
        params = (today, game_date)
    else:
        params = (today,)

    pending_rows = db.fetch_all(query, params)
    if not pending_rows:
        return {
            "pending": 0,
            "settled": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "roi_pct": 0.0,
        }

    actual_rows = get_actual_game_results()
    actual_map = {
        (str(row.get("game_date")), _safe_int(row.get("home_team_id")), _safe_int(row.get("away_team_id"))): row
        for row in actual_rows
    }

    settled = 0
    wins = 0
    losses = 0
    pushes = 0
    total_profit_units = 0.0

    for row in pending_rows:
        key = (
            str(row.get("game_date", "")),
            _safe_int(row.get("home_team_id"), 0),
            _safe_int(row.get("away_team_id"), 0),
        )
        actual = actual_map.get(key)
        if not actual:
            continue

        pick = str(row.get("pick", "")).upper()
        actual_winner = str(actual.get("winner", "")).upper()
        home_score = _safe_float(actual.get("home_score"), 0.0)
        away_score = _safe_float(actual.get("away_score"), 0.0)
        if actual_winner not in {"HOME", "AWAY", "PUSH"}:
            continue

        if pick == "HOME":
            margin_for_pick = home_score - away_score
            ml_line = _safe_int(row.get("vegas_home_ml"), 0)
        else:
            margin_for_pick = away_score - home_score
            ml_line = _safe_int(row.get("vegas_away_ml"), 0)

        if actual_winner == "PUSH":
            model_correct = 0
            profit_units = 0.0
            pushes += 1
        else:
            model_correct = 1 if pick == actual_winner else 0
            payout = _moneyline_payout_multiplier(ml_line)
            profit_units = (payout - 1.0) if (model_correct and payout > 0.0) else (-1.0 if payout > 0.0 else 0.0)
            if model_correct:
                wins += 1
            else:
                losses += 1

        edge = abs(_safe_float(row.get("game_score"), 0.0))
        realized_edge_delta = margin_for_pick - edge
        roi_pct = profit_units * 100.0

        db.execute(
            """
            UPDATE recommendation_snapshot_items
            SET is_settled = 1,
                settled_at = datetime('now'),
                settlement_source = 'historical_results',
                actual_home_score = ?,
                actual_away_score = ?,
                actual_winner = ?,
                model_correct = ?,
                profit_units = ?,
                roi_pct = ?,
                realized_margin_for_pick = ?,
                realized_edge_delta = ?
            WHERE id = ?
            """,
            (
                home_score,
                away_score,
                actual_winner,
                model_correct,
                profit_units,
                roi_pct,
                margin_for_pick,
                realized_edge_delta,
                _safe_int(row.get("id"), 0),
            ),
        )

        settled += 1
        total_profit_units += profit_units

    if callback:
        callback(f"Settled {settled}/{len(pending_rows)} recommendation outcomes")

    roi_pct = (total_profit_units / max(1, settled)) * 100.0
    return {
        "pending": len(pending_rows),
        "settled": settled,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "profit_units": round(total_profit_units, 4),
        "roi_pct": round(roi_pct, 4),
    }
