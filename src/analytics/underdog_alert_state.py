"""Persistent state tracking for underdog alerts (new/resolved detection)."""

import json
from typing import Any, Dict, Iterable, Mapping

from src.database import db


_ALERT_STATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS underdog_alert_state (
    scope_key TEXT NOT NULL,
    signal_key TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    resolved_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    payload TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scope_key, signal_key)
);

CREATE TABLE IF NOT EXISTS underdog_alert_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_key TEXT NOT NULL,
    game_date TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    total_candidates INTEGER NOT NULL DEFAULT 0,
    alert_count INTEGER NOT NULL DEFAULT 0,
    new_count INTEGER NOT NULL DEFAULT 0,
    resolved_count INTEGER NOT NULL DEFAULT 0,
    digest TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_underdog_alert_state_active
    ON underdog_alert_state(scope_key, is_active, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_underdog_alert_runs_scope
    ON underdog_alert_runs(scope_key, snapshot_at DESC);
"""


def _to_json(payload: Any) -> str:
    try:
        return json.dumps(payload or {}, sort_keys=True)
    except Exception:
        return "{}"


def _parse_json(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ensure_underdog_alert_state_schema():
    """Ensure alert-state persistence tables exist."""
    db.execute_script(_ALERT_STATE_SCHEMA_SQL)


def load_active_underdog_alert_state(scope_key: str) -> Dict[str, Dict[str, Any]]:
    """Return active signal map keyed by signal_key."""
    ensure_underdog_alert_state_schema()
    rows = db.fetch_all(
        """
        SELECT signal_key, payload
        FROM underdog_alert_state
        WHERE scope_key = ? AND is_active = 1
        """,
        (scope_key,),
    )
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("signal_key", "")).strip()
        if not key:
            continue
        payload = _parse_json(row.get("payload", "{}"))
        payload.setdefault("signal_key", key)
        result[key] = payload
    return result


def update_underdog_alert_state(
    scope_key: str,
    game_date: str,
    alerts: Iterable[Mapping[str, Any]],
    total_candidates: int,
    digest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Persist current alert snapshot and return new/resolved diff."""
    ensure_underdog_alert_state_schema()
    previous = load_active_underdog_alert_state(scope_key)
    current: Dict[str, Dict[str, Any]] = {}
    for alert in alerts:
        signal_key = str(alert.get("signal_key", "")).strip()
        if not signal_key:
            continue
        current[signal_key] = dict(alert)

    previous_keys = set(previous.keys())
    current_keys = set(current.keys())
    new_keys = current_keys - previous_keys
    resolved_keys = previous_keys - current_keys
    persisting_keys = previous_keys & current_keys

    for signal_key, alert in current.items():
        payload_json = _to_json(alert)
        db.execute(
            """
            INSERT INTO underdog_alert_state (
                scope_key, signal_key, first_seen_at, last_seen_at, resolved_at, is_active, payload
            )
            VALUES (?, ?, datetime('now'), datetime('now'), NULL, 1, ?)
            ON CONFLICT(scope_key, signal_key) DO UPDATE SET
                last_seen_at = datetime('now'),
                resolved_at = NULL,
                is_active = 1,
                payload = excluded.payload
            """,
            (scope_key, signal_key, payload_json),
        )

    for signal_key in resolved_keys:
        db.execute(
            """
            UPDATE underdog_alert_state
            SET is_active = 0,
                resolved_at = datetime('now'),
                last_seen_at = datetime('now')
            WHERE scope_key = ? AND signal_key = ?
            """,
            (scope_key, signal_key),
        )

    db.execute(
        """
        INSERT INTO underdog_alert_runs (
            scope_key, game_date, snapshot_at, total_candidates, alert_count,
            new_count, resolved_count, digest
        )
        VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?)
        """,
        (
            scope_key,
            game_date,
            int(total_candidates),
            len(current_keys),
            len(new_keys),
            len(resolved_keys),
            _to_json(digest),
        ),
    )

    return {
        "scope_key": scope_key,
        "active_count": len(current_keys),
        "new_count": len(new_keys),
        "resolved_count": len(resolved_keys),
        "persisting_count": len(persisting_keys),
        "new_alerts": [current[key] for key in sorted(new_keys)],
        "resolved_alerts": [previous[key] for key in sorted(resolved_keys)],
    }
