"""NBA Fundamentals V2 -- Flask Web Application.

Routes:
  /                          Dashboard (today's games with predictions)
  /underdogs                 Ranked underdog opportunities + filters
  /matchup/<hid>/<aid>/<date>  Game detail with breakdown + sharp panel
  /accuracy                  Backtest results with A/B comparison
  /gamecast                  Live gamecast view (ESPN integration)
  /api/predict  (POST)       JSON prediction endpoint
  /api/underdogs             JSON underdog screener endpoint
  /api/underdogs/alerts/dispatch (POST) Persist alert state + notify new signals
  /api/sync     (POST)       Trigger data sync (background thread)
  /api/sync/odds-today (POST) Trigger odds-only sync for today's games
  /api/gamecast/games        Today's games (ESPN scoreboard)
  /api/gamecast/<game_id>    Full game data (summary, boxscore, plays, odds)
"""

import csv
import hashlib
import io
import json
import logging
import hmac
import os
import re
import secrets
import signal
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from flask import Flask, render_template, jsonify, request, session, abort, Response

from src.analytics.alert_rules import (
    build_underdog_alert_candidates,
    build_underdog_alert_digest,
)
from src.analytics.drift_monitor import evaluate_underdog_drift
from src.analytics.phase_gates import evaluate_phase_acceptance
from src.analytics.recommendation_outcomes import persist_recommendation_snapshot
from src.analytics.underdog_alert_state import update_underdog_alert_state
from src.analytics.underdog_metrics import quality_tier_for_confidence
from src.config import get as get_setting
from src.notifications.models import NotificationCategory, NotificationSeverity
from src.notifications.service import create_notification
from src.utils.timezone_utils import nba_today, nba_game_date_from_utc_iso

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# Secret key for session + CSRF token signing.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

_MATCHUP_ABBR_RE = re.compile(r"^[A-Za-z]{2,4}$")
_MATCHUP_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SYNC_RATE_LIMIT_SECONDS = max(1, int(os.environ.get("NBA_SYNC_RATE_LIMIT_SECONDS", "5")))
_SHUTDOWN_ENABLED = os.environ.get("NBA_WEB_SHUTDOWN_ENABLED", "0") == "1"
_SHUTDOWN_TOKEN = os.environ.get("NBA_SHUTDOWN_TOKEN", "").strip()
_DEPLOY_ENABLED = os.environ.get("NBA_DEPLOY_ENABLED", "0") == "1"
_DEPLOY_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deploy.sh")
_DEPLOY_STATUS_FILE = os.path.join("data", "deploy_status.json")
_STYLE_ASSET_PATH = os.path.join(os.path.dirname(__file__), "static", "style.css")
_CSRF_PROTECTED_ENDPOINTS = {
    "api_predict",
    "api_sync",
    "api_sync_odds_today",
    "api_underdogs_alerts_dispatch",
    "api_shutdown",
    "api_deploy",
}
_UNDERDOG_TIERS = {"ALL", "A", "B", "C"}
_UNDERDOG_PRESETS = {"all", "high_quality", "value_zone", "long_dogs", "balanced"}
_UNDERDOG_SORT_FIELDS = {"rank_score", "confidence", "dog_payout", "edge", "start_time"}
_UNDERDOG_PRESET_LABELS = {
    "all": "All",
    "high_quality": "High Quality",
    "value_zone": "Value Zone",
    "long_dogs": "Long Dogs",
    "balanced": "Balanced",
}
_ADJUSTMENT_DISPLAY_NAMES = {
    "fatigue": "Fatigue",
    "turnover": "Turnovers",
    "rebound": "Rebounds",
    "rating_matchup": "Rating Matchup",
    "ff_efg": "Effective FG%",
    "ff_tov": "Turnover Rate",
    "ff_oreb": "Off. Rebound%",
    "ff_fta": "Free Throw Rate",
    "ff_def_efg": "Opp. Effective FG%",
    "ff_def_tov": "Opp. Turnover Rate",
    "ff_def_oreb": "Opp. Off. Reb%",
    "ff_def_fta": "Opp. Free Throw Rate",
    "clutch": "Clutch Performance",
    "hustle": "Hustle Stats",
    "pace": "Pace Differential",
    "sharp_money": "Sharp Money",
    "sharp_ml": "Sharp Money",
    "interaction_correction": "Interaction Model",
}


def _ensure_csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _asset_version(path: str) -> str:
    try:
        return str(int(os.path.getmtime(path)))
    except OSError:
        return "1"


@app.context_processor
def _inject_template_globals():
    return {
        "csrf_token": _ensure_csrf_token(),
        "shutdown_enabled": _SHUTDOWN_ENABLED,
        "deploy_enabled": _DEPLOY_ENABLED,
        "style_version": _asset_version(_STYLE_ASSET_PATH),
    }


@app.before_request
def _csrf_protect():
    if request.method != "POST":
        return None
    if request.endpoint not in _CSRF_PROTECTED_ENDPOINTS:
        return None
    expected = session.get("_csrf_token", "")
    provided = request.headers.get("X-CSRF-Token", "")
    if not expected or not provided or not hmac.compare_digest(expected, provided):
        return jsonify({"error": "CSRF validation failed"}), 403
    return None


@app.after_request
def _apply_security_headers(response):
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif request.path.startswith("/static/"):
        # Static bundles are immutable between deploys in this local app.
        response.headers["Cache-Control"] = "public, max-age=3600"
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "base-uri 'self'; "
        "frame-ancestors 'self'; "
        "form-action 'self'; "
        "connect-src 'self'; "
        "img-src 'self' data: https:; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:",
    )
    return response


def _is_valid_matchup_path(home_abbr: str, away_abbr: str, date: str) -> bool:
    if not _MATCHUP_ABBR_RE.match(home_abbr or ""):
        return False
    if not _MATCHUP_ABBR_RE.match(away_abbr or ""):
        return False
    if not _MATCHUP_DATE_RE.match(date or ""):
        return False
    try:
        datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _json_error(message: str, status: int):
    return jsonify({"error": message}), status


def _parse_bool_arg(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _safe_float_arg(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _safe_int_arg(
    raw: Any,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _build_date_strip(today: str, selected_date: str, days: int = 7) -> List[Dict[str, Any]]:
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    dates = []
    for i in range(days):
        day = today_dt + timedelta(days=i)
        value = day.strftime("%Y-%m-%d")
        is_today = i == 0
        dates.append({
            "label": (
                day.strftime("%a %-m/%d")
                if os.name != "nt"
                else day.strftime("%a %#m/%d")
            ),
            "day_name": "TODAY" if is_today else day.strftime("%a").upper(),
            "day_num": str(day.day),
            "month": day.strftime("%b").upper(),
            "value": value,
            "is_active": value == selected_date,
            "is_today": is_today,
        })
    return dates


def _parse_underdog_filters() -> Dict[str, Any]:
    preset = str(request.args.get("preset", "all")).strip().lower()
    if preset not in _UNDERDOG_PRESETS:
        preset = "all"

    preset_defaults: Dict[str, Dict[str, Any]] = {
        "all": {
            "tier": "ALL",
            "min_confidence": 0.0,
            "min_payout": 0.0,
            "value_only": False,
            "include_sharp": False,
            "sort_by": "rank_score",
            "sort_dir": "desc",
            "limit": 24,
        },
        "high_quality": {
            "tier": "A",
            "min_confidence": 70.0,
            "min_payout": 1.7,
            "value_only": False,
            "include_sharp": True,
            "sort_by": "confidence",
            "sort_dir": "desc",
            "limit": 24,
        },
        "value_zone": {
            "tier": "ALL",
            "min_confidence": 58.0,
            "min_payout": 1.7,
            "value_only": True,
            "include_sharp": True,
            "sort_by": "rank_score",
            "sort_dir": "desc",
            "limit": 24,
        },
        "long_dogs": {
            "tier": "ALL",
            "min_confidence": 55.0,
            "min_payout": 3.0,
            "value_only": False,
            "include_sharp": True,
            "sort_by": "dog_payout",
            "sort_dir": "desc",
            "limit": 40,
        },
        "balanced": {
            "tier": "ALL",
            "min_confidence": 60.0,
            "min_payout": 1.9,
            "value_only": False,
            "include_sharp": True,
            "sort_by": "rank_score",
            "sort_dir": "desc",
            "limit": 30,
        },
    }
    defaults = preset_defaults[preset]

    today = nba_today()
    selected_date = request.args.get("date", today)
    if not _MATCHUP_DATE_RE.match(selected_date or ""):
        selected_date = today
    else:
        try:
            datetime.strptime(selected_date, "%Y-%m-%d")
        except ValueError:
            selected_date = today

    tier = str(request.args.get("tier", defaults["tier"])).upper()
    if tier not in _UNDERDOG_TIERS:
        tier = str(defaults["tier"])

    min_confidence = _safe_float_arg(
        request.args.get("min_confidence"),
        float(defaults["min_confidence"]),
    )
    min_confidence = max(0.0, min(100.0, min_confidence))

    min_payout = _safe_float_arg(
        request.args.get("min_payout"),
        float(defaults["min_payout"]),
    )
    min_payout = max(0.0, min_payout)

    include_sharp = _parse_bool_arg(
        request.args.get("include_sharp"),
        default=bool(defaults["include_sharp"]),
    )
    value_only = _parse_bool_arg(
        request.args.get("value_only"),
        default=bool(defaults["value_only"]),
    )
    limit = _safe_int_arg(
        request.args.get("limit"),
        default=int(defaults["limit"]),
        min_value=1,
        max_value=200,
    )

    sort_by = str(request.args.get("sort_by", defaults["sort_by"])).strip().lower()
    sort_aliases = {
        "rank": "rank_score",
        "score": "rank_score",
        "edge_score": "rank_score",
        "edge": "edge",
        "game_score": "edge",
        "start": "start_time",
        "start_utc": "start_time",
    }
    sort_by = sort_aliases.get(sort_by, sort_by)
    if sort_by not in _UNDERDOG_SORT_FIELDS:
        sort_by = str(defaults["sort_by"])

    sort_dir = str(request.args.get("sort_dir", defaults["sort_dir"])).strip().lower()
    if sort_dir not in {"asc", "desc"}:
        sort_dir = str(defaults["sort_dir"])

    return {
        "preset": preset,
        "date": selected_date,
        "tier": tier,
        "min_confidence": min_confidence,
        "min_payout": min_payout,
        "include_sharp": include_sharp,
        "value_only": value_only,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "limit": limit,
    }


def _adjustment_display_name(key: Any) -> str:
    text = str(key or "").strip()
    if not text:
        return ""
    return _ADJUSTMENT_DISPLAY_NAMES.get(text, text.replace("_", " ").title())


def _build_pick_drivers(pred: Dict[str, Any], max_items: int = 3) -> List[Dict[str, Any]]:
    raw_adjustments = pred.get("adjustments")
    if not isinstance(raw_adjustments, dict):
        return []

    pick = str(pred.get("pick", "")).upper()
    rows: List[Dict[str, Any]] = []
    for key, value in raw_adjustments.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if abs(numeric) < 0.05:
            continue

        direction = "home" if numeric > 0 else "away" if numeric < 0 else "neutral"
        supports_pick = (
            (pick == "HOME" and numeric > 0)
            or (pick == "AWAY" and numeric < 0)
        )
        rows.append(
            {
                "key": str(key),
                "label": _adjustment_display_name(key),
                "value": round(numeric, 2),
                "direction": direction,
                "supports_pick": supports_pick,
                "abs_value": abs(numeric),
            }
        )

    if not rows:
        return []

    rows.sort(
        key=lambda row: (
            1 if row["supports_pick"] else 0,
            row["abs_value"],
        ),
        reverse=True,
    )
    top_rows = rows[: max(1, int(max_items))]
    for row in top_rows:
        row.pop("abs_value", None)
    return top_rows


def _build_caution_flags(pred: Dict[str, Any], drivers_count: int) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []

    confidence = max(0.0, min(100.0, _safe_float_arg(pred.get("confidence"), 0.0)))
    edge = abs(_safe_float_arg(pred.get("game_score"), 0.0))
    dog_payout = max(0.0, _safe_float_arg(pred.get("dog_payout"), 0.0))
    is_value_zone = bool(pred.get("is_value_zone"))
    sharp_agrees = pred.get("sharp_agrees")
    home_public = _safe_float_arg(pred.get("ml_sharp_home_public"), 0.0)
    home_money = _safe_float_arg(pred.get("ml_sharp_home_money"), 0.0)

    if edge < 3.0:
        flags.append({
            "code": "EDGE_SIZE_RISK",
            "severity": "high",
            "label": "Thin model edge",
            "detail": f"|score| {edge:.2f} < 3.0",
        })
    elif edge < 6.0:
        flags.append({
            "code": "EDGE_SIZE_RISK",
            "severity": "medium",
            "label": "Moderate edge only",
            "detail": f"|score| {edge:.2f} < 6.0",
        })

    if confidence < 55.0:
        flags.append({
            "code": "LOW_CONFIDENCE",
            "severity": "high",
            "label": "Lower confidence tier",
            "detail": f"{confidence:.0f}% < 55%",
        })
    elif confidence < 70.0:
        flags.append({
            "code": "MID_CONFIDENCE",
            "severity": "medium",
            "label": "Mid confidence tier",
            "detail": f"{confidence:.0f}% < 70%",
        })

    if dog_payout <= 0:
        flags.append({
            "code": "MARKET_DATA_GAP",
            "severity": "medium",
            "label": "Missing moneyline payout",
            "detail": "Dog payout unavailable",
        })
    elif dog_payout >= 4.0:
        flags.append({
            "code": "LONG_DOG_VOLATILITY",
            "severity": "high",
            "label": "Very long dog variance",
            "detail": f"Payout {dog_payout:.2f}x",
        })
    elif dog_payout >= 3.0:
        flags.append({
            "code": "LONG_DOG_VOLATILITY",
            "severity": "medium",
            "label": "Long dog variance",
            "detail": f"Payout {dog_payout:.2f}x",
        })

    if not is_value_zone:
        flags.append({
            "code": "OUTSIDE_VALUE_ZONE",
            "severity": "low",
            "label": "Outside value zone",
            "detail": "Market spread not in value band",
        })

    if sharp_agrees is False:
        divergence = abs(home_money - home_public)
        flags.append({
            "code": "SHARP_DISAGREEMENT",
            "severity": "high" if divergence >= 10.0 else "medium",
            "label": "Sharp money disagrees",
            "detail": f"Split divergence {divergence:.0f}pp",
        })
    elif sharp_agrees is None and (home_public <= 0 or home_money <= 0):
        flags.append({
            "code": "SHARP_DATA_GAP",
            "severity": "low",
            "label": "Sharp split unavailable",
            "detail": "Public/money split missing",
        })

    if drivers_count < 2:
        flags.append({
            "code": "WEAK_DRIVER_SIGNAL",
            "severity": "medium",
            "label": "Few strong model drivers",
            "detail": f"{drivers_count} notable driver(s)",
        })

    return flags


def _build_why_pick_payload(pred: Dict[str, Any]) -> Dict[str, Any]:
    pick = str(pred.get("pick", "")).upper()
    pick_team = pred.get("home_team") if pick == "HOME" else pred.get("away_team")
    confidence = max(0.0, min(100.0, _safe_float_arg(pred.get("confidence"), 0.0)))
    edge = abs(_safe_float_arg(pred.get("game_score"), 0.0))
    dog_payout = max(0.0, _safe_float_arg(pred.get("dog_payout"), 0.0))

    drivers = _build_pick_drivers(pred, max_items=3)
    caution_flags = _build_caution_flags(pred, drivers_count=len(drivers))

    top_driver = drivers[0] if drivers else None
    top_driver_text = (
        f" Top driver: {top_driver['label']} ({top_driver['value']:+.2f})."
        if top_driver
        else ""
    )
    summary = (
        f"{pick_team or pick} underdog edge {edge:.1f} "
        f"at {confidence:.0f}% confidence.{top_driver_text}"
    ).strip()

    sharp_divergence = None
    if pred.get("ml_sharp_home_public") is not None and pred.get("ml_sharp_home_money") is not None:
        sharp_divergence = (
            _safe_float_arg(pred.get("ml_sharp_home_money"), 0.0)
            - _safe_float_arg(pred.get("ml_sharp_home_public"), 0.0)
        )

    return {
        "summary": summary,
        "edge": {
            "game_score": round(_safe_float_arg(pred.get("game_score"), 0.0), 3),
            "confidence": round(confidence, 1),
            "tier": pred.get("tier"),
            "pick": pick,
        },
        "market": {
            "vegas_spread": _safe_float_arg(pred.get("vegas_spread"), 0.0),
            "is_value_zone": bool(pred.get("is_value_zone")),
            "dog_payout": round(dog_payout, 3),
        },
        "sharp": {
            "home_public_pct": _safe_float_arg(pred.get("ml_sharp_home_public"), 0.0),
            "home_money_pct": _safe_float_arg(pred.get("ml_sharp_home_money"), 0.0),
            "divergence_pct": sharp_divergence,
            "agrees": pred.get("sharp_agrees"),
        },
        "drivers": drivers,
        "caution_flags": caution_flags,
    }


def _score_underdog_candidate(pred: Dict[str, Any]) -> float:
    confidence = max(0.0, min(100.0, _safe_float_arg(pred.get("confidence"), 0.0)))
    dog_payout = max(0.0, _safe_float_arg(pred.get("dog_payout"), 0.0))
    edge_magnitude = abs(_safe_float_arg(pred.get("game_score"), 0.0))
    score = confidence
    score += min(20.0, edge_magnitude * 0.9)
    score += max(0.0, (dog_payout - 1.5) * 10.0)
    if pred.get("is_value_zone"):
        score += 6.0
    return score


def _rank_underdog_candidates(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for pred in predictions:
        if not pred.get("is_dog_pick"):
            continue
        row = dict(pred)
        confidence = max(0.0, min(100.0, _safe_float_arg(row.get("confidence"), 0.0)))
        dog_payout = max(0.0, _safe_float_arg(row.get("dog_payout"), 0.0))
        row["confidence"] = confidence
        row["dog_payout"] = dog_payout
        row["tier"] = quality_tier_for_confidence(confidence)
        row["rank_score"] = round(_score_underdog_candidate(row), 2)
        why_pick = _build_why_pick_payload(row)
        row["why_pick"] = why_pick
        row["caution_flags"] = why_pick.get("caution_flags", [])
        row["drivers"] = why_pick.get("drivers", [])
        ranked.append(row)

    ranked.sort(
        key=lambda row: (
            row.get("rank_score", 0.0),
            row.get("confidence", 0.0),
            row.get("dog_payout", 0.0),
        ),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def _filter_ranked_underdogs(
    ranked: List[Dict[str, Any]],
    tier: str,
    min_confidence: float,
    min_payout: float,
    value_only: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row in ranked:
        if tier != "ALL" and row.get("tier") != tier:
            continue
        if _safe_float_arg(row.get("confidence"), 0.0) < min_confidence:
            continue
        if _safe_float_arg(row.get("dog_payout"), 0.0) < min_payout:
            continue
        if value_only and not row.get("is_value_zone"):
            continue
        results.append(row)
        if len(results) >= limit:
            break
    return results


def _sort_ranked_underdogs(
    ranked: List[Dict[str, Any]],
    sort_by: str,
    sort_dir: str,
) -> List[Dict[str, Any]]:
    reverse = sort_dir != "asc"
    normalized = sort_by if sort_by in _UNDERDOG_SORT_FIELDS else "rank_score"

    def key_for_row(row: Dict[str, Any]):
        if normalized == "confidence":
            return _safe_float_arg(row.get("confidence"), 0.0)
        if normalized == "dog_payout":
            return _safe_float_arg(row.get("dog_payout"), 0.0)
        if normalized == "edge":
            return abs(_safe_float_arg(row.get("game_score"), 0.0))
        if normalized == "start_time":
            raw = str(row.get("start_utc") or "")
            return raw if raw else "9999-12-31T23:59:59Z"
        return _safe_float_arg(row.get("rank_score"), 0.0)

    return sorted(ranked, key=key_for_row, reverse=reverse)


def _summarize_screened_underdogs(screened: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(screened)
    if total == 0:
        return {
            "count": 0,
            "value_zone_count": 0,
            "avg_confidence": 0.0,
            "avg_payout": 0.0,
            "avg_edge": 0.0,
            "tier_counts": {"A": 0, "B": 0, "C": 0},
        }

    value_zone_count = sum(1 for row in screened if row.get("is_value_zone"))
    tier_counts = {
        "A": sum(1 for row in screened if row.get("tier") == "A"),
        "B": sum(1 for row in screened if row.get("tier") == "B"),
        "C": sum(1 for row in screened if row.get("tier") == "C"),
    }
    avg_confidence = sum(_safe_float_arg(row.get("confidence"), 0.0) for row in screened) / total
    avg_payout = sum(_safe_float_arg(row.get("dog_payout"), 0.0) for row in screened) / total
    avg_edge = (
        sum(abs(_safe_float_arg(row.get("game_score"), 0.0)) for row in screened) / total
    )
    return {
        "count": total,
        "value_zone_count": value_zone_count,
        "avg_confidence": round(avg_confidence, 2),
        "avg_payout": round(avg_payout, 3),
        "avg_edge": round(avg_edge, 3),
        "tier_counts": tier_counts,
    }


def _filters_to_query_params(filters: Dict[str, Any]) -> Dict[str, str]:
    return {
        "preset": str(filters.get("preset", "all")),
        "date": str(filters.get("date", nba_today())),
        "tier": str(filters.get("tier", "ALL")),
        "min_confidence": f"{_safe_float_arg(filters.get('min_confidence'), 0.0):.0f}",
        "min_payout": f"{_safe_float_arg(filters.get('min_payout'), 0.0):.1f}",
        "include_sharp": "1" if filters.get("include_sharp") else "0",
        "value_only": "1" if filters.get("value_only") else "0",
        "sort_by": str(filters.get("sort_by", "rank_score")),
        "sort_dir": str(filters.get("sort_dir", "desc")),
        "limit": str(_safe_int_arg(filters.get("limit"), 24, min_value=1, max_value=200)),
    }


def _underdog_alert_scope_key(filters: Dict[str, Any]) -> str:
    canonical = {
        "date": str(filters.get("date", nba_today())),
        "preset": str(filters.get("preset", "all")),
        "tier": str(filters.get("tier", "ALL")),
        "min_confidence": round(_safe_float_arg(filters.get("min_confidence"), 0.0), 2),
        "min_payout": round(_safe_float_arg(filters.get("min_payout"), 0.0), 3),
        "include_sharp": bool(filters.get("include_sharp")),
        "value_only": bool(filters.get("value_only")),
    }
    encoded = json.dumps(canonical, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:16]
    return f"{canonical['date']}:{digest}"


def _underdog_alert_signal_key(alert: Dict[str, Any]) -> str:
    game_date = str(alert.get("game_date", "")).strip()
    away_ref = str(alert.get("away_team_id") or alert.get("away_team") or "").strip()
    home_ref = str(alert.get("home_team_id") or alert.get("home_team") or "").strip()
    pick = str(alert.get("pick", "")).strip().upper()
    code = str(alert.get("code", "")).strip().upper()
    return f"{game_date}:{away_ref}:{home_ref}:{pick}:{code}"


def _with_underdog_alert_keys(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keyed: List[Dict[str, Any]] = []
    for alert in alerts:
        row = dict(alert)
        row["signal_key"] = _underdog_alert_signal_key(row)
        keyed.append(row)
    return keyed


def _severity_for_underdog_alert(priority: Any) -> str:
    score = _safe_float_arg(priority, 0.0)
    if score >= 90.0:
        return NotificationSeverity.CRITICAL.value
    if score >= 80.0:
        return NotificationSeverity.WARNING.value
    return NotificationSeverity.INFO.value


def _dispatch_underdog_alert_notifications(
    scope_key: str,
    alert_state: Dict[str, Any],
    notify_resolved: bool = False,
) -> List[Dict[str, Any]]:
    created: List[Dict[str, Any]] = []
    new_alerts = alert_state.get("new_alerts", [])
    if isinstance(new_alerts, list):
        for alert in new_alerts:
            confidence = _safe_float_arg(alert.get("confidence"), 0.0)
            payout = _safe_float_arg(alert.get("dog_payout"), 0.0)
            title = f"Underdog Alert: {alert.get('label', 'Signal')}"
            message = (
                f"{alert.get('away_team', '?')} @ {alert.get('home_team', '?')} - "
                f"{str(alert.get('pick', '')).upper()} "
                f"({confidence:.0f}%, {payout:.2f}x)"
            )
            payload = {
                "scope_key": scope_key,
                "signal_key": alert.get("signal_key"),
                "state": "new",
                "alert": alert,
            }
            notification_id = create_notification(
                NotificationCategory.UNDERDOG.value,
                _severity_for_underdog_alert(alert.get("priority")),
                title,
                message,
                data=payload,
            )
            created.append({
                "notification_id": notification_id,
                "signal_key": alert.get("signal_key"),
                "state": "new",
            })

    if notify_resolved:
        resolved_alerts = alert_state.get("resolved_alerts", [])
        if isinstance(resolved_alerts, list):
            for alert in resolved_alerts:
                title = "Underdog Alert Resolved"
                message = (
                    f"{alert.get('away_team', '?')} @ {alert.get('home_team', '?')} "
                    f"{str(alert.get('pick', '')).upper()} signal resolved"
                )
                payload = {
                    "scope_key": scope_key,
                    "signal_key": alert.get("signal_key"),
                    "state": "resolved",
                    "alert": alert,
                }
                notification_id = create_notification(
                    NotificationCategory.UNDERDOG.value,
                    NotificationSeverity.INFO.value,
                    title,
                    message,
                    data=payload,
                )
                created.append({
                    "notification_id": notification_id,
                    "signal_key": alert.get("signal_key"),
                    "state": "resolved",
                })
    return created


def _run_underdog_screen(filters: Dict[str, Any]) -> Dict[str, Any]:
    selected_date = str(filters.get("date", nba_today()))
    predictions = _get_games_for_date(
        selected_date,
        include_sharp=bool(filters.get("include_sharp")),
    )
    ranked_all = _rank_underdog_candidates(predictions)
    sorted_rows = _sort_ranked_underdogs(
        ranked_all,
        sort_by=str(filters.get("sort_by", "rank_score")),
        sort_dir=str(filters.get("sort_dir", "desc")),
    )
    screened = _filter_ranked_underdogs(
        sorted_rows,
        tier=str(filters.get("tier", "ALL")),
        min_confidence=_safe_float_arg(filters.get("min_confidence"), 0.0),
        min_payout=_safe_float_arg(filters.get("min_payout"), 0.0),
        value_only=bool(filters.get("value_only")),
        limit=_safe_int_arg(filters.get("limit"), 24, min_value=1, max_value=200),
    )
    for idx, row in enumerate(screened, start=1):
        row["list_rank"] = idx
    return {
        "ranked_all": ranked_all,
        "screened": screened,
        "summary": _summarize_screened_underdogs(screened),
    }


def _build_underdog_digest_text(
    selected_date: str,
    filters: Dict[str, Any],
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    alerts: List[Dict[str, Any]],
) -> str:
    lines = [
        f"Underdog digest for {selected_date}",
        (
            f"Filters: preset={filters.get('preset')} tier={filters.get('tier')} "
            f"min_conf={_safe_float_arg(filters.get('min_confidence'), 0.0):.0f}% "
            f"min_payout={_safe_float_arg(filters.get('min_payout'), 0.0):.1f}x "
            f"sort={filters.get('sort_by')}:{filters.get('sort_dir')}"
        ),
        (
            f"Screened: {summary.get('count', 0)} | value-zone: "
            f"{summary.get('value_zone_count', 0)} | "
            f"avg_conf: {_safe_float_arg(summary.get('avg_confidence'), 0.0):.1f}% | "
            f"avg_payout: {_safe_float_arg(summary.get('avg_payout'), 0.0):.2f}x"
        ),
        f"Alerts: {len(alerts)}",
        "",
        "Top underdogs:",
    ]

    for row in rows[: min(10, len(rows))]:
        pick = str(row.get("pick", "")).upper()
        pick_team = row.get("home_team") if pick == "HOME" else row.get("away_team")
        why_pick = row.get("why_pick") if isinstance(row.get("why_pick"), dict) else {}
        lines.append(
            (
                f"- #{row.get('list_rank', '?')} {row.get('away_team')} @ {row.get('home_team')}: "
                f"{pick_team} ({_safe_float_arg(row.get('confidence'), 0.0):.0f}%, "
                f"{_safe_float_arg(row.get('dog_payout'), 0.0):.2f}x, "
                f"tier {row.get('tier')}, score {_safe_float_arg(row.get('rank_score'), 0.0):.1f}) "
                f"- {why_pick.get('summary', '')}"
            ).strip()
        )

    if alerts:
        lines.append("")
        lines.append("Alert candidates:")
        for alert in alerts[: min(8, len(alerts))]:
            lines.append(
                (
                    f"- {alert.get('label')}: {alert.get('away_team')} @ {alert.get('home_team')} "
                    f"({float(alert.get('confidence', 0.0)):.0f}%, {float(alert.get('dog_payout', 0.0)):.2f}x)"
                )
            )

    return "\n".join(lines)


def _build_phase_acceptance_report_meta(raw_report: Any) -> Dict[str, Any]:
    report_path = ""
    latest_path = ""
    if isinstance(raw_report, dict):
        report_path = str(raw_report.get("report_path", "") or "").strip()
        latest_path = str(raw_report.get("latest_path", "") or "").strip()

    report_dir = str(
        get_setting("weekly_frontier_report_dir", "data/reports") or "data/reports"
    ).strip() or "data/reports"
    default_latest = os.path.join(report_dir, "phase_acceptance_latest.json")
    if not latest_path:
        latest_path = default_latest
    if not report_path:
        report_path = latest_path

    return {
        "report_path": report_path,
        "latest_path": latest_path,
        "exists": bool(
            (report_path and os.path.exists(report_path))
            or (latest_path and os.path.exists(latest_path))
        ),
    }


def _enrich_predictions_with_start_times(predictions: List[Dict[str, Any]], game_date: str):
    if not predictions:
        return

    start_lookup: Dict[str, str] = {}
    try:
        from src.data.gamecast import fetch_espn_scoreboard

        for game in fetch_espn_scoreboard(game_date):
            key = f"{game.get('away_team', '')}@{game.get('home_team', '')}"
            start_lookup[key] = str(game.get("date", "") or "")
    except Exception as e:
        logger.debug("ESPN start-time enrichment failed: %s", e)

    missing_keys = []
    for pred in predictions:
        key = f"{pred.get('away_team', '')}@{pred.get('home_team', '')}"
        start_utc = start_lookup.get(key, "")
        pred["start_utc"] = start_utc
        if not start_utc:
            missing_keys.append(key)

    if missing_keys:
        try:
            from src.data.nba_fetcher import fetch_nba_cdn_schedule

            for game in fetch_nba_cdn_schedule():
                if game.get("game_date") != game_date:
                    continue
                key = f"{game.get('away_team', '')}@{game.get('home_team', '')}"
                if key in start_lookup and start_lookup[key]:
                    continue
                start_lookup[key] = str(game.get("game_time_utc", "") or "")
        except Exception as e:
            logger.debug("CDN start-time enrichment failed: %s", e)

        for pred in predictions:
            if pred.get("start_utc"):
                continue
            key = f"{pred.get('away_team', '')}@{pred.get('home_team', '')}"
            pred["start_utc"] = start_lookup.get(key, "")

    predictions.sort(
        key=lambda pred: (
            pred.get("start_utc") or "9999-12-31T23:59:59Z",
            pred.get("away_team", ""),
            pred.get("home_team", ""),
        )
    )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _get_todays_games() -> List[Dict[str, Any]]:
    """Fetch today's games from game_odds and run predictions."""
    return _get_games_for_date(nba_today(), include_sharp=False)


def _get_games_for_date(game_date: str, include_sharp: bool = False) -> List[Dict[str, Any]]:
    """Fetch game_odds rows for a date and run predictions."""
    from src.database import db
    from src.analytics.stats_engine import get_team_abbreviations, get_team_names

    abbr_map = get_team_abbreviations()
    name_map = get_team_names()

    games = db.fetch_all(
        "SELECT home_team_id, away_team_id, spread, home_moneyline, away_moneyline "
        "FROM game_odds WHERE game_date = ?",
        (game_date,),
    )

    predictions = []
    for g in games:
        home_id = g["home_team_id"]
        away_id = g["away_team_id"]
        try:
            pred_dict = _run_prediction(
                home_id,
                away_id,
                game_date,
                include_sharp=include_sharp,
            )
            predictions.append(pred_dict)
        except Exception as e:
            logger.warning("Failed to predict %s vs %s: %s", home_id, away_id, e)
            # Still show the game card with basic info
            predictions.append({
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": abbr_map.get(home_id, str(home_id)),
                "away_team": abbr_map.get(away_id, str(away_id)),
                "home_name": name_map.get(home_id, ""),
                "away_name": name_map.get(away_id, ""),
                "game_date": game_date,
                "pick": "",
                "confidence": 0,
                "projected_home_pts": 0,
                "projected_away_pts": 0,
                "is_dog_pick": False,
                "is_value_zone": False,
                "dog_payout": 0.0,
                "game_score": 0,
                "vegas_spread": g.get("spread") or 0,
                "error": "Prediction unavailable",
            })

    _enrich_predictions_with_start_times(predictions, game_date)
    return predictions


def _count_missing_odds_for_today(game_date: str) -> Optional[int]:
    """Best-effort count of today's scoreboard matchups still missing odds."""
    try:
        from src.data.gamecast import fetch_espn_scoreboard
        from src.analytics.stats_engine import get_team_abbreviations
        from src.database import db
    except Exception:
        logger.debug("Missing-odds check imports unavailable", exc_info=True)
        return None

    try:
        scoreboard_games = fetch_espn_scoreboard()
    except Exception:
        logger.debug("Could not load scoreboard for missing-odds check", exc_info=True)
        return None

    id_to_abbr = get_team_abbreviations() or {}
    abbr_to_id = {}
    for team_id, abbr in id_to_abbr.items():
        if abbr is None:
            continue
        try:
            abbr_to_id[str(abbr).upper()] = int(team_id)
        except (TypeError, ValueError):
            continue

    matchups = []
    seen = set()
    for game in scoreboard_games:
        home_abbr = str(game.get("home_team", "")).upper().strip()
        away_abbr = str(game.get("away_team", "")).upper().strip()
        home_id = abbr_to_id.get(home_abbr)
        away_id = abbr_to_id.get(away_abbr)
        if not home_id or not away_id:
            continue
        key = (home_id, away_id)
        if key in seen:
            continue
        seen.add(key)
        matchups.append(key)

    if not matchups:
        return 0

    missing_count = 0
    for home_id, away_id in matchups:
        odds_row = db.fetch_one(
            """
            SELECT spread, over_under, home_moneyline, away_moneyline
            FROM game_odds
            WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?
            """,
            (game_date, home_id, away_id),
        )
        if not odds_row:
            missing_count += 1
            continue
        has_any_odds = any(
            odds_row.get(field) is not None
            for field in ("spread", "over_under", "home_moneyline", "away_moneyline")
        )
        if not has_any_odds:
            missing_count += 1

    return missing_count


def _format_starters_out(starters_out: List[Dict[str, Any]]) -> List[str]:
    """Format starters-out entries for compact UI display."""
    formatted: List[str] = []
    for entry in starters_out or []:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        status = str(entry.get("status") or "").strip()
        if status and status.lower() != "out":
            formatted.append(f"{name} ({status})")
        else:
            formatted.append(name)
    return formatted


def _build_matchup_team_context(
    home_id: int,
    away_id: int,
    game_date: str,
) -> Dict[str, Dict[str, Any]]:
    """Build per-team context used by matchup views."""
    from src.analytics.team_context import get_team_display_context

    return {
        "home": get_team_display_context(
            home_id, game_date=game_date, include_starters_out=True
        ),
        "away": get_team_display_context(
            away_id, game_date=game_date, include_starters_out=True
        ),
    }


def _attach_team_context(
    pred_dict: Dict[str, Any],
    team_context: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attach display context fields to prediction dict."""
    if not team_context:
        return pred_dict

    home_ctx = team_context.get("home", {})
    away_ctx = team_context.get("away", {})

    pred_dict.update({
        "home_record": home_ctx.get("record", ""),
        "away_record": away_ctx.get("record", ""),
        "home_streak": home_ctx.get("streak", ""),
        "away_streak": away_ctx.get("streak", ""),
        "home_days_since_last_game": home_ctx.get("days_since_last_game"),
        "away_days_since_last_game": away_ctx.get("days_since_last_game"),
        "home_last_game_text": home_ctx.get("last_game_text", ""),
        "away_last_game_text": away_ctx.get("last_game_text", ""),
        "home_last_game_short": home_ctx.get("last_game_short", ""),
        "away_last_game_short": away_ctx.get("last_game_short", ""),
        "home_starters_out": _format_starters_out(home_ctx.get("starters_out", [])),
        "away_starters_out": _format_starters_out(away_ctx.get("starters_out", [])),
        "home_starters_confirmed": home_ctx.get("starters_confirmed", False),
        "away_starters_confirmed": away_ctx.get("starters_confirmed", False),
        "home_minutes_restricted": home_ctx.get("minutes_restricted", []),
        "away_minutes_restricted": away_ctx.get("minutes_restricted", []),
    })
    return pred_dict


def _fetch_live_score(home_abbr: str, away_abbr: str, game_date: str) -> Optional[Dict[str, Any]]:
    """Return live/final score data from ESPN if game is in progress or completed.

    Only fetches for today's date (ESPN scoreboard won't have historical data).
    Returns None for pre-game or if no matching game found.
    """
    today = nba_today()
    if game_date != today:
        return None
    try:
        from src.data.gamecast import fetch_espn_scoreboard
        for g in fetch_espn_scoreboard(game_date):
            if (g.get("home_team", "").upper() == home_abbr.upper()
                    and g.get("away_team", "").upper() == away_abbr.upper()):
                if g.get("state", "pre") == "pre":
                    return None
                return {
                    "state": g["state"],
                    "short_detail": g.get("short_detail", ""),
                    "home_score": g.get("home_score", 0),
                    "away_score": g.get("away_score", 0),
                    "home_abbr": home_abbr.upper(),
                    "away_abbr": away_abbr.upper(),
                }
    except Exception as e:
        logger.debug("Live score fetch failed: %s", e)
    return None


def _run_prediction(home_id: int, away_id: int, game_date: str,
                    include_sharp: bool = False,
                    team_context: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Run a single prediction and return as dict."""
    from src.analytics.prediction import predict_matchup
    from src.analytics.stats_engine import get_team_names

    pred = predict_matchup(home_id, away_id, game_date, include_sharp=include_sharp)
    pred_dict = asdict(pred)
    name_map = get_team_names()
    pred_dict["home_name"] = name_map.get(home_id, pred_dict.get("home_team", ""))
    pred_dict["away_name"] = name_map.get(away_id, pred_dict.get("away_team", ""))
    return _attach_team_context(pred_dict, team_context=team_context)


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Dashboard: today's games with model picks, confidence, upset flags."""
    today = nba_today()
    try:
        predictions = _get_todays_games()
    except Exception as e:
        logger.error("Dashboard error: %s", e, exc_info=True)
        predictions = []

    # Enrich predictions with ESPN start times for sorting
    espn_games = []
    try:
        from src.data.gamecast import fetch_espn_scoreboard
        espn_list = fetch_espn_scoreboard()
        if predictions:
            # Build lookup: "AWAY@HOME" -> espn game
            espn_lookup = {}
            for g in espn_list:
                espn_lookup[f"{g['away_team']}@{g['home_team']}"] = g
            for pred in predictions:
                key = f"{pred.get('away_team', '')}@{pred.get('home_team', '')}"
                eg = espn_lookup.get(key, {})
                pred["start_utc"] = eg.get("date", "")
            predictions.sort(key=lambda p: p.get("start_utc", ""))
        else:
            espn_games = sorted(espn_list, key=lambda g: g.get("date", ""))
    except Exception as e:
        logger.warning("ESPN scoreboard enrichment failed: %s", e)

    return render_template(
        "dashboard.html",
        predictions=predictions,
        espn_games=espn_games,
        today=today,
        game_count=len(predictions) or len(espn_games),
    )


@app.route("/underdogs")
def underdogs():
    """Ranked underdog opportunities with date + quality filters."""
    today = nba_today()
    filters = _parse_underdog_filters()
    selected_date = filters["date"]
    dates = _build_date_strip(today, selected_date)

    ranked = []
    total_candidates = 0
    summary = _summarize_screened_underdogs([])
    alerts_preview: List[Dict[str, Any]] = []
    alert_digest = build_underdog_alert_digest([], total_candidates=0)
    error = None
    try:
        screened = _run_underdog_screen(filters)
        ranked = screened["screened"]
        total_candidates = len(screened["ranked_all"])
        summary = screened["summary"]
        alerts_preview = build_underdog_alert_candidates(ranked, max_items=6)
        alert_digest = build_underdog_alert_digest(
            alerts_preview,
            total_candidates=total_candidates,
        )
    except Exception as e:
        logger.error("Underdogs page error: %s", e, exc_info=True)
        error = "Unable to load underdog opportunities right now."

    export_query = urlencode(_filters_to_query_params(filters))
    preset_options = [
        {"value": key, "label": _UNDERDOG_PRESET_LABELS.get(key, key.title())}
        for key in ("all", "high_quality", "value_zone", "long_dogs", "balanced")
    ]

    return render_template(
        "underdogs.html",
        predictions=ranked,
        filters=filters,
        today=today,
        selected_date=selected_date,
        dates=dates,
        game_count=len(ranked),
        total_candidates=total_candidates,
        summary=summary,
        alerts_preview=alerts_preview,
        alert_digest=alert_digest,
        export_query=export_query,
        preset_options=preset_options,
        error=error,
    )


@app.route("/matchup")
def matchup_picker():
    """Game picker for matchup predictions — shows games for selected date."""
    today = nba_today()
    selected_date = request.args.get("date", today)
    if not _MATCHUP_DATE_RE.match(selected_date):
        selected_date = today

    # Build 7-day date strip (today + 6 days)
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    dates = []
    for i in range(7):
        d = today_dt + timedelta(days=i)
        val = d.strftime("%Y-%m-%d")
        dates.append({
            "day_name": "TODAY" if i == 0 else d.strftime("%a").upper(),
            "day_num": str(d.day),
            "month": d.strftime("%b").upper(),
            "value": val,
            "is_active": val == selected_date,
            "is_today": i == 0,
        })

    # Merge ESPN scoreboard + NBA CDN schedule for selected date
    games = []
    seen = set()

    try:
        from src.data.gamecast import fetch_espn_scoreboard
        for g in fetch_espn_scoreboard(selected_date):
            key = f"{g['away_team']}@{g['home_team']}"
            seen.add(key)
            games.append({
                "home_abbr": g["home_team"],
                "away_abbr": g["away_team"],
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "status": g.get("short_detail") or g.get("status"),
                "state": g.get("state"),
                "start_utc": g.get("date", ""),
            })
    except Exception as e:
        logger.error("Matchup picker ESPN error: %s", e, exc_info=True)

    try:
        from src.data.nba_fetcher import fetch_nba_cdn_schedule
        for g in fetch_nba_cdn_schedule():
            if g["game_date"] != selected_date:
                continue
            key = f"{g['away_team']}@{g['home_team']}"
            if key in seen:
                continue
            games.append({
                "home_abbr": g["home_team"],
                "away_abbr": g["away_team"],
                "home_score": None,
                "away_score": None,
                "status": g.get("status_text") or g.get("game_time") or "",
                "state": "pre",
                "start_utc": g.get("game_time_utc", ""),
            })
    except Exception as e:
        logger.error("Matchup picker CDN error: %s", e, exc_info=True)

    games.sort(key=lambda g: g.get("start_utc", ""))

    return render_template(
        "matchup_picker.html",
        games=games,
        today=today,
        selected_date=selected_date,
        dates=dates,
    )


@app.route("/matchup/<home_abbr>/<away_abbr>/<date>")
def matchup_by_abbr(home_abbr, away_abbr, date):
    """Matchup detail by team abbreviation — resolves to IDs internally."""
    if not _is_valid_matchup_path(home_abbr, away_abbr, date):
        abort(404)

    from src.database import db as _db

    error = None
    fund_pred = None
    sharp_pred = None
    home_id = 0
    away_id = 0

    try:
        # Direct DB lookup — no memory store dependency
        rows = _db.fetch_all("SELECT team_id, abbreviation FROM teams")
        abbr_to_id = {r["abbreviation"]: r["team_id"] for r in rows}
        home_id = abbr_to_id.get(home_abbr.upper(), 0)
        away_id = abbr_to_id.get(away_abbr.upper(), 0)

        if not home_id or not away_id:
            found = sorted(abbr_to_id.keys())
            raise ValueError(
                f"Could not resolve teams: {home_abbr.upper()}, {away_abbr.upper()}. "
                f"DB has {len(found)} teams: {', '.join(found[:10])}..."
            )

        team_context = {}
        try:
            team_context = _build_matchup_team_context(home_id, away_id, date)
        except Exception as ctx_err:
            logger.debug("Matchup team context unavailable: %s", ctx_err)
        fund_pred = _run_prediction(
            home_id, away_id, date, include_sharp=False, team_context=team_context
        )
        sharp_pred = _run_prediction(
            home_id, away_id, date, include_sharp=True, team_context=team_context
        )
    except Exception as e:
        logger.error("Matchup by abbr error: %s", e, exc_info=True)
        error = "Unable to load matchup details right now."

    if fund_pred and fund_pred.get("adjustments"):
        sorted_adj = sorted(
            fund_pred["adjustments"].items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    else:
        sorted_adj = []

    live_score = _fetch_live_score(home_abbr.upper(), away_abbr.upper(), date)

    interaction_detail = None
    if fund_pred:
        interaction_detail = fund_pred.get("interaction_detail")

    return render_template(
        "matchup.html",
        fund=fund_pred,
        sharp=sharp_pred,
        sorted_adjustments=sorted_adj,
        interaction_detail=interaction_detail,
        date=date,
        home_id=home_id,
        away_id=away_id,
        error=error,
        live_score=live_score,
    )


@app.route("/matchup/<int:home_id>/<int:away_id>/<date>")
def matchup_detail(home_id, away_id, date):
    """Game detail with breakdown + sharp money panel."""
    if not _MATCHUP_DATE_RE.match(date or ""):
        abort(404)
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        abort(404)

    error = None
    fund_pred = None
    sharp_pred = None

    try:
        team_context = {}
        try:
            team_context = _build_matchup_team_context(home_id, away_id, date)
        except Exception as ctx_err:
            logger.debug("Matchup team context unavailable: %s", ctx_err)
        fund_pred = _run_prediction(
            home_id, away_id, date, include_sharp=False, team_context=team_context
        )
        sharp_pred = _run_prediction(
            home_id, away_id, date, include_sharp=True, team_context=team_context
        )
    except Exception as e:
        logger.error("Matchup detail error: %s", e, exc_info=True)
        error = "Unable to load matchup details right now."

    # Sort adjustments by absolute magnitude for breakdown table
    if fund_pred and fund_pred.get("adjustments"):
        sorted_adj = sorted(
            fund_pred["adjustments"].items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    else:
        sorted_adj = []

    # Live score — resolve abbreviations from prediction result
    live_score = None
    if fund_pred:
        live_score = _fetch_live_score(
            fund_pred.get("home_team", ""),
            fund_pred.get("away_team", ""),
            date,
        )

    interaction_detail = None
    if fund_pred:
        interaction_detail = fund_pred.get("interaction_detail")

    return render_template(
        "matchup.html",
        fund=fund_pred,
        sharp=sharp_pred,
        sorted_adjustments=sorted_adj,
        interaction_detail=interaction_detail,
        date=date,
        home_id=home_id,
        away_id=away_id,
        error=error,
        live_score=live_score,
    )


@app.route("/accuracy")
def accuracy():
    """Backtest results with A/B comparison."""
    error = None
    results = None
    phase_acceptance: Dict[str, Any] = {}
    phase_acceptance_report = _build_phase_acceptance_report_meta({})
    drift: Dict[str, Any] = {}

    try:
        from src.analytics.backtester import run_backtest
        results = run_backtest()
    except Exception as e:
        logger.error("Accuracy page error: %s", e, exc_info=True)
        error = "Unable to load accuracy metrics right now."

    fund = results.get("fundamentals", {}) if results else {}
    sharp = results.get("sharp", {}) if results else {}
    comparison = results.get("comparison", {}) if results else {}
    if fund or sharp:
        drift_raw = results.get("drift", {}) if isinstance(results, dict) else {}
        if isinstance(drift_raw, dict) and drift_raw:
            drift = drift_raw
        else:
            drift = {
                "fundamentals": evaluate_underdog_drift(fund),
                "sharp": evaluate_underdog_drift(sharp),
            }

        phase_acceptance_raw = (
            results.get("phase_acceptance", {}) if isinstance(results, dict) else {}
        )
        if isinstance(phase_acceptance_raw, dict) and phase_acceptance_raw:
            phase_acceptance = phase_acceptance_raw
        else:
            phase_acceptance = evaluate_phase_acceptance(
                {
                    "fundamentals": fund,
                    "sharp": sharp,
                    "comparison": comparison,
                    "drift": drift,
                }
            )

        phase_acceptance_report = _build_phase_acceptance_report_meta(
            results.get("phase_acceptance_report", {}) if isinstance(results, dict) else {}
        )

    return render_template(
        "accuracy.html",
        fund=fund,
        sharp=sharp,
        comparison=comparison,
        drift=drift,
        phase_acceptance=phase_acceptance,
        phase_acceptance_report=phase_acceptance_report,
        error=error,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON prediction endpoint.

    Accepts JSON body: {home_id, away_id, date, include_sharp (optional)}
    Returns prediction as JSON.
    """
    if not request.is_json:
        return _json_error("Content-Type must be application/json", 415)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _json_error("Invalid JSON payload", 400)

    home_id = data.get("home_id")
    away_id = data.get("away_id")
    game_date = data.get("date")
    include_sharp = bool(data.get("include_sharp", False))

    if not home_id or not away_id or not game_date:
        return _json_error("Missing required fields: home_id, away_id, date", 400)
    if not _MATCHUP_DATE_RE.match(str(game_date)):
        return _json_error("Invalid date format. Expected YYYY-MM-DD.", 400)
    try:
        home_id = int(home_id)
        away_id = int(away_id)
    except (TypeError, ValueError):
        return _json_error("home_id and away_id must be integers.", 400)

    try:
        pred = _run_prediction(home_id, away_id, game_date,
                               include_sharp=include_sharp)
        return jsonify(pred)
    except Exception as e:
        logger.error("API predict error: %s", e, exc_info=True)
        return _json_error("Prediction failed. Please try again.", 500)


@app.route("/api/underdogs")
def api_underdogs():
    """JSON underdog screener endpoint with ranking + filters."""
    filters = _parse_underdog_filters()
    selected_date = filters["date"]
    try:
        screened = _run_underdog_screen(filters)
        ranked_all = screened["ranked_all"]
        rows = screened["screened"]
        summary = screened["summary"]
        alerts = _with_underdog_alert_keys(
            build_underdog_alert_candidates(rows, max_items=12)
        )
        alert_digest = build_underdog_alert_digest(
            alerts,
            total_candidates=len(ranked_all),
        )
        return jsonify({
            "date": selected_date,
            "filters": filters,
            "total_candidates": len(ranked_all),
            "count": len(rows),
            "summary": summary,
            "alerts": alerts,
            "alert_digest": alert_digest,
            "underdogs": rows,
        })
    except Exception as e:
        logger.error("API underdogs error: %s", e, exc_info=True)
        return _json_error("Unable to load underdog opportunities right now.", 500)


@app.route("/api/underdogs/export.csv")
def api_underdogs_export_csv():
    """Export filtered underdog candidates as CSV."""
    filters = _parse_underdog_filters()
    selected_date = str(filters.get("date", nba_today()))
    try:
        screened = _run_underdog_screen(filters)
        rows = screened["screened"]
        output = io.StringIO()
        fieldnames = [
            "list_rank",
            "rank",
            "tier",
            "home_team",
            "away_team",
            "pick",
            "confidence",
            "game_score",
            "rank_score",
            "dog_payout",
            "is_value_zone",
            "vegas_spread",
            "start_utc",
            "why_pick_summary",
            "caution_flags",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            why_pick = row.get("why_pick") if isinstance(row.get("why_pick"), dict) else {}
            caution_labels = [
                str(flag.get("label", "")).strip()
                for flag in row.get("caution_flags", [])
                if isinstance(flag, dict) and str(flag.get("label", "")).strip()
            ]
            writer.writerow(
                {
                    "list_rank": row.get("list_rank", ""),
                    "rank": row.get("rank", ""),
                    "tier": row.get("tier", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "pick": row.get("pick", ""),
                    "confidence": row.get("confidence", 0.0),
                    "game_score": row.get("game_score", 0.0),
                    "rank_score": row.get("rank_score", 0.0),
                    "dog_payout": row.get("dog_payout", 0.0),
                    "is_value_zone": bool(row.get("is_value_zone")),
                    "vegas_spread": row.get("vegas_spread", 0.0),
                    "start_utc": row.get("start_utc", ""),
                    "why_pick_summary": why_pick.get("summary", ""),
                    "caution_flags": "; ".join(caution_labels),
                }
            )

        filename = f"underdogs_{selected_date}.csv"
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error("API underdogs CSV export error: %s", e, exc_info=True)
        return _json_error("Unable to export underdog opportunities right now.", 500)


@app.route("/api/underdogs/alerts")
def api_underdogs_alerts():
    """Alert-focused underdog endpoint using current screener filters."""
    filters = _parse_underdog_filters()
    selected_date = str(filters.get("date", nba_today()))
    try:
        screened = _run_underdog_screen(filters)
        ranked_all = screened["ranked_all"]
        rows = screened["screened"]
        alerts = _with_underdog_alert_keys(
            build_underdog_alert_candidates(rows, max_items=20)
        )
        digest = build_underdog_alert_digest(
            alerts,
            total_candidates=len(ranked_all),
        )
        return jsonify({
            "date": selected_date,
            "filters": filters,
            "scope_key": _underdog_alert_scope_key(filters),
            "alert_digest": digest,
            "alerts": alerts,
        })
    except Exception as e:
        logger.error("API underdogs alerts error: %s", e, exc_info=True)
        return _json_error("Unable to load underdog alerts right now.", 500)


@app.route("/api/underdogs/alerts/dispatch", methods=["POST"])
def api_underdogs_alerts_dispatch():
    """Persist alert state + dispatch notifications for newly detected signals."""
    filters = _parse_underdog_filters()
    selected_date = str(filters.get("date", nba_today()))
    notify_resolved = _parse_bool_arg(request.args.get("notify_resolved"), default=False)
    max_items = _safe_int_arg(
        request.args.get("max_items"),
        default=20,
        min_value=1,
        max_value=100,
    )

    try:
        screened = _run_underdog_screen(filters)
        ranked_all = screened["ranked_all"]
        rows = screened["screened"]
        alerts = _with_underdog_alert_keys(
            build_underdog_alert_candidates(rows, max_items=max_items)
        )
        digest = build_underdog_alert_digest(
            alerts,
            total_candidates=len(ranked_all),
        )
        scope_key = _underdog_alert_scope_key(filters)
        state = update_underdog_alert_state(
            scope_key=scope_key,
            game_date=selected_date,
            alerts=alerts,
            total_candidates=len(ranked_all),
            digest=digest,
        )
        snapshot = persist_recommendation_snapshot(
            game_date=selected_date,
            scope_key=scope_key,
            filters=filters,
            rows=rows,
            total_candidates=len(ranked_all),
            summary=screened.get("summary"),
            alert_digest=digest,
        )
        notifications = _dispatch_underdog_alert_notifications(
            scope_key=scope_key,
            alert_state=state,
            notify_resolved=notify_resolved,
        )
        return jsonify(
            {
                "date": selected_date,
                "filters": filters,
                "scope_key": scope_key,
                "total_candidates": len(ranked_all),
                "count": len(rows),
                "summary": screened.get("summary", {}),
                "alert_digest": digest,
                "alerts": alerts,
                "state": state,
                "snapshot": snapshot,
                "notifications": notifications,
            }
        )
    except Exception as e:
        logger.error("API underdogs alert dispatch error: %s", e, exc_info=True)
        return _json_error("Unable to dispatch underdog alerts right now.", 500)


@app.route("/api/underdogs/digest")
def api_underdogs_digest():
    """Text + JSON digest for automation scripts."""
    filters = _parse_underdog_filters()
    selected_date = str(filters.get("date", nba_today()))
    try:
        screened = _run_underdog_screen(filters)
        ranked_all = screened["ranked_all"]
        rows = screened["screened"]
        summary = screened["summary"]
        alerts = _with_underdog_alert_keys(
            build_underdog_alert_candidates(rows, max_items=12)
        )
        digest = build_underdog_alert_digest(
            alerts,
            total_candidates=len(ranked_all),
        )
        text = _build_underdog_digest_text(
            selected_date=selected_date,
            filters=filters,
            rows=rows,
            summary=summary,
            alerts=alerts,
        )
        return jsonify({
            "date": selected_date,
            "filters": filters,
            "summary": summary,
            "alert_digest": digest,
            "alerts": alerts,
            "digest_text": text,
        })
    except Exception as e:
        logger.error("API underdogs digest error: %s", e, exc_info=True)
        return _json_error("Unable to build underdog digest right now.", 500)


# Background sync state
_sync_lock = threading.Lock()
_sync_running = False
_sync_status = "idle"
_last_sync_request_at = 0.0


@app.route("/api/sync", methods=["POST"])
def api_sync():
    """Trigger data sync (returns immediately, runs in background)."""
    global _sync_running, _sync_status, _last_sync_request_at

    now = time.time()
    with _sync_lock:
        since_last = now - _last_sync_request_at
        if since_last < _SYNC_RATE_LIMIT_SECONDS:
            wait_sec = int(max(1, _SYNC_RATE_LIMIT_SECONDS - since_last))
            return jsonify({
                "status": "rate_limited",
                "message": f"Please wait {wait_sec}s before starting another sync.",
            }), 429
        if _sync_running:
            return jsonify({"status": "already_running", "message": _sync_status})
        _sync_running = True
        _sync_status = "Starting sync..."
        _last_sync_request_at = now

    def _run_sync():
        global _sync_running, _sync_status
        try:
            from src.analytics.pipeline import run_pipeline
            _sync_status = "Pipeline running..."
            run_pipeline(callback=lambda msg: _update_sync_status(msg))
            _sync_status = "Sync complete"
        except Exception as e:
            logger.error("Background sync error: %s", e, exc_info=True)
            _sync_status = "Sync failed. See server logs."
        finally:
            with _sync_lock:
                _sync_running = False

    thread = threading.Thread(target=_run_sync, name="web-sync", daemon=True)
    thread.start()

    return jsonify({"status": "started", "message": "Sync started in background"})


@app.route("/api/sync/odds-today", methods=["POST"])
def api_sync_odds_today():
    """Trigger odds-only sync for today + tomorrow (background thread)."""
    global _sync_running, _sync_status, _last_sync_request_at

    now = time.time()
    with _sync_lock:
        since_last = now - _last_sync_request_at
        if since_last < _SYNC_RATE_LIMIT_SECONDS:
            wait_sec = int(max(1, _SYNC_RATE_LIMIT_SECONDS - since_last))
            return jsonify({
                "status": "rate_limited",
                "message": f"Please wait {wait_sec}s before starting another sync.",
            }), 429
        if _sync_running:
            return jsonify({"status": "already_running", "message": _sync_status})
        _sync_running = True
        _sync_status = "Starting odds-only sync..."
        _last_sync_request_at = now

    def _run_odds_sync():
        global _sync_running, _sync_status

        try:
            from src.data.odds_sync import sync_upcoming_odds
            from src.utils.timezone_utils import nba_today, nba_tomorrow

            today = nba_today()
            tomorrow = nba_tomorrow()

            missing_before = _count_missing_odds_for_today(today)
            if missing_before is not None:
                _sync_status = (
                    f"Odds sync running for {today} + {tomorrow} "
                    f"({missing_before} missing matchup(s) for today)..."
                )
            else:
                _sync_status = f"Odds sync running for {today} + {tomorrow}..."

            saved_count = sync_upcoming_odds(
                callback=_update_sync_status,
            )
            missing_after = _count_missing_odds_for_today(today)

            if missing_before is not None and missing_after is not None:
                filled = max(0, missing_before - missing_after)
            else:
                filled = None

            if saved_count > 0:
                if missing_after is None:
                    _sync_status = (
                        f"Odds sync complete. Updated {saved_count} game(s) for {today} + {tomorrow}."
                    )
                elif missing_after == 0:
                    _sync_status = (
                        f"Odds sync complete. Filled all missing today matchups "
                        f"and updated {saved_count} game(s) for {today} + {tomorrow}."
                    )
                elif filled is not None and filled > 0:
                    _sync_status = (
                        f"Odds sync complete. Filled {filled} missing matchup(s); "
                        f"{missing_after} still missing for today."
                    )
                else:
                    _sync_status = (
                        f"Odds sync complete. Updated {saved_count} game(s); "
                        f"{missing_after} matchup(s) still missing for today."
                    )
            else:
                if missing_after is None:
                    _sync_status = "Odds sync complete. No new odds returned."
                else:
                    _sync_status = (
                        f"Odds sync complete. No new odds returned; "
                        f"{missing_after} matchup(s) still missing for today."
                    )
        except Exception as e:
            logger.error("Background odds sync error: %s", e, exc_info=True)
            _sync_status = "Odds sync failed. See server logs."
        finally:
            with _sync_lock:
                _sync_running = False

    thread = threading.Thread(target=_run_odds_sync, name="web-odds-sync", daemon=True)
    thread.start()

    return jsonify({
        "status": "started",
        "message": "Odds sync for today + tomorrow started in background",
    })


@app.route("/api/sync/status")
def api_sync_status():
    """Check background sync status."""
    return jsonify({
        "running": _sync_running,
        "status": _sync_status,
    })


# ── Minutes restriction management ─────────────────────────────

@app.route("/api/injuries/minutes-cap", methods=["GET"])
def api_get_minutes_caps():
    """List all players currently on minutes restrictions."""
    rows = db.fetch_all(
        "SELECT player_id, player_name, team_id, status, reason, minutes_cap "
        "FROM injuries WHERE minutes_cap IS NOT NULL"
    )
    return jsonify([dict(r) for r in (rows or [])])


@app.route("/api/injuries/minutes-cap", methods=["POST"])
def api_set_minutes_cap():
    """Set or update a minutes restriction via manual injury override.

    Body: {"player_id": int, "player_name": str, "team_id": int,
           "minutes_cap": int, "status": str (optional), "reason": str (optional)}
    """
    data = request.get_json(silent=True) or {}
    player_id = data.get("player_id")
    player_name = data.get("player_name", "")
    team_id = data.get("team_id", 0)
    minutes_cap = data.get("minutes_cap")
    status = data.get("status", "Available")
    reason = data.get("reason", "Minutes restriction")

    if not player_id or minutes_cap is None:
        return jsonify({"error": "player_id and minutes_cap are required"}), 400

    try:
        from src.data.injury_scraper import add_manual_injury
        add_manual_injury(
            player_id=int(player_id),
            player_name=str(player_name),
            team_id=int(team_id),
            status=status,
            reason=reason,
            minutes_cap=int(minutes_cap),
        )
        return jsonify({"status": "ok", "player_id": player_id, "minutes_cap": minutes_cap})
    except Exception as e:
        logger.error("Failed to set minutes cap: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/injuries/minutes-cap/<int:player_id>", methods=["DELETE"])
def api_remove_minutes_cap(player_id: int):
    """Remove a manual minutes restriction for a player."""
    try:
        from src.data.injury_scraper import remove_manual_injury
        remove_manual_injury(player_id)
        return jsonify({"status": "ok", "player_id": player_id})
    except Exception as e:
        logger.error("Failed to remove minutes cap: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/status")
def api_cache_status():
    """Cache invalidation/debug state."""
    try:
        from src.analytics.cache_registry import get_cache_registry_state
        return jsonify(get_cache_registry_state())
    except Exception as e:
        logger.error("Cache status error: %s", e, exc_info=True)
        return _json_error("Unable to load cache status right now.", 500)


def _update_sync_status(msg: str):
    global _sync_status
    _sync_status = msg


@app.route("/gamecast")
@app.route("/gamecast/<game_id>")
def gamecast(game_id=None):
    """Live gamecast view with ESPN integration."""
    return render_template("gamecast.html", initial_game_id=game_id or "")


@app.route("/api/gamecast/games")
def api_gamecast_games():
    """Today's games from ESPN scoreboard."""
    from src.data.gamecast import fetch_espn_scoreboard
    try:
        games = fetch_espn_scoreboard()
        return jsonify({"games": games})
    except Exception as e:
        logger.error("Gamecast games error: %s", e, exc_info=True)
        return jsonify({"games": [], "error": "Unable to load game list right now."}), 500


@app.route("/api/gamecast/<game_id>")
def api_gamecast_data(game_id):
    """Full game data for gamecast (summary, boxscore, plays, odds)."""
    from src.data.gamecast import fetch_espn_game_summary
    from src.utils.team_mapper import normalize_espn_abbr
    try:
        summary = fetch_espn_game_summary(game_id)
        if not summary:
            return jsonify({"error": "No data available"}), 404
        result = _parse_game_summary(summary, game_id, normalize_espn_abbr)
        return jsonify(result)
    except Exception as e:
        logger.error("Gamecast data error for %s: %s", game_id, e, exc_info=True)
        return _json_error("Unable to load gamecast data right now.", 500)


def _parse_game_summary(summary, game_id, normalize_abbr):
    """Parse ESPN game summary into clean JSON for gamecast."""

    def _as_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_optional_float(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().rstrip("%")
            if not value:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_logo(team_obj):
        logos = team_obj.get("logos", [])
        if isinstance(logos, list) and logos:
            first = logos[0]
            if isinstance(first, dict):
                href = first.get("href", "")
                if href:
                    return href
        return team_obj.get("logo", "") or ""

    def _extract_total_record(competitor_obj):
        records = competitor_obj.get("record") or competitor_obj.get("records") or []
        if not isinstance(records, list):
            return ""
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if rec.get("type") == "total":
                return str(rec.get("summary", "") or rec.get("displayValue", "")).strip()
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("name", "")).lower() == "overall":
                return str(rec.get("summary", "") or rec.get("displayValue", "")).strip()
        return ""

    def _is_team_in_bonus(competitor_obj):
        fouls = competitor_obj.get("fouls", {})
        if isinstance(fouls, dict):
            bonus_state = str(fouls.get("bonusState", "")).strip().lower()
            if bonus_state and ("bonus" in bonus_state or "double" in bonus_state):
                return True
            if _as_int(fouls.get("foulsToGive"), 1) <= 0:
                return True
        lines = competitor_obj.get("linescores", [])
        return bool(lines and isinstance(lines[-1], dict) and lines[-1].get("isBonus", False))

    result = {
        "game_id": game_id,
        "header": {"home": {}, "away": {}, "status": {}},
        "boxscore": {"away": None, "home": None},
        "plays": [],
        "odds": {},
        "predictor": {"home_pct": 50.0, "away_pct": 50.0},
        "model_prediction": None,
        "flow_stats": {
            "home_drives": {"scored": 0, "total": 0, "display": "0/0"},
            "away_drives": {"scored": 0, "total": 0, "display": "0/0"},
            "home_possessions": {"scored": 0, "total": 0, "display": "0/0"},
            "away_possessions": {"scored": 0, "total": 0, "display": "0/0"},
            "current_run": "",
        },
        "live_event": {
            "type": "",
            "event_key": "",
            "text": "",
            "timeout_seconds": 0,
            "substitution": {"in": "", "out": ""},
        },
    }

    # ── Header ──
    header = summary.get("header", {})
    competitions = header.get("competitions", [{}])
    game_date_for_context = nba_today()
    if competitions:
        comp = competitions[0]
        raw_comp_date = str(comp.get("date", "") or "")
        if raw_comp_date:
            game_date_for_context = nba_game_date_from_utc_iso(raw_comp_date)

        get_team_display_context = None
        try:
            from src.analytics.team_context import get_team_display_context
        except Exception:
            logger.debug("team_context import unavailable for gamecast parse", exc_info=True)
            get_team_display_context = None

        abbr_to_id = {}
        try:
            from src.analytics.stats_engine import get_team_abbreviations

            abbr_map = get_team_abbreviations()
            abbr_to_id = {
                str(v).upper(): _as_int(k, 0)
                for k, v in (abbr_map or {}).items()
            }
        except Exception:
            logger.debug("team abbreviation map unavailable for gamecast parse", exc_info=True)
            abbr_to_id = {}

        status = comp.get("status", {})
        result["header"]["status"] = {
            "state": status.get("type", {}).get("state", ""),
            "period": _as_int(status.get("period", 0), 0),
            "clock": status.get("displayClock", "0:00"),
            "detail": status.get("type", {}).get("shortDetail", ""),
            "description": status.get("type", {}).get("description", ""),
        }
        for c in comp.get("competitors", []):
            side = c.get("homeAway", "")
            if side not in ("home", "away"):
                continue
            team = c.get("team", {})
            team_abbr = normalize_abbr(team.get("abbreviation", ""))
            linescores = c.get("linescores", [])
            bonus = _is_team_in_bonus(c)
            team_id_int = _as_int(abbr_to_id.get(str(team_abbr).upper(), 0), 0)
            ctx = {}
            if get_team_display_context and team_id_int:
                try:
                    ctx = get_team_display_context(
                        team_id_int,
                        game_date=game_date_for_context,
                        include_starters_out=True,
                    )
                except Exception:
                    logger.debug(
                        "team_context fetch failed for team_id=%s date=%s",
                        team_id_int,
                        game_date_for_context,
                        exc_info=True,
                    )
                    ctx = {}
            record = _extract_total_record(c) or ctx.get("record", "")
            result["header"][side] = {
                "abbr": team_abbr,
                "name": team.get("displayName", ""),
                "id": str(team.get("id", "") or ""),
                "db_id": str(team_id_int or ""),
                "logo": _extract_logo(team),
                "record": record,
                "streak": ctx.get("streak", ""),
                "days_since_last_game": ctx.get("days_since_last_game"),
                "last_game_text": ctx.get("last_game_text", ""),
                "starters_out": ctx.get("starters_out", []),
                "score": _as_int(c.get("score", 0), 0),
                "timeouts_remaining": _as_int(c.get("timeoutsRemaining", -1), -1),
                "bonus": bonus,
                "fouls": 0,
                "linescores": [_as_int(q.get("displayValue", 0), 0)
                               for q in linescores],
            }

    # ── Boxscore ──
    for team_block in summary.get("boxscore", {}).get("players", []):
        team = team_block.get("team", {})
        team_id = str(team.get("id", "") or "")
        side = None
        if str(result["header"]["home"].get("id", "")) == team_id:
            side = "home"
        elif str(result["header"]["away"].get("id", "")) == team_id:
            side = "away"
        else:
            continue

        stats_block = (team_block.get("statistics") or [{}])[0]
        labels = stats_block.get("labels", [])
        pf_index = labels.index("PF") if "PF" in labels else -1
        team_fouls = 0
        players = []

        for ath in stats_block.get("athletes", []):
            info = ath.get("athlete", {})
            raw_stats = ath.get("stats", [])
            stat_dict = {}
            for i, label in enumerate(labels):
                stat_dict[label] = raw_stats[i] if i < len(raw_stats) else ""

            if pf_index >= 0 and pf_index < len(raw_stats):
                team_fouls += _as_int(raw_stats[pf_index], 0)

            headshot = info.get("headshot", {})
            headshot_url = ""
            if isinstance(headshot, dict):
                headshot_url = headshot.get("href", "")

            players.append({
                "name": info.get("displayName", ""),
                "id": info.get("id", ""),
                "active": bool(ath.get("active", False)),
                "starter": bool(ath.get("starter", False)),
                "headshot": headshot_url,
                "stats": stat_dict,
            })

        result["header"][side]["fouls"] = team_fouls
        result["boxscore"][side] = {
            "team": normalize_abbr(team.get("abbreviation", "")),
            "labels": labels,
            "players": players,
        }

    # ── Plays ──
    home_id = str(result["header"]["home"].get("id", ""))
    away_id = str(result["header"]["away"].get("id", ""))
    team_meta = {
        home_id: {
            "abbr": result["header"]["home"].get("abbr", ""),
            "logo": result["header"]["home"].get("logo", ""),
            "side": "home",
        },
        away_id: {
            "abbr": result["header"]["away"].get("abbr", ""),
            "logo": result["header"]["away"].get("logo", ""),
            "side": "away",
        },
    }

    for entry in summary.get("plays", []):
        if not isinstance(entry, dict):
            continue
        items = entry.get("items")
        play_list = items if isinstance(items, list) else [entry]
        for item in play_list:
            text = item.get("text", "")
            if not text:
                continue

            team = item.get("team", {})
            team_id = str(team.get("id", "") if isinstance(team, dict) else "")
            clock_raw = item.get("clock", {})
            period_raw = item.get("period", {})
            clock_val = (
                clock_raw.get("displayValue", "")
                if isinstance(clock_raw, dict)
                else str(clock_raw or "")
            )
            period_val = (
                _as_int(period_raw.get("number", 0), 0)
                if isinstance(period_raw, dict)
                else _as_int(period_raw, 0)
            )
            meta = team_meta.get(team_id, {})
            away_score = item.get("awayScore")
            home_score = item.get("homeScore")
            result["plays"].append({
                "text": text,
                "team_id": team_id,
                "team_abbr": meta.get("abbr", ""),
                "team_logo": meta.get("logo", ""),
                "team_side": meta.get("side", ""),
                "clock": clock_val,
                "period": period_val,
                "scoring": bool(item.get("scoringPlay", False)),
                "away_score": (
                    _as_int(away_score, 0)
                    if away_score not in (None, "")
                    else None
                ),
                "home_score": (
                    _as_int(home_score, 0)
                    if home_score not in (None, "")
                    else None
                ),
                "coordinate": item.get("coordinate", {}),
                "shootingPlay": bool(item.get("shootingPlay", False)),
                "scoreValue": _as_int(item.get("scoreValue", 0), 0),
            })

    # ── Live odds refresh (score-gated) ──
    home_abbr = result["header"]["home"].get("abbr", "")
    away_abbr = result["header"]["away"].get("abbr", "")
    _status_state = result["header"]["status"].get("state", "")
    if _status_state == "in" and home_abbr and away_abbr:
        try:
            from src.data.odds_sync import maybe_refresh_live_odds
            _h_score = result["header"]["home"].get("score", 0)
            _a_score = result["header"]["away"].get("score", 0)
            _period = result["header"]["status"].get("period", 0)
            maybe_refresh_live_odds(
                home_abbr, away_abbr, _h_score, _a_score,
                _status_state, _period, game_date_for_context,
            )
        except Exception:
            logger.debug("Live odds refresh failed in gamecast", exc_info=True)

    # ── Odds (DB first, fallback to ESPN pickcenter) ──
    # Use synced DB odds to avoid hitting ActionNetwork API on every poll.
    if home_abbr and away_abbr:
        try:
            from src.database import db as _db
            h_id = abbr_to_id.get(home_abbr.upper())
            a_id = abbr_to_id.get(away_abbr.upper())
            if h_id and a_id:
                _odds_row = _db.fetch_one(
                    """SELECT spread, over_under, home_moneyline, away_moneyline,
                              spread_home_public, spread_away_public,
                              spread_home_money, spread_away_money,
                              ml_home_public, ml_away_public,
                              ml_home_money, ml_away_money,
                              opening_spread, spread_movement,
                              provider, fetched_at
                       FROM game_odds
                       WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?""",
                    (game_date_for_context, h_id, a_id),
                )
                if _odds_row and (_odds_row.get("spread") is not None or _odds_row.get("over_under") is not None):
                    _sp = _odds_row["spread"]
                    result["odds"] = {
                        "spread": f"{_sp:+.1f}" if _sp is not None else "N/A",
                        "over_under": _odds_row.get("over_under"),
                        "home_moneyline": _odds_row.get("home_moneyline"),
                        "away_moneyline": _odds_row.get("away_moneyline"),
                        "provider": _odds_row.get("provider") or "DB",
                        "fetched_at": _odds_row.get("fetched_at"),
                        "spread_home_public": _as_optional_float(_odds_row.get("spread_home_public")),
                        "spread_away_public": _as_optional_float(_odds_row.get("spread_away_public")),
                        "spread_home_money": _as_optional_float(_odds_row.get("spread_home_money")),
                        "spread_away_money": _as_optional_float(_odds_row.get("spread_away_money")),
                        "ml_home_public": _as_optional_float(_odds_row.get("ml_home_public")),
                        "ml_away_public": _as_optional_float(_odds_row.get("ml_away_public")),
                        "ml_home_money": _as_optional_float(_odds_row.get("ml_home_money")),
                        "ml_away_money": _as_optional_float(_odds_row.get("ml_away_money")),
                        "opening_spread": _odds_row.get("opening_spread"),
                        "spread_movement": _odds_row.get("spread_movement"),
                    }
        except Exception:
            logger.debug("DB odds lookup failed for gamecast", exc_info=True)
    if not result["odds"]:
        pickcenter = summary.get("pickcenter", [])
        if pickcenter:
            pc = pickcenter[0]
            result["odds"] = {
                "spread": pc.get("details", ""),
                "over_under": pc.get("overUnder"),
                "home_moneyline": pc.get("homeTeamOdds", {}).get("moneyLine"),
                "away_moneyline": pc.get("awayTeamOdds", {}).get("moneyLine"),
                "provider": pc.get("provider", {}).get("name", "ESPN"),
            }

    # ── Predictor ──
    predictor = summary.get("predictor", {})
    if predictor:
        result["predictor"] = {
            "home_pct": float(
                predictor.get("homeTeam", {}).get("gameProjection", 50.0)),
            "away_pct": float(
                predictor.get("awayTeam", {}).get("gameProjection", 50.0)),
        }

    # ── Flow stats (desktop parity subset) ──
    home_drives = away_drives = 0
    home_drives_scored = away_drives_scored = 0
    current_run_team = None
    current_run_pts = 0

    for play in result["plays"]:
        text = str(play.get("text", "")).lower()
        team_id = str(play.get("team_id", ""))
        scoring = bool(play.get("scoring", False))

        if "driving" in text:
            if team_id == home_id:
                home_drives += 1
                if scoring:
                    home_drives_scored += 1
            elif team_id == away_id:
                away_drives += 1
                if scoring:
                    away_drives_scored += 1

        if not scoring:
            continue

        pts = _as_int(play.get("scoreValue", 0), 0)
        if pts <= 0:
            if "free throw" in text:
                pts = 1
            elif "3-pt" in text or "three point" in text:
                pts = 3
            else:
                pts = 2

        if team_id == current_run_team:
            current_run_pts += pts
        else:
            current_run_team = team_id
            current_run_pts = pts

    run_text = ""
    if current_run_pts >= 5 and current_run_team:
        if current_run_team == home_id:
            run_text = f"{home_abbr} on a {current_run_pts}-0 run"
        elif current_run_team == away_id:
            run_text = f"{away_abbr} on a {current_run_pts}-0 run"

    home_poss = away_poss = 0
    home_poss_scored = away_poss_scored = 0
    for team_block in summary.get("boxscore", {}).get("teams", []):
        block_team_id = str(team_block.get("team", {}).get("id", ""))
        if block_team_id == home_id:
            side = "home"
        elif block_team_id == away_id:
            side = "away"
        else:
            continue
        stats_blocks = team_block.get("statistics", [])
        if not stats_blocks:
            continue

        stats_dict = {}
        for stat in stats_blocks:
            if "abbreviation" in stat:
                stats_dict[stat["abbreviation"]] = stat.get("displayValue")
            elif "label" in stat:
                stats_dict[stat["label"]] = stat.get("displayValue")

        try:
            fg_parts = str(stats_dict.get("FG", "0-0")).split("-")
            fgm = _as_int(fg_parts[0] if fg_parts else 0, 0)
            fga = _as_int(fg_parts[1] if len(fg_parts) > 1 else 0, 0)
            ft_parts = str(stats_dict.get("FT", "0-0")).split("-")
            ftm = _as_int(ft_parts[0] if ft_parts else 0, 0)
            fta = _as_int(ft_parts[1] if len(ft_parts) > 1 else 0, 0)
            orb = _as_int(
                stats_dict.get("OR", stats_dict.get("Offensive Rebounds", 0)),
                0,
            )
            tov = _as_int(
                stats_dict.get("TO", stats_dict.get("Turnovers", 0)),
                0,
            )
            possessions = round(fga + 0.44 * fta - orb + tov)
            scored_possessions = round(fgm + (ftm / 2))
            if side == "home":
                home_poss = possessions
                home_poss_scored = scored_possessions
            else:
                away_poss = possessions
                away_poss_scored = scored_possessions
        except Exception:
            logger.debug("Flow possession parse failed for side=%s", side, exc_info=True)
            continue

    result["flow_stats"] = {
        "home_drives": {
            "scored": home_drives_scored,
            "total": home_drives,
            "display": f"{home_drives_scored}/{home_drives}",
        },
        "away_drives": {
            "scored": away_drives_scored,
            "total": away_drives,
            "display": f"{away_drives_scored}/{away_drives}",
        },
        "home_possessions": {
            "scored": home_poss_scored,
            "total": home_poss,
            "display": f"{home_poss_scored}/{home_poss}",
        },
        "away_possessions": {
            "scored": away_poss_scored,
            "total": away_poss,
            "display": f"{away_poss_scored}/{away_poss}",
        },
        "current_run": run_text,
    }

    # ── Live event hints for web overlays ──
    status_state = result["header"]["status"].get("state", "")
    if result["plays"]:
        latest = result["plays"][-1]
        latest_text = str(latest.get("text", ""))
        lower_text = latest_text.lower()
        event_key = f"{latest.get('period', 0)}:{latest.get('clock', '')}:{latest_text}"
        if status_state == "in" and "timeout" in lower_text:
            result["live_event"] = {
                "type": "timeout",
                "event_key": event_key,
                "text": latest_text,
                "timeout_seconds": 75,
                "substitution": {"in": "", "out": ""},
            }
        elif "enters the game for" in lower_text:
            token = " enters the game for "
            token_idx = lower_text.find(token)
            in_name = ""
            out_name = ""
            if token_idx >= 0:
                in_name = latest_text[:token_idx].strip(" .")
                out_name = latest_text[token_idx + len(token):].strip(" .")
            result["live_event"] = {
                "type": "substitution",
                "event_key": event_key,
                "text": latest_text,
                "timeout_seconds": 0,
                "substitution": {"in": in_name, "out": out_name},
            }

    # ── Model prediction (best-effort) ──
    try:
        from src.analytics.stats_engine import get_team_abbreviations
        abbr_map = get_team_abbreviations()
        abbr_to_id = {v: k for k, v in abbr_map.items()}
        h_id = abbr_to_id.get(home_abbr)
        a_id = abbr_to_id.get(away_abbr)
        if h_id and a_id:
            today = nba_today()
            result["model_prediction"] = _run_prediction(
                int(h_id), int(a_id), today)
    except Exception as e:
        logger.debug("Model prediction unavailable: %s", e)

    return result


@app.route("/api/deploy", methods=["POST"])
def api_deploy():
    """Trigger a git-pull deploy via deploy.sh (runs detached)."""
    if not _DEPLOY_ENABLED:
        return _json_error("Deploy endpoint is disabled.", 403)

    # Check if a deploy is already running
    try:
        with open(_DEPLOY_STATUS_FILE) as f:
            status = json.load(f)
        if status.get("status") == "running" and time.time() - status.get("ts", 0) < 120:
            return jsonify({"status": "already_running", "message": status.get("message", "")})
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    branch = (request.json or {}).get("branch", "")
    if branch and not re.match(r"^[A-Za-z0-9._/-]+$", branch):
        return _json_error("Invalid branch name.", 400)

    cmd = ["bash", _DEPLOY_SCRIPT]
    if branch:
        cmd.append(branch)

    logger.info("Deploy triggered via web UI (branch=%s)", branch or "current")
    subprocess.Popen(cmd, start_new_session=True)
    return jsonify({"status": "started"})


@app.route("/api/deploy/status")
def api_deploy_status():
    """Return current deploy status from the status file."""
    try:
        with open(_DEPLOY_STATUS_FILE) as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"status": "idle"})


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """Shut down the web server and stop background services."""
    if not _SHUTDOWN_ENABLED:
        return _json_error("Shutdown endpoint is disabled.", 403)
    if _SHUTDOWN_TOKEN:
        provided = request.headers.get("X-Shutdown-Token", "")
        if not provided or not hmac.compare_digest(provided, _SHUTDOWN_TOKEN):
            return _json_error("Invalid shutdown token.", 403)

    from src.bootstrap import shutdown
    logger.info("Shutdown requested via web UI")
    shutdown()

    def _exit():
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Timer(0.5, _exit).start()
    return jsonify({"status": "shutting_down"})


# ──────────────────────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────────────────────

@app.errorhandler(404)
def handle_not_found(_err):
    if request.path.startswith("/api/"):
        return _json_error("Not found", 404)
    return render_template("404.html"), 404


@app.errorhandler(500)
def handle_server_error(_err):
    if request.path.startswith("/api/"):
        return _json_error("Internal server error", 500)
    return render_template("500.html"), 500


# ──────────────────────────────────────────────────────────────
# Template filters
# ──────────────────────────────────────────────────────────────

@app.template_filter("sign")
def sign_filter(value):
    """Format a number with explicit +/- sign."""
    try:
        v = float(value)
        if v > 0:
            return f"+{v:.1f}"
        return f"{v:.1f}"
    except (ValueError, TypeError):
        return str(value)


@app.template_filter("pct")
def pct_filter(value):
    """Format a number as percentage."""
    try:
        return f"{float(value):.1f}%"
    except (ValueError, TypeError):
        return "N/A"


@app.template_filter("adj_name")
def adj_name_filter(key):
    """Convert adjustment key to display name."""
    return _adjustment_display_name(key)
