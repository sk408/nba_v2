"""NBA Fundamentals V2 -- Flask Web Application.

Routes:
  /                          Dashboard (today's games with predictions)
  /matchup/<hid>/<aid>/<date>  Game detail with breakdown + sharp panel
  /accuracy                  Backtest results with A/B comparison
  /gamecast                  Live gamecast view (ESPN integration)
  /api/predict  (POST)       JSON prediction endpoint
  /api/sync     (POST)       Trigger data sync (background thread)
  /api/gamecast/games        Today's games (ESPN scoreboard)
  /api/gamecast/<game_id>    Full game data (summary, boxscore, plays, odds)
"""

import logging
import hmac
import os
import re
import secrets
import signal
import threading
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, jsonify, request, session, abort

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
_CSRF_PROTECTED_ENDPOINTS = {"api_predict", "api_sync", "api_shutdown"}


def _ensure_csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def _inject_template_globals():
    return {
        "csrf_token": _ensure_csrf_token(),
        "shutdown_enabled": _SHUTDOWN_ENABLED,
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


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _get_todays_games() -> List[Dict[str, Any]]:
    """Fetch today's games from game_odds and run predictions."""
    from src.database import db
    from src.analytics.prediction import predict_matchup
    from src.analytics.stats_engine import get_team_abbreviations, get_team_names

    today = datetime.now().strftime("%Y-%m-%d")
    abbr_map = get_team_abbreviations()
    name_map = get_team_names()

    games = db.fetch_all(
        "SELECT home_team_id, away_team_id, spread, home_moneyline, away_moneyline "
        "FROM game_odds WHERE game_date = ?",
        (today,),
    )

    predictions = []
    for g in games:
        home_id = g["home_team_id"]
        away_id = g["away_team_id"]
        try:
            pred = predict_matchup(home_id, away_id, today)
            pred_dict = asdict(pred)
            # Add full team names for display
            pred_dict["home_name"] = name_map.get(home_id, pred_dict.get("home_team", ""))
            pred_dict["away_name"] = name_map.get(away_id, pred_dict.get("away_team", ""))
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
                "game_date": today,
                "pick": "",
                "confidence": 0,
                "projected_home_pts": 0,
                "projected_away_pts": 0,
                "is_dog_pick": False,
                "is_value_zone": False,
                "game_score": 0,
                "vegas_spread": g.get("spread") or 0,
                "error": "Prediction unavailable",
            })

    return predictions


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
    })
    return pred_dict


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
    today = datetime.now().strftime("%Y-%m-%d")
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


@app.route("/matchup")
def matchup_picker():
    """Game picker for matchup predictions — always shows ESPN games."""
    today = datetime.now().strftime("%Y-%m-%d")
    games = []

    try:
        from src.data.gamecast import fetch_espn_scoreboard
        for g in fetch_espn_scoreboard():
            games.append({
                "home_abbr": g["home_team"],
                "away_abbr": g["away_team"],
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "status": g.get("short_detail") or g.get("status"),
                "state": g.get("state"),
                "start_utc": g.get("date", ""),
            })
        games.sort(key=lambda g: g.get("start_utc", ""))
    except Exception as e:
        logger.error("Matchup picker error: %s", e, exc_info=True)

    return render_template("matchup_picker.html", games=games, today=today)


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

    return render_template(
        "matchup.html",
        fund=fund_pred,
        sharp=sharp_pred,
        sorted_adjustments=sorted_adj,
        date=date,
        home_id=home_id,
        away_id=away_id,
        error=error,
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

    return render_template(
        "matchup.html",
        fund=fund_pred,
        sharp=sharp_pred,
        sorted_adjustments=sorted_adj,
        date=date,
        home_id=home_id,
        away_id=away_id,
        error=error,
    )


@app.route("/accuracy")
def accuracy():
    """Backtest results with A/B comparison."""
    error = None
    results = None

    try:
        from src.analytics.backtester import run_backtest
        results = run_backtest()
    except Exception as e:
        logger.error("Accuracy page error: %s", e, exc_info=True)
        error = "Unable to load accuracy metrics right now."

    fund = results.get("fundamentals", {}) if results else {}
    sharp = results.get("sharp", {}) if results else {}
    comparison = results.get("comparison", {}) if results else {}

    return render_template(
        "accuracy.html",
        fund=fund,
        sharp=sharp,
        comparison=comparison,
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


@app.route("/api/sync/status")
def api_sync_status():
    """Check background sync status."""
    return jsonify({
        "running": _sync_running,
        "status": _sync_status,
    })


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
    from src.data.gamecast import fetch_espn_game_summary, get_actionnetwork_odds
    from src.utils.team_mapper import normalize_espn_abbr
    try:
        summary = fetch_espn_game_summary(game_id)
        if not summary:
            return jsonify({"error": "No data available"}), 404
        result = _parse_game_summary(summary, game_id, normalize_espn_abbr,
                                     get_actionnetwork_odds)
        return jsonify(result)
    except Exception as e:
        logger.error("Gamecast data error for %s: %s", game_id, e, exc_info=True)
        return _json_error("Unable to load gamecast data right now.", 500)


def _parse_game_summary(summary, game_id, normalize_abbr, get_an_odds):
    """Parse ESPN game summary into clean JSON for gamecast."""

    def _as_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

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
    game_date_for_context = datetime.now().strftime("%Y-%m-%d")
    if competitions:
        comp = competitions[0]
        raw_comp_date = str(comp.get("date", "") or "")
        if len(raw_comp_date) >= 10:
            game_date_for_context = raw_comp_date[:10]

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

    # ── Odds (try Action Network, fallback to ESPN pickcenter) ──
    home_abbr = result["header"]["home"].get("abbr", "")
    away_abbr = result["header"]["away"].get("abbr", "")
    if home_abbr and away_abbr:
        try:
            an_odds = get_an_odds(home_abbr, away_abbr)
            if an_odds:
                result["odds"] = an_odds
        except Exception:
            logger.debug(
                "ActionNetwork odds parse failed for %s vs %s",
                home_abbr,
                away_abbr,
                exc_info=True,
            )
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
            today = datetime.now().strftime("%Y-%m-%d")
            result["model_prediction"] = _run_prediction(
                int(h_id), int(a_id), today)
    except Exception as e:
        logger.debug("Model prediction unavailable: %s", e)

    return result


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
    names = {
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
    }
    return names.get(key, key.replace("_", " ").title())
