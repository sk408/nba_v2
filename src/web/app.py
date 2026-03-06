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
import os
import signal
import threading
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

from flask import Flask, render_template, jsonify, request

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# Secret key for session/flash (not critical -- we don't use sessions heavily)
app.secret_key = os.urandom(24)


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
                "error": str(e),
            })

    return predictions


def _run_prediction(home_id: int, away_id: int, game_date: str,
                    include_sharp: bool = False) -> Dict[str, Any]:
    """Run a single prediction and return as dict."""
    from src.analytics.prediction import predict_matchup
    from src.analytics.stats_engine import get_team_names

    pred = predict_matchup(home_id, away_id, game_date, include_sharp=include_sharp)
    pred_dict = asdict(pred)
    name_map = get_team_names()
    pred_dict["home_name"] = name_map.get(home_id, pred_dict.get("home_team", ""))
    pred_dict["away_name"] = name_map.get(away_id, pred_dict.get("away_team", ""))
    return pred_dict


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

        fund_pred = _run_prediction(home_id, away_id, date, include_sharp=False)
        sharp_pred = _run_prediction(home_id, away_id, date, include_sharp=True)
    except Exception as e:
        logger.error("Matchup by abbr error: %s", e, exc_info=True)
        error = str(e)

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
    error = None
    fund_pred = None
    sharp_pred = None

    try:
        fund_pred = _run_prediction(home_id, away_id, date, include_sharp=False)
        sharp_pred = _run_prediction(home_id, away_id, date, include_sharp=True)
    except Exception as e:
        logger.error("Matchup detail error: %s", e, exc_info=True)
        error = str(e)

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
        error = str(e)

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
    data = request.get_json(force=True, silent=True) or {}
    home_id = data.get("home_id")
    away_id = data.get("away_id")
    game_date = data.get("date")
    include_sharp = bool(data.get("include_sharp", False))

    if not home_id or not away_id or not game_date:
        return jsonify({"error": "Missing required fields: home_id, away_id, date"}), 400

    try:
        pred = _run_prediction(int(home_id), int(away_id), game_date,
                               include_sharp=include_sharp)
        return jsonify(pred)
    except Exception as e:
        logger.error("API predict error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# Background sync state
_sync_lock = threading.Lock()
_sync_running = False
_sync_status = "idle"


@app.route("/api/sync", methods=["POST"])
def api_sync():
    """Trigger data sync (returns immediately, runs in background)."""
    global _sync_running, _sync_status

    with _sync_lock:
        if _sync_running:
            return jsonify({"status": "already_running", "message": _sync_status})
        _sync_running = True
        _sync_status = "Starting sync..."

    def _run_sync():
        global _sync_running, _sync_status
        try:
            from src.analytics.pipeline import run_pipeline
            _sync_status = "Pipeline running..."
            run_pipeline(callback=lambda msg: _update_sync_status(msg))
            _sync_status = "Sync complete"
        except Exception as e:
            logger.error("Background sync error: %s", e, exc_info=True)
            _sync_status = f"Sync failed: {e}"
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


def _update_sync_status(msg: str):
    global _sync_status
    _sync_status = msg


@app.route("/gamecast")
def gamecast():
    """Live gamecast view with ESPN integration."""
    return render_template("gamecast.html")


@app.route("/api/gamecast/games")
def api_gamecast_games():
    """Today's games from ESPN scoreboard."""
    from src.data.gamecast import fetch_espn_scoreboard
    try:
        games = fetch_espn_scoreboard()
        return jsonify({"games": games})
    except Exception as e:
        logger.error("Gamecast games error: %s", e, exc_info=True)
        return jsonify({"games": [], "error": str(e)})


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
        return jsonify({"error": str(e)}), 500


def _parse_game_summary(summary, game_id, normalize_abbr, get_an_odds):
    """Parse ESPN game summary into clean JSON for gamecast."""
    result = {
        "game_id": game_id,
        "header": {"home": {}, "away": {}, "status": {}},
        "boxscore": {"away": None, "home": None},
        "plays": [],
        "odds": {},
        "predictor": {"home_pct": 50.0, "away_pct": 50.0},
        "model_prediction": None,
    }

    # ── Header ──
    header = summary.get("header", {})
    competitions = header.get("competitions", [{}])
    if competitions:
        comp = competitions[0]
        status = comp.get("status", {})
        result["header"]["status"] = {
            "state": status.get("type", {}).get("state", ""),
            "period": status.get("period", 0),
            "clock": status.get("displayClock", "0:00"),
            "detail": status.get("type", {}).get("shortDetail", ""),
            "description": status.get("type", {}).get("description", ""),
        }
        for c in comp.get("competitors", []):
            side = c.get("homeAway", "")
            if side not in ("home", "away"):
                continue
            team = c.get("team", {})
            result["header"][side] = {
                "abbr": normalize_abbr(team.get("abbreviation", "")),
                "name": team.get("displayName", ""),
                "id": team.get("id", ""),
                "score": int(c.get("score", 0) or 0),
                "linescores": [int(q.get("displayValue", 0) or 0)
                               for q in c.get("linescores", [])],
            }

    # ── Boxscore ──
    for team_block in summary.get("boxscore", {}).get("players", []):
        team = team_block.get("team", {})
        team_id = team.get("id", "")
        side = None
        if result["header"]["home"].get("id") == team_id:
            side = "home"
        elif result["header"]["away"].get("id") == team_id:
            side = "away"
        else:
            continue
        stats_block = (team_block.get("statistics") or [{}])[0]
        labels = stats_block.get("labels", [])
        players = []
        for ath in stats_block.get("athletes", []):
            info = ath.get("athlete", {})
            raw_stats = ath.get("stats", [])
            stat_dict = {}
            for i, label in enumerate(labels):
                stat_dict[label] = raw_stats[i] if i < len(raw_stats) else ""
            players.append({
                "name": info.get("displayName", ""),
                "id": info.get("id", ""),
                "active": ath.get("active", False),
                "starter": ath.get("starter", False),
                "stats": stat_dict,
            })
        result["boxscore"][side] = {
            "team": normalize_abbr(team.get("abbreviation", "")),
            "labels": labels,
            "players": players,
        }

    # ── Plays ──
    for entry in summary.get("plays", []):
        if not isinstance(entry, dict):
            continue
        # Plays may be nested in "items" or be standalone
        items = entry.get("items")
        if isinstance(items, list):
            play_list = items
        else:
            play_list = [entry]
        for item in play_list:
            text = item.get("text", "")
            if not text:
                continue
            team = item.get("team", {})
            clock = item.get("clock", {})
            period = item.get("period", {})
            result["plays"].append({
                "text": text,
                "team_id": (team.get("id", "") if isinstance(team, dict)
                            else ""),
                "clock": (clock.get("displayValue", "")
                          if isinstance(clock, dict) else str(clock)),
                "period": (period.get("number", 0)
                           if isinstance(period, dict)
                           else int(period) if period else 0),
                "scoring": bool(item.get("scoringPlay", False)),
                "away_score": item.get("awayScore", 0),
                "home_score": item.get("homeScore", 0),
                "coordinate": item.get("coordinate", {}),
                "shootingPlay": bool(item.get("shootingPlay", False)),
                "scoreValue": int(item.get("scoreValue", 0) or 0),
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
            pass
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
    from src.bootstrap import shutdown
    logger.info("Shutdown requested via web UI")
    shutdown()

    def _exit():
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Timer(0.5, _exit).start()
    return jsonify({"status": "shutting_down"})


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
