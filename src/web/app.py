"""NBA Fundamentals V2 -- Flask Web Application.

Routes:
  /                          Dashboard (today's games with predictions)
  /matchup/<hid>/<aid>/<date>  Game detail with breakdown + sharp panel
  /accuracy                  Backtest results with A/B comparison
  /api/predict  (POST)       JSON prediction endpoint
  /api/sync     (POST)       Trigger data sync (background thread)
"""

import logging
import os
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

    return render_template(
        "dashboard.html",
        predictions=predictions,
        today=today,
        game_count=len(predictions),
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
