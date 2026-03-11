"""Backtester with A/B comparison (fundamentals vs fundamentals+sharp).

Runs both prediction modes on all historical games and produces:
 - Per-game detailed results (pick, confidence, upset detection, ML profit)
 - Aggregate metrics (winner%, upset accuracy, ML ROI, spread MAE)
 - A/B comparison: which picks sharp money flipped, and whether those flips helped
"""

import csv
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from src.analytics.prediction import (
    GameInput, Prediction, predict, precompute_all_games,
    get_actual_game_results,
)
from src.analytics.optimizer import VectorizedGames
from src.analytics.thresholds import MODEL_PICK_EDGE_THRESHOLD, ACTUAL_WIN_THRESHOLD
from src.analytics.weight_config import WeightConfig, get_weight_config
from src.analytics.stats_engine import get_team_abbreviations

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Cache settings
# ──────────────────────────────────────────────────────────────

_BACKTEST_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "backtest_cache"
)
_BACKTEST_CACHE_TTL = 3600  # 1 hour
ONE_POSSESSION_DOG_MARGIN = 3.0
LONG_DOG_MIN_PAYOUT = 3.0
_BACKTEST_METRICS_VERSION = 2

# In-memory cache (module-level singleton)
_mem_cache: Optional[Dict[str, Any]] = None
_mem_cache_hash: Optional[str] = None
_mem_cache_lock = threading.Lock()


def _backtest_source_fingerprint() -> str:
    """Fingerprint source tables so cache invalidates on corrected data."""
    from src.database import db

    ps = db.fetch_one(
        "SELECT COUNT(*) AS c, COALESCE(MAX(game_date), '') AS max_date, "
        "COALESCE(SUM(points), 0.0) AS points_sum FROM player_stats"
    ) or {}
    odds = db.fetch_one(
        "SELECT COUNT(*) AS c, COALESCE(MAX(game_date), '') AS max_date, "
        "COALESCE(SUM(COALESCE(spread, 0.0)), 0.0) AS spread_sum FROM game_odds"
    ) or {}
    payload = json.dumps({"ps": ps, "odds": odds}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _get_competitive_dog_margin() -> float:
    """Read competitive-dog margin setting with fallback."""
    from src.config import get as get_setting
    try:
        return max(0.0, float(get_setting("optimizer_competitive_dog_margin", 7.5)))
    except (TypeError, ValueError):
        return 7.5


def _get_long_dog_min_payout() -> float:
    """Read minimum long-dog payout multiplier with fallback."""
    from src.config import get as get_setting
    try:
        return max(1.0, float(get_setting("optimizer_save_long_dog_min_payout", 3.0)))
    except (TypeError, ValueError):
        return 3.0


def _get_long_dog_onepos_margin() -> float:
    """Read long-dog one-possession margin with fallback."""
    from src.config import get as get_setting
    try:
        return max(
            0.0,
            float(
                get_setting(
                    "optimizer_save_long_dog_onepos_margin",
                    ONE_POSSESSION_DOG_MARGIN,
                )
            ),
        )
    except (TypeError, ValueError):
        return ONE_POSSESSION_DOG_MARGIN


def _cache_hash(
    n_games: int,
    source_fingerprint: str,
    w: WeightConfig,
    competitive_dog_margin: float,
    long_dog_min_payout: float,
    long_dog_onepos_margin: float,
) -> str:
    """Deterministic hash of game count + weight config for cache invalidation."""
    payload = json.dumps(
        {
            "n": n_games,
            "src": source_fingerprint,
            "w": w.to_dict(),
            "competitive_dog_margin": competitive_dog_margin,
            "long_dog_min_payout": long_dog_min_payout,
            "long_dog_onepos_margin": long_dog_onepos_margin,
            "metrics_v": _BACKTEST_METRICS_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _disk_cache_path(h: str) -> str:
    return os.path.join(_BACKTEST_CACHE_DIR, f"backtest_{h}.pkl")


def _load_disk_cache(h: str) -> Optional[Dict[str, Any]]:
    path = _disk_cache_path(h)
    if not os.path.isfile(path):
        return None
    try:
        age = time.time() - os.path.getmtime(path)
        if age > _BACKTEST_CACHE_TTL:
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load backtest cache: %s", e)
        return None


def _save_disk_cache(h: str, data: Dict[str, Any]):
    try:
        os.makedirs(_BACKTEST_CACHE_DIR, exist_ok=True)
        path = _disk_cache_path(h)
        bak_path = f"{path}.bak"
        if os.path.exists(path):
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.replace(path, bak_path)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.warning("Failed to save backtest cache: %s", e)


def invalidate_backtest_cache():
    """Clear in-memory backtest cache. Call after optimizer run or data sync."""
    global _mem_cache, _mem_cache_hash
    with _mem_cache_lock:
        _mem_cache = None
        _mem_cache_hash = None


# ──────────────────────────────────────────────────────────────
# ML payout helpers
# ──────────────────────────────────────────────────────────────

def _ml_payout_multiplier(ml_line: int) -> float:
    """Convert American moneyline to decimal payout multiplier.

    +150 means you win $150 on a $100 bet -> multiplier = 2.50
    -200 means you bet $200 to win $100 -> multiplier = 1.50
    """
    if ml_line == 0:
        return 0.0
    if ml_line > 0:
        return 1.0 + ml_line / 100.0
    return 1.0 + 100.0 / abs(ml_line)


# ──────────────────────────────────────────────────────────────
# Per-game result builder
# ──────────────────────────────────────────────────────────────

def _build_per_game_result(
    game: GameInput,
    pred: Prediction,
    abbr: Dict[int, str],
    competitive_dog_margin: float,
    long_dog_min_payout: float,
    long_dog_onepos_margin: float,
) -> Dict[str, Any]:
    """Build a single per-game result dict from a GameInput + Prediction."""
    home_abbr = abbr.get(game.home_team_id, str(game.home_team_id))
    away_abbr = abbr.get(game.away_team_id, str(game.away_team_id))

    actual_spread = game.actual_home_score - game.actual_away_score
    if actual_spread > ACTUAL_WIN_THRESHOLD:
        actual_winner = "HOME"
    elif actual_spread < -ACTUAL_WIN_THRESHOLD:
        actual_winner = "AWAY"
    else:
        actual_winner = "PUSH"

    model_correct = (
        (pred.pick == "HOME" and actual_winner == "HOME")
        or (pred.pick == "AWAY" and actual_winner == "AWAY")
        or (actual_winner == "PUSH" and abs(pred.game_score) <= 3.0)
    )

    # Upset identification: model disagrees with Vegas favorite
    vegas_fav_home = game.vegas_spread < 0  # negative spread = home favored
    model_picks_home = pred.game_score > MODEL_PICK_EDGE_THRESHOLD
    is_upset_pick = model_picks_home != vegas_fav_home
    upset_correct = is_upset_pick and model_correct
    dog_actual_margin = -actual_spread if vegas_fav_home else actual_spread
    competitive_dog = is_upset_pick and (dog_actual_margin >= -competitive_dog_margin)
    one_possession_dog = is_upset_pick and (dog_actual_margin >= -ONE_POSSESSION_DOG_MARGIN)

    # ML payout for model's pick
    if model_picks_home:
        ml_line = game.vegas_home_ml
    else:
        ml_line = game.vegas_away_ml
    ml_payout = _ml_payout_multiplier(ml_line)
    long_dog_pick = is_upset_pick and (ml_payout >= long_dog_min_payout)
    long_dog_onepos = long_dog_pick and (dog_actual_margin >= -long_dog_onepos_margin)

    # Profit: +payout-1 if correct, -1 if wrong
    if ml_payout > 0:
        if model_correct:
            ml_profit = ml_payout - 1.0
        else:
            ml_profit = -1.0
    else:
        ml_profit = 0.0

    return {
        "game_date": game.game_date,
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "actual_home_score": game.actual_home_score,
        "actual_away_score": game.actual_away_score,
        "actual_winner": actual_winner,
        "model_pick": pred.pick,
        "model_correct": model_correct,
        "game_score": pred.game_score,
        "confidence": pred.confidence,
        "vegas_spread": game.vegas_spread,
        "is_upset_pick": is_upset_pick,
        "upset_correct": upset_correct,
        "competitive_dog": competitive_dog,
        "one_possession_dog": one_possession_dog,
        "long_dog_pick": long_dog_pick,
        "long_dog_onepos": long_dog_onepos,
        "ml_payout": ml_payout,
        "ml_profit": ml_profit,
    }


# ──────────────────────────────────────────────────────────────
# Aggregate metrics from per-game results
# ──────────────────────────────────────────────────────────────

def _aggregate_from_per_game(per_game: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from a list of per-game result dicts."""
    if not per_game:
        return {
            "winner_pct": 0.0,
            "favorites_pct": 0.0,
            "beats_favorites": False,
            "upset_rate": 0.0,
            "upset_accuracy": 0.0,
            "upset_count": 0,
            "upset_correct": 0,
            "competitive_dog_rate": 0.0,
            "competitive_dog_count": 0,
            "one_possession_dog_rate": 0.0,
            "one_possession_dog_count": 0,
            "long_dog_onepos_rate": 0.0,
            "long_dog_onepos_count": 0,
            "long_dog_count": 0,
            "ml_roi": 0.0,
            "ml_win_rate": 0.0,
            "spread_mae": 0.0,
            "total_games": 0,
        }

    total = len(per_game)
    correct = sum(1 for g in per_game if g["model_correct"])
    winner_pct = correct / total * 100.0

    # Favorites baseline: how often does the Vegas favorite win?
    fav_correct = 0
    non_push = 0
    for g in per_game:
        actual = g["actual_winner"]
        if actual == "PUSH":
            continue
        non_push += 1
        vegas_fav_home = g["vegas_spread"] < 0
        if (vegas_fav_home and actual == "HOME") or (not vegas_fav_home and actual == "AWAY"):
            fav_correct += 1
    favorites_pct = fav_correct / max(1, non_push) * 100.0

    # Upset metrics
    upset_count = sum(1 for g in per_game if g["is_upset_pick"])
    upset_correct_count = sum(1 for g in per_game if g["upset_correct"])
    upset_rate = upset_count / max(1, total) * 100.0
    upset_accuracy = upset_correct_count / max(1, upset_count) * 100.0
    competitive_dog_count = sum(1 for g in per_game if g.get("competitive_dog"))
    competitive_dog_rate = competitive_dog_count / max(1, upset_count) * 100.0
    one_possession_dog_count = sum(1 for g in per_game if g.get("one_possession_dog"))
    one_possession_dog_rate = one_possession_dog_count / max(1, upset_count) * 100.0
    long_dog_count = sum(1 for g in per_game if g.get("long_dog_pick"))
    long_dog_onepos_count = sum(1 for g in per_game if g.get("long_dog_onepos"))
    long_dog_onepos_rate = long_dog_onepos_count / max(1, long_dog_count) * 100.0

    # ML ROI
    bets = [g for g in per_game if g["ml_payout"] > 0]
    if bets:
        total_profit = sum(g["ml_profit"] for g in bets)
        ml_roi = total_profit / len(bets) * 100.0
        ml_wins = sum(1 for g in bets if g["ml_profit"] > 0)
        ml_win_rate = ml_wins / len(bets) * 100.0
    else:
        ml_roi = 0.0
        ml_win_rate = 0.0

    # Spread MAE (game_score vs actual spread)
    errors = []
    for g in per_game:
        actual_spread = g["actual_home_score"] - g["actual_away_score"]
        errors.append(abs(g["game_score"] - actual_spread))
    spread_mae = sum(errors) / max(1, len(errors))

    return {
        "winner_pct": winner_pct,
        "favorites_pct": favorites_pct,
        "beats_favorites": winner_pct > favorites_pct,
        "upset_rate": upset_rate,
        "upset_accuracy": upset_accuracy,
        "upset_count": upset_count,
        "upset_correct": upset_correct_count,
        "competitive_dog_rate": competitive_dog_rate,
        "competitive_dog_count": competitive_dog_count,
        "one_possession_dog_rate": one_possession_dog_rate,
        "one_possession_dog_count": one_possession_dog_count,
        "long_dog_onepos_rate": long_dog_onepos_rate,
        "long_dog_onepos_count": long_dog_onepos_count,
        "long_dog_count": long_dog_count,
        "ml_roi": ml_roi,
        "ml_win_rate": ml_win_rate,
        "spread_mae": spread_mae,
        "total_games": total,
    }


# ──────────────────────────────────────────────────────────────
# run_backtest() -- main entry point
# ──────────────────────────────────────────────────────────────

def run_backtest(
    games: Optional[List[GameInput]] = None,
    callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run full backtest with both fundamentals and sharp modes.

    Returns dict with keys: "fundamentals", "sharp", "comparison".
    Each mode has aggregate metrics + per_game list.
    Results are cached to disk + memory keyed on source fingerprint + weights.
    """
    global _mem_cache, _mem_cache_hash

    # Get precomputed games if not provided
    if games is None:
        if callback:
            callback("Loading precomputed games...")
        games = precompute_all_games(callback=callback)

    if not games:
        if callback:
            callback("No games available for backtesting")
        return {"fundamentals": {}, "sharp": {}, "comparison": {}}

    w = get_weight_config()
    source_fingerprint = _backtest_source_fingerprint()
    competitive_dog_margin = _get_competitive_dog_margin()
    long_dog_min_payout = _get_long_dog_min_payout()
    long_dog_onepos_margin = _get_long_dog_onepos_margin()
    h = _cache_hash(
        len(games),
        source_fingerprint,
        w,
        competitive_dog_margin,
        long_dog_min_payout,
        long_dog_onepos_margin,
    )

    # Check memory cache
    with _mem_cache_lock:
        if _mem_cache is not None and _mem_cache_hash == h:
            if callback:
                callback(f"Backtest results loaded from memory cache ({len(games)} games)")
            return _mem_cache

    # Check disk cache
    cached = _load_disk_cache(h)
    if cached is not None:
        with _mem_cache_lock:
            _mem_cache = cached
            _mem_cache_hash = h
        if callback:
            callback(f"Backtest results loaded from disk cache ({len(games)} games)")
        return cached

    if callback:
        callback(f"Running backtest on {len(games)} games...")

    # ── Step 1: Fast vectorized aggregate evaluation ──
    vg = VectorizedGames(games)
    fund_agg = vg.evaluate(w, include_sharp=False)
    sharp_agg = vg.evaluate(w, include_sharp=True)

    if callback:
        callback(f"  Vectorized: Fund winner={fund_agg['winner_pct']:.1f}%, "
                 f"Sharp winner={sharp_agg['winner_pct']:.1f}%")

    # ── Step 2: Per-game detailed results (both modes) ──
    abbr = get_team_abbreviations()

    fund_per_game = []
    sharp_per_game = []

    total = len(games)
    log_interval = max(1, total // 10)

    for i, game in enumerate(games):
        # Fundamentals-only prediction
        fund_pred = predict(game, w, include_sharp=False)
        fund_result = _build_per_game_result(
            game,
            fund_pred,
            abbr,
            competitive_dog_margin,
            long_dog_min_payout,
            long_dog_onepos_margin,
        )
        fund_per_game.append(fund_result)

        # Sharp prediction
        sharp_pred = predict(game, w, include_sharp=True)
        sharp_result = _build_per_game_result(
            game,
            sharp_pred,
            abbr,
            competitive_dog_margin,
            long_dog_min_payout,
            long_dog_onepos_margin,
        )

        # Tag whether sharp flipped this pick
        sharp_result["sharp_flipped"] = fund_pred.pick != sharp_pred.pick
        sharp_per_game.append(sharp_result)

        if callback and (i + 1) % log_interval == 0:
            callback(f"  Per-game: {i + 1}/{total} games processed")

    if callback:
        callback(f"  Per-game: {total}/{total} games processed")

    # ── Step 3: Aggregate from per-game results ──
    fund_metrics = _aggregate_from_per_game(fund_per_game)
    sharp_metrics = _aggregate_from_per_game(sharp_per_game)
    fund_metrics["competitive_dog_margin"] = competitive_dog_margin
    sharp_metrics["competitive_dog_margin"] = competitive_dog_margin
    fund_metrics["one_possession_dog_margin"] = ONE_POSSESSION_DOG_MARGIN
    sharp_metrics["one_possession_dog_margin"] = ONE_POSSESSION_DOG_MARGIN
    fund_metrics["long_dog_min_payout"] = long_dog_min_payout
    sharp_metrics["long_dog_min_payout"] = long_dog_min_payout
    fund_metrics["long_dog_onepos_margin"] = long_dog_onepos_margin
    sharp_metrics["long_dog_onepos_margin"] = long_dog_onepos_margin

    # ── Step 4: A/B comparison ──
    sharp_flipped_picks = sum(1 for g in sharp_per_game if g.get("sharp_flipped"))
    sharp_flipped_correct = sum(
        1 for g in sharp_per_game
        if g.get("sharp_flipped") and g["model_correct"]
    )
    sharp_flipped_accuracy = (
        sharp_flipped_correct / max(1, sharp_flipped_picks) * 100.0
    )
    sharp_net_value = sharp_metrics["winner_pct"] - fund_metrics["winner_pct"]

    comparison = {
        "sharp_flipped_picks": sharp_flipped_picks,
        "sharp_flipped_correct": sharp_flipped_correct,
        "sharp_flipped_accuracy": sharp_flipped_accuracy,
        "sharp_net_value": sharp_net_value,
    }

    if callback:
        callback(f"  Comparison: Sharp flipped {sharp_flipped_picks} picks, "
                 f"{sharp_flipped_correct} correct "
                 f"({sharp_flipped_accuracy:.1f}%), "
                 f"net value: {sharp_net_value:+.1f}% winner")

    # ── Build result ──
    result = {
        "fundamentals": {
            **fund_metrics,
            "per_game": fund_per_game,
        },
        "sharp": {
            **sharp_metrics,
            "per_game": sharp_per_game,
        },
        "comparison": comparison,
    }

    # ── Cache ──
    with _mem_cache_lock:
        _mem_cache = result
        _mem_cache_hash = h
    _save_disk_cache(h, result)

    if callback:
        callback(f"Backtest complete: {fund_metrics['total_games']} games, "
                 f"Fund={fund_metrics['winner_pct']:.1f}%, "
                 f"Sharp={sharp_metrics['winner_pct']:.1f}%")

    return result


# ──────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────

def export_backtest_csv(results: Dict[str, Any], filepath: str, mode: str = "fundamentals"):
    """Export per-game results to CSV for analysis.

    Args:
        results: Output of run_backtest().
        filepath: Path to write the CSV file.
        mode: "fundamentals" or "sharp" -- which per_game list to export.
    """
    per_game = results.get(mode, {}).get("per_game", [])
    if not per_game:
        logger.warning("No per-game results for mode '%s'", mode)
        return

    fieldnames = [
        "game_date", "home_team", "away_team",
        "actual_home_score", "actual_away_score", "actual_winner",
        "model_pick", "model_correct",
        "game_score", "confidence", "vegas_spread",
        "is_upset_pick", "upset_correct", "competitive_dog", "one_possession_dog",
        "long_dog_pick", "long_dog_onepos",
        "ml_payout", "ml_profit",
    ]

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in per_game:
            writer.writerow(row)

    logger.info("Exported %d games to %s", len(per_game), filepath)


# ──────────────────────────────────────────────────────────────
# Quick summary (for pipeline/UI)
# ──────────────────────────────────────────────────────────────

def backtest_summary(results: Optional[Dict[str, Any]] = None,
                     callback: Optional[Callable] = None) -> str:
    """Return a human-readable summary string from backtest results.

    If results not provided, runs the backtest first.
    """
    if results is None:
        results = run_backtest(callback=callback)

    f = results.get("fundamentals", {})
    s = results.get("sharp", {})
    c = results.get("comparison", {})

    lines = [
        "=== Backtest Results ===",
        f"Total games: {f.get('total_games', 0)}",
        "",
        "--- Fundamentals Only ---",
        f"  Winner%:       {f.get('winner_pct', 0):.1f}%",
        f"  Favorites%:    {f.get('favorites_pct', 0):.1f}%",
        f"  Beats favs:    {'YES' if f.get('beats_favorites') else 'NO'}",
        f"  Upset rate:    {f.get('upset_rate', 0):.1f}% "
        f"({f.get('upset_count', 0)} picks)",
        f"  Upset acc:     {f.get('upset_accuracy', 0):.1f}% "
        f"({f.get('upset_correct', 0)} correct)",
        f"  Comp dog rate: {f.get('competitive_dog_rate', 0):.1f}% "
        f"({f.get('competitive_dog_count', 0)} of {f.get('upset_count', 0)}), "
        f"margin <= {f.get('competitive_dog_margin', 7.5):.1f}",
        f"  One-pos dogs:  {f.get('one_possession_dog_rate', 0):.1f}% "
        f"({f.get('one_possession_dog_count', 0)} of {f.get('upset_count', 0)}), "
        f"margin <= {f.get('one_possession_dog_margin', ONE_POSSESSION_DOG_MARGIN):.1f}",
        f"  Long-dog 1-pos:{f.get('long_dog_onepos_rate', 0):.1f}% "
        f"({f.get('long_dog_onepos_count', 0)} of {f.get('long_dog_count', 0)}), "
        f"payout >= {f.get('long_dog_min_payout', LONG_DOG_MIN_PAYOUT):.2f}x, "
        f"margin <= {f.get('long_dog_onepos_margin', ONE_POSSESSION_DOG_MARGIN):.1f}",
        f"  ML ROI:        {f.get('ml_roi', 0):+.1f}%",
        f"  Spread MAE:    {f.get('spread_mae', 0):.1f}",
        "",
        "--- Fundamentals + Sharp ---",
        f"  Winner%:       {s.get('winner_pct', 0):.1f}%",
        f"  Upset rate:    {s.get('upset_rate', 0):.1f}% "
        f"({s.get('upset_count', 0)} picks)",
        f"  Upset acc:     {s.get('upset_accuracy', 0):.1f}% "
        f"({s.get('upset_correct', 0)} correct)",
        f"  Comp dog rate: {s.get('competitive_dog_rate', 0):.1f}% "
        f"({s.get('competitive_dog_count', 0)} of {s.get('upset_count', 0)}), "
        f"margin <= {s.get('competitive_dog_margin', 7.5):.1f}",
        f"  One-pos dogs:  {s.get('one_possession_dog_rate', 0):.1f}% "
        f"({s.get('one_possession_dog_count', 0)} of {s.get('upset_count', 0)}), "
        f"margin <= {s.get('one_possession_dog_margin', ONE_POSSESSION_DOG_MARGIN):.1f}",
        f"  Long-dog 1-pos:{s.get('long_dog_onepos_rate', 0):.1f}% "
        f"({s.get('long_dog_onepos_count', 0)} of {s.get('long_dog_count', 0)}), "
        f"payout >= {s.get('long_dog_min_payout', LONG_DOG_MIN_PAYOUT):.2f}x, "
        f"margin <= {s.get('long_dog_onepos_margin', ONE_POSSESSION_DOG_MARGIN):.1f}",
        f"  ML ROI:        {s.get('ml_roi', 0):+.1f}%",
        "",
        "--- Comparison ---",
        f"  Sharp flipped: {c.get('sharp_flipped_picks', 0)} picks",
        f"  Flipped correct: {c.get('sharp_flipped_correct', 0)} "
        f"({c.get('sharp_flipped_accuracy', 0):.1f}%)",
        f"  Net value:     {c.get('sharp_net_value', 0):+.1f}% winner",
    ]
    return "\n".join(lines)
