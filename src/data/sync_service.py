"""Central data sync orchestrator: full_sync() with 8 sub-steps."""

import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

from src.database import db
from src.config import get_season
from src.data import nba_fetcher, injury_scraper

logger = logging.getLogger(__name__)


def _get_sync_meta(step_name: str) -> dict:
    """Get sync metadata for a step."""
    row = db.fetch_one("SELECT * FROM sync_meta WHERE step_name = ?", (step_name,))
    if row:
        return dict(row)
    return {"step_name": step_name, "last_synced_at": "", "game_count_at_sync": 0,
            "last_game_date_at_sync": "", "extra": ""}


def _set_sync_meta(step_name: str, game_count: int = 0, last_game_date: str = ""):
    """Update sync metadata."""
    now = datetime.now().isoformat()
    db.execute(
        """INSERT INTO sync_meta (step_name, last_synced_at, game_count_at_sync, last_game_date_at_sync)
           VALUES (?,?,?,?)
           ON CONFLICT(step_name) DO UPDATE SET
             last_synced_at=excluded.last_synced_at,
             game_count_at_sync=excluded.game_count_at_sync,
             last_game_date_at_sync=excluded.last_game_date_at_sync""",
        (step_name, now, game_count, last_game_date)
    )


def _is_fresh(step_name: str, hours: int = 24) -> bool:
    """Check if a sync step is still fresh."""
    meta = _get_sync_meta(step_name)
    if not meta["last_synced_at"]:
        return False
    try:
        last = datetime.fromisoformat(meta["last_synced_at"])
        return (datetime.now() - last).total_seconds() < hours * 3600
    except (ValueError, TypeError):
        return False


def _get_game_count() -> int:
    row = db.fetch_one("SELECT COUNT(*) as cnt FROM player_stats")
    return row["cnt"] if row else 0


def _get_last_game_date() -> str:
    row = db.fetch_one("SELECT MAX(game_date) as d FROM player_stats")
    return row["d"] if row and row["d"] else ""


def clear_sync_cache():
    """Delete all freshness metadata so next sync re-fetches everything.

    NOTE: Precompute cache is NOT cleared here — it rebuilds incrementally
    and is never invalidated by weight/tuning/sync changes.
    """
    db.execute("DELETE FROM sync_meta")
    db.execute("DELETE FROM player_sync_cache")
    # Invalidate compute caches (NOT precompute — it's incremental & independent)
    # These modules may not exist yet during first-boot; skip gracefully.
    try:
        from src.analytics.prediction import invalidate_residual_cache, invalidate_elo_cache
        invalidate_residual_cache()
        invalidate_elo_cache()
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from src.analytics.stats_engine import invalidate_stats_caches
        invalidate_stats_caches()
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from src.analytics.prediction_quality import invalidate_odds_cache
        invalidate_odds_cache()
    except (ImportError, ModuleNotFoundError):
        pass
    logger.info("Cleared sync freshness caches (precompute cache preserved)")


def nuke_synced_data(callback: Optional[Callable] = None):
    """Delete ALL synced data from the database and disk caches.

    This wipes every table populated by the sync pipeline so that a
    subsequent full_sync(force=True) re-fetches everything from scratch.
    Preserves the database schema (tables/indexes remain).
    """
    import os, glob, shutil

    def emit(msg):
        if callback:
            callback(msg)
        logger.info(msg)

    emit("Nuking all synced data...")

    # 1. Clear all DB tables populated by sync
    tables = [
        "player_stats", "player_sync_cache", "players", "teams",
        "team_metrics", "player_impact", "injuries", "injury_history",
        "game_odds", "game_quarter_scores", "sync_meta",
        # Also clear derived/tuning tables so they rebuild cleanly
        "team_tuning", "model_weights", "team_weight_overrides",
        "predictions",
    ]
    for table in tables:
        try:
            db.execute(f"DELETE FROM {table}")
            emit(f"  Cleared table: {table}")
        except Exception as e:
            emit(f"  Skipped table {table}: {e}")

    # 2. Delete disk caches (precompute cache preserved — incremental & independent)
    cache_paths = [
        os.path.join("data", "pipeline_state.json"),
    ]
    for path in cache_paths:
        if os.path.exists(path):
            os.remove(path)
            emit(f"  Deleted: {path}")

    # Backtest cache directory
    bt_cache_dir = os.path.join("data", "backtest_cache")
    if os.path.isdir(bt_cache_dir):
        shutil.rmtree(bt_cache_dir)
        emit(f"  Deleted: {bt_cache_dir}/")

    # ML model files
    ml_dir = os.path.join("data", "ml_models")
    if os.path.isdir(ml_dir):
        for f in glob.glob(os.path.join(ml_dir, "*")):
            os.remove(f)
        emit(f"  Cleared: {ml_dir}/")

    # Sensitivity CSVs
    for csv in glob.glob(os.path.join("data", "sensitivity_*.csv")):
        os.remove(csv)
        emit(f"  Deleted: {csv}")

    # 3. Invalidate all in-memory caches
    try:
        from src.analytics.prediction import invalidate_residual_cache, invalidate_tuning_cache, invalidate_elo_cache
        from src.analytics.stats_engine import invalidate_stats_caches
        from src.analytics.prediction_quality import invalidate_odds_cache
        from src.analytics.weight_config import invalidate_weight_cache
        from src.analytics.backtester import invalidate_actual_results_cache
        invalidate_residual_cache()
        invalidate_tuning_cache()
        invalidate_elo_cache()
        invalidate_stats_caches()
        invalidate_odds_cache()
        invalidate_weight_cache()
        invalidate_actual_results_cache()
    except Exception as e:
        emit(f"  Cache invalidation warning: {e}")

    emit("Nuke complete! All synced data cleared. Run Force Full Sync to re-fetch.")


def sync_reference_data(callback: Optional[Callable] = None, force: bool = False):
    """Step 1: Sync teams and players."""
    if not force and _is_fresh("reference_data", 24):
        team_count = db.fetch_one("SELECT COUNT(*) as cnt FROM teams")
        player_count = db.fetch_one("SELECT COUNT(*) as cnt FROM players")
        if team_count and team_count["cnt"] >= 30 and player_count and player_count["cnt"] > 100:
            if callback:
                callback("Reference data is fresh, skipping...")
            return
    if callback:
        callback("Fetching NBA teams...")
    teams = nba_fetcher.fetch_teams()
    nba_fetcher.save_teams(teams)
    if callback:
        callback(f"Saved {len(teams)} teams. Fetching rosters...")

    team_rows = db.fetch_all("SELECT team_id FROM teams")
    total = len(team_rows)
    for i, trow in enumerate(team_rows):
        tid = trow["team_id"]
        players = nba_fetcher.fetch_players(tid)
        nba_fetcher.save_players(players)
        if callback and (i + 1) % 5 == 0:
            callback(f"Rosters: {i + 1}/{total} teams...")

    _set_sync_meta("reference_data")
    if callback:
        callback("Reference data sync complete")


def sync_player_game_logs(callback: Optional[Callable] = None, force: bool = False):
    """Step 2: Sync player game logs using bulk fetch (1 API call).

    Uses LeagueGameLog to fetch ALL player logs in a single request
    instead of 320+ individual PlayerGameLog calls.
    """
    if callback:
        callback("Syncing player game logs...")

    now = datetime.now()

    # Step-level freshness: skip if synced within configured hours (unless force)
    from src.config import get as get_setting
    freshness_hours = int(get_setting("sync_freshness_hours", 4))
    if not force and _is_fresh("player_game_logs", freshness_hours):
        if callback:
            callback(f"Game logs are fresh (synced < {freshness_hours}h ago), skipping...")
        return

    # Determine incremental date range
    date_from_str = None
    if not force:
        last_date = _get_last_game_date()
        if last_date:
            # Fetch from 1 day before last known game to catch stragglers
            fetch_from = datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days=1)
            date_from_str = fetch_from.strftime("%m/%d/%Y")  # API expects MM/DD/YYYY

    if date_from_str:
        if callback:
            callback(f"Bulk fetch: game logs from {date_from_str}...")
    else:
        if callback:
            callback("Bulk fetch: all game logs for the season...")

    logs = nba_fetcher.fetch_bulk_game_logs(date_from=date_from_str)

    if logs:
        nba_fetcher.save_game_logs(logs)

        # Update player_sync_cache for all players in the bulk fetch
        from collections import defaultdict
        player_info = defaultdict(lambda: {"latest": "", "count": 0})
        for log in logs:
            pid = log["player_id"]
            player_info[pid]["count"] += 1
            if log["game_date"] > player_info[pid]["latest"]:
                player_info[pid]["latest"] = log["game_date"]

        now_iso = now.isoformat()
        cache_batch = [
            (pid, now_iso, info["count"], info["latest"])
            for pid, info in player_info.items()
        ]
        db.execute_many(
            """INSERT INTO player_sync_cache (player_id, last_synced_at, games_synced, latest_game_date)
               VALUES (?,?,?,?)
               ON CONFLICT(player_id) DO UPDATE SET
                 last_synced_at=excluded.last_synced_at,
                 games_synced=excluded.games_synced,
                 latest_game_date=MAX(excluded.latest_game_date,
                     COALESCE(player_sync_cache.latest_game_date, ''))""",
            cache_batch
        )

        if callback:
            callback(f"Saved {len(logs)} game log entries for {len(player_info)} players")
    else:
        if callback:
            callback("No new game logs found")

    _set_sync_meta("player_game_logs", _get_game_count(), _get_last_game_date())
    if callback:
        callback("Player game logs sync complete")


def sync_injuries_step(callback: Optional[Callable] = None, force: bool = False):
    """Step 3: Sync current injuries."""
    count = injury_scraper.sync_injuries(callback)
    _set_sync_meta("injuries")
    if callback:
        callback(f"Injury sync complete: {count} injuries")


def sync_injury_history(callback: Optional[Callable] = None, force: bool = False):
    """Step 4: Build injury history from game log gaps."""
    meta = _get_sync_meta("injury_history")
    current_gc = _get_game_count()
    if not force and _is_fresh("injury_history", 168) and meta.get("game_count_at_sync", 0) == current_gc:
        if callback:
            callback("Injury history is fresh, skipping...")
        return

    if callback:
        callback("Building injury history from game logs...")

    from src.analytics.injury_history import infer_injuries_from_logs
    result = infer_injuries_from_logs(callback=callback)
    count = result.get("records", 0) if isinstance(result, dict) else 0
    _set_sync_meta("injury_history", current_gc, _get_last_game_date())
    if callback:
        callback(f"Injury history built: {count} records")


def sync_team_metrics(callback: Optional[Callable] = None, force: bool = False):
    """Step 5: Sync team advanced metrics (8 API calls)."""
    meta = _get_sync_meta("team_metrics")
    current_gc = _get_game_count()
    if not force and _is_fresh("team_metrics", 168) and meta.get("game_count_at_sync", 0) == current_gc:
        if callback:
            callback("Team metrics are fresh, skipping...")
        return

    if callback:
        callback("Fetching team estimated metrics...")
    season = get_season()
    now = datetime.now().isoformat()

    # 1. Estimated metrics
    est_metrics = nba_fetcher.fetch_team_estimated_metrics()
    for m in est_metrics:
        tid = m["team_id"]
        db.execute(
            """INSERT INTO team_metrics (team_id, season, gp, w, l, w_pct,
                 e_off_rating, e_def_rating, e_net_rating, e_pace,
                 e_ast_ratio, e_oreb_pct, e_dreb_pct, e_reb_pct, e_tm_tov_pct, last_synced_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(team_id, season) DO UPDATE SET
                 gp=excluded.gp, w=excluded.w, l=excluded.l, w_pct=excluded.w_pct,
                 e_off_rating=excluded.e_off_rating, e_def_rating=excluded.e_def_rating,
                 e_net_rating=excluded.e_net_rating, e_pace=excluded.e_pace,
                 e_ast_ratio=excluded.e_ast_ratio, e_oreb_pct=excluded.e_oreb_pct,
                 e_dreb_pct=excluded.e_dreb_pct, e_reb_pct=excluded.e_reb_pct,
                 e_tm_tov_pct=excluded.e_tm_tov_pct, last_synced_at=excluded.last_synced_at""",
            (tid, season, m["gp"], m["w"], m["l"], m["w_pct"],
             m["e_off_rating"], m["e_def_rating"], m["e_net_rating"], m["e_pace"],
             m["e_ast_ratio"], m["e_oreb_pct"], m["e_dreb_pct"], m["e_reb_pct"],
             m["e_tm_tov_pct"], now)
        )

    if callback:
        callback("Fetching advanced team stats...")

    # 2. Advanced
    _update_team_dash_stats(season, "Advanced", "", now, callback)

    # 3. Four Factors
    if callback:
        callback("Fetching four factors...")
    _update_team_dash_stats(season, "Four Factors", "", now, callback)

    # 4. Opponent
    if callback:
        callback("Fetching opponent stats...")
    _update_opponent_stats(season, now, callback)

    # 5. Home splits
    if callback:
        callback("Fetching home splits...")
    _update_home_road_stats(season, "Home", now, callback)

    # 6. Road splits
    if callback:
        callback("Fetching road splits...")
    _update_home_road_stats(season, "Road", now, callback)

    # 7. Clutch
    if callback:
        callback("Fetching clutch stats...")
    _update_clutch_stats(season, now, callback)

    # 8. Hustle
    if callback:
        callback("Fetching hustle stats...")
    _update_hustle_stats(season, now, callback)

    # 9. Process stats (Misc: paint, fast break, second chance, off turnovers)
    if callback:
        callback("Fetching process stats...")
    _update_misc_stats(season, now, callback)

    _set_sync_meta("team_metrics", current_gc, _get_last_game_date())
    if callback:
        callback("Team metrics sync complete")


def _update_team_dash_stats(season, measure, location, now, callback):
    data = nba_fetcher.fetch_league_dash_team_stats(measure_type=measure, location=location, season=season)
    for row in data:
        tid = row.get("TEAM_ID", 0)
        if not tid:
            continue
        if measure == "Advanced":
            db.execute(
                """UPDATE team_metrics SET
                     off_rating=?, def_rating=?, net_rating=?, pace=?,
                     efg_pct=?, ts_pct=?, ast_ratio=?, ast_to=?,
                     oreb_pct=?, dreb_pct=?, reb_pct=?, tm_tov_pct=?, pie=?,
                     last_synced_at=?
                   WHERE team_id=? AND season=?""",
                (row.get("OFF_RATING", 0), row.get("DEF_RATING", 0),
                 row.get("NET_RATING", 0), row.get("PACE", 0),
                 row.get("EFG_PCT", 0), row.get("TS_PCT", 0),
                 row.get("AST_RATIO", 0), row.get("AST_TO", 0),
                 row.get("OREB_PCT", 0), row.get("DREB_PCT", 0),
                 row.get("REB_PCT", 0), row.get("TM_TOV_PCT", 0),
                 row.get("PIE", 0), now, tid, season)
            )
        elif measure == "Four Factors":
            db.execute(
                """UPDATE team_metrics SET
                     ff_efg_pct=?, ff_fta_rate=?, ff_tm_tov_pct=?, ff_oreb_pct=?,
                     opp_efg_pct=?, opp_fta_rate=?, opp_tm_tov_pct=?, opp_oreb_pct=?,
                     last_synced_at=?
                   WHERE team_id=? AND season=?""",
                (row.get("EFG_PCT", 0), row.get("FTA_RATE", 0),
                 row.get("TM_TOV_PCT", 0), row.get("OREB_PCT", 0),
                 row.get("OPP_EFG_PCT", 0), row.get("OPP_FTA_RATE", 0),
                 row.get("OPP_TOV_PCT", 0), row.get("OPP_OREB_PCT", 0),
                 now, tid, season)
            )


def _update_opponent_stats(season, now, callback):
    data = nba_fetcher.fetch_league_dash_team_stats(measure_type="Opponent", season=season)
    for row in data:
        tid = row.get("TEAM_ID", 0)
        if not tid:
            continue
        db.execute(
            """UPDATE team_metrics SET
                 opp_pts=?, opp_fg_pct=?, opp_fg3_pct=?, opp_ft_pct=?, last_synced_at=?
               WHERE team_id=? AND season=?""",
            (row.get("OPP_PTS", 0), row.get("OPP_FG_PCT", 0),
             row.get("OPP_FG3_PCT", 0), row.get("OPP_FT_PCT", 0),
             now, tid, season)
        )


def _update_home_road_stats(season, location, now, callback):
    data = nba_fetcher.fetch_league_dash_team_stats(measure_type="Base", location=location, season=season)
    prefix = "home" if location == "Home" else "road"
    assert prefix in ("home", "road"), f"unexpected location: {location}"

    # Base measure_type doesn't include OPP_PTS — fetch Opponent with location filter
    opp_data = nba_fetcher.fetch_league_dash_team_stats(measure_type="Opponent", location=location, season=season)
    opp_pts_map = {row.get("TEAM_ID", 0): row.get("OPP_PTS", 0) or 0 for row in opp_data}

    for row in data:
        tid = row.get("TEAM_ID", 0)
        if not tid:
            continue
        gp = row.get("GP", 0)
        w = row.get("W", 0)
        l_val = row.get("L", 0)
        pts = row.get("PTS", 0)
        opp_pts = opp_pts_map.get(tid, 0)
        db.execute(
            f"""UPDATE team_metrics SET
                 {prefix}_gp=?, {prefix}_w=?, {prefix}_l=?,
                 {prefix}_pts=?, {prefix}_opp_pts=?, last_synced_at=?
               WHERE team_id=? AND season=?""",
            (gp, w, l_val, pts, opp_pts, now, tid, season)
        )


def _update_clutch_stats(season, now, callback):
    data = nba_fetcher.fetch_team_clutch_stats(season=season)
    for row in data:
        tid = row.get("TEAM_ID", 0)
        if not tid:
            continue
        db.execute(
            """UPDATE team_metrics SET
                 clutch_gp=?, clutch_w=?, clutch_l=?,
                 clutch_net_rating=?, clutch_efg_pct=?, clutch_ts_pct=?,
                 last_synced_at=?
               WHERE team_id=? AND season=?""",
            (row.get("GP", 0), row.get("W", 0), row.get("L", 0),
             row.get("NET_RATING", 0), row.get("EFG_PCT", 0), row.get("TS_PCT", 0),
             now, tid, season)
        )


def _update_hustle_stats(season, now, callback):
    data = nba_fetcher.fetch_team_hustle_stats(season=season)
    for row in data:
        tid = row.get("TEAM_ID", 0)
        if not tid:
            continue
        db.execute(
            """UPDATE team_metrics SET
                 deflections=?, loose_balls_recovered=?, contested_shots=?,
                 charges_drawn=?, screen_assists=?, last_synced_at=?
               WHERE team_id=? AND season=?""",
            (row.get("DEFLECTIONS", 0), row.get("LOOSE_BALLS_RECOVERED", 0),
             row.get("CONTESTED_SHOTS", 0), row.get("CHARGES_DRAWN", 0),
             row.get("SCREEN_ASSISTS", 0), now, tid, season)
        )


def _update_misc_stats(season, now, callback):
    """Fetch process stats (Misc measure): paint, fast break, second chance, off turnovers."""
    data = nba_fetcher.fetch_league_dash_team_stats(measure_type="Misc", season=season)
    for row in data:
        tid = row.get("TEAM_ID", 0)
        if not tid:
            continue
        db.execute(
            """UPDATE team_metrics SET
                 points_in_paint=?, fast_break_pts=?, second_chance_pts=?, pts_off_tov=?,
                 opp_pts_paint=?, opp_pts_fb=?, opp_pts_2nd_chance=?, opp_pts_off_tov=?,
                 last_synced_at=?
               WHERE team_id=? AND season=?""",
            (row.get("PTS_PAINT", 0), row.get("PTS_FB", 0),
             row.get("PTS_2ND_CHANCE", 0), row.get("PTS_OFF_TOV", 0),
             row.get("OPP_PTS_PAINT", 0), row.get("OPP_PTS_FB", 0),
             row.get("OPP_PTS_2ND_CHANCE", 0), row.get("OPP_PTS_OFF_TOV", 0),
             now, tid, season)
        )


def sync_player_impact(callback: Optional[Callable] = None, force: bool = False):
    """Step 6: Sync player impact (estimated metrics + on/off)."""
    meta = _get_sync_meta("player_impact")
    current_gc = _get_game_count()
    if not force and _is_fresh("player_impact", 168) and meta.get("game_count_at_sync", 0) == current_gc:
        if callback:
            callback("Player impact is fresh, skipping...")
        return

    season = get_season()
    now = datetime.now().isoformat()

    if callback:
        callback("Fetching player estimated metrics...")
    est = nba_fetcher.fetch_player_estimated_metrics()
    est_map = {int(r.get("PLAYER_ID", 0)): r for r in est}

    if callback:
        callback("Fetching player on/off data per team...")
    team_rows = db.fetch_all("SELECT team_id FROM teams")
    on_off_map = {}
    for i, trow in enumerate(team_rows):
        tid = trow["team_id"]
        data = nba_fetcher.fetch_player_on_off(tid)
        for rec in data.get("on", []):
            pid = int(rec.get("VS_PLAYER_ID", rec.get("PLAYER_ID", 0)))
            on_off_map.setdefault(pid, {})["on"] = rec
        for rec in data.get("off", []):
            pid = int(rec.get("VS_PLAYER_ID", rec.get("PLAYER_ID", 0)))
            on_off_map.setdefault(pid, {}).setdefault("off", rec)
        if callback and (i + 1) % 10 == 0:
            callback(f"On/off: {i + 1}/{len(team_rows)} teams...")

    players = db.fetch_all("SELECT player_id, team_id FROM players")
    batch = []
    for p in players:
        pid = p["player_id"]
        tid = p["team_id"]
        e = est_map.get(pid, {})
        oo = on_off_map.get(pid, {})
        on_data = oo.get("on", {})
        off_data = oo.get("off", {})

        on_off_r = float(on_data.get("OFF_RATING", 0) or 0)
        on_def_r = float(on_data.get("DEF_RATING", 0) or 0)
        on_net = float(on_data.get("NET_RATING", 0) or 0)
        off_off_r = float(off_data.get("OFF_RATING", 0) or 0)
        off_def_r = float(off_data.get("DEF_RATING", 0) or 0)
        off_net = float(off_data.get("NET_RATING", 0) or 0)
        net_diff = on_net - off_net
        on_min = float(on_data.get("MIN", 0) or 0)

        batch.append((
            pid, tid, season,
            on_off_r, on_def_r, on_net,
            off_off_r, off_def_r, off_net,
            net_diff, on_min,
            float(e.get("E_USG_PCT", 0) or 0),
            float(e.get("E_OFF_RATING", 0) or 0),
            float(e.get("E_DEF_RATING", 0) or 0),
            float(e.get("E_NET_RATING", 0) or 0),
            float(e.get("E_PACE", 0) or 0),
            float(e.get("E_AST_RATIO", 0) or 0),
            float(e.get("E_OREB_PCT", 0) or 0),
            float(e.get("E_DREB_PCT", 0) or 0),
            now,
        ))
    if batch:
        db.execute_many(
            """INSERT INTO player_impact
               (player_id, team_id, season,
                on_court_off_rating, on_court_def_rating, on_court_net_rating,
                off_court_off_rating, off_court_def_rating, off_court_net_rating,
                net_rating_diff, on_court_minutes,
                e_usg_pct, e_off_rating, e_def_rating, e_net_rating,
                e_pace, e_ast_ratio, e_oreb_pct, e_dreb_pct, last_synced_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(player_id, season) DO UPDATE SET
                 team_id=excluded.team_id,
                 on_court_off_rating=excluded.on_court_off_rating,
                 on_court_def_rating=excluded.on_court_def_rating,
                 on_court_net_rating=excluded.on_court_net_rating,
                 off_court_off_rating=excluded.off_court_off_rating,
                 off_court_def_rating=excluded.off_court_def_rating,
                 off_court_net_rating=excluded.off_court_net_rating,
                 net_rating_diff=excluded.net_rating_diff,
                 on_court_minutes=excluded.on_court_minutes,
                 e_usg_pct=excluded.e_usg_pct,
                 e_off_rating=excluded.e_off_rating,
                 e_def_rating=excluded.e_def_rating,
                 e_net_rating=excluded.e_net_rating,
                 e_pace=excluded.e_pace,
                 e_ast_ratio=excluded.e_ast_ratio,
                 e_oreb_pct=excluded.e_oreb_pct,
                 e_dreb_pct=excluded.e_dreb_pct,
                 last_synced_at=excluded.last_synced_at""",
            batch,
        )

    _set_sync_meta("player_impact", current_gc, _get_last_game_date())
    if callback:
        callback("Player impact sync complete")


def sync_historical_odds(callback: Optional[Callable] = None, force: bool = False):
    """Step 7: Sync Vegas odds for recent games."""
    from src.data.odds_sync import backfill_odds
    
    meta = _get_sync_meta("odds_sync")
    current_gc = _get_game_count()
    if not force and _is_fresh("odds_sync", 24) and meta.get("game_count_at_sync", 0) == current_gc:
        if callback:
            callback("Odds sync is fresh, skipping...")
        return

    if callback:
        callback("Syncing historical Vegas odds...")
        
    count = backfill_odds(callback=callback)
    _set_sync_meta("odds_sync", current_gc, _get_last_game_date())
    
    if callback:
        callback(f"Odds sync complete: {count} games updated.")


def sync_historical_seasons(callback: Optional[Callable] = None, force: bool = False):
    """Sync game logs and team metrics for historical seasons (one-time fetch)."""
    from src.config import get_historical_seasons

    historical = get_historical_seasons()
    if not historical:
        if callback:
            callback("No historical seasons configured, skipping...")
        return

    for hist_season in historical:
        step_key = f"historical_{hist_season}"

        # Historical data is immutable — only fetch once unless forced
        if not force and _is_fresh(step_key, hours=8760):  # 1 year freshness
            if callback:
                callback(f"Historical season {hist_season} already synced, skipping...")
            continue

        if callback:
            callback(f"Fetching game logs for {hist_season}...")

        # 1. Bulk game logs
        logs = nba_fetcher.fetch_bulk_game_logs(season=hist_season)
        if logs:
            nba_fetcher.save_game_logs(logs, season=hist_season)
            if callback:
                callback(f"  Saved {len(logs)} game logs for {hist_season}")
        else:
            if callback:
                callback(f"  No game logs found for {hist_season}")

        # 2. Team metrics for this season
        if callback:
            callback(f"Fetching team metrics for {hist_season}...")

        season = hist_season
        now = datetime.now().isoformat()

        # Estimated metrics — insert rows so UPDATE calls below have rows to update
        est_metrics = nba_fetcher.fetch_team_estimated_metrics(season=season)
        for m in est_metrics:
            tid = m["team_id"]
            db.execute(
                """INSERT INTO team_metrics (team_id, season, gp, w, l, w_pct,
                     e_off_rating, e_def_rating, e_net_rating, e_pace,
                     e_ast_ratio, e_oreb_pct, e_dreb_pct, e_reb_pct, e_tm_tov_pct, last_synced_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(team_id, season) DO UPDATE SET
                     gp=excluded.gp, w=excluded.w, l=excluded.l, w_pct=excluded.w_pct,
                     e_off_rating=excluded.e_off_rating, e_def_rating=excluded.e_def_rating,
                     e_net_rating=excluded.e_net_rating, e_pace=excluded.e_pace,
                     e_ast_ratio=excluded.e_ast_ratio, e_oreb_pct=excluded.e_oreb_pct,
                     e_dreb_pct=excluded.e_dreb_pct, e_reb_pct=excluded.e_reb_pct,
                     e_tm_tov_pct=excluded.e_tm_tov_pct, last_synced_at=excluded.last_synced_at""",
                (tid, season, m["gp"], m["w"], m["l"], m["w_pct"],
                 m["e_off_rating"], m["e_def_rating"], m["e_net_rating"], m["e_pace"],
                 m["e_ast_ratio"], m["e_oreb_pct"], m["e_dreb_pct"], m["e_reb_pct"],
                 m["e_tm_tov_pct"], now)
            )

        # Advanced, Four Factors, Opponent, Home/Road, Clutch, Hustle
        _update_team_dash_stats(season, "Advanced", "", now, callback)
        _update_team_dash_stats(season, "Four Factors", "", now, callback)
        _update_opponent_stats(season, now, callback)
        _update_home_road_stats(season, "Home", now, callback)
        _update_home_road_stats(season, "Road", now, callback)
        _update_clutch_stats(season, now, callback)
        _update_hustle_stats(season, now, callback)

        _set_sync_meta(step_key, _get_game_count(), _get_last_game_date())
        if callback:
            callback(f"Historical season {hist_season} sync complete")


def full_sync(callback: Optional[Callable] = None, force: bool = False):
    """Full 8-step data sync.

    Args:
        force: If True, bypass all freshness checks and re-fetch everything.
    """
    if force:
        if callback:
            callback("Force mode: clearing sync caches...")
        clear_sync_cache()

    steps = [
        ("1/8 Reference data", sync_reference_data),
        ("2/8 Player game logs", sync_player_game_logs),
        ("3/8 Historical seasons", sync_historical_seasons),
        ("4/8 Injuries", sync_injuries_step),
        ("5/8 Injury history", sync_injury_history),
        ("6/8 Team metrics", sync_team_metrics),
        ("7/8 Player impact", sync_player_impact),
        ("8/8 Vegas odds", sync_historical_odds),
    ]
    for label, func in steps:
        if callback:
            callback(f"=== {label} ===")
        try:
            func(callback=callback, force=force)
        except Exception as e:
            logger.error(f"Error in {label}: {e}")
            if callback:
                callback(f"ERROR in {label}: {e}")
    if callback:
        callback("Full data sync complete!")
