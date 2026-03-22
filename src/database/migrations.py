"""Database schema migrations — 20 tables, indexes, init_db()."""

import logging

from src.database.db import execute_script, execute, execute_many, fetch_all, fetch_one

_log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    abbreviation TEXT NOT NULL UNIQUE,
    conference TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    position TEXT,
    is_injured INTEGER NOT NULL DEFAULT 0,
    injury_note TEXT,
    height TEXT,
    weight TEXT,
    age INTEGER,
    experience INTEGER,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    opponent_team_id INTEGER NOT NULL,
    is_home INTEGER NOT NULL,
    game_date DATE NOT NULL,
    game_id TEXT,
    season TEXT NOT NULL DEFAULT '2025-26',
    points REAL NOT NULL,
    rebounds REAL NOT NULL,
    assists REAL NOT NULL,
    minutes REAL NOT NULL,
    steals REAL DEFAULT 0,
    blocks REAL DEFAULT 0,
    turnovers REAL DEFAULT 0,
    fg_made INTEGER DEFAULT 0,
    fg_attempted INTEGER DEFAULT 0,
    fg3_made INTEGER DEFAULT 0,
    fg3_attempted INTEGER DEFAULT 0,
    ft_made INTEGER DEFAULT 0,
    ft_attempted INTEGER DEFAULT 0,
    oreb REAL DEFAULT 0,
    dreb REAL DEFAULT 0,
    plus_minus REAL DEFAULT 0,
    win_loss TEXT,
    personal_fouls REAL DEFAULT 0,
    UNIQUE(player_id, opponent_team_id, game_date, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (opponent_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    predicted_spread REAL NOT NULL,
    predicted_total REAL NOT NULL,
    actual_spread REAL,
    actual_total REAL,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS live_games (
    game_id TEXT PRIMARY KEY,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    start_time_utc TEXT,
    status TEXT,
    period INTEGER,
    clock TEXT,
    home_score INTEGER,
    away_score INTEGER,
    last_updated TEXT,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS team_metrics (
    team_id INTEGER NOT NULL,
    season TEXT NOT NULL,
    gp INTEGER DEFAULT 0,
    w INTEGER DEFAULT 0,
    l INTEGER DEFAULT 0,
    w_pct REAL DEFAULT 0,
    e_off_rating REAL DEFAULT 0,
    e_def_rating REAL DEFAULT 0,
    e_net_rating REAL DEFAULT 0,
    e_pace REAL DEFAULT 0,
    e_ast_ratio REAL DEFAULT 0,
    e_oreb_pct REAL DEFAULT 0,
    e_dreb_pct REAL DEFAULT 0,
    e_reb_pct REAL DEFAULT 0,
    e_tm_tov_pct REAL DEFAULT 0,
    off_rating REAL DEFAULT 0,
    def_rating REAL DEFAULT 0,
    net_rating REAL DEFAULT 0,
    pace REAL DEFAULT 0,
    efg_pct REAL DEFAULT 0,
    ts_pct REAL DEFAULT 0,
    ast_ratio REAL DEFAULT 0,
    ast_to REAL DEFAULT 0,
    oreb_pct REAL DEFAULT 0,
    dreb_pct REAL DEFAULT 0,
    reb_pct REAL DEFAULT 0,
    tm_tov_pct REAL DEFAULT 0,
    pie REAL DEFAULT 0,
    ff_efg_pct REAL DEFAULT 0,
    ff_fta_rate REAL DEFAULT 0,
    ff_tm_tov_pct REAL DEFAULT 0,
    ff_oreb_pct REAL DEFAULT 0,
    opp_efg_pct REAL DEFAULT 0,
    opp_fta_rate REAL DEFAULT 0,
    opp_tm_tov_pct REAL DEFAULT 0,
    opp_oreb_pct REAL DEFAULT 0,
    opp_pts REAL DEFAULT 0,
    opp_fg_pct REAL DEFAULT 0,
    opp_fg3_pct REAL DEFAULT 0,
    opp_ft_pct REAL DEFAULT 0,
    clutch_gp INTEGER DEFAULT 0,
    clutch_w INTEGER DEFAULT 0,
    clutch_l INTEGER DEFAULT 0,
    clutch_net_rating REAL DEFAULT 0,
    clutch_efg_pct REAL DEFAULT 0,
    clutch_ts_pct REAL DEFAULT 0,
    deflections REAL DEFAULT 0,
    loose_balls_recovered REAL DEFAULT 0,
    contested_shots REAL DEFAULT 0,
    charges_drawn REAL DEFAULT 0,
    screen_assists REAL DEFAULT 0,
    home_gp INTEGER DEFAULT 0,
    home_w INTEGER DEFAULT 0,
    home_l INTEGER DEFAULT 0,
    home_pts REAL DEFAULT 0,
    home_opp_pts REAL DEFAULT 0,
    road_gp INTEGER DEFAULT 0,
    road_w INTEGER DEFAULT 0,
    road_l INTEGER DEFAULT 0,
    road_pts REAL DEFAULT 0,
    road_opp_pts REAL DEFAULT 0,
    last_synced_at TEXT,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_impact (
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season TEXT NOT NULL,
    on_court_off_rating REAL DEFAULT 0,
    on_court_def_rating REAL DEFAULT 0,
    on_court_net_rating REAL DEFAULT 0,
    off_court_off_rating REAL DEFAULT 0,
    off_court_def_rating REAL DEFAULT 0,
    off_court_net_rating REAL DEFAULT 0,
    net_rating_diff REAL DEFAULT 0,
    on_court_minutes REAL DEFAULT 0,
    e_usg_pct REAL DEFAULT 0,
    e_off_rating REAL DEFAULT 0,
    e_def_rating REAL DEFAULT 0,
    e_net_rating REAL DEFAULT 0,
    e_pace REAL DEFAULT 0,
    e_ast_ratio REAL DEFAULT 0,
    e_oreb_pct REAL DEFAULT 0,
    e_dreb_pct REAL DEFAULT 0,
    last_synced_at TEXT,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS injury_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    was_out INTEGER NOT NULL DEFAULT 1,
    avg_minutes REAL,
    reason TEXT,
    UNIQUE(player_id, game_date),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS injury_status_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    log_date TEXT NOT NULL,
    status_level TEXT NOT NULL,
    injury_keyword TEXT DEFAULT '',
    injury_detail TEXT DEFAULT '',
    next_game_date TEXT,
    did_play INTEGER,
    UNIQUE(player_id, log_date, status_level),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS team_tuning (
    team_id INTEGER PRIMARY KEY,
    home_pts_correction REAL DEFAULT 0.0,
    away_pts_correction REAL DEFAULT 0.0,
    games_analyzed INTEGER DEFAULT 0,
    avg_spread_error_before REAL DEFAULT 0.0,
    avg_total_error_before REAL DEFAULT 0.0,
    last_tuned_at TEXT,
    tuning_mode TEXT DEFAULT 'classic',
    tuning_version TEXT DEFAULT 'v1_classic',
    tuning_sample_size INTEGER DEFAULT 0,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS model_weights (
    key TEXT PRIMARY KEY,
    value REAL
);

CREATE TABLE IF NOT EXISTS team_weight_overrides (
    team_id INTEGER,
    key TEXT,
    value REAL,
    PRIMARY KEY (team_id, key)
);

CREATE TABLE IF NOT EXISTS game_quarter_scores (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    q1 INTEGER,
    q2 INTEGER,
    q3 INTEGER,
    q4 INTEGER,
    ot INTEGER DEFAULT 0,
    final_score INTEGER,
    game_date TEXT,
    is_home INTEGER DEFAULT 0,
    PRIMARY KEY (game_id, team_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_sync_cache (
    player_id INTEGER PRIMARY KEY,
    last_synced_at TEXT NOT NULL,
    games_synced INTEGER DEFAULT 0,
    latest_game_date DATE
);

CREATE TABLE IF NOT EXISTS sync_meta (
    step_name TEXT PRIMARY KEY,
    last_synced_at TEXT NOT NULL,
    game_count_at_sync INTEGER DEFAULT 0,
    last_game_date_at_sync TEXT DEFAULT '',
    extra TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS injuries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    player_name TEXT NOT NULL,
    team_id INTEGER,
    status TEXT NOT NULL DEFAULT 'Out',
    reason TEXT DEFAULT '',
    expected_return TEXT DEFAULT '',
    source TEXT DEFAULT 'scraped',
    injury_keyword TEXT DEFAULT '',
    updated_at TEXT,
    UNIQUE(player_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS confirmed_lineups (
    game_date TEXT NOT NULL,
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    player_name TEXT DEFAULT '',
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (game_date, game_id, team_id, player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    data TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    read INTEGER NOT NULL DEFAULT 0
);

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

CREATE TABLE IF NOT EXISTS game_odds (
    game_date DATE NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    spread REAL,
    over_under REAL,
    home_moneyline INTEGER,
    away_moneyline INTEGER,
    provider TEXT DEFAULT 'espn',
    fetched_at TEXT,
    opening_spread REAL,
    opening_moneyline INTEGER,
    spread_home_public INTEGER,
    spread_away_public INTEGER,
    spread_home_money INTEGER,
    spread_away_money INTEGER,
    ml_home_public INTEGER,
    ml_away_public INTEGER,
    ml_home_money INTEGER,
    ml_away_money INTEGER,
    num_bets INTEGER,
    PRIMARY KEY (game_date, home_team_id, away_team_id),
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS arenas (
    team_id    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    city       TEXT NOT NULL,
    lat        REAL NOT NULL,
    lon        REAL NOT NULL,
    altitude_ft INTEGER NOT NULL DEFAULT 0,
    timezone   TEXT NOT NULL DEFAULT 'US/Eastern'
);

CREATE TABLE IF NOT EXISTS referees (
    referee_name       TEXT NOT NULL,
    season             TEXT NOT NULL DEFAULT '2025-26',
    games_officiated   INTEGER DEFAULT 0,
    home_win_pct       REAL DEFAULT 50.0,
    total_points_pg    REAL DEFAULT 215.0,
    fouls_per_game     REAL DEFAULT 38.0,
    foul_differential  REAL DEFAULT 0.0,
    home_foul_pct      REAL DEFAULT 50.0,
    road_foul_pct      REAL DEFAULT 50.0,
    last_synced_at     TEXT,
    PRIMARY KEY (referee_name, season)
);

CREATE TABLE IF NOT EXISTS game_referees (
    game_date       TEXT NOT NULL,
    home_team_id    INTEGER NOT NULL,
    away_team_id    INTEGER NOT NULL,
    referee_name    TEXT NOT NULL,
    PRIMARY KEY (game_date, home_team_id, referee_name)
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    team_id    INTEGER NOT NULL,
    game_date  TEXT NOT NULL,
    season     TEXT NOT NULL DEFAULT '',
    elo        REAL NOT NULL DEFAULT 1500.0,
    PRIMARY KEY (team_id, game_date)
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_player_stats_player_date ON player_stats(player_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_player_stats_matchup ON player_stats(opponent_team_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_player_stats_game_id ON player_stats(game_id, is_home);
CREATE INDEX IF NOT EXISTS idx_predictions_matchup ON predictions(home_team_id, away_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_injury_history_team_date ON injury_history(team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_injury_history_player ON injury_history(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_injury_status_log_player ON injury_status_log(player_id, log_date);
CREATE INDEX IF NOT EXISTS idx_injury_status_log_status ON injury_status_log(status_level, did_play);
CREATE INDEX IF NOT EXISTS idx_injury_status_log_team ON injury_status_log(team_id, log_date);
CREATE INDEX IF NOT EXISTS idx_player_sync_cache_date ON player_sync_cache(last_synced_at);
CREATE INDEX IF NOT EXISTS idx_player_impact_team ON player_impact(team_id, season);
CREATE INDEX IF NOT EXISTS idx_quarter_scores_team_date ON game_quarter_scores(team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON notifications(read, id DESC);
CREATE INDEX IF NOT EXISTS idx_rec_items_unsettled ON recommendation_snapshot_items(is_settled, game_date);
CREATE INDEX IF NOT EXISTS idx_rec_items_matchup ON recommendation_snapshot_items(game_date, home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_rec_runs_scope_date ON recommendation_snapshot_runs(scope_key, game_date, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team_id);
CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(player_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_stats(season, game_date);
CREATE INDEX IF NOT EXISTS idx_elo_ratings_date ON elo_ratings(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_elo_ratings_team_season_date ON elo_ratings(team_id, season, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_game_referees_date ON game_referees(game_date, home_team_id);
CREATE INDEX IF NOT EXISTS idx_referees_season ON referees(season);
"""


def _migrate_player_stats_season():
    """Add season column to player_stats if missing (v2 migration)."""
    try:
        execute("ALTER TABLE player_stats ADD COLUMN season TEXT NOT NULL DEFAULT '2025-26'")
        _log.info("Added 'season' column to player_stats")
    except Exception:
        _log.debug("player_stats.season column already present", exc_info=True)


def init_db():
    """Create all tables and indexes."""
    execute_script(SCHEMA_SQL)
    _run_column_migrations()
    _migrate_player_stats_season()
    _backfill_elo_season()
    _backfill_player_stats_team_id()
    execute_script(INDEXES_SQL)


def _run_column_migrations():
    """Add columns that may be missing in older databases."""
    _add_column_if_missing("injuries", "expected_return", "TEXT DEFAULT ''")
    _add_column_if_missing("game_odds", "spread_home_public", "INTEGER")
    _add_column_if_missing("game_odds", "spread_away_public", "INTEGER")
    _add_column_if_missing("game_odds", "spread_home_money", "INTEGER")
    _add_column_if_missing("game_odds", "spread_away_money", "INTEGER")
    _add_column_if_missing("game_odds", "ml_home_public", "INTEGER")
    _add_column_if_missing("game_odds", "ml_away_public", "INTEGER")
    _add_column_if_missing("game_odds", "ml_home_money", "INTEGER")
    _add_column_if_missing("game_odds", "ml_away_money", "INTEGER")
    # New feature columns: SRS, Pythagorean, paint/fastbreak/second-chance
    _add_column_if_missing("team_metrics", "srs", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "pythag_wins", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "points_in_paint", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "fast_break_pts", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "second_chance_pts", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "pts_off_tov", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "opp_pts_paint", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "opp_pts_fb", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "opp_pts_2nd_chance", "REAL DEFAULT 0.0")
    _add_column_if_missing("team_metrics", "opp_pts_off_tov", "REAL DEFAULT 0.0")
    # New feature columns: player advanced metrics
    _add_column_if_missing("injuries", "minutes_cap", "INTEGER DEFAULT NULL")
    _add_column_if_missing("injury_history", "minutes_cap", "INTEGER DEFAULT NULL")
    _add_column_if_missing("player_impact", "vorp", "REAL DEFAULT 0.0")
    _add_column_if_missing("player_impact", "bpm", "REAL DEFAULT 0.0")
    _add_column_if_missing("player_impact", "ws_per_48", "REAL DEFAULT 0.0")
    # New feature column: spread movement
    _add_column_if_missing("game_odds", "spread_movement", "REAL DEFAULT 0.0")
    _add_column_if_missing("game_odds", "num_bets", "INTEGER")
    _add_column_if_missing("player_stats", "team_id", "INTEGER")
    _add_column_if_missing("elo_ratings", "season", "TEXT DEFAULT ''")
    try:
        execute("CREATE INDEX IF NOT EXISTS idx_player_stats_team_date ON player_stats(team_id, game_date DESC)")
    except Exception:
        _log.debug("idx_player_stats_team_date creation skipped", exc_info=True)
    try:
        execute("CREATE INDEX IF NOT EXISTS idx_elo_ratings_team_season_date ON elo_ratings(team_id, season, game_date DESC)")
    except Exception:
        _log.debug("idx_elo_ratings_team_season_date creation skipped", exc_info=True)
    _rename_notifications_body_to_message()
    _fix_game_date_formats()


def _season_for_game_date(game_date: str) -> str:
    try:
        year = int(game_date[:4])
        month = int(game_date[5:7])
        if month >= 7:
            return f"{year}-{str(year + 1)[2:]}"
        return f"{year - 1}-{str(year)[2:]}"
    except Exception:
        return ""


def _backfill_elo_season():
    """Populate elo_ratings.season for legacy rows."""
    try:
        rows = fetch_all(
            "SELECT team_id, game_date FROM elo_ratings WHERE COALESCE(season, '') = ''"
        )
    except Exception:
        _log.debug("elo season backfill query skipped", exc_info=True)
        return

    if not rows:
        return

    updates = []
    for row in rows:
        season = _season_for_game_date(row.get("game_date", ""))
        if not season:
            continue
        updates.append((season, row["team_id"], row["game_date"]))

    if not updates:
        return

    try:
        execute_many(
            "UPDATE elo_ratings SET season = ? WHERE team_id = ? AND game_date = ?",
            updates,
        )
        _log.info("Backfilled elo_ratings.season for %d rows", len(updates))
    except Exception as e:
        _log.debug("elo season backfill skipped: %s", e)


def _add_column_if_missing(table: str, column: str, col_type: str):
    """Safely add a column to an existing table."""
    try:
        cols = fetch_all(f"PRAGMA table_info({table})")
        if not any(c["name"] == column for c in cols):
            execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception:
        _log.debug(
            "Column migration skipped for %s.%s",
            table,
            column,
            exc_info=True,
        )  # table may not exist yet


def _rename_notifications_body_to_message():
    """Migrate notifications table from 'body' column to 'message' column."""
    try:
        cols = fetch_all("PRAGMA table_info(notifications)")
        col_names = [c["name"] for c in cols]
        if "body" in col_names and "message" not in col_names:
            execute("ALTER TABLE notifications RENAME COLUMN body TO message")
            _log.info("Renamed notifications.body -> notifications.message")
        elif "body" not in col_names and "message" not in col_names:
            execute("ALTER TABLE notifications ADD COLUMN message TEXT NOT NULL DEFAULT ''")
            _log.info("Added notifications.message column")
    except Exception as e:
        _log.debug("notifications migration skipped: %s", e)


def _fix_game_date_formats():
    """One-time migration: deduplicate & normalise player_stats game dates.

    The old nba_fetcher used ``str(GAME_DATE)[:10]`` which truncated
    dates like 'Oct 31, 2025' to 'Oct 31, 20'.  Later fetches with the
    fixed ``_normalize_game_date()`` wrote correct ISO dates, so the DB
    can contain BOTH formats for the same (player, game).

    Strategy:
      Phase 0 – Delete text-format rows when the ISO version already
                exists (most common case after a force-sync).
      Phase 1 – Convert any remaining text-format rows that have NO ISO
                duplicate (i.e. never re-fetched).
      Phase 2 – Fix rows with wrong year (< 2024).
      Phase 3 – Create a unique index on (player_id, game_id) to prevent
                future duplicates regardless of date format.
    """
    from src.database.db import execute_many as _exec_many

    try:
        from datetime import datetime as _dt

        # Determine the correct season years from config
        try:
            from src.config import get_season
            season_str = get_season()  # e.g. "2025-26"
            season_start_year = int(season_str.split("-")[0])  # 2025
            season_end_year = season_start_year + 1             # 2026
        except Exception:
            season_start_year = _dt.now().year
            season_end_year = season_start_year + 1

        def _correct_year(parsed_dt):
            """Assign the right year based on month: Oct-Dec = start year, Jan-Sep = end year."""
            if parsed_dt.month >= 10:  # Oct, Nov, Dec
                return parsed_dt.replace(year=season_start_year)
            else:  # Jan–Sep
                return parsed_dt.replace(year=season_end_year)

        # ── Phase 0: Delete text-date duplicates where ISO row exists ──
        # A row is a duplicate if:
        #   - its game_date does NOT look like YYYY-MM-DD
        #   - another row with the same (player_id, game_id) already has
        #     a properly-formatted ISO date
        dup_ids = fetch_all("""
            SELECT bad.id
            FROM player_stats bad
            JOIN player_stats good
              ON good.player_id = bad.player_id
             AND good.game_id   = bad.game_id
             AND good.id       != bad.id
             AND good.game_date LIKE '____-__-__'
            WHERE bad.game_date NOT LIKE '____-__-__'
        """)
        if dup_ids:
            id_list = [r["id"] for r in dup_ids]
            # Delete in chunks to avoid SQL variable limit
            chunk = 500
            for i in range(0, len(id_list), chunk):
                batch = id_list[i:i + chunk]
                placeholders = ",".join("?" * len(batch))
                execute(f"DELETE FROM player_stats WHERE id IN ({placeholders})", batch)
            _log.info("Phase 0: Deleted %d duplicate text-date rows", len(id_list))

        # ── Phase 1: Convert remaining text-format dates ──
        bad_rows = fetch_all(
            "SELECT id, game_date FROM player_stats WHERE game_date NOT LIKE '____-__-__' LIMIT 20000"
        )
        if bad_rows:
            updates = []
            for r in bad_rows:
                raw = r["game_date"]
                for fmt in ("%b %d, %y", "%b %d, %Y", "%B %d, %Y"):
                    try:
                        parsed = _dt.strptime(raw, fmt)
                        corrected = _correct_year(parsed)
                        updates.append((corrected.strftime("%Y-%m-%d"), r["id"]))
                        break
                    except ValueError:
                        continue

            if updates:
                _exec_many(
                    "UPDATE player_stats SET game_date = ? WHERE id = ?",
                    updates
                )
                _log.info("Phase 1: Migrated %d remaining text dates to YYYY-MM-DD", len(updates))

        # ── Phase 2: Fix dates with wrong year — ONLY for current season ──
        # Only re-date rows from the current season that had truncated 2-digit years.
        # Historical seasons (season != current) must keep their original dates.
        cutoff = f"{season_start_year - 1}-01-01"
        current_season_str = f"{season_start_year}-{str(season_end_year)[-2:]}"
        wrong_year = fetch_all(
            "SELECT id, game_date FROM player_stats WHERE game_date < ? AND season = ? LIMIT 20000",
            (cutoff, current_season_str)
        )
        if wrong_year:
            updates2 = []
            for r in wrong_year:
                try:
                    parsed = _dt.strptime(r["game_date"], "%Y-%m-%d")
                    corrected = _correct_year(parsed)
                    updates2.append((corrected.strftime("%Y-%m-%d"), r["id"]))
                except ValueError:
                    continue

            if updates2:
                _exec_many(
                    "UPDATE player_stats SET game_date = ? WHERE id = ?",
                    updates2
                )
                _log.info(
                    "Phase 2: Re-dated %d rows from wrong year to %d/%d season",
                    len(updates2), season_start_year, season_end_year)

        # ── Phase 2b: Fix historical seasons with dates shifted to current season ──
        # Undo damage from previous Phase 2 runs that remapped all old dates.
        # For each non-current season, derive the correct year from the season string.
        hist_seasons = fetch_all(
            "SELECT DISTINCT season FROM player_stats WHERE season != ?",
            (current_season_str,)
        )
        for hs in (hist_seasons or []):
            s = hs["season"]  # e.g. "2019-20"
            try:
                hist_start = int(s.split("-")[0])
                hist_end = hist_start + 1
            except (ValueError, IndexError):
                continue

            # Find rows where the date year doesn't match the season
            mismatched = fetch_all(
                """SELECT id, game_date FROM player_stats
                   WHERE season = ?
                   AND game_date LIKE '____-__-__'
                   AND CAST(SUBSTR(game_date, 1, 4) AS INTEGER) NOT BETWEEN ? AND ?
                   LIMIT 50000""",
                (s, hist_start, hist_end)
            )
            if not mismatched:
                continue

            fixes = []
            for r in mismatched:
                try:
                    parsed = _dt.strptime(r["game_date"], "%Y-%m-%d")
                    if parsed.month >= 10:  # Oct-Dec = season start year
                        fixed = parsed.replace(year=hist_start)
                    else:  # Jan-Sep = season end year
                        fixed = parsed.replace(year=hist_end)
                    fixes.append((fixed.strftime("%Y-%m-%d"), r["id"]))
                except ValueError:
                    continue

            if fixes:
                _exec_many(
                    "UPDATE player_stats SET game_date = ? WHERE id = ?",
                    fixes
                )
                _log.info("Phase 2b: Fixed %d dates for season %s (restored to %d/%d)",
                          len(fixes), s, hist_start, hist_end)

        # ── Phase 3: Add unique index on (player_id, game_id) ──
        # This prevents future duplicates regardless of date format.
        # If duplicates somehow remain, dedupe first: keep the row with
        # the ISO-formatted date (or the lower id as tiebreaker).
        remaining_dups = fetch_all("""
            SELECT player_id, game_id, MIN(id) as keep_id, COUNT(*) as cnt
            FROM player_stats
            GROUP BY player_id, game_id
            HAVING cnt > 1
        """)
        if remaining_dups:
            for rd in remaining_dups:
                execute(
                    "DELETE FROM player_stats WHERE player_id = ? AND game_id = ? AND id != ?",
                    (rd["player_id"], rd["game_id"], rd["keep_id"])
                )
            _log.info("Phase 3: Removed %d leftover duplicate groups", len(remaining_dups))

        try:
            execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_player_stats_player_game "
                    "ON player_stats(player_id, game_id)")
            _log.info("Phase 3: Created unique index on (player_id, game_id)")
        except Exception:
            _log.debug("Phase 3 unique index creation skipped", exc_info=True)

    except Exception as exc:
        _log.warning("_fix_game_date_formats failed: %s", exc)


def _backfill_player_stats_team_id():
    """Backfill team_id in player_stats from game context.

    For home players (is_home=1): team_id = opponent_team_id of away players in same game.
    For away players (is_home=0): team_id = opponent_team_id of home players in same game.
    """
    row = fetch_one("SELECT COUNT(*) as cnt FROM player_stats WHERE team_id IS NULL")
    if not row or row["cnt"] == 0:
        return

    _log.info("Backfilling team_id for %d player_stats rows...", row["cnt"])

    # Home players: their team = opponent of away players in same game
    execute("""
        UPDATE player_stats SET team_id = (
            SELECT DISTINCT ps2.opponent_team_id
            FROM player_stats ps2
            WHERE ps2.game_id = player_stats.game_id
              AND ps2.is_home = 0
            LIMIT 1
        )
        WHERE is_home = 1 AND team_id IS NULL AND game_id IS NOT NULL
    """)

    # Away players: their team = opponent of home players in same game
    execute("""
        UPDATE player_stats SET team_id = (
            SELECT DISTINCT ps2.opponent_team_id
            FROM player_stats ps2
            WHERE ps2.game_id = player_stats.game_id
              AND ps2.is_home = 1
            LIMIT 1
        )
        WHERE is_home = 0 AND team_id IS NULL AND game_id IS NOT NULL
    """)

    # Fallback for any remaining NULLs: use players.team_id
    execute("""
        UPDATE player_stats SET team_id = (
            SELECT p.team_id FROM players p WHERE p.player_id = player_stats.player_id
        )
        WHERE team_id IS NULL
    """)

    remaining = fetch_one("SELECT COUNT(*) as cnt FROM player_stats WHERE team_id IS NULL")
    _log.info("Backfill complete. Remaining NULLs: %d", remaining["cnt"] if remaining else 0)


def reset_db():
    """Drop and recreate all tables."""
    from src.database.db import delete_database
    delete_database()
    init_db()
    # Ensure in-memory DB reflects the fresh schema
    from src.database.db import reload_memory
    reload_memory()


def get_table_counts() -> dict:
    """Return row counts for key tables."""
    tables = ["teams", "players", "player_stats", "predictions",
              "team_metrics", "player_impact", "injuries", "injury_history",
              "injury_status_log", "team_tuning", "notifications",
              "recommendation_snapshot_runs", "recommendation_snapshot_items", "game_odds",
              "arenas", "referees", "game_referees", "elo_ratings",
              "confirmed_lineups"]
    counts = {}
    for t in tables:
        try:
            row = fetch_all(f"SELECT COUNT(*) as cnt FROM {t}")
            counts[t] = row[0]["cnt"] if row else 0
        except Exception:
            counts[t] = 0
    return counts
