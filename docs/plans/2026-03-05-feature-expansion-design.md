# NBA Fundamentals V2 — Feature Expansion Design

**Date:** 2026-03-05
**Architecture:** Layer Cake (add features as adjustment terms in existing formula)
**Goal:** Maximize both win prediction accuracy AND betting profitability (ATS/ROI)

---

## 1. New Data Sources

### 1A. Arena Data (`src/data/arenas.py`) — Static, one-time
- `ARENAS` dict: all 30 NBA arenas with lat, lon, altitude_ft, timezone
- Haversine distance computation (no external dependency)
- Timezone crossing detection from arena timezone differences

### 1B. Basketball-Reference Scraper (`src/data/bbref_scraper.py`)
- **Team-level:** SRS (Simple Rating System), Pythagorean W%
- **Player-level:** VORP, BPM, WS/48
- BeautifulSoup scraping, 3-second delays, cached daily
- Rate limit: ~20 req/min, enforce 3s between requests

### 1C. NBAstuffer Referee Scraper (`src/data/referee_scraper.py`)
- Fouls/game, home win%, foul differential, total points/game per referee
- Stored in `referees` table, scraped once daily during sync
- BeautifulSoup, standard HTML tables

### 1D. Referee Game Assignments
- Use `nba_api` `ScoreboardV2` endpoint (includes referee names for today's/recent games)
- Cross-reference with NBAstuffer tendency data
- NBA.com referee page is JS-rendered and unreliable — skipped

### 1E. The Odds API — SKIPPED
- User chose to keep Action Network as sole odds source
- Opening/closing line tracking via existing `game_odds.opening_spread` column

---

## 2. New Database Tables

### `arenas`
```sql
CREATE TABLE IF NOT EXISTS arenas (
    team_id    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    city       TEXT NOT NULL,
    lat        REAL NOT NULL,
    lon        REAL NOT NULL,
    altitude_ft INTEGER NOT NULL DEFAULT 0,
    timezone   TEXT NOT NULL DEFAULT 'US/Eastern'
);
```

### `referees`
```sql
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
```

### `game_referees`
```sql
CREATE TABLE IF NOT EXISTS game_referees (
    game_date       TEXT NOT NULL,
    home_team_id    INTEGER NOT NULL,
    away_team_id    INTEGER NOT NULL,
    referee_name    TEXT NOT NULL,
    PRIMARY KEY (game_date, home_team_id, referee_name)
);
```

### `elo_ratings`
```sql
CREATE TABLE IF NOT EXISTS elo_ratings (
    team_id    INTEGER NOT NULL,
    game_date  TEXT NOT NULL,
    elo        REAL NOT NULL DEFAULT 1500.0,
    PRIMARY KEY (team_id, game_date)
);
```

### Modified: `team_metrics` — add columns
- `srs REAL DEFAULT 0.0`
- `pythag_wins REAL DEFAULT 0.0`
- `points_in_paint REAL DEFAULT 0.0`
- `fast_break_pts REAL DEFAULT 0.0`
- `second_chance_pts REAL DEFAULT 0.0`

### Modified: `player_impact` — add columns
- `vorp REAL DEFAULT 0.0`
- `bpm REAL DEFAULT 0.0`
- `ws_per_48 REAL DEFAULT 0.0`

### Modified: `game_odds` — add column
- `spread_movement REAL DEFAULT 0.0` (closing - opening spread)

---

## 3. New Features & GameInput Fields

### 3A. Elo Ratings
- Init all teams at 1500 at season start
- K-factor = 20, logistic expected score
- **GameInput:** `home_elo: float`, `away_elo: float`
- **Weight:** `elo_diff_mult` (0.0-5.0, default 1.0)

### 3B. Travel & Geography
- Haversine distance between arenas
- Timezone crossings (absolute hour difference)
- Cumulative miles in last 7 days
- **GameInput:** `home_travel_miles: float`, `away_travel_miles: float`, `home_tz_crossings: int`, `away_tz_crossings: int`, `home_cum_travel_7d: float`, `away_cum_travel_7d: float`
- **Weights:** `travel_dist_mult` (0.0-3.0, default 0.5), `timezone_crossing_mult` (0.0-5.0, default 1.0)

### 3C. Momentum / Streaks
- Win/loss streak capped at +/-10
- Margin-of-victory trend (avg MOV last 5 games)
- **GameInput:** `home_streak: int`, `away_streak: int`, `home_mov_trend: float`, `away_mov_trend: float`
- **Weights:** `momentum_streak_mult` (0.0-3.0, default 0.3), `mov_trend_mult` (0.0-2.0, default 0.2)

### 3D. Injury Impact (VORP-based)
- Sum VORP of all out/doubtful players
- Weighted by play probability (Out=0.0, Doubtful=0.15, etc.)
- **GameInput:** `home_injury_vorp_lost: float`, `away_injury_vorp_lost: float`
- **Weight:** `injury_vorp_mult` (0.0-5.0, default 1.0)

### 3E. Referee Tendencies
- Crew average fouls/game → total adjustment
- Crew average home win% bias → spread adjustment
- **GameInput:** `ref_crew_fouls_pg: float`, `ref_crew_home_bias: float`
- **Weights:** `ref_fouls_mult` (0.0-3.0, default 0.3), `ref_home_bias_mult` (0.0-3.0, default 0.3)

### 3F. Spread Sharp Money (already in DB)
- Divergence: spread_home_money - spread_home_public
- **GameInput:** `spread_sharp_edge: float`
- **Weight:** `sharp_spread_weight` (0.0-15.0, default 1.0)

### 3G. Schedule Spots
- Lookahead: next opponent is top-8 team by win%
- Letdown: previous game was vs top-8 AND won
- Road trip game number, home stand game number
- **GameInput:** `home_lookahead: bool`, `away_lookahead: bool`, `home_letdown: bool`, `away_letdown: bool`, `home_road_trip_game: int`, `away_road_trip_game: int`
- **Weights:** `lookahead_penalty` (0.0-3.0, default 0.5), `letdown_penalty` (0.0-3.0, default 0.5)

### 3H. Pace Differential
- |home_pace - away_pace| → variance/total adjustment
- **GameInput:** `pace_diff: float` (already computable from existing data)
- **Weight:** `pace_mismatch_mult` (0.0-2.0, default 0.2) — applied to total, not spread

### 3I. SRS / Pythagorean Wins
- SRS from BBRef — combines MOV with strength-of-schedule
- Pythagorean W% — exposes over/under-performing teams
- **GameInput:** `home_srs: float`, `away_srs: float`, `home_pythag_wpct: float`, `away_pythag_wpct: float`
- **Weight:** `srs_diff_mult` (0.0-3.0, default 0.5)

### 3J. Player On/Off Differential (already in DB)
- Sum of net_rating_diff weighted by minutes for active players
- **GameInput:** `home_onoff_impact: float`, `away_onoff_impact: float`
- **Weight:** `onoff_impact_mult` (0.0-3.0, default 0.5)

---

## 4. Prediction Formula Changes

### Existing adjustments (1-13): unchanged

### New adjustments (14-25):

```
14. Elo:          += (home_elo - away_elo) / 400 * elo_diff_mult
15. Travel:       -= (away_travel_penalty - home_travel_penalty) * travel_dist_mult
                  where travel_penalty = miles / 1000.0 (normalized)
16. Timezone:     -= (away_tz_crossings - home_tz_crossings) * timezone_crossing_mult
17. Momentum:     += (home_streak - away_streak) * momentum_streak_mult
18. MOV trend:    += (home_mov_trend - away_mov_trend) * mov_trend_mult
19. Injury VORP:  += (away_vorp_lost - home_vorp_lost) * injury_vorp_mult
20. Ref bias:     += ref_crew_home_bias * ref_home_bias_mult
21. Spread sharp: += spread_sharp_edge / 100.0 * sharp_spread_weight
22. Schedule:     -= (home_lookahead * lookahead_penalty + home_letdown * letdown_penalty)
                  += (away_lookahead * lookahead_penalty + away_letdown * letdown_penalty)
23. SRS:          += (home_srs - away_srs) * srs_diff_mult
24. On/Off:       += (home_onoff_impact - away_onoff_impact) * onoff_impact_mult
25. Pace (total): total_adj += |home_pace - away_pace| * pace_mismatch_mult
    Ref fouls:    total_adj += (ref_crew_fouls_pg - 38.0) * ref_fouls_mult
```

### VectorizedGames expansion
- Add NumPy arrays for all new fields
- Maintain formula parity with predict()

---

## 5. Weight Config Changes

### New parameters (~20):
```python
# Elo
elo_diff_mult: float = 1.0

# Travel & Geography
travel_dist_mult: float = 0.5
timezone_crossing_mult: float = 1.0

# Momentum
momentum_streak_mult: float = 0.3
mov_trend_mult: float = 0.2

# Injury VORP
injury_vorp_mult: float = 1.0

# Referee
ref_fouls_mult: float = 0.3
ref_home_bias_mult: float = 0.3

# Spread Sharp Money
sharp_spread_weight: float = 1.0

# Schedule Spots
lookahead_penalty: float = 0.5
letdown_penalty: float = 0.5

# SRS
srs_diff_mult: float = 0.5

# On/Off Impact
onoff_impact_mult: float = 0.5

# Pace Mismatch
pace_mismatch_mult: float = 0.2
```

Total: ~31 existing + ~14 new = ~45 parameters
(Some feature groups share weights to keep parameter count manageable)

OPTIMIZER_RANGES and CD_RANGES updated for all new parameters.

---

## 6. Pipeline Changes

### Overnight phases:
```
Phase 1 (Full Pipeline):
  1. Backup
  2. Sync teams, players, stats, odds, injuries (existing)
  3. Sync BBRef stats (SRS, Pythagorean, VORP, BPM, WS/48)       NEW
  4. Sync referee tendencies from NBAstuffer                       NEW
  5. Sync referee assignments for recent/today's games             NEW
  6. Compute Elo ratings for all historical games                  NEW
  7. Precompute (includes all new GameInput fields)
  8. Optimize fundamentals (3000 trials, ~45 parameters)
  9. Optimize sharp (3000 trials)
  10. Backtest

Phase 2+ (Loops):
  - Optimize fundamentals -> optimize sharp -> backtest
```

Travel/timezone, momentum, schedule spots computed during precompute (step 7).
Injury VORP computed at prediction time using current injuries + BBRef VORP data.

---

## 7. Loss Function Enhancement

```python
# Current
loss = -(winner_pct + upset_accuracy * upset_rate / 100.0 * 0.5)

# Enhanced
ats_bonus = ats_win_pct * 0.3
loss = -(winner_pct + upset_accuracy * upset_rate / 100.0 * 0.5 + ats_bonus)
```

ATS win% computed by comparing predicted spread vs vegas spread vs actual result.

---

## 8. Files Summary

### New files (3):
- `src/data/arenas.py` — static arena data + haversine
- `src/data/bbref_scraper.py` — Basketball-Reference scraper
- `src/data/referee_scraper.py` — NBAstuffer referee scraper

### Modified files (~8):
- `src/database/migrations.py` — new tables + column additions
- `src/analytics/weight_config.py` — new parameters + ranges
- `src/analytics/prediction.py` — new GameInput fields + adjustment layers + VectorizedGames arrays
- `src/analytics/optimizer.py` — VectorizedGames expansion + loss function
- `src/analytics/stats_engine.py` — travel, momentum, schedule spot computation
- `src/analytics/pipeline.py` — new sync steps + Elo computation
- `src/data/nba_fetcher.py` — referee assignment extraction
- `src/data/odds_sync.py` — spread movement computation

### Parameter count: ~31 existing + ~14 new = ~45 total
### GameInput fields: ~40 existing + ~25 new = ~65 total
### Adjustment layers: 13 existing + 12 new = 25 total
