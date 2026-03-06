# Feature Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand prediction model from 13 to 25 adjustment layers with Elo, travel, momentum, injury VORP, referee tendencies, spread sharp money, schedule spots, SRS, player on/off impact, and pace mismatch.

**Architecture:** Layer Cake — add new features as adjustment terms in the existing `predict()` formula. No ML ensemble. Single optimization loop via Optuna TPE (~45 weight parameters).

**Tech Stack:** Python 3.12, SQLite, NumPy, BeautifulSoup, nba_api, Flask. No new heavy dependencies.

**Branch:** `stable-v1` preserves pre-expansion state. All work on `master`.

---

## Task 1: Database Schema — New Tables

**Files:**
- Modify: `src/database/migrations.py`

**Step 1: Add new table definitions to SCHEMA_SQL**

Append these CREATE TABLE statements after the existing tables in `SCHEMA_SQL`:

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
    elo        REAL NOT NULL DEFAULT 1500.0,
    PRIMARY KEY (team_id, game_date)
);
```

**Step 2: Add ALTER TABLE statements for new columns**

Add to the `_run_migrations()` function (which uses `ALTER TABLE ... ADD COLUMN` with try/except for idempotency):

```python
# team_metrics expansions
_add_column("team_metrics", "srs", "REAL DEFAULT 0.0")
_add_column("team_metrics", "pythag_wins", "REAL DEFAULT 0.0")
_add_column("team_metrics", "points_in_paint", "REAL DEFAULT 0.0")
_add_column("team_metrics", "fast_break_pts", "REAL DEFAULT 0.0")
_add_column("team_metrics", "second_chance_pts", "REAL DEFAULT 0.0")

# player_impact expansions
_add_column("player_impact", "vorp", "REAL DEFAULT 0.0")
_add_column("player_impact", "bpm", "REAL DEFAULT 0.0")
_add_column("player_impact", "ws_per_48", "REAL DEFAULT 0.0")

# game_odds expansion
_add_column("game_odds", "spread_movement", "REAL DEFAULT 0.0")
```

**Step 3: Verify**

Run: `python -c "from src.database.migrations import init_db; init_db(); print('OK')"`
Expected: "OK" with no errors. New tables created, columns added.

**Step 4: Commit**

```bash
git add src/database/migrations.py
git commit -m "feat: add schema for arenas, referees, elo_ratings tables and new columns"
```

---

## Task 2: Arena Data Module

**Files:**
- Create: `src/data/arenas.py`

**Step 1: Create the arenas module**

This is a static data file with all 30 NBA arenas plus a haversine distance function. No external API calls needed.

```python
"""Static NBA arena data and travel distance computation."""

import math
from typing import Dict, Tuple

# All 30 NBA arenas — lat/lon/altitude/timezone
# Team IDs match nba_api team IDs
ARENAS: Dict[int, Dict] = {
    1610612737: {"abbr": "ATL", "name": "State Farm Arena",       "city": "Atlanta",       "lat": 33.757, "lon": -84.396,  "altitude_ft": 1050, "tz": "US/Eastern"},
    1610612738: {"abbr": "BOS", "name": "TD Garden",              "city": "Boston",        "lat": 42.366, "lon": -71.062,  "altitude_ft": 20,   "tz": "US/Eastern"},
    1610612751: {"abbr": "BKN", "name": "Barclays Center",        "city": "Brooklyn",      "lat": 40.683, "lon": -73.975,  "altitude_ft": 30,   "tz": "US/Eastern"},
    1610612766: {"abbr": "CHA", "name": "Spectrum Center",        "city": "Charlotte",     "lat": 35.225, "lon": -80.839,  "altitude_ft": 751,  "tz": "US/Eastern"},
    1610612741: {"abbr": "CHI", "name": "United Center",          "city": "Chicago",       "lat": 41.881, "lon": -87.674,  "altitude_ft": 594,  "tz": "US/Central"},
    1610612739: {"abbr": "CLE", "name": "Rocket Mortgage FieldHouse","city": "Cleveland",   "lat": 41.497, "lon": -81.688,  "altitude_ft": 653,  "tz": "US/Eastern"},
    1610612742: {"abbr": "DAL", "name": "American Airlines Center","city": "Dallas",        "lat": 32.790, "lon": -96.810,  "altitude_ft": 430,  "tz": "US/Central"},
    1610612743: {"abbr": "DEN", "name": "Ball Arena",             "city": "Denver",        "lat": 39.749, "lon": -104.999, "altitude_ft": 5289, "tz": "US/Mountain"},
    1610612765: {"abbr": "DET", "name": "Little Caesars Arena",   "city": "Detroit",       "lat": 42.341, "lon": -83.055,  "altitude_ft": 600,  "tz": "US/Eastern"},
    1610612744: {"abbr": "GSW", "name": "Chase Center",           "city": "San Francisco", "lat": 37.768, "lon": -122.388, "altitude_ft": 10,   "tz": "US/Pacific"},
    1610612745: {"abbr": "HOU", "name": "Toyota Center",          "city": "Houston",       "lat": 29.751, "lon": -95.362,  "altitude_ft": 50,   "tz": "US/Central"},
    1610612754: {"abbr": "IND", "name": "Gainbridge Fieldhouse",  "city": "Indianapolis",  "lat": 39.764, "lon": -86.155,  "altitude_ft": 715,  "tz": "US/Eastern"},
    1610612746: {"abbr": "LAC", "name": "Intuit Dome",            "city": "Inglewood",     "lat": 33.958, "lon": -118.341, "altitude_ft": 115,  "tz": "US/Pacific"},
    1610612747: {"abbr": "LAL", "name": "Crypto.com Arena",       "city": "Los Angeles",   "lat": 34.043, "lon": -118.267, "altitude_ft": 270,  "tz": "US/Pacific"},
    1610612763: {"abbr": "MEM", "name": "FedExForum",             "city": "Memphis",       "lat": 35.138, "lon": -90.051,  "altitude_ft": 337,  "tz": "US/Central"},
    1610612748: {"abbr": "MIA", "name": "Kaseya Center",          "city": "Miami",         "lat": 25.781, "lon": -80.187,  "altitude_ft": 7,    "tz": "US/Eastern"},
    1610612749: {"abbr": "MIL", "name": "Fiserv Forum",           "city": "Milwaukee",     "lat": 43.045, "lon": -87.917,  "altitude_ft": 617,  "tz": "US/Central"},
    1610612750: {"abbr": "MIN", "name": "Target Center",          "city": "Minneapolis",   "lat": 44.980, "lon": -93.276,  "altitude_ft": 830,  "tz": "US/Central"},
    1610612740: {"abbr": "NOP", "name": "Smoothie King Center",   "city": "New Orleans",   "lat": 29.949, "lon": -90.082,  "altitude_ft": 3,    "tz": "US/Central"},
    1610612752: {"abbr": "NYK", "name": "Madison Square Garden",  "city": "New York",      "lat": 40.751, "lon": -73.994,  "altitude_ft": 33,   "tz": "US/Eastern"},
    1610612760: {"abbr": "OKC", "name": "Paycom Center",          "city": "Oklahoma City", "lat": 35.463, "lon": -97.515,  "altitude_ft": 1201, "tz": "US/Central"},
    1610612753: {"abbr": "ORL", "name": "Kia Center",             "city": "Orlando",       "lat": 28.539, "lon": -81.384,  "altitude_ft": 82,   "tz": "US/Eastern"},
    1610612755: {"abbr": "PHI", "name": "Wells Fargo Center",     "city": "Philadelphia",  "lat": 39.901, "lon": -75.172,  "altitude_ft": 39,   "tz": "US/Eastern"},
    1610612756: {"abbr": "PHX", "name": "Footprint Center",       "city": "Phoenix",       "lat": 33.446, "lon": -112.071, "altitude_ft": 1086, "tz": "US/Mountain"},
    1610612757: {"abbr": "POR", "name": "Moda Center",            "city": "Portland",      "lat": 45.532, "lon": -122.667, "altitude_ft": 50,   "tz": "US/Pacific"},
    1610612758: {"abbr": "SAC", "name": "Golden 1 Center",        "city": "Sacramento",    "lat": 38.580, "lon": -121.500, "altitude_ft": 30,   "tz": "US/Pacific"},
    1610612759: {"abbr": "SAS", "name": "Frost Bank Center",      "city": "San Antonio",   "lat": 29.427, "lon": -98.438,  "altitude_ft": 650,  "tz": "US/Central"},
    1610612761: {"abbr": "TOR", "name": "Scotiabank Arena",       "city": "Toronto",       "lat": 43.643, "lon": -79.379,  "altitude_ft": 249,  "tz": "US/Eastern"},
    1610612762: {"abbr": "UTA", "name": "Delta Center",           "city": "Salt Lake City","lat": 40.768, "lon": -111.901, "altitude_ft": 4226, "tz": "US/Mountain"},
    1610612764: {"abbr": "WAS", "name": "Capital One Arena",      "city": "Washington",    "lat": 38.898, "lon": -77.021,  "altitude_ft": 72,   "tz": "US/Eastern"},
}

# Timezone UTC offsets (standard time) for crossing calculation
_TZ_OFFSETS = {
    "US/Eastern": -5, "US/Central": -6, "US/Mountain": -7, "US/Pacific": -8,
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def travel_distance(from_team_id: int, to_team_id: int) -> float:
    """Miles between two teams' arenas. Returns 0 if team not found."""
    a = ARENAS.get(from_team_id)
    b = ARENAS.get(to_team_id)
    if not a or not b:
        return 0.0
    return haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])


def timezone_crossings(from_team_id: int, to_team_id: int) -> int:
    """Absolute timezone hour difference between two arenas."""
    a = ARENAS.get(from_team_id)
    b = ARENAS.get(to_team_id)
    if not a or not b:
        return 0
    return abs(_TZ_OFFSETS.get(a["tz"], -5) - _TZ_OFFSETS.get(b["tz"], -5))


def get_altitude(team_id: int) -> int:
    """Altitude in feet for a team's arena. Returns 0 if not found."""
    a = ARENAS.get(team_id)
    return a["altitude_ft"] if a else 0


def seed_arenas_table():
    """Populate the arenas DB table from static data. Idempotent."""
    from src.database import db
    for tid, info in ARENAS.items():
        db.execute(
            "INSERT OR REPLACE INTO arenas (team_id, name, city, lat, lon, altitude_ft, timezone) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, info["name"], info["city"], info["lat"], info["lon"],
             info["altitude_ft"], info["tz"]),
        )
```

**Step 2: Verify**

Run: `python -c "from src.data.arenas import travel_distance, timezone_crossings; print(f'LAL→BOS: {travel_distance(1610612747, 1610612738):.0f} mi, {timezone_crossings(1610612747, 1610612738)} tz'); print(f'LAL→LAC: {travel_distance(1610612747, 1610612746):.0f} mi')"`

Expected: ~2600 mi LAL→BOS, 3 tz crossings, ~11 mi LAL→LAC

**Step 3: Commit**

```bash
git add src/data/arenas.py
git commit -m "feat: add arena data module with haversine distance and timezone crossings"
```

---

## Task 3: Basketball-Reference Scraper

**Files:**
- Create: `src/data/bbref_scraper.py`

**Step 1: Create the BBRef scraper**

Scrapes team-level (SRS, Pythagorean W%) and player-level (VORP, BPM, WS/48) stats. Uses BeautifulSoup on static HTML tables. Respects 3-second rate limits.

```python
"""Basketball-Reference scraper for advanced metrics not in nba_api.

Fetches: SRS, Pythagorean W% (team-level), VORP, BPM, WS/48 (player-level).
Rate-limited to 3s between requests. Cached daily.
"""

import logging
import re
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Comment

from src.database import db

_log = logging.getLogger(__name__)
_BASE = "https://www.basketball-reference.com"
_DELAY = 3.0  # seconds between requests
_HEADERS = {"User-Agent": "NBAFundamentalsV2/1.0 (research)"}

# Map BBRef team abbreviations to nba_api team IDs
_BBREF_ABBR_TO_ID = {
    "ATL": 1610612737, "BOS": 1610612738, "BRK": 1610612751, "CHO": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHO": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}


def _fetch_page(url: str) -> Optional[BeautifulSoup]:
    """Fetch a BBRef page with rate limiting."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        time.sleep(_DELAY)
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        _log.warning("BBRef fetch failed: %s — %s", url, e)
        return None


def _parse_float(text: str) -> float:
    """Parse a float from BBRef table cell, returning 0.0 on failure."""
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return 0.0


def fetch_team_advanced(season_end_year: int = 2026) -> List[Dict]:
    """Fetch SRS and Pythagorean W% for all teams.

    Args:
        season_end_year: e.g., 2026 for the 2025-26 season.

    Returns:
        List of dicts with keys: team_id, srs, pythag_wins
    """
    url = f"{_BASE}/leagues/NBA_{season_end_year}.html"
    soup = _fetch_page(url)
    if not soup:
        return []

    results = []

    # BBRef hides some tables in HTML comments — extract them
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    all_html = str(soup)
    for c in comments:
        all_html += str(c)
    full_soup = BeautifulSoup(all_html, "html.parser")

    # Look for the misc stats table (contains SRS and Pythagorean W/L)
    misc_table = full_soup.find("table", id="misc_stats")
    if not misc_table:
        # Fallback: search for table with SRS header
        for table in full_soup.find_all("table"):
            headers = [th.get_text() for th in table.find_all("th")]
            if "SRS" in headers:
                misc_table = table
                break

    if not misc_table:
        _log.warning("Could not find misc_stats table on BBRef")
        return []

    # Parse header indices
    header_row = misc_table.find("thead")
    if header_row:
        headers = [th.get_text().strip() for th in header_row.find_all("th")]
    else:
        headers = []

    srs_idx = headers.index("SRS") if "SRS" in headers else None
    pw_idx = headers.index("PW") if "PW" in headers else None  # Pythagorean Wins

    tbody = misc_table.find("tbody")
    if not tbody:
        return []

    for row in tbody.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 3:
            continue

        # Team name is in the first th/td with a link
        team_link = row.find("a")
        if not team_link:
            continue

        # Extract BBRef abbreviation from URL: /teams/BOS/2026.html
        href = team_link.get("href", "")
        match = re.search(r"/teams/([A-Z]+)/", href)
        if not match:
            continue
        bbref_abbr = match.group(1)
        team_id = _BBREF_ABBR_TO_ID.get(bbref_abbr)
        if not team_id:
            continue

        texts = [c.get_text().strip() for c in cells]

        srs = _parse_float(texts[srs_idx]) if srs_idx and srs_idx < len(texts) else 0.0
        pw = _parse_float(texts[pw_idx]) if pw_idx and pw_idx < len(texts) else 0.0

        results.append({"team_id": team_id, "srs": srs, "pythag_wins": pw})

    _log.info("BBRef: fetched team advanced stats for %d teams", len(results))
    return results


def fetch_player_advanced(season_end_year: int = 2026) -> List[Dict]:
    """Fetch VORP, BPM, WS/48 for all players.

    Returns:
        List of dicts with keys: player_name, vorp, bpm, ws_per_48
    """
    url = f"{_BASE}/leagues/NBA_{season_end_year}_advanced.html"
    soup = _fetch_page(url)
    if not soup:
        return []

    table = soup.find("table", id="advanced_stats")
    if not table:
        # Fallback
        for t in soup.find_all("table"):
            headers = [th.get_text() for th in t.find_all("th")]
            if "VORP" in headers:
                table = t
                break

    if not table:
        _log.warning("Could not find advanced_stats table on BBRef")
        return []

    header_row = table.find("thead")
    headers = [th.get_text().strip() for th in header_row.find_all("th")] if header_row else []

    vorp_idx = headers.index("VORP") if "VORP" in headers else None
    bpm_idx = headers.index("BPM") if "BPM" in headers else None
    ws48_idx = headers.index("WS/48") if "WS/48" in headers else None

    results = []
    seen = set()  # Dedupe players who were traded (appear multiple times)

    tbody = table.find("tbody")
    if not tbody:
        return []

    for row in tbody.find_all("tr"):
        if row.get("class") and "thead" in row.get("class", []):
            continue  # Skip sub-header rows
        cells = row.find_all(["th", "td"])
        if len(cells) < 5:
            continue

        player_link = row.find("a")
        if not player_link:
            continue

        player_name = player_link.get_text().strip()
        if player_name in seen:
            continue  # Keep first row (TOT for traded players)
        seen.add(player_name)

        texts = [c.get_text().strip() for c in cells]

        # Get team abbreviation
        team_td = cells[4] if len(cells) > 4 else None  # Tm column is usually index 4
        team_abbr = team_td.get_text().strip() if team_td else ""
        if team_abbr == "TOT":
            team_abbr = ""  # Traded player total — won't map to a single team

        vorp = _parse_float(texts[vorp_idx]) if vorp_idx and vorp_idx < len(texts) else 0.0
        bpm = _parse_float(texts[bpm_idx]) if bpm_idx and bpm_idx < len(texts) else 0.0
        ws48 = _parse_float(texts[ws48_idx]) if ws48_idx and ws48_idx < len(texts) else 0.0

        results.append({
            "player_name": player_name,
            "team_abbr": team_abbr,
            "vorp": vorp,
            "bpm": bpm,
            "ws_per_48": ws48,
        })

    _log.info("BBRef: fetched advanced stats for %d players", len(results))
    return results


def sync_bbref_stats(season: str = "2025-26"):
    """Sync BBRef advanced stats into the database.

    Updates team_metrics (srs, pythag_wins) and player_impact (vorp, bpm, ws_per_48).
    """
    # Determine season end year from season string
    parts = season.split("-")
    season_end_year = int("20" + parts[1]) if len(parts) == 2 else 2026

    # Team advanced stats
    team_stats = fetch_team_advanced(season_end_year)
    for ts in team_stats:
        db.execute(
            "UPDATE team_metrics SET srs = ?, pythag_wins = ? "
            "WHERE team_id = ? AND season = ?",
            (ts["srs"], ts["pythag_wins"], ts["team_id"], season),
        )
    _log.info("BBRef: updated team_metrics for %d teams", len(team_stats))

    # Player advanced stats — match by name to player_impact table
    player_stats = fetch_player_advanced(season_end_year)
    updated = 0
    for ps in player_stats:
        # Match player by name (BBRef names should match nba_api names closely)
        result = db.execute(
            "UPDATE player_impact SET vorp = ?, bpm = ?, ws_per_48 = ? "
            "WHERE season = ? AND player_id IN "
            "(SELECT player_id FROM players WHERE name = ?)",
            (ps["vorp"], ps["bpm"], ps["ws_per_48"], season, ps["player_name"]),
        )
        if result:
            updated += 1
    _log.info("BBRef: updated player_impact for %d players", updated)
```

**Step 2: Verify**

Run: `python -c "from src.data.bbref_scraper import fetch_team_advanced; data = fetch_team_advanced(2026); print(f'{len(data)} teams'); print(data[:3] if data else 'empty')"`

Expected: 30 teams with SRS and pythag_wins values.

**Step 3: Commit**

```bash
git add src/data/bbref_scraper.py
git commit -m "feat: add Basketball-Reference scraper for SRS, VORP, BPM, WS/48"
```

---

## Task 4: NBAstuffer Referee Scraper

**Files:**
- Create: `src/data/referee_scraper.py`

**Step 1: Create the referee scraper**

```python
"""NBAstuffer referee tendency scraper.

Fetches: fouls/game, home win%, foul differential, total points/game per referee.
Source: https://www.nbastuffer.com/YYYY-YYYY-nba-referee-stats/
"""

import logging
import re
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from src.database import db

_log = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "NBAFundamentalsV2/1.0 (research)"}


def _parse_float(text: str, default: float = 0.0) -> float:
    try:
        cleaned = re.sub(r"[%,]", "", text.strip())
        return float(cleaned)
    except (ValueError, AttributeError):
        return default


def fetch_referee_stats(season: str = "2025-26") -> List[Dict]:
    """Fetch referee tendency data from NBAstuffer.

    Args:
        season: NBA season string, e.g. "2025-26"

    Returns:
        List of dicts with referee data.
    """
    url = f"https://www.nbastuffer.com/{season}-nba-referee-stats/"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        _log.warning("NBAstuffer fetch failed: %s", e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the main data table
    table = soup.find("table")
    if not table:
        _log.warning("No table found on NBAstuffer referee page")
        return []

    # Parse headers
    header_row = table.find("thead")
    if header_row:
        headers = [th.get_text().strip().lower() for th in header_row.find_all("th")]
    else:
        # Fallback: first row
        first_row = table.find("tr")
        headers = [td.get_text().strip().lower() for td in first_row.find_all(["th", "td"])] if first_row else []

    # Map header names to indices (NBAstuffer headers may vary slightly)
    def find_idx(keywords):
        for i, h in enumerate(headers):
            if any(k in h for k in keywords):
                return i
        return None

    name_idx = find_idx(["referee", "name"])
    games_idx = find_idx(["game", "gp"])
    home_win_idx = find_idx(["home", "win"])
    total_pts_idx = find_idx(["total", "point", "ppg"])
    fouls_idx = find_idx(["foul", "fpg"])
    foul_diff_idx = find_idx(["differential", "diff"])

    if name_idx is None:
        _log.warning("Could not find referee name column in NBAstuffer table")
        return []

    results = []
    tbody = table.find("tbody") or table

    for row in tbody.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 3:
            continue

        texts = [c.get_text().strip() for c in cells]
        name = texts[name_idx] if name_idx < len(texts) else ""
        if not name or name.lower() in ("referee", "name", ""):
            continue

        ref = {
            "referee_name": name,
            "games_officiated": int(_parse_float(texts[games_idx])) if games_idx and games_idx < len(texts) else 0,
            "home_win_pct": _parse_float(texts[home_win_idx]) if home_win_idx and home_win_idx < len(texts) else 50.0,
            "total_points_pg": _parse_float(texts[total_pts_idx]) if total_pts_idx and total_pts_idx < len(texts) else 215.0,
            "fouls_per_game": _parse_float(texts[fouls_idx]) if fouls_idx and fouls_idx < len(texts) else 38.0,
            "foul_differential": _parse_float(texts[foul_diff_idx]) if foul_diff_idx and foul_diff_idx < len(texts) else 0.0,
        }
        results.append(ref)

    _log.info("NBAstuffer: fetched stats for %d referees", len(results))
    return results


def sync_referee_stats(season: str = "2025-26"):
    """Sync referee stats into the referees table."""
    refs = fetch_referee_stats(season)
    for r in refs:
        db.execute(
            "INSERT INTO referees (referee_name, season, games_officiated, home_win_pct, "
            "total_points_pg, fouls_per_game, foul_differential, last_synced_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(referee_name, season) DO UPDATE SET "
            "games_officiated=excluded.games_officiated, home_win_pct=excluded.home_win_pct, "
            "total_points_pg=excluded.total_points_pg, fouls_per_game=excluded.fouls_per_game, "
            "foul_differential=excluded.foul_differential, last_synced_at=excluded.last_synced_at",
            (r["referee_name"], season, r["games_officiated"], r["home_win_pct"],
             r["total_points_pg"], r["fouls_per_game"], r["foul_differential"]),
        )
    _log.info("Synced %d referees to DB", len(refs))
    return len(refs)
```

**Step 2: Verify**

Run: `python -c "from src.data.referee_scraper import fetch_referee_stats; data = fetch_referee_stats(); print(f'{len(data)} refs'); print(data[:2] if data else 'empty')"`

Expected: 70+ referees with fouls_per_game, home_win_pct, etc.

**Step 3: Commit**

```bash
git add src/data/referee_scraper.py
git commit -m "feat: add NBAstuffer referee tendency scraper"
```

---

## Task 5: Elo Rating Computation

**Files:**
- Create: `src/analytics/elo.py`

**Step 1: Create Elo module**

```python
"""Elo rating computation for NBA teams.

Standard Elo with K=20, logistic expected score, home court adjustment.
Computed from historical game results, stored in elo_ratings table.
"""

import logging
from typing import Dict

from src.database import db

_log = logging.getLogger(__name__)

_K = 20.0              # K-factor
_HOME_ELO_ADV = 70.0   # Elo points of home advantage (~3.5 pts spread equivalent)
_INIT_ELO = 1500.0
_SEASON_REGRESS = 0.75  # Regress toward mean between seasons


def _expected(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_all_elo(season: str = "2025-26"):
    """Compute Elo ratings for all historical games and store in DB.

    Processes games chronologically. For the target season, initializes
    from the end of the previous season (with regression toward mean).
    """
    # Get all completed games with results, ordered by date
    rows = db.fetch_all(
        "SELECT DISTINCT ps.game_date, ps.opponent_team_id AS away_id, "
        "       (SELECT team_id FROM players WHERE player_id = ps.player_id) AS home_id, "
        "       ps.is_home "
        "FROM player_stats ps "
        "WHERE ps.season = ? AND ps.is_home = 1 "
        "ORDER BY ps.game_date",
        (season,),
    )

    if not rows:
        _log.warning("No games found for Elo computation in season %s", season)
        return

    # Build game results from player_stats
    # Group by (game_date, home_team, away_team) to get unique games
    games = []
    seen = set()
    for r in rows:
        key = (r["game_date"], r.get("home_id"), r["away_id"])
        if key in seen or not key[1]:
            continue
        seen.add(key)
        games.append(key)

    # Get actual results from game data
    game_results = []
    for game_date, home_id, away_id in games:
        # Get home team result from player_stats win_loss
        row = db.fetch_all(
            "SELECT win_loss FROM player_stats "
            "WHERE game_date = ? AND is_home = 1 AND season = ? "
            "LIMIT 1",
            (game_date, season),
        )
        if row:
            home_won = 1.0 if row[0].get("win_loss", "L") == "W" else 0.0
        else:
            continue
        game_results.append((game_date, home_id, away_id, home_won))

    # Initialize Elo ratings
    elos: Dict[int, float] = {}
    all_team_ids = set()
    for _, h, a, _ in game_results:
        all_team_ids.add(h)
        all_team_ids.add(a)
    for tid in all_team_ids:
        elos[tid] = _INIT_ELO

    # Clear existing Elo data for this season
    db.execute("DELETE FROM elo_ratings WHERE game_date IN "
               "(SELECT DISTINCT game_date FROM player_stats WHERE season = ?)",
               (season,))

    # Process games chronologically
    batch = []
    for game_date, home_id, away_id, home_won in game_results:
        h_elo = elos.get(home_id, _INIT_ELO)
        a_elo = elos.get(away_id, _INIT_ELO)

        # Expected score (with home advantage)
        h_exp = _expected(h_elo + _HOME_ELO_ADV, a_elo)
        a_exp = 1.0 - h_exp

        # Update
        h_new = h_elo + _K * (home_won - h_exp)
        a_new = a_elo + _K * ((1.0 - home_won) - a_exp)

        elos[home_id] = h_new
        elos[away_id] = a_new

        # Store post-game Elo
        batch.append((home_id, game_date, h_new))
        batch.append((away_id, game_date, a_new))

    # Batch insert
    for tid, gd, elo in batch:
        db.execute(
            "INSERT OR REPLACE INTO elo_ratings (team_id, game_date, elo) VALUES (?, ?, ?)",
            (tid, gd, elo),
        )

    _log.info("Elo: computed ratings for %d games, %d teams", len(game_results), len(all_team_ids))


def get_team_elo(team_id: int, game_date: str, season: str = "2025-26") -> float:
    """Get the most recent Elo rating for a team before a given date.

    Returns 1500.0 if no prior rating exists.
    """
    row = db.fetch_all(
        "SELECT elo FROM elo_ratings WHERE team_id = ? AND game_date < ? "
        "ORDER BY game_date DESC LIMIT 1",
        (team_id, game_date),
    )
    if row:
        return row[0]["elo"]
    return _INIT_ELO
```

**Step 2: Verify**

Run: `python -c "from src.database.migrations import init_db; init_db(); from src.analytics.elo import compute_all_elo; compute_all_elo()"`

Expected: "Elo: computed ratings for N games, 30 teams"

**Step 3: Commit**

```bash
git add src/analytics/elo.py
git commit -m "feat: add Elo rating computation module"
```

---

## Task 6: Feature Computation Functions

**Files:**
- Modify: `src/analytics/stats_engine.py`

**Step 1: Add travel computation**

Add these functions to `stats_engine.py`:

```python
def compute_travel(team_id: int, game_date: str, opponent_team_id: int,
                   is_home: bool) -> Dict:
    """Compute travel distance and timezone crossings for a team.

    Returns dict with: travel_miles, tz_crossings, cum_travel_7d
    """
    from src.data.arenas import travel_distance, timezone_crossings

    if is_home:
        # Home team — find where their last game was
        prev = db.fetch_all(
            "SELECT DISTINCT opponent_team_id, is_home FROM player_stats "
            "WHERE game_date < ? AND season = ? "
            "AND player_id IN (SELECT player_id FROM players WHERE team_id = ?) "
            "ORDER BY game_date DESC LIMIT 1",
            (game_date, _game_date_to_season(game_date), team_id),
        )
        if prev and not prev[0]["is_home"]:
            # Last game was away — traveled from opponent's city back home
            last_away_opp = prev[0]["opponent_team_id"]
            miles = travel_distance(last_away_opp, team_id)
        else:
            miles = 0.0
        tz = 0  # Home team, no timezone crossing
    else:
        # Away team — traveled from home (or last away game) to this game
        miles = travel_distance(team_id, opponent_team_id)
        tz = timezone_crossings(team_id, opponent_team_id)

    # Cumulative travel last 7 days
    from datetime import datetime, timedelta
    d = datetime.strptime(game_date, "%Y-%m-%d")
    week_ago = (d - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_games = db.fetch_all(
        "SELECT DISTINCT game_date, opponent_team_id, is_home FROM player_stats "
        "WHERE game_date >= ? AND game_date < ? "
        "AND player_id IN (SELECT player_id FROM players WHERE team_id = ?) "
        "ORDER BY game_date",
        (week_ago, game_date, team_id),
    )
    cum = 0.0
    prev_loc = team_id  # Start from home
    for g in recent_games:
        dest = g["opponent_team_id"] if not g["is_home"] else team_id
        cum += travel_distance(prev_loc, dest)
        prev_loc = dest
    cum += miles  # Add current game travel

    return {"travel_miles": miles, "tz_crossings": tz, "cum_travel_7d": cum}
```

**Step 2: Add momentum computation**

```python
def compute_momentum(team_id: int, game_date: str) -> Dict:
    """Compute momentum features: win streak, margin-of-victory trend.

    Returns dict with: streak (positive=wins, negative=losses, capped ±10),
                       mov_trend (avg margin of victory over last 5 games)
    """
    season = _game_date_to_season(game_date)
    # Get recent game results (last 10 for streak, last 5 for MOV)
    rows = db.fetch_all(
        "SELECT game_date, win_loss, points, "
        "       (SELECT AVG(points) FROM player_stats ps2 "
        "        WHERE ps2.game_date = ps.game_date AND ps2.opponent_team_id = ps.opponent_team_id "
        "        AND ps2.is_home != ps.is_home AND ps2.season = ps.season) AS opp_pts "
        "FROM (SELECT DISTINCT game_date, win_loss, "
        "      (SELECT SUM(points) FROM player_stats ps3 "
        "       WHERE ps3.game_date = player_stats.game_date "
        "       AND ps3.season = player_stats.season "
        "       AND ps3.player_id IN (SELECT player_id FROM players WHERE team_id = ?)"
        "      ) AS points, opponent_team_id, is_home, season "
        "      FROM player_stats "
        "      WHERE season = ? AND game_date < ? "
        "      AND player_id IN (SELECT player_id FROM players WHERE team_id = ?) "
        "      ORDER BY game_date DESC LIMIT 10) ps",
        (team_id, season, game_date, team_id),
    )

    # Simple approach: use win_loss from player_stats
    recent_wl = db.fetch_all(
        "SELECT DISTINCT game_date, win_loss FROM player_stats "
        "WHERE season = ? AND game_date < ? "
        "AND player_id IN (SELECT player_id FROM players WHERE team_id = ?) "
        "ORDER BY game_date DESC LIMIT 10",
        (season, game_date, team_id),
    )

    # Streak
    streak = 0
    for r in recent_wl:
        wl = r.get("win_loss", "L")
        if streak == 0:
            streak = 1 if wl == "W" else -1
        elif (streak > 0 and wl == "W") or (streak < 0 and wl == "L"):
            streak += 1 if streak > 0 else -1
        else:
            break
    streak = max(-10, min(10, streak))

    # MOV trend (average margin over last 5 games) — simplified
    # Use actual scores from game_quarter_scores if available
    mov_rows = db.fetch_all(
        "SELECT final_score, "
        "       (SELECT final_score FROM game_quarter_scores gqs2 "
        "        WHERE gqs2.game_id = gqs.game_id AND gqs2.team_id != gqs.team_id) AS opp_score "
        "FROM game_quarter_scores gqs "
        "WHERE team_id = ? AND game_date < ? "
        "ORDER BY game_date DESC LIMIT 5",
        (team_id, game_date),
    )

    if mov_rows:
        margins = [r["final_score"] - (r.get("opp_score") or r["final_score"])
                   for r in mov_rows if r.get("opp_score")]
        mov_trend = sum(margins) / len(margins) if margins else 0.0
    else:
        mov_trend = 0.0

    return {"streak": streak, "mov_trend": mov_trend}
```

**Step 3: Add schedule spot detection**

```python
def compute_schedule_spots(team_id: int, game_date: str,
                           opponent_team_id: int) -> Dict:
    """Detect schedule spot situations: lookahead, letdown, road trip length.

    Returns dict with: lookahead (bool), letdown (bool),
                       road_trip_game (int, 0 if home)
    """
    season = _game_date_to_season(game_date)

    # Get top-8 teams by win% for this season
    top_teams = db.fetch_all(
        "SELECT team_id FROM team_metrics WHERE season = ? "
        "ORDER BY w_pct DESC LIMIT 8",
        (season,),
    )
    top_ids = {r["team_id"] for r in top_teams}

    # Next game opponent (for lookahead)
    next_game = db.fetch_all(
        "SELECT DISTINCT opponent_team_id FROM player_stats "
        "WHERE season = ? AND game_date > ? "
        "AND player_id IN (SELECT player_id FROM players WHERE team_id = ?) "
        "ORDER BY game_date ASC LIMIT 1",
        (season, game_date, team_id),
    )
    next_opp = next_game[0]["opponent_team_id"] if next_game else 0
    lookahead = next_opp in top_ids and opponent_team_id not in top_ids

    # Previous game (for letdown)
    prev_game = db.fetch_all(
        "SELECT DISTINCT opponent_team_id, win_loss FROM player_stats "
        "WHERE season = ? AND game_date < ? "
        "AND player_id IN (SELECT player_id FROM players WHERE team_id = ?) "
        "ORDER BY game_date DESC LIMIT 1",
        (season, game_date, team_id),
    )
    letdown = False
    if prev_game:
        prev_opp = prev_game[0]["opponent_team_id"]
        prev_won = prev_game[0].get("win_loss", "L") == "W"
        letdown = prev_opp in top_ids and prev_won and opponent_team_id not in top_ids

    # Road trip game number
    road_trip_game = 0
    recent = db.fetch_all(
        "SELECT DISTINCT game_date, is_home FROM player_stats "
        "WHERE season = ? AND game_date <= ? "
        "AND player_id IN (SELECT player_id FROM players WHERE team_id = ?) "
        "ORDER BY game_date DESC LIMIT 10",
        (season, game_date, team_id),
    )
    for r in recent:
        if not r["is_home"]:
            road_trip_game += 1
        else:
            break

    return {
        "lookahead": lookahead,
        "letdown": letdown,
        "road_trip_game": road_trip_game,
    }
```

**Step 4: Add helper for game_date → season**

If `_game_date_to_season` doesn't already exist in stats_engine.py, add:

```python
def _game_date_to_season(game_date: str) -> str:
    """Map YYYY-MM-DD to NBA season string."""
    from src.config import get_season
    return get_season()
```

Note: This function likely already exists in `prediction.py`. Import it if needed.

**Step 5: Verify**

Run a quick smoke test with a known team and date.

**Step 6: Commit**

```bash
git add src/analytics/stats_engine.py
git commit -m "feat: add travel, momentum, and schedule spot computation functions"
```

---

## Task 7: Expand GameInput and Prediction Dataclasses

**Files:**
- Modify: `src/analytics/prediction.py` (GameInput and Prediction dataclasses)

**Step 1: Add new fields to GameInput**

Add after the existing `vegas_away_ml` field (line ~91):

```python
    # ── New V2.1 features ──
    # Elo ratings
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    # Travel & Geography
    home_travel_miles: float = 0.0
    away_travel_miles: float = 0.0
    home_tz_crossings: int = 0
    away_tz_crossings: int = 0
    home_cum_travel_7d: float = 0.0
    away_cum_travel_7d: float = 0.0
    # Momentum
    home_streak: int = 0
    away_streak: int = 0
    home_mov_trend: float = 0.0
    away_mov_trend: float = 0.0
    # Injury VORP
    home_injury_vorp_lost: float = 0.0
    away_injury_vorp_lost: float = 0.0
    # Referee
    ref_crew_fouls_pg: float = 38.0
    ref_crew_home_bias: float = 50.0
    # Spread sharp money
    spread_sharp_edge: float = 0.0
    # Schedule spots
    home_lookahead: bool = False
    away_lookahead: bool = False
    home_letdown: bool = False
    away_letdown: bool = False
    home_road_trip_game: int = 0
    away_road_trip_game: int = 0
    # SRS / Pythagorean
    home_srs: float = 0.0
    away_srs: float = 0.0
    home_pythag_wpct: float = 0.5
    away_pythag_wpct: float = 0.5
    # Player On/Off impact
    home_onoff_impact: float = 0.0
    away_onoff_impact: float = 0.0
    # Pace differential (computed from existing pace fields)
    pace_diff: float = 0.0
```

**Step 2: Verify**

Run: `python -c "from src.analytics.prediction import GameInput; g = GameInput(); print(f'home_elo={g.home_elo}, away_travel_miles={g.away_travel_miles}')"`

Expected: Defaults printed correctly. Also note: this will auto-invalidate the precompute cache via `_precompute_schema_version()` since GameInput fields changed.

**Step 3: Commit**

```bash
git add src/analytics/prediction.py
git commit -m "feat: expand GameInput with 25 new feature fields"
```

---

## Task 8: Expand WeightConfig

**Files:**
- Modify: `src/analytics/weight_config.py`

**Step 1: Add new weight parameters to WeightConfig dataclass**

Add after `sharp_ml_weight` (line ~76):

```python
    # ── New V2.1 weights ──
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

    # Pace Mismatch (total adjustment)
    pace_mismatch_mult: float = 0.2
```

**Step 2: Add new ranges to OPTIMIZER_RANGES**

Add to the `OPTIMIZER_RANGES` dict:

```python
    "elo_diff_mult": (0.0, 5.0),
    "travel_dist_mult": (0.0, 3.0),
    "timezone_crossing_mult": (0.0, 5.0),
    "momentum_streak_mult": (0.0, 3.0),
    "mov_trend_mult": (0.0, 2.0),
    "injury_vorp_mult": (0.0, 5.0),
    "ref_fouls_mult": (0.0, 3.0),
    "ref_home_bias_mult": (0.0, 3.0),
    "sharp_spread_weight": (0.0, 15.0),
    "lookahead_penalty": (0.0, 3.0),
    "letdown_penalty": (0.0, 3.0),
    "srs_diff_mult": (0.0, 3.0),
    "onoff_impact_mult": (0.0, 3.0),
    "pace_mismatch_mult": (0.0, 2.0),
```

**Step 3: Add to SHARP_RANGES**

Add `sharp_spread_weight` to SHARP_RANGES (it's a market signal):

```python
SHARP_RANGES = {
    "sharp_ml_weight": (0.0, 15.0),
    "sharp_spread_weight": (0.0, 15.0),
}
```

**Step 4: Add new ranges to CD_RANGES**

Add wider versions of each new parameter:

```python
    "elo_diff_mult": (0.0, 7.0),
    "travel_dist_mult": (0.0, 5.0),
    "timezone_crossing_mult": (0.0, 7.0),
    "momentum_streak_mult": (0.0, 5.0),
    "mov_trend_mult": (0.0, 3.0),
    "injury_vorp_mult": (0.0, 7.0),
    "ref_fouls_mult": (0.0, 5.0),
    "ref_home_bias_mult": (0.0, 5.0),
    "sharp_spread_weight": (0.0, 20.0),
    "lookahead_penalty": (0.0, 5.0),
    "letdown_penalty": (0.0, 5.0),
    "srs_diff_mult": (0.0, 5.0),
    "onoff_impact_mult": (0.0, 5.0),
    "pace_mismatch_mult": (0.0, 3.0),
```

**Step 5: Verify**

Run: `python -c "from src.analytics.weight_config import WeightConfig, OPTIMIZER_RANGES; w = WeightConfig(); print(f'params: {len(w.to_dict())}'); print(f'opt ranges: {len(OPTIMIZER_RANGES)}')"`

Expected: params ~45, opt ranges ~39

**Step 6: Commit**

```bash
git add src/analytics/weight_config.py
git commit -m "feat: add 14 new weight parameters with optimizer and CD ranges"
```

---

## Task 9: Expand predict() Formula

**Files:**
- Modify: `src/analytics/prediction.py` (the `predict()` function)

**Step 1: Add new adjustment layers after sharp ML (line ~236)**

Insert after the existing sharp money ML block and before the total/projected scores section:

```python
    # ── New V2.1 adjustment layers ──

    # 14. Elo differential
    elo_adj = (game.home_elo - game.away_elo) / 400.0 * w.elo_diff_mult
    game_score += elo_adj
    pred.adjustments["elo"] = elo_adj

    # 15. Travel fatigue
    travel_adj = -((game.away_travel_miles / 1000.0) - (game.home_travel_miles / 1000.0)) * w.travel_dist_mult
    game_score += travel_adj
    pred.adjustments["travel"] = travel_adj

    # 16. Timezone crossing
    tz_adj = -(game.away_tz_crossings - game.home_tz_crossings) * w.timezone_crossing_mult
    game_score += tz_adj
    pred.adjustments["timezone"] = tz_adj

    # 17. Momentum (win/loss streak)
    momentum_adj = (game.home_streak - game.away_streak) * w.momentum_streak_mult
    game_score += momentum_adj
    pred.adjustments["momentum"] = momentum_adj

    # 18. Margin-of-victory trend
    mov_adj = (game.home_mov_trend - game.away_mov_trend) * w.mov_trend_mult
    game_score += mov_adj
    pred.adjustments["mov_trend"] = mov_adj

    # 19. Injury VORP impact
    injury_adj = (game.away_injury_vorp_lost - game.home_injury_vorp_lost) * w.injury_vorp_mult
    game_score += injury_adj
    pred.adjustments["injury_vorp"] = injury_adj

    # 20. Referee home bias
    # Normalize: 50% = neutral, >50% = favors home
    ref_bias_adj = (game.ref_crew_home_bias - 50.0) / 50.0 * w.ref_home_bias_mult
    game_score += ref_bias_adj
    pred.adjustments["ref_home_bias"] = ref_bias_adj

    # 21. Spread sharp money (always, not gated by include_sharp)
    if game.spread_sharp_edge:
        spread_sharp_adj = game.spread_sharp_edge / 100.0 * w.sharp_spread_weight
        game_score += spread_sharp_adj
        pred.adjustments["sharp_spread"] = spread_sharp_adj

    # 22. Schedule spots
    sched_adj = (-(game.home_lookahead * w.lookahead_penalty +
                   game.home_letdown * w.letdown_penalty)
                 + (game.away_lookahead * w.lookahead_penalty +
                    game.away_letdown * w.letdown_penalty))
    game_score += sched_adj
    pred.adjustments["schedule_spots"] = sched_adj

    # 23. SRS differential
    srs_adj = (game.home_srs - game.away_srs) * w.srs_diff_mult
    game_score += srs_adj
    pred.adjustments["srs"] = srs_adj

    # 24. Player On/Off impact
    onoff_adj = (game.home_onoff_impact - game.away_onoff_impact) * w.onoff_impact_mult
    game_score += onoff_adj
    pred.adjustments["onoff_impact"] = onoff_adj
```

**Step 2: Add pace mismatch and ref fouls to the total adjustment section**

In the total calculation section (around line ~239-252), add:

```python
    # Pace mismatch total adjustment
    total += abs(game.home_pace - game.away_pace) * w.pace_mismatch_mult

    # Referee fouls total adjustment (more fouls = more FTs = higher scoring)
    total += (game.ref_crew_fouls_pg - 38.0) * w.ref_fouls_mult
```

**Step 3: Verify**

Run: `python -c "from src.analytics.prediction import predict, GameInput; from src.analytics.weight_config import WeightConfig; g = GameInput(home_elo=1550, away_elo=1450, home_streak=5, away_streak=-2); p = predict(g, WeightConfig()); print(f'score={p.game_score:.2f}, adjustments={p.adjustments}')"`

Expected: game_score includes elo, momentum, etc. adjustments shown in dict.

**Step 4: Commit**

```bash
git add src/analytics/prediction.py
git commit -m "feat: add 12 new adjustment layers to predict() formula"
```

---

## Task 10: Expand VectorizedGames in optimizer.py

**Files:**
- Modify: `src/analytics/optimizer.py` (VectorizedGames class)

**Step 1: Add new NumPy arrays to VectorizedGames.__init__()**

In the constructor where it converts GameInput fields to NumPy arrays, add:

```python
    # V2.1 features
    self.home_elo = np.array([g.home_elo for g in games])
    self.away_elo = np.array([g.away_elo for g in games])
    self.home_travel_miles = np.array([g.home_travel_miles for g in games])
    self.away_travel_miles = np.array([g.away_travel_miles for g in games])
    self.home_tz_crossings = np.array([g.home_tz_crossings for g in games], dtype=float)
    self.away_tz_crossings = np.array([g.away_tz_crossings for g in games], dtype=float)
    self.home_streak = np.array([g.home_streak for g in games], dtype=float)
    self.away_streak = np.array([g.away_streak for g in games], dtype=float)
    self.home_mov_trend = np.array([g.home_mov_trend for g in games])
    self.away_mov_trend = np.array([g.away_mov_trend for g in games])
    self.home_injury_vorp = np.array([g.home_injury_vorp_lost for g in games])
    self.away_injury_vorp = np.array([g.away_injury_vorp_lost for g in games])
    self.ref_crew_fouls_pg = np.array([g.ref_crew_fouls_pg for g in games])
    self.ref_crew_home_bias = np.array([g.ref_crew_home_bias for g in games])
    self.spread_sharp_edge = np.array([g.spread_sharp_edge for g in games])
    self.home_lookahead = np.array([float(g.home_lookahead) for g in games])
    self.away_lookahead = np.array([float(g.away_lookahead) for g in games])
    self.home_letdown = np.array([float(g.home_letdown) for g in games])
    self.away_letdown = np.array([float(g.away_letdown) for g in games])
    self.home_srs = np.array([g.home_srs for g in games])
    self.away_srs = np.array([g.away_srs for g in games])
    self.home_onoff = np.array([g.home_onoff_impact for g in games])
    self.away_onoff = np.array([g.away_onoff_impact for g in games])
    self.pace_diff = np.array([abs(g.home_pace - g.away_pace) for g in games])
```

**Step 2: Add corresponding vectorized adjustments to evaluate()**

In the `evaluate()` method, after the existing sharp ML calculation, add:

```python
    # V2.1 vectorized adjustments
    game_score += (self.home_elo - self.away_elo) / 400.0 * w.elo_diff_mult
    game_score -= ((self.away_travel_miles / 1000.0) - (self.home_travel_miles / 1000.0)) * w.travel_dist_mult
    game_score -= (self.away_tz_crossings - self.home_tz_crossings) * w.timezone_crossing_mult
    game_score += (self.home_streak - self.away_streak) * w.momentum_streak_mult
    game_score += (self.home_mov_trend - self.away_mov_trend) * w.mov_trend_mult
    game_score += (self.away_injury_vorp - self.home_injury_vorp) * w.injury_vorp_mult
    game_score += (self.ref_crew_home_bias - 50.0) / 50.0 * w.ref_home_bias_mult
    game_score += self.spread_sharp_edge / 100.0 * w.sharp_spread_weight
    game_score += (-(self.home_lookahead * w.lookahead_penalty + self.home_letdown * w.letdown_penalty)
                   + (self.away_lookahead * w.lookahead_penalty + self.away_letdown * w.letdown_penalty))
    game_score += (self.home_srs - self.away_srs) * w.srs_diff_mult
    game_score += (self.home_onoff - self.away_onoff) * w.onoff_impact_mult
```

And in the total adjustment section:

```python
    total += self.pace_diff * w.pace_mismatch_mult
    total += (self.ref_crew_fouls_pg - 38.0) * w.ref_fouls_mult
```

**Step 3: Verify**

Run: `python -c "from src.analytics.optimizer import VectorizedGames; from src.analytics.prediction import GameInput; from src.analytics.weight_config import WeightConfig; vg = VectorizedGames([GameInput(home_elo=1550)]); print(f'home_elo array: {vg.home_elo}')"`

Expected: array([1550.])

**Step 4: Commit**

```bash
git add src/analytics/optimizer.py
git commit -m "feat: expand VectorizedGames with 24 new feature arrays"
```

---

## Task 11: Precompute Pipeline — Populate New GameInput Fields

**Files:**
- Modify: `src/analytics/prediction.py` (the `precompute_all_games()` and `predict_matchup()` functions)

**Step 1: Update precompute_all_games() to populate new fields**

In the section where `GameInput` is constructed for each historical game, add:

```python
    # Elo ratings
    from src.analytics.elo import get_team_elo
    gi.home_elo = get_team_elo(home_id, game_date, season)
    gi.away_elo = get_team_elo(away_id, game_date, season)

    # Travel & geography
    home_travel = compute_travel(home_id, game_date, away_id, is_home=True)
    away_travel = compute_travel(away_id, game_date, home_id, is_home=False)
    gi.home_travel_miles = home_travel["travel_miles"]
    gi.away_travel_miles = away_travel["travel_miles"]
    gi.home_tz_crossings = home_travel["tz_crossings"]
    gi.away_tz_crossings = away_travel["tz_crossings"]
    gi.home_cum_travel_7d = home_travel["cum_travel_7d"]
    gi.away_cum_travel_7d = away_travel["cum_travel_7d"]

    # Momentum
    home_momentum = compute_momentum(home_id, game_date)
    away_momentum = compute_momentum(away_id, game_date)
    gi.home_streak = home_momentum["streak"]
    gi.away_streak = away_momentum["streak"]
    gi.home_mov_trend = home_momentum["mov_trend"]
    gi.away_mov_trend = away_momentum["mov_trend"]

    # Schedule spots
    home_sched = compute_schedule_spots(home_id, game_date, away_id)
    away_sched = compute_schedule_spots(away_id, game_date, home_id)
    gi.home_lookahead = home_sched["lookahead"]
    gi.away_lookahead = away_sched["lookahead"]
    gi.home_letdown = home_sched["letdown"]
    gi.away_letdown = away_sched["letdown"]
    gi.home_road_trip_game = home_sched["road_trip_game"]
    gi.away_road_trip_game = away_sched["road_trip_game"]

    # SRS / Pythagorean (from team_metrics)
    h_tm = _get_team_metrics(home_id, season)
    a_tm = _get_team_metrics(away_id, season)
    gi.home_srs = h_tm.get("srs", 0.0) if h_tm else 0.0
    gi.away_srs = a_tm.get("srs", 0.0) if a_tm else 0.0

    # Spread sharp money (from game_odds)
    odds_row = db.fetch_all(
        "SELECT spread_home_money, spread_home_public FROM game_odds "
        "WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
        (game_date, home_id, away_id),
    )
    if odds_row:
        sm = odds_row[0].get("spread_home_money", 0) or 0
        sp = odds_row[0].get("spread_home_public", 0) or 0
        gi.spread_sharp_edge = float(sm - sp)

    # Player On/Off impact (sum of net_rating_diff for active players)
    for side, tid in [("home", home_id), ("away", away_id)]:
        impact_rows = db.fetch_all(
            "SELECT pi.net_rating_diff, pi.on_court_minutes "
            "FROM player_impact pi "
            "JOIN players p ON pi.player_id = p.player_id "
            "WHERE pi.season = ? AND p.team_id = ? AND pi.on_court_minutes > 0",
            (season, tid),
        )
        total_impact = sum(
            r["net_rating_diff"] * min(r["on_court_minutes"], 30) / 30.0
            for r in impact_rows
            if r["net_rating_diff"] is not None
        ) if impact_rows else 0.0
        setattr(gi, f"{side}_onoff_impact", total_impact)

    # Pace differential
    gi.pace_diff = abs(gi.home_pace - gi.away_pace)
```

**Step 2: Update predict_matchup() similarly**

Apply the same field population in `predict_matchup()` (the live prediction path), with the addition of:

```python
    # Injury VORP (live only — uses current injury report)
    injured = _load_current_injuries(home_id, away_id)
    for side, tid in [("home", home_id), ("away", away_id)]:
        vorp_lost = 0.0
        for pid, play_prob in injured.items():
            # Check if player is on this team
            player_row = db.fetch_all(
                "SELECT team_id FROM players WHERE player_id = ?", (pid,)
            )
            if player_row and player_row[0]["team_id"] == tid:
                # Get player VORP
                vorp_row = db.fetch_all(
                    "SELECT vorp FROM player_impact WHERE player_id = ? AND season = ?",
                    (pid, season),
                )
                if vorp_row:
                    vorp_lost += vorp_row[0]["vorp"] * (1.0 - play_prob)
        setattr(gi, f"{side}_injury_vorp_lost", vorp_lost)

    # Referee data (live only — today's assignments)
    ref_rows = db.fetch_all(
        "SELECT r.fouls_per_game, r.home_win_pct FROM game_referees gr "
        "JOIN referees r ON gr.referee_name = r.referee_name "
        "WHERE gr.game_date = ? AND gr.home_team_id = ?",
        (game_date, home_id),
    )
    if ref_rows:
        gi.ref_crew_fouls_pg = sum(r["fouls_per_game"] for r in ref_rows) / len(ref_rows)
        gi.ref_crew_home_bias = sum(r["home_win_pct"] for r in ref_rows) / len(ref_rows)
```

**Step 3: Commit**

```bash
git add src/analytics/prediction.py
git commit -m "feat: populate new GameInput fields in precompute and predict_matchup"
```

---

## Task 12: Pipeline Integration — New Sync Steps

**Files:**
- Modify: `src/analytics/pipeline.py`

**Step 1: Add new sync steps to the pipeline**

In `run_overnight()`, after the existing sync step and before precompute, add:

```python
    # ── New V2.1 sync steps ──

    # Seed arena data (one-time, idempotent)
    from src.data.arenas import seed_arenas_table
    seed_arenas_table()

    # Sync BBRef advanced stats
    if not _is_fresh("bbref_sync", max_age_hours=24):
        _log.info("Syncing Basketball-Reference stats...")
        from src.data.bbref_scraper import sync_bbref_stats
        sync_bbref_stats(season)
        _mark_step_done("bbref_sync")

    # Sync referee tendencies
    if not _is_fresh("referee_sync", max_age_hours=24):
        _log.info("Syncing referee tendencies...")
        from src.data.referee_scraper import sync_referee_stats
        sync_referee_stats(season)
        _mark_step_done("referee_sync")

    # Compute Elo ratings (must run before precompute)
    _log.info("Computing Elo ratings...")
    from src.analytics.elo import compute_all_elo
    compute_all_elo(season)
```

**Step 2: Verify**

Run: `python -c "from src.analytics.pipeline import run_overnight; print('Pipeline imports OK')"`

Expected: No import errors.

**Step 3: Commit**

```bash
git add src/analytics/pipeline.py
git commit -m "feat: add BBRef, referee, Elo sync steps to overnight pipeline"
```

---

## Task 13: Loss Function Enhancement

**Files:**
- Modify: `src/analytics/optimizer.py` (evaluate method and loss calculation)

**Step 1: Add ATS evaluation to VectorizedGames.evaluate()**

In the evaluate method, after computing winner_pct and before the loss calculation, add ATS computation:

```python
    # ATS (Against The Spread) evaluation
    # Predicted spread = game_score, vegas spread = self.vegas_spread
    # ATS win: (predicted covers) matches (actual covers)
    has_spread = self.vegas_spread != 0
    if np.any(has_spread):
        pred_covers_home = game_score[has_spread] > 0  # model says home wins
        actual_margin = self.actual_spread[has_spread]  # actual home margin
        vegas_sp = self.vegas_spread[has_spread]
        actual_covers_home = (actual_margin + vegas_sp) > 0
        ats_correct = pred_covers_home == actual_covers_home
        ats_win_pct = float(np.mean(ats_correct)) * 100.0
    else:
        ats_win_pct = 50.0
```

**Step 2: Update loss function**

```python
    # Enhanced loss with ATS bonus
    ats_bonus = ats_win_pct * 0.3
    loss = -(winner_pct + upset_accuracy * upset_rate / 100.0 * upset_bonus_mult + ats_bonus)
```

Add `ats_win_pct` to the returned metrics dict.

**Step 3: Commit**

```bash
git add src/analytics/optimizer.py
git commit -m "feat: add ATS win% to loss function for betting optimization"
```

---

## Task 14: Integration Verification

**Step 1: Clear precompute cache**

The GameInput schema change will auto-invalidate, but explicitly clear:

```bash
python -c "import os; [os.remove(f'data/cache/{f}') for f in os.listdir('data/cache') if f.endswith('.pkl')]"
```

**Step 2: Run a full overnight pass (short budget)**

```bash
python overnight.py --hours 0.5 --reset-weights
```

Expected: Full pipeline runs — sync, BBRef, referees, Elo, precompute (with new fields), optimize (with ~45 params), backtest. Should complete without errors.

**Step 3: Run a live prediction**

```bash
python -c "
from src.database.migrations import init_db
init_db()
from src.analytics.prediction import predict_matchup
p = predict_matchup(1610612747, 1610612738, '2026-03-05')  # LAL vs BOS
print(f'Pick: {p.pick}, Confidence: {p.confidence:.1f}%')
print(f'Score: {p.projected_away_pts:.0f} - {p.projected_home_pts:.0f}')
print(f'Adjustments: {p.adjustments}')
"
```

Expected: Prediction with all 25 adjustment terms visible in the adjustments dict.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes for feature expansion"
```

---

## Task 15: Final Commit and Push

**Step 1: Verify all changes**

```bash
git diff stable-v1..master --stat
```

Review the diff to ensure nothing unexpected.

**Step 2: Push**

```bash
git push origin master
```

**Step 3: Run extended overnight on desktop**

```bash
python overnight.py --hours 8 --reset-weights
```

This will optimize all ~45 parameters from scratch. The first run with reset weights will take the full 8 hours to converge.

---

## Summary of All Tasks

| # | Task | New Files | Modified Files |
|---|------|-----------|----------------|
| 1 | DB Schema | — | migrations.py |
| 2 | Arena Data | arenas.py | — |
| 3 | BBRef Scraper | bbref_scraper.py | — |
| 4 | Referee Scraper | referee_scraper.py | — |
| 5 | Elo Module | elo.py | — |
| 6 | Feature Functions | — | stats_engine.py |
| 7 | GameInput Expansion | — | prediction.py |
| 8 | WeightConfig Expansion | — | weight_config.py |
| 9 | predict() Formula | — | prediction.py |
| 10 | VectorizedGames | — | optimizer.py |
| 11 | Precompute Population | — | prediction.py |
| 12 | Pipeline Integration | — | pipeline.py |
| 13 | Loss Function | — | optimizer.py |
| 14 | Integration Verification | — | (fixes as needed) |
| 15 | Final Push | — | — |

**Total: 4 new files, ~8 modified files, ~14 new weight parameters, ~25 new GameInput fields, 12 new adjustment layers.**
