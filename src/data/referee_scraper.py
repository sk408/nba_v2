"""NBAstuffer referee tendency scraper.

Scrapes referee stats from NBAstuffer.com and syncs them into the local
``referees`` table.  Uses flexible header matching so minor column-name
changes on the site do not break the parser.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from src.database import db

logger = logging.getLogger(__name__)

_UA = "NBAFundamentalsV2/1.0 (research)"
_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_float(value: str) -> Optional[float]:
    """Strip %, commas, whitespace and convert to float.  Returns None on failure."""
    if not value or not value.strip():
        return None
    cleaned = value.strip().replace("%", "").replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _match_header(text: str) -> Optional[str]:
    """Map a raw header string to a canonical column name using keyword matching.

    Returns one of the canonical column names or None if no match.
    """
    low = text.strip().lower()

    # Order matters: check more specific patterns before generic ones.
    if re.search(r"differential|diff", low):
        return "foul_differential"
    if re.search(r"home.*win|win.*pct|home.*pct|hw%", low):
        return "home_win_pct"
    if re.search(r"total.*point|point.*game|ppg|pts|total.*pg", low):
        return "total_points_pg"
    if re.search(r"foul.*game|fouls.*pg|fpg|fouls.*per", low):
        return "fouls_per_game"
    if re.search(r"game|gp|officiated|#games|games$", low):
        return "games_officiated"
    if re.search(r"referee|ref|name|official", low):
        return "referee_name"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_referee_stats(season: str = "2025-26") -> List[Dict]:
    """Scrape referee stats from NBAstuffer for *season*.

    Parameters
    ----------
    season : str
        Season slug used in the NBAstuffer URL, e.g. ``"2025-26"``.

    Returns
    -------
    list[dict]
        Each dict contains at least ``referee_name`` plus whichever numeric
        columns were found on the page (``games_officiated``, ``home_win_pct``,
        ``total_points_pg``, ``fouls_per_game``, ``foul_differential``).
    """
    url = f"https://www.nbastuffer.com/{season}-nba-referee-stats/"
    logger.info("Fetching referee stats from %s", url)

    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the stats table — NBAstuffer uses a standard <table> element.
    table = soup.find("table")
    if table is None:
        logger.warning("No <table> found on %s", url)
        return []

    # ---- Build column map from header row --------------------------------
    header_row = table.find("thead")
    if header_row:
        header_cells = header_row.find_all(["th", "td"])
    else:
        # Fallback: first <tr> might be the header
        first_row = table.find("tr")
        header_cells = first_row.find_all(["th", "td"]) if first_row else []

    col_map: Dict[int, str] = {}
    for idx, cell in enumerate(header_cells):
        canonical = _match_header(cell.get_text(strip=True))
        if canonical:
            col_map[idx] = canonical

    if "referee_name" not in col_map.values():
        # If we could not identify the referee name column, assume the first
        # text column is the name.
        for idx, cell in enumerate(header_cells):
            if idx not in col_map:
                col_map[idx] = "referee_name"
                break

    logger.debug("Column map: %s", col_map)

    # ---- Parse data rows -------------------------------------------------
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

    results: List[Dict] = []
    numeric_cols = {
        "games_officiated",
        "home_win_pct",
        "total_points_pg",
        "fouls_per_game",
        "foul_differential",
    }

    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        entry: Dict = {}
        for idx, cell in enumerate(cells):
            col_name = col_map.get(idx)
            if col_name is None:
                continue
            raw = cell.get_text(strip=True)
            if col_name in numeric_cols:
                val = _parse_float(raw)
                if val is not None:
                    # games_officiated should be int
                    if col_name == "games_officiated":
                        entry[col_name] = int(val)
                    else:
                        entry[col_name] = val
            else:
                entry[col_name] = raw

        # Skip empty / header-like rows
        name = entry.get("referee_name", "").strip()
        if not name or name.lower() in ("referee", "name", "official", ""):
            continue

        results.append(entry)

    logger.info("Parsed %d referees for season %s", len(results), season)
    return results


def sync_referee_stats(season: str = "2025-26") -> int:
    """Fetch referee stats and upsert into the ``referees`` table.

    Parameters
    ----------
    season : str
        Season slug, e.g. ``"2025-26"``.

    Returns
    -------
    int
        Number of referee rows synced.
    """
    data = fetch_referee_stats(season)
    if not data:
        logger.warning("No referee data to sync for season %s", season)
        return 0

    now = datetime.now().isoformat()
    count = 0

    for ref in data:
        name = ref.get("referee_name")
        if not name:
            continue

        db.execute(
            """INSERT INTO referees
                   (referee_name, season, games_officiated, home_win_pct,
                    total_points_pg, fouls_per_game, foul_differential,
                    last_synced_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(referee_name, season) DO UPDATE SET
                   games_officiated  = excluded.games_officiated,
                   home_win_pct      = excluded.home_win_pct,
                   total_points_pg   = excluded.total_points_pg,
                   fouls_per_game    = excluded.fouls_per_game,
                   foul_differential = excluded.foul_differential,
                   last_synced_at    = excluded.last_synced_at""",
            (
                name,
                season,
                ref.get("games_officiated", 0),
                ref.get("home_win_pct", 50.0),
                ref.get("total_points_pg", 215.0),
                ref.get("fouls_per_game", 38.0),
                ref.get("foul_differential", 0.0),
                now,
            ),
        )
        count += 1

    logger.info("Synced %d referees for season %s", count, season)
    return count
