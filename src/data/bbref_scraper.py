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
