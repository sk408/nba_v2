"""Injury scraping: ESPN → CBS → RotoWire waterfall, plus manual overrides.

Name normalization: all player names are stripped of diacritical marks
(e.g. Dončić → Doncic) before matching to the DB, so accent differences
between sources and stored roster data never cause missed matches.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from src.database import db

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
MANUAL_INJURIES_PATH = Path("data") / "manual_injuries.json"

# ── TTL cache for HTTP scrapes (avoid hammering sources) ──
_scrape_cache: Dict[str, Any] = {}
_scrape_cache_ts: Dict[str, float] = {}
_SCRAPE_TTL = 3600  # 60 minutes


def _get_cached_scrape(source: str) -> Optional[List[Dict[str, Any]]]:
    """Return cached scrape result if still fresh."""
    import time
    if source in _scrape_cache:
        age = time.time() - _scrape_cache_ts.get(source, 0)
        if age < _SCRAPE_TTL:
            logger.info("Using cached %s injuries (%.0fs old)", source, age)
            return _scrape_cache[source]
    return None


def _set_cached_scrape(source: str, data: List[Dict[str, Any]]):
    """Store scrape result in TTL cache."""
    import time
    _scrape_cache[source] = data
    _scrape_cache_ts[source] = time.time()


def invalidate_injury_scrape_cache():
    """Clear all cached scrape results."""
    _scrape_cache.clear()
    _scrape_cache_ts.clear()


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Strip diacritical marks and normalize whitespace.

    Dončić → Doncic, Jokić → Jokic, Hernangómez → Hernangomez, etc.
    Also collapses multiple spaces and strips leading/trailing whitespace.
    """
    # NFKD decomposes accented characters into base + combining marks
    nfkd = unicodedata.normalize("NFKD", name)
    # Drop combining marks (category 'M')
    ascii_name = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    # Collapse whitespace
    return " ".join(ascii_name.split())


def _normalize_status(raw: str) -> str:
    """Normalize injury status to canonical form."""
    low = raw.strip().lower()
    if low in ("out", "o"):
        return "Out"
    if low in ("doubtful", "d"):
        return "Doubtful"
    if low in ("questionable", "q"):
        return "Questionable"
    if low in ("probable", "p"):
        return "Probable"
    if "day" in low:
        return "Day-To-Day"
    if low in ("gtd", "game time", "game-time decision"):
        return "GTD"
    if low in ("available",):
        return "Available"
    return raw.title() if raw else "Out"


def _classify_keyword(detail: str) -> str:
    """Classify injury detail into keyword category."""
    low = detail.lower()
    # Order matters — check specific body parts before general terms.
    # "achilles" before "back" (since "back" matches "coming back", etc.)
    # Surgery/season-ending checked before "rest" (to avoid "rest of the season")
    keywords = [
        "concussion",
        "hamstring", "quad", "calf", "groin", "ankle", "foot", "toe",
        "knee", "acl", "mcl", "meniscus", "hip", "achilles",
        "shoulder", "elbow", "wrist", "hand", "finger", "thigh", "leg",
        "rib", "chest", "abdomen", "neck", "eye",
        "fracture", "surgery", "sprain", "strain",
        "soreness", "contusion",
        "personal", "suspension", "illness",
    ]
    for kw in keywords:
        if kw in low:
            if kw in ("acl", "mcl", "meniscus"):
                return "knee"
            return kw
    # "rest" — only match actual "rest" (load management), not "rest of the season"
    if re.search(r"\brest\b", low) and "rest of" not in low and "remainder" not in low:
        return "rest"
    # "back" — use word boundary to avoid "coming back", "setback", etc.
    if re.search(r"\bback\b", low) and not re.search(r"(come|coming|set|hold|held|get|got)\s*back", low):
        return "back"
    return "other"


def scrape_espn_injuries() -> List[Dict[str, Any]]:
    """Scrape injuries from ESPN.

    ESPN table columns (as of 2025):
      Name | Position | Est. Return | Status | Detail
    """
    cached = _get_cached_scrape("ESPN")
    if cached is not None:
        return cached

    url = "https://www.espn.com/nba/injuries"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        injuries = []
        # Parse ESPN injury tables
        tables = soup.find_all("div", class_="ResponsiveTable")
        for table in tables:
            # Get team name from header
            header = table.find_previous("div", class_="injuries__teamHeader")
            team_name = ""
            if header:
                team_link = header.find("a")
                if team_link:
                    team_name = team_link.get_text(strip=True)
            rows = table.find_all("tr")
            for row in rows[1:]:  # skip header row
                cols = row.find_all("td")
                if len(cols) >= 5:
                    # 5-column layout: Name | Pos | Return Date | Status | Detail
                    name = normalize_name(cols[0].get_text(strip=True))
                    # cols[1] = position (F/G/C) — skip
                    expected_return = cols[2].get_text(strip=True)  # e.g. "Feb 26", "Mar 4"
                    status = cols[3].get_text(strip=True)
                    detail = cols[4].get_text(strip=True)
                elif len(cols) >= 3:
                    # Fallback for older/different layout
                    name = normalize_name(cols[0].get_text(strip=True))
                    expected_return = ""
                    status = cols[1].get_text(strip=True)
                    detail = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                    # Detect if status is actually a position (single letter)
                    if status in ("F", "G", "C", "F-C", "G-F", "F-G", "C-F"):
                        # Columns shifted — try to recover
                        if len(cols) >= 4:
                            expected_return = cols[2].get_text(strip=True)
                            status = cols[3].get_text(strip=True) if len(cols) > 3 else "Out"
                            detail = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                        else:
                            status = "Out"
                            detail = ""
                else:
                    continue

                # Clean up the detail — often has "Feb 18: The Hawks announced..."
                # Extract the actual injury info after the date prefix
                detail_clean = re.sub(r"^[A-Z][a-z]{2}\s+\d{1,2}:\s*", "", detail)

                injuries.append({
                    "name": name,
                    "team": team_name,
                    "status": _normalize_status(status),
                    "detail": detail_clean if detail_clean else detail,
                    "expected_return": expected_return,
                    "keyword": _classify_keyword(detail),
                    "source": "ESPN",
                })
        _set_cached_scrape("ESPN", injuries)
        return injuries
    except Exception as e:
        logger.warning(f"ESPN injury scrape failed: {e}")
        return []


def _clean_cbs_name(raw: str) -> str:
    """Fix CBS doubled names like 'J. KumingaJonathan Kuminga' → 'Jonathan Kuminga'.

    CBS often concatenates abbreviated + full name without separator.
    Handles edge cases: suffixes (Jr., II, III), McNames, D'Angelo, etc.
    """
    raw = raw.strip()
    if not raw:
        return raw

    # Strategy 1: lowercase→uppercase transition (handles most names)
    # e.g. "J. KumingaJonathan Kuminga" — split at "a" → "J"
    matches = list(re.finditer(r"[a-z]([A-Z][a-z])", raw))
    for m in reversed(matches):
        idx = m.start() + 1
        full_part = raw[idx:]
        if " " in full_part.strip():
            return normalize_name(full_part.strip())

    # Strategy 2: abbreviated "X. " prefix with suffix gluing
    # e.g. "D. Lively IIDereck Lively II", "J. Jackson Jr.Jaren Jackson Jr."
    initial_m = re.match(r'^[A-Z]\.\s*', raw)
    if initial_m:
        rest = raw[initial_m.end():]
        # Look for suffix (Jr./Sr./II/III/IV/V) immediately followed by uppercase
        split_m = re.search(r'(?:Jr\.|Sr\.|III|II|IV|V)([A-Z])', rest)
        if split_m:
            full_name = rest[split_m.start(1):]
            if " " in full_name.strip():
                return normalize_name(full_name.strip())
        # Also try direct uppercase→uppercase without suffix
        # e.g. "RussellD'Angelo Russell" → split at "lD"
        split_m2 = re.search(r'([a-z])([A-Z])', rest)
        if split_m2:
            full_name = rest[split_m2.start(2):]
            if " " in full_name.strip():
                return normalize_name(full_name.strip())

    # Fallback: return as-is with normalization
    return normalize_name(raw)


def scrape_cbs_injuries() -> List[Dict[str, Any]]:
    """Scrape injuries from CBS Sports.

    CBS table columns (as of 2025):
      Name | Position | Updated Date | Injury Type
    Note: CBS does NOT provide an explicit status (Out/Day-To-Day/etc).
    We default to "Out" and refine if the injury text hints at status.
    """
    cached = _get_cached_scrape("CBS")
    if cached is not None:
        return cached

    url = "https://www.cbssports.com/nba/injuries/"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        injuries = []
        tables = soup.find_all("table", class_="TableBase-table")
        for table in tables:
            team_header = table.find_previous("div", class_="TeamHeader")
            team_name = ""
            if team_header:
                span = team_header.find("span")
                if span:
                    team_name = span.get_text(strip=True)
            rows = table.find_all("tr")
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue

                # CBS layout: cols[0]=Name (often doubled), cols[1]=Pos,
                # cols[2]=Updated/Status (often a date), cols[3]=Injury type
                raw_name = cols[0].get_text(strip=True)
                name = _clean_cbs_name(raw_name)
                # cols[1] = position — skip
                raw_status = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                injury_type = cols[3].get_text(strip=True) if len(cols) > 3 else ""

                # CBS "status" column is often a date like "Wed, Feb 18"
                # Detect if it looks like a date rather than a real status
                is_date = bool(re.match(
                    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
                    raw_status
                ))

                if is_date or not raw_status:
                    # No real status from CBS — infer from injury text
                    status = _infer_status_from_text(injury_type)
                else:
                    status = _normalize_status(raw_status)

                # Use injury_type as the detail
                detail = injury_type if injury_type else raw_status

                injuries.append({
                    "name": name,
                    "team": team_name,
                    "status": status,
                    "detail": detail,
                    "keyword": _classify_keyword(detail),
                    "source": "CBS",
                })
        _set_cached_scrape("CBS", injuries)
        return injuries
    except Exception as e:
        logger.warning(f"CBS injury scrape failed: {e}")
        return []


def _infer_status_from_text(text: str) -> str:
    """Infer injury status from description text when source doesn't provide one."""
    low = text.lower()
    if any(w in low for w in ("season-ending", "surgery", "torn acl", "torn achilles",
                               "indefinitely", "remainder of")):
        return "Out"
    if any(w in low for w in ("day-to-day", "dtd", "game-time")):
        return "Day-To-Day"
    if "questionable" in low:
        return "Questionable"
    if "probable" in low:
        return "Probable"
    if "doubtful" in low:
        return "Doubtful"
    return "Out"  # conservative default


def scrape_rotowire_injuries() -> List[Dict[str, Any]]:
    """Scrape injuries from RotoWire (fallback — may return 0 if site changed)."""
    cached = _get_cached_scrape("RotoWire")
    if cached is not None:
        return cached

    url = "https://www.rotowire.com/basketball/injury-report.php"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        injuries = []

        # Try multiple possible table selectors (site layout changes often)
        table = (
            soup.find("table", class_="injury-report")
            or soup.find("div", class_="injury-report")
            or soup.find("table", {"id": "injury-report"})
        )
        if table:
            rows = table.find_all("tr")
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    name = normalize_name(cols[0].get_text(strip=True))
                    team = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                    status = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                    detail = cols[3].get_text(strip=True) if len(cols) > 3 else ""
                    injuries.append({
                        "name": name,
                        "team": team,
                        "status": _normalize_status(status),
                        "detail": detail,
                        "keyword": _classify_keyword(detail),
                        "source": "RotoWire",
                    })
        _set_cached_scrape("RotoWire", injuries)
        return injuries
    except Exception as e:
        logger.warning(f"RotoWire injury scrape failed: {e}")
        return []


def load_manual_injuries() -> List[Dict[str, Any]]:
    """Load manual injury overrides from JSON file."""
    if not MANUAL_INJURIES_PATH.exists():
        return []
    try:
        with open(MANUAL_INJURIES_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Failed to load manual injuries: %s", e)
        return []


def save_manual_injury(entry: Dict[str, Any]):
    """Add a manual injury entry."""
    MANUAL_INJURIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    injuries = load_manual_injuries()
    injuries.append(entry)
    with open(MANUAL_INJURIES_PATH, "w") as f:
        json.dump(injuries, f, indent=2)


def add_manual_injury(player_id: int, player_name: str, team_id: int,
                      status: str = "Out", reason: str = ""):
    """Convenience wrapper: add a manual injury by individual fields."""
    entry = {
        "player_id": player_id,
        "name": player_name,
        "team_id": team_id,
        "status": status,
        "reason": reason,
    }
    save_manual_injury(entry)


def remove_manual_injury(player_id_or_name):
    """Remove a manual injury entry by player_id (int) or player_name (str)."""
    injuries = load_manual_injuries()
    if isinstance(player_id_or_name, int):
        injuries = [i for i in injuries if i.get("player_id") != player_id_or_name]
    else:
        injuries = [i for i in injuries
                    if i.get("name", "").lower() != str(player_id_or_name).lower()]
    MANUAL_INJURIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_INJURIES_PATH, "w") as f:
        json.dump(injuries, f, indent=2)


def scrape_all_injuries() -> List[Dict[str, Any]]:
    """Scrape from all sources and merge (ESPN primary, CBS fills gaps).

    ESPN is the highest-quality source (has real status like Out/Day-To-Day).
    CBS provides additional players ESPN might miss but lacks status info.
    RotoWire is a last-resort fallback.
    """
    combined = []
    seen_names: set = set()

    # Primary: ESPN
    espn = scrape_espn_injuries()
    if espn:
        logger.info(f"Got {len(espn)} injuries from ESPN")
        for inj in espn:
            key = inj["name"].lower()
            if key not in seen_names:
                seen_names.add(key)
                combined.append(inj)

    # Secondary: CBS fills gaps (players ESPN missed)
    cbs = scrape_cbs_injuries()
    if cbs:
        logger.info(f"Got {len(cbs)} injuries from CBS")
        for inj in cbs:
            key = inj["name"].lower()
            if key not in seen_names:
                seen_names.add(key)
                combined.append(inj)

    # Fallback: RotoWire
    if not combined:
        rw = scrape_rotowire_injuries()
        if rw:
            logger.info(f"Got {len(rw)} injuries from RotoWire")
            combined = rw

    if combined:
        logger.info(f"Total merged injuries: {len(combined)}")
    else:
        logger.warning("All injury scrapers failed or returned 0 results")

    return combined


def _match_player_id(name: str) -> Optional[int]:
    """Try to find player_id by name match with accent normalization.

    Handles: Dončić/Doncic, Jokić/Jokic, Hernangómez/Hernangomez, etc.
    """
    norm = normalize_name(name).lower()

    # Exact match (on normalized name)
    row = db.fetch_one("SELECT player_id, name FROM players WHERE LOWER(name) = ?", (norm,))
    if row:
        return row["player_id"]

    # Try matching against normalized DB names
    # Fetch all players and compare with normalization
    parts = norm.split()
    if len(parts) >= 2:
        last = parts[-1]
        rows = db.fetch_all(
            "SELECT player_id, name FROM players WHERE LOWER(name) LIKE ?",
            (f"%{last}%",)
        )
        for r in rows:
            db_norm = normalize_name(r["name"]).lower()
            # Check all parts match
            if all(p in db_norm for p in parts):
                return r["player_id"]
            # Also try matching "First Last" directly
            if db_norm == norm:
                return r["player_id"]

    # Last resort: single-part name search (handles "Giannis" etc.)
    if len(parts) == 1:
        rows = db.fetch_all(
            "SELECT player_id, name FROM players WHERE LOWER(name) LIKE ?",
            (f"%{norm}%",)
        )
        if len(rows) == 1:
            return rows[0]["player_id"]

    return None


def sync_injuries(callback=None) -> int:
    """Full injury sync: scrape + manual overrides → update DB."""
    if callback:
        callback("Scraping injury data...")

    scraped = scrape_all_injuries()
    manual = load_manual_injuries()
    now = datetime.now().isoformat()

    # Clear current injury status
    db.execute("UPDATE players SET is_injured = 0, injury_note = NULL")
    db.execute("DELETE FROM injuries")

    all_injuries = manual + scraped  # manual takes precedence (processed first)
    seen_players = set()
    count = 0

    for inj in all_injuries:
        name = normalize_name(inj.get("name", ""))
        pid = inj.get("player_id") or _match_player_id(name)
        if not pid or pid in seen_players:
            continue
        seen_players.add(pid)

        status = inj.get("status", "Out")
        detail = inj.get("detail", "")
        expected_return = inj.get("expected_return", "")
        keyword = inj.get("keyword", _classify_keyword(detail))
        source = "manual" if inj in manual else "scraped"

        # Update player
        db.execute(
            "UPDATE players SET is_injured = 1, injury_note = ? WHERE player_id = ?",
            (f"{status} - {detail}", pid)
        )

        # Get team_id (from player record, or fall back to scraped team name)
        p_row = db.fetch_one("SELECT team_id FROM players WHERE player_id = ?", (pid,))
        team_id = p_row["team_id"] if p_row else 0
        if not team_id and inj.get("team"):
            t_row = db.fetch_one(
                "SELECT team_id FROM teams WHERE name = ? OR abbreviation = ? COLLATE NOCASE",
                (inj["team"], inj["team"]),
            )
            if t_row:
                team_id = t_row["team_id"]

        # Insert into injuries table
        db.execute(
            """INSERT INTO injuries
               (player_id, player_name, team_id, status, reason, expected_return,
                source, injury_keyword, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(player_id) DO UPDATE SET
                 player_name=excluded.player_name,
                 team_id=excluded.team_id,
                 status=excluded.status,
                 reason=excluded.reason,
                 expected_return=excluded.expected_return,
                 source=excluded.source,
                 injury_keyword=excluded.injury_keyword,
                 updated_at=excluded.updated_at""",
            (pid, name, team_id, status, detail, expected_return, source, keyword, now)
        )

        # Log to injury_status_log
        db.execute(
            """INSERT INTO injury_status_log
               (player_id, team_id, log_date, status_level, injury_keyword, injury_detail)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(player_id, log_date, status_level) DO UPDATE SET
                 injury_keyword=excluded.injury_keyword,
                 injury_detail=excluded.injury_detail""",
            (pid, team_id, now[:10], status, keyword, detail)
        )
        count += 1

    if callback:
        callback(f"Updated {count} player injuries")
    return count
