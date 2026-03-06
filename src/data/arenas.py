"""NBA arena data: locations, altitudes, timezones, and travel helpers."""

import math
import logging

from src.database.db import execute, fetch_all

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All 30 NBA arenas keyed by nba_api team_id
# ---------------------------------------------------------------------------

ARENAS = {
    1610612737: {  # ATL
        "abbr": "ATL",
        "name": "State Farm Arena",
        "city": "Atlanta, GA",
        "lat": 33.7573,
        "lon": -84.3963,
        "altitude_ft": 1050,
        "tz": "US/Eastern",
    },
    1610612738: {  # BOS
        "abbr": "BOS",
        "name": "TD Garden",
        "city": "Boston, MA",
        "lat": 42.3662,
        "lon": -71.0621,
        "altitude_ft": 20,
        "tz": "US/Eastern",
    },
    1610612751: {  # BKN
        "abbr": "BKN",
        "name": "Barclays Center",
        "city": "Brooklyn, NY",
        "lat": 40.6826,
        "lon": -73.9754,
        "altitude_ft": 30,
        "tz": "US/Eastern",
    },
    1610612766: {  # CHA
        "abbr": "CHA",
        "name": "Spectrum Center",
        "city": "Charlotte, NC",
        "lat": 35.2251,
        "lon": -80.8392,
        "altitude_ft": 751,
        "tz": "US/Eastern",
    },
    1610612741: {  # CHI
        "abbr": "CHI",
        "name": "United Center",
        "city": "Chicago, IL",
        "lat": 41.8807,
        "lon": -87.6742,
        "altitude_ft": 594,
        "tz": "US/Central",
    },
    1610612739: {  # CLE
        "abbr": "CLE",
        "name": "Rocket Mortgage FieldHouse",
        "city": "Cleveland, OH",
        "lat": 41.4965,
        "lon": -81.6882,
        "altitude_ft": 653,
        "tz": "US/Eastern",
    },
    1610612742: {  # DAL
        "abbr": "DAL",
        "name": "American Airlines Center",
        "city": "Dallas, TX",
        "lat": 32.7905,
        "lon": -96.8103,
        "altitude_ft": 430,
        "tz": "US/Central",
    },
    1610612743: {  # DEN
        "abbr": "DEN",
        "name": "Ball Arena",
        "city": "Denver, CO",
        "lat": 39.7487,
        "lon": -105.0077,
        "altitude_ft": 5280,
        "tz": "US/Mountain",
    },
    1610612765: {  # DET
        "abbr": "DET",
        "name": "Little Caesars Arena",
        "city": "Detroit, MI",
        "lat": 42.3411,
        "lon": -83.0553,
        "altitude_ft": 600,
        "tz": "US/Eastern",
    },
    1610612744: {  # GSW
        "abbr": "GSW",
        "name": "Chase Center",
        "city": "San Francisco, CA",
        "lat": 37.7680,
        "lon": -122.3877,
        "altitude_ft": 5,
        "tz": "US/Pacific",
    },
    1610612745: {  # HOU
        "abbr": "HOU",
        "name": "Toyota Center",
        "city": "Houston, TX",
        "lat": 29.7508,
        "lon": -95.3621,
        "altitude_ft": 80,
        "tz": "US/Central",
    },
    1610612754: {  # IND
        "abbr": "IND",
        "name": "Gainbridge Fieldhouse",
        "city": "Indianapolis, IN",
        "lat": 39.7640,
        "lon": -86.1555,
        "altitude_ft": 715,
        "tz": "US/Eastern",
    },
    1610612746: {  # LAC
        "abbr": "LAC",
        "name": "Intuit Dome",
        "city": "Inglewood, CA",
        "lat": 33.9517,
        "lon": -118.3412,
        "altitude_ft": 110,
        "tz": "US/Pacific",
    },
    1610612747: {  # LAL
        "abbr": "LAL",
        "name": "Crypto.com Arena",
        "city": "Los Angeles, CA",
        "lat": 34.0430,
        "lon": -118.2673,
        "altitude_ft": 270,
        "tz": "US/Pacific",
    },
    1610612763: {  # MEM
        "abbr": "MEM",
        "name": "FedExForum",
        "city": "Memphis, TN",
        "lat": 35.1382,
        "lon": -90.0507,
        "altitude_ft": 337,
        "tz": "US/Central",
    },
    1610612748: {  # MIA
        "abbr": "MIA",
        "name": "Kaseya Center",
        "city": "Miami, FL",
        "lat": 25.7814,
        "lon": -80.1870,
        "altitude_ft": 7,
        "tz": "US/Eastern",
    },
    1610612749: {  # MIL
        "abbr": "MIL",
        "name": "Fiserv Forum",
        "city": "Milwaukee, WI",
        "lat": 43.0451,
        "lon": -87.9174,
        "altitude_ft": 617,
        "tz": "US/Central",
    },
    1610612750: {  # MIN
        "abbr": "MIN",
        "name": "Target Center",
        "city": "Minneapolis, MN",
        "lat": 44.9795,
        "lon": -93.2761,
        "altitude_ft": 830,
        "tz": "US/Central",
    },
    1610612740: {  # NOP
        "abbr": "NOP",
        "name": "Smoothie King Center",
        "city": "New Orleans, LA",
        "lat": 29.9490,
        "lon": -90.0821,
        "altitude_ft": 3,
        "tz": "US/Central",
    },
    1610612752: {  # NYK
        "abbr": "NYK",
        "name": "Madison Square Garden",
        "city": "New York, NY",
        "lat": 40.7505,
        "lon": -73.9934,
        "altitude_ft": 33,
        "tz": "US/Eastern",
    },
    1610612760: {  # OKC
        "abbr": "OKC",
        "name": "Paycom Center",
        "city": "Oklahoma City, OK",
        "lat": 35.4634,
        "lon": -97.5151,
        "altitude_ft": 1201,
        "tz": "US/Central",
    },
    1610612753: {  # ORL
        "abbr": "ORL",
        "name": "Kia Center",
        "city": "Orlando, FL",
        "lat": 28.5392,
        "lon": -81.3839,
        "altitude_ft": 82,
        "tz": "US/Eastern",
    },
    1610612755: {  # PHI
        "abbr": "PHI",
        "name": "Wells Fargo Center",
        "city": "Philadelphia, PA",
        "lat": 39.9012,
        "lon": -75.1720,
        "altitude_ft": 39,
        "tz": "US/Eastern",
    },
    1610612756: {  # PHX
        "abbr": "PHX",
        "name": "Footprint Center",
        "city": "Phoenix, AZ",
        "lat": 33.4457,
        "lon": -112.0712,
        "altitude_ft": 1086,
        "tz": "US/Mountain",
    },
    1610612757: {  # POR
        "abbr": "POR",
        "name": "Moda Center",
        "city": "Portland, OR",
        "lat": 45.5316,
        "lon": -122.6668,
        "altitude_ft": 50,
        "tz": "US/Pacific",
    },
    1610612758: {  # SAC
        "abbr": "SAC",
        "name": "Golden 1 Center",
        "city": "Sacramento, CA",
        "lat": 38.5802,
        "lon": -121.4997,
        "altitude_ft": 30,
        "tz": "US/Pacific",
    },
    1610612759: {  # SAS
        "abbr": "SAS",
        "name": "Frost Bank Center",
        "city": "San Antonio, TX",
        "lat": 29.4270,
        "lon": -98.4375,
        "altitude_ft": 650,
        "tz": "US/Central",
    },
    1610612761: {  # TOR
        "abbr": "TOR",
        "name": "Scotiabank Arena",
        "city": "Toronto, ON",
        "lat": 43.6435,
        "lon": -79.3791,
        "altitude_ft": 249,
        "tz": "US/Eastern",
    },
    1610612762: {  # UTA
        "abbr": "UTA",
        "name": "Delta Center",
        "city": "Salt Lake City, UT",
        "lat": 40.7683,
        "lon": -111.9011,
        "altitude_ft": 4226,
        "tz": "US/Mountain",
    },
    1610612764: {  # WAS
        "abbr": "WAS",
        "name": "Capital One Arena",
        "city": "Washington, DC",
        "lat": 38.8981,
        "lon": -77.0209,
        "altitude_ft": 25,
        "tz": "US/Eastern",
    },
}

# ---------------------------------------------------------------------------
# Timezone UTC offsets (standard time)
# ---------------------------------------------------------------------------

TZ_OFFSETS = {
    "US/Eastern": -5,
    "US/Central": -6,
    "US/Mountain": -7,
    "US/Pacific": -8,
}

# ---------------------------------------------------------------------------
# Haversine and travel helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def travel_distance(from_team_id: int, to_team_id: int) -> float:
    """Miles between two arenas (by nba_api team_id)."""
    a = ARENAS[from_team_id]
    b = ARENAS[to_team_id]
    return haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])


def timezone_crossings(from_team_id: int, to_team_id: int) -> int:
    """Absolute timezone hour difference between two arenas."""
    tz_a = TZ_OFFSETS[ARENAS[from_team_id]["tz"]]
    tz_b = TZ_OFFSETS[ARENAS[to_team_id]["tz"]]
    return abs(tz_a - tz_b)


def get_altitude(team_id: int) -> int:
    """Return altitude in feet for an arena."""
    return ARENAS[team_id]["altitude_ft"]


# ---------------------------------------------------------------------------
# Seed the DB arenas table
# ---------------------------------------------------------------------------

def seed_arenas_table() -> None:
    """INSERT OR REPLACE all 30 arenas into the database arenas table."""
    for team_id, info in ARENAS.items():
        execute(
            """INSERT OR REPLACE INTO arenas
               (team_id, name, city, lat, lon, altitude_ft, timezone)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (team_id, info["name"], info["city"],
             info["lat"], info["lon"], info["altitude_ft"], info["tz"]),
        )
    _log.info("Seeded %d arenas into database", len(ARENAS))
