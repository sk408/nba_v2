"""Centralized team abbreviation mappings for external data sources."""

# ESPN uses shorter/different abbreviations than the NBA standard stored in our DB
ESPN_TO_NBA: dict[str, str] = {
    "GS":   "GSW",
    "SA":   "SAS",
    "NY":   "NYK",
    "NO":   "NOP",
    "WSH":  "WAS",
    "UTAH": "UTA",
    "PHO":  "PHX",
    "BK":   "BKN",
}

# Action Network abbreviation differences
ACTION_TO_NBA: dict[str, str] = {
    "NO":   "NOP",
    "NY":   "NYK",
    "SA":   "SAS",
    "WSH":  "WAS",
    "GS":   "GSW",
    "UTAH": "UTA",
}


def normalize_espn_abbr(abbr: str) -> str:
    """Translate an ESPN abbreviation to the NBA/DB standard."""
    return ESPN_TO_NBA.get(abbr.upper(), abbr)


def normalize_action_abbr(abbr: str) -> str:
    """Translate an Action Network abbreviation to the NBA/DB standard."""
    if not abbr:
        return ""
    return ACTION_TO_NBA.get(abbr.upper(), abbr.upper())
