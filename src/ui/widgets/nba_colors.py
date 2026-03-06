"""NBA team primary/secondary colors keyed by team_id."""

# fmt: off
TEAM_COLORS = {
    1610612737: ("#E03A3E", "#C1D32F"),  # ATL Hawks
    1610612738: ("#007A33", "#BA9653"),  # BOS Celtics
    1610612739: ("#860038", "#041E42"),  # CLE Cavaliers
    1610612740: ("#0C2340", "#C8102E"),  # NOP Pelicans
    1610612741: ("#CE1141", "#000000"),  # CHI Bulls
    1610612742: ("#00538C", "#002B5E"),  # DAL Mavericks
    1610612743: ("#0E2240", "#FEC524"),  # DEN Nuggets
    1610612744: ("#1D428A", "#FFC72C"),  # GSW Warriors
    1610612745: ("#CE1141", "#000000"),  # HOU Rockets
    1610612746: ("#C8102E", "#1D428A"),  # LAC Clippers
    1610612747: ("#552583", "#FDB927"),  # LAL Lakers
    1610612748: ("#98002E", "#F9A01B"),  # MIA Heat
    1610612749: ("#00471B", "#EEE1C6"),  # MIL Bucks
    1610612750: ("#0C2340", "#236192"),  # MIN Timberwolves
    1610612751: ("#000000", "#FFFFFF"),  # BKN Nets
    1610612752: ("#006BB6", "#F58426"),  # NYK Knicks
    1610612753: ("#0077C0", "#000000"),  # ORL Magic
    1610612754: ("#002D62", "#FDBB30"),  # IND Pacers
    1610612755: ("#006BB6", "#ED174C"),  # PHI 76ers
    1610612756: ("#1D1160", "#E56020"),  # PHX Suns
    1610612757: ("#E03A3E", "#000000"),  # POR Trail Blazers
    1610612758: ("#5A2D81", "#63727A"),  # SAC Kings
    1610612759: ("#C4CED4", "#000000"),  # SAS Spurs
    1610612760: ("#007AC1", "#EF6100"),  # OKC Thunder
    1610612761: ("#CE1141", "#000000"),  # TOR Raptors
    1610612762: ("#002B5C", "#00471B"),  # UTA Jazz
    1610612763: ("#5D76A9", "#12173F"),  # MEM Grizzlies
    1610612764: ("#002B5C", "#E31837"),  # WAS Wizards
    1610612765: ("#C8102E", "#1D42BA"),  # DET Pistons
    1610612766: ("#1D1160", "#00788C"),  # CHA Hornets
}
# fmt: on

# Fallback colors for unknown teams
_DEFAULT_PRIMARY = "#3b82f6"
_DEFAULT_SECONDARY = "#1e293b"


def get_team_colors(team_id: int) -> tuple:
    """Return (primary, secondary) hex colors for a team."""
    return TEAM_COLORS.get(team_id, (_DEFAULT_PRIMARY, _DEFAULT_SECONDARY))


def _relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a hex color (0.0 = black, 1.0 = white)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255

    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def ensure_visible(hex_color: str, min_luminance: float = 0.08) -> str:
    """Lighten a hex color if it's too dark for the app's dark background.

    Blends toward white until the color reaches *min_luminance*.
    Returns the original color unchanged if already bright enough.
    """
    if _relative_luminance(hex_color) >= min_luminance:
        return hex_color

    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    # Iteratively blend toward white (fast — converges in <20 steps)
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        nr = int(r + (255 - r) * mid)
        ng = int(g + (255 - g) * mid)
        nb = int(b + (255 - b) * mid)
        candidate = f"#{nr:02x}{ng:02x}{nb:02x}"
        if _relative_luminance(candidate) < min_luminance:
            lo = mid
        else:
            hi = mid

    t = hi
    nr = int(r + (255 - r) * t)
    ng = int(g + (255 - g) * t)
    nb = int(b + (255 - b) * t)
    return f"#{nr:02x}{ng:02x}{nb:02x}"
