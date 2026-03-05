"""Browser-like HTTP headers and nba_api monkey-patching."""

import importlib

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Host": "stats.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}


def patch_nba_api_headers():
    """Monkey-patch nba_api to use browser-like headers globally."""
    try:
        http_mod = importlib.import_module("nba_api.library.http")
        if hasattr(http_mod, "STATS_HEADERS"):
            http_mod.STATS_HEADERS.update(HEADERS)
        # Also try the older location
        try:
            nba_http = importlib.import_module("nba_api.stats.library.http")
            if hasattr(nba_http, "STATS_HEADERS"):
                nba_http.STATS_HEADERS.update(HEADERS)
        except (ImportError, ModuleNotFoundError):
            pass
    except (ImportError, ModuleNotFoundError):
        pass  # nba_api not installed – graceful degradation
