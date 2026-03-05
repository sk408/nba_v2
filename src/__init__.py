"""NBA Fundamentals V2 – top-level package."""

# Header patches will be added when the data layer is copied (Task 3).
try:
    from src.data._http_headers import patch_nba_api_headers
    patch_nba_api_headers()
except ImportError:
    pass
