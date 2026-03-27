# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Fundamentals V2 is a multi-tier NBA analytics and prediction platform with three entry points:
- **`web.py`** — Flask web dashboard (port 5050) for predictions, odds, and gamecast
- **`desktop.py`** — PySide6 desktop app with live gamecast, pipeline monitoring, settings
- **`overnight.py`** — CLI for the 11-step overnight optimization pipeline (Rich TUI)

Tech stack: Python 3.10+, Flask, PySide6 (Qt6), SQLite (WAL mode with in-memory read cache), Optuna, nba_api, BeautifulSoup4/Cloudscraper.

## Common Commands

```bash
# Run web app (local dev, debug mode)
python web.py

# Run desktop app
python desktop.py

# Run overnight pipeline
python overnight.py --hours 7 --plain

# Lint
ruff check src/

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_pipeline.py -v

# Run a specific test
pytest tests/test_pipeline.py::test_function_name -v

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Entry Points and Bootstrap

All three entry points share initialization via `src/bootstrap.py`, which:
1. Patches NBA API headers
2. Initializes the database + runs migrations (`src/database/migrations.py`)
3. Starts the injury monitor (5-min polling)
4. Starts daily automation (scheduled pipeline + git commits)

The web app under Gunicorn runs bootstrap in `post_fork` (see `gunicorn.conf.py`).

### Database (`src/database/`)

Hybrid SQLite architecture in `db.py`:
- **In-memory** shared SQLite DB serves all reads (loaded at startup via `sqlite3.backup()`)
- **Thread-local** disk connections handle writes with WAL mode
- Writes applied to both disk and memory atomically
- Readers-writer lock (`_RWLock`): multiple concurrent readers, exclusive writer
- Gunicorn uses 2 sync workers to avoid SQLite write contention

Schema: ~20 tables defined in `migrations.py` (teams, players, player_stats, predictions, game_odds, elo_ratings, etc.). Auto-initialized on startup via `init_db()`.

### Analytics Pipeline (`src/analytics/pipeline.py`)

11-step orchestrator run by `overnight.py`:
1. backup → 2. sync → 3. settle_recommendations → 4. seed_arenas → 5. bbref_sync → 6. referee_sync → 7. elo_compute → 8. precompute → 9. optimize_fundamentals → 10. optimize_sharp → 11. backtest

Key modules: `optimizer.py` (Optuna hyperparameter tuning), `prediction.py` (scoring engine), `backtester.py` (A/B comparison), `weight_config.py` (feature family tuning), `elo.py` (team ratings).

### Data Ingestion (`src/data/`)

`sync_service.py` orchestrates all ingestion. Sources: NBA API (`nba_fetcher.py`), DraftKings/FanDuel odds (`odds_sync.py`), ESPN injuries (`injury_scraper.py`), Basketball-Reference (`bbref_scraper.py`), ESPN live gamecast (`gamecast.py`). HTTP client in `http_client.py` handles rate limiting, retries, and Cloudflare bypass.

### Web App (`src/web/app.py`)

Single Flask app file (~2750 LOC) with ~15 routes. Templates in `src/web/templates/`, styles in `src/web/static/style.css`.

### Desktop App (`src/ui/`)

PySide6 tab-based layout in `main_window.py`. Views in `src/ui/views/`, custom widgets in `src/ui/widgets/` (court visualization, scoreboard, play feed).

### Configuration (`src/config.py`)

JSON-backed runtime settings stored in `data/app_settings.json`. Accessed via `src.config.get(key, default)`. Covers season, optimizer params, API rate limits, notification settings, theme.

## Environment Variables

Required in `.env` (not in git):
- `FLASK_SECRET_KEY` — session signing
- `NBA_WEB_DEBUG` — 0 (production) or 1 (dev)
- `NBA_DEPLOY_ENABLED` — 0 or 1

## Linting & Style

- Ruff: line-length 120, target Python 3.10, rules: E, F, W, I
- Config in `pyproject.toml`

## CI

GitHub Actions (`.github/workflows/ci.yml`): Python 3.10, `ruff check src/`, `pytest tests/ -v`.

## Deployment

VPS with Gunicorn + Nginx reverse proxy (port 8443) + Cloudflare SSL. systemd service `nba-web`. See `DEPLOYMENT.md` for full details. Overnight pipeline runs via cron at 2:00 AM UTC.

## Key Data Files

- `data/nba_analytics.db` — main SQLite database
- `data/app_settings.json` — runtime config
- `data/pipeline_state.json` — pipeline step status and timing
- `data/optuna_studies.db` — hyperparameter optimization trials (large, gitignored)

## NBA Date Boundary

The NBA "day" rolls over at 6:00 AM ET, not midnight. See `src/utils/timezone_utils.py`.
