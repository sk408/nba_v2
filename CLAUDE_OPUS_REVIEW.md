# NBA V2 - Comprehensive Code Review & Improvement Plan
### By Claude Opus 4.6 | March 9, 2026

---

## Executive Summary

This is a **production-grade NBA analytics and prediction platform** with three interfaces (PySide6 desktop, Flask web, Rich CLI), a 10-step data pipeline, Optuna-powered CMA-ES weight optimization, and a dual-DB SQLite architecture. The codebase spans ~58 Python modules and ~15,000+ lines of code.

**Overall Grade: B**

**Strengths:** Excellent modular architecture, consistent coding style, sophisticated analytics engine, smart dual-DB design with reader-writer locks, graceful degradation patterns throughout.

**Critical Gaps:** Zero test coverage, incomplete `requirements.txt`, undefined functions that crash at runtime, missing web security headers, and known architectural issues around stale reads and season leakage.

---

## Table of Contents

1. [Critical Bugs (Fix Immediately)](#1-critical-bugs)
2. [Security Vulnerabilities](#2-security-vulnerabilities)
3. [Architecture Issues](#3-architecture-issues)
4. [Performance Problems](#4-performance-problems)
5. [Code Quality & Technical Debt](#5-code-quality--technical-debt)
6. [Missing Infrastructure](#6-missing-infrastructure)
7. [UI/UX Issues](#7-uiux-issues)
8. [Feature Ideas & Enhancements](#8-feature-ideas--enhancements)
9. [Prioritized Action Plan](#9-prioritized-action-plan)

---

## 1. Critical Bugs

### BUG-01: Undefined Functions in Score Calibration (CRASH)
**File:** `src/analytics/score_calibration.py` (lines 431, 441)
**Severity:** CRITICAL - Runtime crash

`_fit_mode_payload()` calls two functions that **do not exist anywhere in the codebase**:
- `_blend_near_spread()` (line 431)
- `_apply_team_point_ranges_arrays()` (line 441)

Also references undefined settings: `near_spread_identity_band`, `near_spread_deadband`, `near_spread_raw_weight`, and undefined variable `team_point_ranges`.

**Impact:** Score calibration crashes at runtime. This is a complete blocker for the score realism optimization feature.

**Fix:** Either implement the missing functions or disable the code paths with a feature flag until they're ready.

---

### BUG-02: Parameter Mismatch in Pipeline seed_arenas Step
**File:** `src/analytics/pipeline.py` (line 159 vs 336)
**Severity:** HIGH

`run_seed_arenas()` doesn't accept an `is_cancelled` parameter, but the pipeline passes one. This could cause a `TypeError` at runtime depending on how the function signature handles extra kwargs.

**Fix:** Add `is_cancelled=None` parameter to `run_seed_arenas()` or remove it from the call site.

---

### BUG-03: Race Condition in Prediction Cache
**File:** `src/analytics/prediction.py` (lines 931-933)
**Severity:** MEDIUM

```python
with _mem_pc_lock:
    if _mem_pc_cache is not None and _mem_pc_schema == schema:
        return _mem_pc_cache
# Lock released here - another thread can modify cache before return is used
```

The lock is released before the caller uses the returned cache reference. Another thread could invalidate or rebuild the cache between release and use.

**Fix:** Return a copy of the cache, or hold the lock longer via a context manager pattern.

---

### BUG-04: Tab Fade Animation Race Condition
**File:** `src/ui/main_window.py` (lines 135-148)
**Severity:** LOW-MEDIUM

Rapid tab switching creates new `QGraphicsOpacityEffect` instances on each switch. Old animation refs are overwritten as `self._tab_fade_effect` / `self._tab_fade_anim`. If the user switches tabs fast enough, effects conflict and the `finished.connect()` callback may fire on a stale widget.

**Fix:** Guard with `if self._tab_fade_anim and self._tab_fade_anim.state() == QAbstractAnimation.Running: self._tab_fade_anim.stop()` and reuse effects instead of creating new ones.

---

### BUG-05: Notification Panel Memory Leak
**File:** `src/ui/notification_widget.py` (lines 130-133)
**Severity:** MEDIUM

`deleteLater()` doesn't guarantee immediate cleanup. If the panel refreshes rapidly (polling every 30s), widgets pile up in memory before the event loop processes deletions.

**Fix:** Use `widget.setParent(None)` before `deleteLater()` to immediately detach from layout, or batch-clear only when the panel is visible.

---

## 2. Security Vulnerabilities

### SEC-01: Missing CSRF Protection on All POST Endpoints
**File:** `src/web/app.py` (lines 368, 399, 938)
**Severity:** CRITICAL

No CSRF token validation on `/api/predict`, `/api/sync`, or `/api/shutdown`. An attacker could forge POST requests from external sites to trigger predictions, data syncs, or **shut down the server entirely**.

**Fix:**
```python
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)
```
Or add manual token validation for API endpoints.

---

### SEC-02: Missing Security Headers
**File:** `src/web/app.py`
**Severity:** CRITICAL

No `X-Frame-Options`, `Content-Security-Policy`, `X-Content-Type-Options`, or `Strict-Transport-Security` headers. Vulnerable to clickjacking, MIME sniffing, and XSS.

**Fix:**
```python
@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
    return response
```

---

### SEC-03: Unsafe JSON Parsing with `force=True`
**File:** `src/web/app.py` (line 375)
**Severity:** HIGH

```python
data = request.get_json(force=True, silent=True) or {}
```

`force=True` parses ANY request body as JSON regardless of Content-Type header, potentially bypassing client-side validation.

**Fix:** Remove `force=True`. Use proper `Content-Type: application/json` headers.

---

### SEC-04: Error Messages Expose Internal Details
**File:** `src/web/app.py` (lines 273, 317, 389)
**Severity:** HIGH

Raw Python exception messages (`str(e)`) are passed directly to templates and JSON responses. This exposes database schema, file paths, and internal API errors to users.

**Fix:** Return generic user-friendly messages. Log the full exception server-side.

---

### SEC-05: No Custom Error Pages (404, 500)
**File:** `src/web/app.py`
**Severity:** MEDIUM

Flask's default error pages expose framework version information. No `@app.errorhandler()` decorators found.

**Fix:** Create custom error templates and register handlers for 404, 500, etc.

---

### SEC-06: Regenerating Secret Key on Every Restart
**File:** `src/web/app.py` (line 33)
**Severity:** MEDIUM

```python
app.secret_key = os.urandom(24)
```

This invalidates all sessions and signed cookies on every server restart.

**Fix:** `app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))`

---

### SEC-07: No Rate Limiting on API Endpoints
**File:** `src/web/app.py`
**Severity:** MEDIUM

`/api/sync` spawns background threads with no limit. An attacker could trigger unlimited sync operations causing resource exhaustion.

**Fix:** Use Flask-Limiter or add a simple in-memory rate limit check.

---

### SEC-08: No Input Validation on Route Parameters
**File:** `src/web/app.py` (lines 236-259)
**Severity:** MEDIUM

`/matchup/<home_abbr>/<away_abbr>/<date>` - the `date` parameter is never validated. No regex, no format check, no length limit.

**Fix:** Validate with `re.match(r'^\d{4}-\d{2}-\d{2}$', date)` before processing.

---

### SEC-09: Unescaped ESPN ID in onclick Handler
**File:** `src/web/templates/gamecast.html` (line 214)
**Severity:** LOW-MEDIUM

```javascript
<button onclick="GC.select('${g.espn_id}')">
```

`g.espn_id` is NOT escaped. If it contains quotes or special characters, the onclick handler breaks and could potentially execute injected code.

**Fix:** Use `data-` attributes and event listeners instead of inline onclick.

---

## 3. Architecture Issues

### ARCH-01: Stale-Read Architecture in Write-Heavy Workflows
**Severity:** HIGH

`thread_local_db()` creates isolated SQLite snapshots that don't see concurrent writes from other threads. When used in pipeline/sync/overnight flows that expect fresh reads after writes, data appears stale.

**Affected:** `src/database/db.py`, `src/ui/views/pipeline_view.py`, `src/ui/workers.py`, `overnight.py`

**Fix:** Use the shared DB instance for write-heavy flows, or refresh thread-local connections after write operations.

---

### ARCH-02: Historical Season Leakage in Feature Generation
**Severity:** HIGH

Feature builders call `get_season()` (which returns the *current* season) instead of using the game's actual season. Elo lookup also ignores the season parameter. This means historical backtests contaminate features with current-season context.

**Affected:** `src/analytics/stats_engine.py`, `src/analytics/prediction.py`, `src/analytics/elo.py`

**Fix:** Thread the game's season through all feature-building functions. Add assertion checks.

---

### ARCH-03: Pipeline Step Count Mismatch (Core vs UI)
**Severity:** MEDIUM

Core pipeline defines 10 steps. The UI only tracks 6. This causes the progress bar to be inaccurate and step status indicators to be misaligned.

**Affected:** `src/analytics/pipeline.py` vs `src/ui/views/pipeline_view.py`

**Fix:** Sync the step definitions. Consider a shared `PIPELINE_STEPS` constant both modules import.

---

### ARCH-04: Optimizer vs Prediction Threshold Mismatch
**Severity:** MEDIUM

The optimizer classifies winners using `> 0.5 / < -0.5` thresholds, but live prediction uses `> 0 / < 0`. This means the optimizer optimizes for a different decision boundary than what's used in production.

**Fix:** Unify thresholds. Use a single shared constant.

---

### ARCH-05: No Global State Manager for UI
**Severity:** MEDIUM

Each view maintains its own caches (`_prediction_cache`, `_game_data_cache`, `_espn_headshot_data`). Multiple threads write to shared dicts without synchronization. No centralized state bus or store pattern.

**Fix:** Consider a simple observable state store that views subscribe to, or at minimum add locks to all shared mutable state.

---

### ARCH-06: Historical Team Attribution Instability
**Severity:** MEDIUM

`player_stats` infers team via `JOIN players` but `players.team_id` is mutable (updates when players are traded). Historical queries return incorrect team attribution for traded players.

**Fix:** The `team_id` column was recently added to `player_stats` (commit `7b3e58b`). Verify the backfill is complete and all queries use it.

---

### ARCH-07: Odds Backfill Skips Games Too Aggressively
**Severity:** LOW-MEDIUM

Date-level `NOT IN` query skips ALL games on a date if ANY game on that date has odds. Should use game-level `(game_date, home_team_id, away_team_id)` completeness checking.

**Fix:** Change from date-level to game-level existence check in `odds_sync.py`.

---

## 4. Performance Problems

### PERF-01: Loading ALL Player Stats Into Memory (No WHERE Clause)
**File:** `src/analytics/prediction.py` (lines 1046-1059)
**Severity:** HIGH

```python
rows = db.fetch_all("""
    SELECT player_id, game_id, game_date, is_home, points, minutes
    FROM player_stats ORDER BY game_date
""")
```

Loads ALL player stats for ALL players for ALL seasons with no filter. For a multi-season dataset, this could be 100K+ rows loaded into memory at once.

**Fix:** Add `WHERE season = ?` or `WHERE game_date >= ?` filters. Consider pagination or streaming.

---

### PERF-02: Unbounded Module-Level Caches
**Files:** `src/analytics/prediction.py`, `src/analytics/stats_engine.py`, `src/ui/views/gamecast_view.py`
**Severity:** MEDIUM

Multiple caches grow without bound:
- `_mem_pc_cache` - entire precomputed games cache
- `_mem_ctx_cache` - entire context cache
- `_splits_cache`, `_streak_cache`, `_fatigue_cache` - no size limits
- `_espn_headshot_data` - stores all player photos indefinitely
- `_game_data_cache` - parsed games with no TTL

**Fix:** Add `maxsize` to caches or use `functools.lru_cache` with bounded sizes. Add TTL-based eviction.

---

### PERF-03: VectorizedGames Memory Usage
**File:** `src/analytics/optimizer.py` (lines 483-650)
**Severity:** MEDIUM

Creates separate numpy arrays for every game input feature. With 5000+ games and 49 features, this allocates ~500MB+ of RAM as separate arrays rather than a single structured array.

**Fix:** Use a single numpy structured array or a pandas DataFrame with typed columns.

---

### PERF-04: No Static File Caching Headers
**File:** `src/web/app.py`
**Severity:** LOW-MEDIUM

The 2678-line CSS file (69KB) is served without cache headers. Every page load re-downloads it.

**Fix:** `app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000` or add `Cache-Control` headers.

---

### PERF-05: Injury Scraper Cache Race Condition
**File:** `src/data/injury_scraper.py` (lines 27-47)
**Severity:** LOW

Global `_scrape_cache` dict has no lock. Multiple threads could trigger redundant scrapes in the same TTL window.

**Fix:** Add `threading.Lock()` around cache reads and writes.

---

## 5. Code Quality & Technical Debt

### QUAL-01: Duplicated Utility Functions
**Severity:** MEDIUM

`_safe_float_setting()`, `_safe_int_setting()`, `_safe_bool_setting()` are defined independently in 3+ files:
- `src/analytics/optimizer.py`
- `src/analytics/score_calibration.py`
- `src/analytics/backtester.py`

**Fix:** Move to a shared `src/utils/settings_helpers.py` module.

---

### QUAL-02: 50+ Hardcoded Magic Numbers
**Severity:** MEDIUM

Examples scattered across the codebase:
- `_DECAY = 0.9`, `_PACE_FALLBACK = 98.0`, `_LEAGUE_AVG_PPG = 112.0` (stats_engine.py)
- `K = 20.0`, `HOME_ELO_ADV = 70.0`, `INIT_ELO = 1500.0` (elo.py)
- `WALK_FORWARD_SPLIT = 0.80`, `COMPRESSION_RATIO_FLOOR = 0.55` (optimizer.py)
- `score_calibration_bins = 15`, `spread_cap = 28.0` (score_calibration.py)
- Court dimensions, font sizes, padding values throughout UI

**Fix:** Move analytical constants to config or a dedicated `constants.py`. UI magic numbers to theme constants.

---

### QUAL-03: Broad Exception Handling (Silent Failures)
**Severity:** MEDIUM

8+ locations use `except Exception: pass` or `except Exception:` without logging:
- `src/analytics/prediction.py` lines 506, 689, 714, 779
- `src/analytics/optimizer.py` line 1222
- `src/web/app.py` lines 60-87, 176-208

**Fix:** At minimum, log with `logger.debug()`. Prefer catching specific exception types.

---

### QUAL-04: Three Orphaned TODO Comments
**Severity:** LOW

All three are identical: `# TODO: re-implement odds cache invalidation when prediction_quality module is restored`
- `src/data/odds_sync.py:135`
- `src/data/sync_service.py:81`
- `src/data/sync_service.py:146`

The `prediction_quality` module doesn't exist. Either implement it or remove the TODOs.

---

### QUAL-05: Inconsistent Commit Messages
**Severity:** LOW (but matters for maintainability)

Recent commits mix styles:
- `"PRE SCORE OPTIMIEZ.."` (all caps, misspelled, no prefix)
- `"MALFORM FIX"` (all caps, vague)
- `"TEMP HOLD"` (repeated 2x, no context)
- `"feat: raise upset bonus cap..."` (proper conventional commit)
- `"fix: stabilize optimizer save-gate..."` (proper)

**Fix:** Adopt conventional commits consistently (`feat:`, `fix:`, `perf:`, `chore:`).

---

### QUAL-06: No Linter/Formatter Configuration
**Severity:** LOW

No `pyproject.toml`, `setup.cfg`, `.flake8`, `.pylintrc`, `ruff.toml`, or `black.toml` found. The code is surprisingly consistent anyway, but enforcement would prevent drift.

**Fix:** Add `ruff` config for linting + formatting. Minimal setup:
```toml
# pyproject.toml
[tool.ruff]
line-length = 120
target-version = "py310"
```

---

### QUAL-07: No Random Seed in Optimizer
**File:** `src/analytics/optimizer.py`
**Severity:** LOW

`random.seed()` is never set, so optimization results are non-deterministic across runs. Makes debugging and reproducibility harder.

**Fix:** Add `optuna.logging.set_verbosity(optuna.logging.WARNING)` and set `seed` in CMA-ES sampler.

---

## 6. Missing Infrastructure

### INFRA-01: Zero Test Coverage
**Severity:** CRITICAL

No test files, no test framework, no `pytest.ini`, no `conftest.py`. A production analytics system with 15,000+ lines of code has zero automated tests.

**Priority targets for testing:**
1. `prediction.py` - core prediction logic
2. `optimizer.py` - loss function and save gate logic
3. `backtester.py` - walk-forward correctness
4. `db.py` - read/write/lock behavior
5. `elo.py` - mathematical correctness
6. `sync_service.py` - freshness checks and incremental sync

**Fix:** Set up pytest with fixtures for a test database. Start with unit tests for pure functions, then integration tests for the pipeline.

---

### INFRA-02: Severely Incomplete `requirements.txt`
**Severity:** CRITICAL

**Declared:**
```
flask, requests, beautifulsoup4, numpy, pandas
```

**Actually used but NOT declared:**
- `PySide6` (22 files import it)
- `optuna` (CMA-ES optimization engine)
- `rich` (overnight.py TUI)
- `nba_api` (NBA stats API wrapper)
- `cloudscraper` (Cloudflare bypass for Basketball Reference)
- `websocket-client` (ESPN Fastcast WebSocket)
- `scipy` (Optuna dependency)

**Fix:** Complete the requirements.txt or migrate to `pyproject.toml` with dependency groups:
```toml
[project]
dependencies = ["flask", "requests", "beautifulsoup4", "numpy", "pandas", "nba_api", "cloudscraper", "websocket-client"]

[project.optional-dependencies]
desktop = ["PySide6"]
optimizer = ["optuna", "scipy"]
cli = ["rich"]
dev = ["pytest", "ruff"]
```

---

### INFRA-03: No CI/CD Pipeline
**Severity:** HIGH

No GitHub Actions, no Docker, no pre-commit hooks. No automated linting, testing, or deployment.

**Fix:** Add a minimal GitHub Actions workflow:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: ruff check src/
      - run: pytest tests/ -v
```

---

### INFRA-04: No Environment Variable / Secrets Management
**Severity:** MEDIUM

No `.env` file support. No `python-dotenv` usage. The web app's secret key is randomly generated (SEC-06). If this ever needs API keys (e.g., premium data sources), there's no infrastructure for it.

**Fix:** Add `python-dotenv` support in `src/config.py` with env var overrides for sensitive settings.

---

## 7. UI/UX Issues

### UX-01: Hard-Coded Minimum Window Size
**File:** `src/ui/main_window.py` (line 48)
**Severity:** MEDIUM

```python
self.setMinimumSize(1200, 800)
```

Breaks on small displays, netbooks, and tablets. No responsive scaling.

**Fix:** Calculate minimum from `QScreen.availableGeometry()` and set to 80% of screen size or a sensible minimum.

---

### UX-02: No Keyboard Navigation
**Severity:** MEDIUM

Tab switching is mouse-only. No `Alt+` shortcuts defined. Combo boxes and buttons aren't easily keyboard-accessible. Inaccessible for keyboard-only users.

**Fix:** Add `QShortcut` bindings: `Alt+1` through `Alt+5` for tabs, `Ctrl+R` for refresh, etc.

---

### UX-03: Color-Only Information Encoding
**Severity:** MEDIUM (Accessibility)

- Dog pick badge uses amber color only (no pattern/icon)
- Direction indicators use green/red without text labels
- Court widget shot chart uses red/green only

Colorblind users (~8% of males) cannot distinguish this information.

**Fix:** Add shape/icon/text indicators alongside color. Use colorblind-safe palettes.

---

### UX-04: No Font Scaling for High-DPI
**Severity:** LOW-MEDIUM

Font sizes hardcoded in pixels (`26px`, `14px`, etc.). No `QScreen` DPI detection. On 4K displays, text may be illegibly small; on low-res displays, too large.

**Fix:** Use `QFontMetrics` or scale fonts based on logical DPI.

---

### UX-05: No Export Functionality
**Severity:** LOW

No way to export predictions, backtest results, or analytics data. Users can only view in-app.

**Fix:** Add "Export to CSV" buttons on accuracy_view and matchup_view.

---

### UX-06: Vague Error Messages
**Severity:** LOW

When predictions fail, users see raw Python exceptions in the status bar (e.g., `"IndexError: list index out of range"` instead of `"Game data unavailable"`). No retry button.

**Fix:** Map common exceptions to user-friendly messages. Add a "Retry" action.

---

### UX-07: OLED Theme Implementation is Brittle
**File:** `src/ui/theme.py`
**Severity:** LOW

OLED mode works by string-replacing `#0b0f19` with `#000000` in the QSS stylesheet. If any color is defined differently or new colors are added, they won't be updated.

**Fix:** Use CSS variables (QSS custom properties) or a theme dict that generates the stylesheet.

---

## 8. Feature Ideas & Enhancements

### FEAT-01: Prediction Confidence Intervals
Instead of point estimates for spread/total, show confidence intervals (e.g., "LAL -3.5 +/- 2.1"). The optimizer already has the data to compute this from historical prediction variance.

### FEAT-02: Historical Trend Sparklines
Add tiny sparklines in the matchup view showing each team's last 10 games (W/L streak, scoring trend, defensive rating trend). The data is already in `player_stats` and `team_metrics`.

### FEAT-03: Live Bet Tracker
Since the app already tracks odds and predictions, add a virtual bet tracker that shows "if you had bet every model pick, here's your P&L over time." The backtester already computes ML ROI - surface it as a running chart.

### FEAT-04: Model Explanation View
Show *why* the model picked a team: which features contributed most to the prediction. Could be a simple waterfall chart showing feature weights * values for each game.

### FEAT-05: Push Notifications for Injury Alerts
The injury monitor polls every 5 minutes. Add optional desktop notifications (via `QSystemTrayIcon.showMessage()`) or push to a webhook (Slack/Discord) when a key player's status changes.

### FEAT-06: Head-to-Head Historical Matchup Context
When viewing a matchup (e.g., BOS vs LAL), show the last 5-10 meetings between these teams with spreads, scores, and whether the model would have been correct.

### FEAT-07: Referee Impact Overlay
The data layer already scrapes referee stats. Surface this in the matchup view: "Tonight's officials tend to call X% more fouls than average, historically favoring home teams by Y points."

### FEAT-08: Auto-Backtest After Optimization
After the optimizer saves new weights, automatically run a quick backtest and show the before/after comparison. The pipeline already has both steps - just wire them together with a comparison view.

### FEAT-09: Dark/Light Theme Toggle (Not Just OLED)
Currently only dark mode + OLED variant. Add a proper light theme for daytime use. Many users prefer light themes for readability.

### FEAT-10: Mobile-Responsive Web Dashboard
The Flask web app has decent responsive CSS (breakpoints at 900px, 768px, 600px, 480px), but could be enhanced into a proper PWA with service workers for offline viewing of cached predictions.

### FEAT-11: Database Migration System
Currently schema is defined in `migrations.py` as CREATE TABLE IF NOT EXISTS statements. As the schema evolves, there's no versioned migration system. Consider Alembic or a simple version-stamped migration approach.

### FEAT-12: Prediction Leaderboard / History
Track all predictions made and their outcomes over time. Show a calendar view of daily picks with hit rates, longest win/loss streaks, and best/worst picks. The `predictions` table exists but isn't surfaced in a historical view.

---

## 9. Prioritized Action Plan

### Phase 1: Stop the Bleeding (1-2 days)
| # | Task | Severity | Effort |
|---|------|----------|--------|
| 1 | Fix undefined functions in `score_calibration.py` (BUG-01) | CRITICAL | 2h |
| 2 | Complete `requirements.txt` with all actual dependencies (INFRA-02) | CRITICAL | 30m |
| 3 | Add CSRF protection to Flask POST endpoints (SEC-01) | CRITICAL | 1h |
| 4 | Add security headers to Flask responses (SEC-02) | CRITICAL | 30m |
| 5 | Fix error message exposure - return generic messages (SEC-04) | HIGH | 1h |
| 6 | Fix `force=True` JSON parsing (SEC-03) | HIGH | 15m |
| 7 | Add custom 404/500 error pages (SEC-05) | MEDIUM | 1h |
| 8 | Fix pipeline parameter mismatch (BUG-02) | HIGH | 15m |

### Phase 2: Foundation (3-5 days)
| # | Task | Severity | Effort |
|---|------|----------|--------|
| 9 | Set up pytest + write tests for `prediction.py`, `elo.py`, `db.py` (INFRA-01) | CRITICAL | 2d |
| 10 | Fix stale-read architecture in write-heavy flows (ARCH-01) | HIGH | 4h |
| 11 | Fix historical season leakage in feature generation (ARCH-02) | HIGH | 4h |
| 12 | Unify optimizer vs prediction thresholds (ARCH-04) | MEDIUM | 1h |
| 13 | Sync pipeline step count between core and UI (ARCH-03) | MEDIUM | 2h |
| 14 | Add `WHERE` clause to bulk player_stats load (PERF-01) | HIGH | 1h |
| 15 | Add rate limiting to Flask API endpoints (SEC-07) | MEDIUM | 1h |

### Phase 3: Hardening (1 week)
| # | Task | Severity | Effort |
|---|------|----------|--------|
| 16 | Add route parameter validation (SEC-08) | MEDIUM | 2h |
| 17 | Bound all module-level caches with maxsize/TTL (PERF-02) | MEDIUM | 3h |
| 18 | Consolidate `_safe_*_setting()` to shared utility (QUAL-01) | MEDIUM | 1h |
| 19 | Move magic numbers to config/constants (QUAL-02) | MEDIUM | 3h |
| 20 | Replace broad `except Exception:` with specific types (QUAL-03) | MEDIUM | 2h |
| 21 | Fix prediction cache race condition (BUG-03) | MEDIUM | 1h |
| 22 | Add ruff linter configuration (QUAL-06) | LOW | 30m |
| 23 | Set up GitHub Actions CI (INFRA-03) | HIGH | 2h |
| 24 | Add `.env` support for secrets (INFRA-04) | MEDIUM | 1h |

### Phase 4: Polish & Features (ongoing)
| # | Task | Severity | Effort |
|---|------|----------|--------|
| 25 | Add keyboard navigation to desktop UI (UX-02) | MEDIUM | 3h |
| 26 | Fix colorblind accessibility (UX-03) | MEDIUM | 2h |
| 27 | Add responsive window sizing (UX-01) | MEDIUM | 2h |
| 28 | Add static file caching headers (PERF-04) | LOW | 30m |
| 29 | Implement prediction confidence intervals (FEAT-01) | LOW | 4h |
| 30 | Add CSV export (FEAT-05/UX-05) | LOW | 2h |
| 31 | Add historical trend sparklines (FEAT-02) | LOW | 4h |
| 32 | Add model explanation waterfall chart (FEAT-04) | LOW | 6h |
| 33 | Implement virtual bet tracker (FEAT-03) | LOW | 4h |
| 34 | Add referee impact overlay (FEAT-07) | LOW | 3h |

---

## Appendix: File-by-File Quality Assessment

| Module | Files | Quality | Key Issues |
|--------|-------|---------|------------|
| `src/analytics/` | 13 | B+ | Undefined functions in calibration, magic numbers, broad exceptions |
| `src/data/` | 11 | A- | Clean, well-structured, good retry/fallback patterns |
| `src/database/` | 3 | A | Excellent dual-DB architecture, proper locking |
| `src/ui/` | 17 | B | Memory leaks, no keyboard nav, magic numbers |
| `src/web/` | 7 | C+ | Missing CSRF, security headers, error exposure |
| `src/notifications/` | 3 | B+ | Clean, well-structured |
| `src/utils/` | 1 | B | Functional but thin |
| Entry points | 3 | B+ | Good bootstrap sequence |
| Config/Settings | 2 | A- | Comprehensive, thread-safe |
| **Overall** | **58** | **B** | Solid architecture, needs security hardening and tests |

---

## Appendix: Positive Highlights

Not everything is problems! Here's what's done well:

1. **Dual-DB Architecture** - In-memory SQLite for reads + on-disk for durability is clever and fast
2. **Reader-Writer Lock** - Proper concurrent access with writer preference
3. **Waterfall Scraping** - ESPN -> CBS -> RotoWire fallback chain for injury data
4. **Exponential Backoff** - Proper retry logic with `_safe_get()` pattern
5. **Walk-Forward Optimization** - Correct 80/20 train/test split prevents overfitting
6. **Multi-Criteria Save Gate** - 10+ checks before accepting new weights prevents regression
7. **Parameterized SQL Everywhere** - Zero SQL injection risk found
8. **Graceful Degradation** - `try/except` import guards for optional UI components
9. **Thread-Safe Config** - Locks around settings reads/writes
10. **Consistent Code Style** - PEP 8 compliant across 58 modules despite no linter
11. **Modular Architecture** - Clean separation: analytics, data, database, UI, web, notifications
12. **Three UI Targets** - Desktop, web, and CLI all share the same core engine
13. **Comprehensive Settings** - 95+ configurable parameters with sensible defaults
14. **Pipeline State Persistence** - Resumes after crashes, tracks per-step timing

---

*Generated by Claude Opus 4.6 after analyzing 58 Python modules, 6 HTML templates, 2678 lines of CSS, and 50+ configuration/data files across the NBA V2 codebase.*
