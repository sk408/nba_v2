@echo off
REM Sync odds locally (ActionNetwork works from home IP) then push DB to VPS
cd /d "%~dp0"

echo [sync_odds] Running local odds sync...
python -c "from src.bootstrap import bootstrap; bootstrap(); from src.data.odds_sync import sync_odds_for_date; from src.utils.timezone_utils import nba_today; d = nba_today(); print(f'Syncing odds for {d}...'); n = sync_odds_for_date(d, callback=print); print(f'Done: {n} games saved.')"
if errorlevel 1 (
    echo [sync_odds] Odds sync failed.
    pause
    exit /b 1
)

echo.
echo [sync_odds] Uploading database to VPS...
python deploy\sync_db.py
if errorlevel 1 (
    echo [sync_odds] DB upload failed.
    pause
    exit /b 1
)

echo.
echo [sync_odds] All done!
pause
