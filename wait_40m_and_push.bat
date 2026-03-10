@echo off
setlocal
cd /d "%~dp0"

echo Waiting 40 minutes...
timeout /t 2400 /nobreak >nul

echo Running git add . && git commit -m "ODDS SYNC" && git push
git add .
git commit -m "ODDS SYNC"
git push

endlocal
