@echo off
setlocal
cd /d "%~dp0"

set "CONTAINER_NAME=lightpanda"

where docker >nul 2>nul
if errorlevel 1 (
    echo [lightpanda] Docker CLI not found. Nothing to stop.
    exit /b 1
)

docker inspect %CONTAINER_NAME% >nul 2>nul
if errorlevel 1 (
    echo [lightpanda] Container %CONTAINER_NAME% does not exist. Nothing to stop.
    exit /b 0
)

echo [lightpanda] Stopping and removing container...
docker rm -f %CONTAINER_NAME% >nul
if errorlevel 1 (
    echo [lightpanda] Failed to stop/remove %CONTAINER_NAME%.
    exit /b 1
)

echo [lightpanda] Container removed.
exit /b 0
