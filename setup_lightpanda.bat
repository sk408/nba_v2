@echo off
setlocal
cd /d "%~dp0"

set "CONTAINER_NAME=lightpanda"
set "IMAGE_NAME=lightpanda/browser:nightly"
set "CDP_PORT=9222"
set "CONTAINER_CMD=/bin/lightpanda serve --host 0.0.0.0 --port 9222 --log_level info --timeout 3600"

where docker >nul 2>nul
if errorlevel 1 (
    echo [lightpanda] Docker CLI not found. Install Docker Desktop first.
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo [lightpanda] Docker daemon is not running. Start Docker Desktop and retry.
    exit /b 1
)

echo [lightpanda] Pulling latest image...
docker pull %IMAGE_NAME%
if errorlevel 1 (
    echo [lightpanda] Failed to pull image.
    exit /b 1
)

docker inspect %CONTAINER_NAME% >nul 2>nul
if not errorlevel 1 (
    echo [lightpanda] Removing existing container...
    docker rm -f %CONTAINER_NAME% >nul 2>nul
    if errorlevel 1 (
        echo [lightpanda] Failed to remove existing container.
        exit /b 1
    )
)

echo [lightpanda] Starting container on port %CDP_PORT%...
docker run -d --name %CONTAINER_NAME% -p %CDP_PORT%:9222 %IMAGE_NAME% %CONTAINER_CMD% >nul
if errorlevel 1 (
    echo [lightpanda] Failed to start container.
    exit /b 1
)

call :verify_endpoint
if errorlevel 1 exit /b 1

echo [lightpanda] Setup complete.
exit /b 0

:verify_endpoint
echo [lightpanda] Verifying CDP endpoint...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%CDP_PORT%/json/version' -TimeoutSec 5; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
    echo [lightpanda] CDP endpoint check failed at http://127.0.0.1:%CDP_PORT%/json/version
    echo [lightpanda] Check container logs with: docker logs %CONTAINER_NAME%
    exit /b 1
)
echo [lightpanda] CDP endpoint is reachable.
exit /b 0
