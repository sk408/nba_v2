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

docker inspect %CONTAINER_NAME% >nul 2>nul
if errorlevel 1 goto create_container

for /f "delims=" %%i in ('docker inspect -f "{{json .Config.Cmd}}" %CONTAINER_NAME%') do set "CURRENT_CMD=%%i"
echo %CURRENT_CMD% | findstr /c:"--timeout" >nul
if errorlevel 1 (
    echo [lightpanda] Existing container uses old command. Recreating with stable CDP timeout...
    docker rm -f %CONTAINER_NAME% >nul 2>nul
    if errorlevel 1 (
        echo [lightpanda] Failed to remove old container.
        exit /b 1
    )
    goto create_container
)

for /f %%i in ('docker inspect -f "{{.State.Running}}" %CONTAINER_NAME%') do set "IS_RUNNING=%%i"
if /i "%IS_RUNNING%"=="true" (
    echo [lightpanda] Container is already running.
) else (
    echo [lightpanda] Starting existing container...
    docker start %CONTAINER_NAME% >nul
    if errorlevel 1 (
        echo [lightpanda] Failed to start existing container.
        exit /b 1
    )
)
goto verify_endpoint

:create_container
echo [lightpanda] Container not found. Creating from %IMAGE_NAME%...
docker run -d --name %CONTAINER_NAME% -p %CDP_PORT%:9222 %IMAGE_NAME% %CONTAINER_CMD% >nul
if errorlevel 1 (
    echo [lightpanda] Failed to create container.
    echo [lightpanda] Run setup_lightpanda.bat to pull/update the image first.
    exit /b 1
)

:verify_endpoint
echo [lightpanda] Verifying CDP endpoint...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%CDP_PORT%/json/version' -TimeoutSec 5; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
    echo [lightpanda] CDP endpoint check failed at http://127.0.0.1:%CDP_PORT%/json/version
    echo [lightpanda] Check container logs with: docker logs %CONTAINER_NAME%
    exit /b 1
)

echo [lightpanda] Ready.
exit /b 0
