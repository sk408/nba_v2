@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo [lightpanda-node] Usage: discover_endpoints_lightpanda.bat "https://target.url" [output.json]
    exit /b 1
)

set "TARGET_URL=%~1"
set "OUTPUT_FILE=%~2"
set "NODE_TOOL_DIR=tools\lightpanda-node"
set "NODE_CONTAINER_NAME=lightpanda-node"
set "NODE_IMAGE=lightpanda/browser:nightly"
set "NODE_CDP_PORT=9223"
set "NODE_CDP_ENDPOINT=ws://127.0.0.1:%NODE_CDP_PORT%/"
set "OUTPUT_ARG="

where node >nul 2>nul
if errorlevel 1 (
    echo [lightpanda-node] Node.js not found. Install Node 18+ first.
    exit /b 1
)

where docker >nul 2>nul
if errorlevel 1 (
    echo [lightpanda-node] Docker CLI not found. Install Docker Desktop first.
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo [lightpanda-node] Docker daemon is not running. Start Docker Desktop and retry.
    exit /b 1
)

if not exist "%NODE_TOOL_DIR%\package.json" (
    echo [lightpanda-node] Missing %NODE_TOOL_DIR%\package.json
    exit /b 1
)

if not exist "%NODE_TOOL_DIR%\node_modules\puppeteer-core" (
    echo [lightpanda-node] Installing node dependencies...
    pushd "%NODE_TOOL_DIR%"
    npm install
    if errorlevel 1 (
        popd
        echo [lightpanda-node] npm install failed.
        exit /b 1
    )
    popd
)

docker inspect %NODE_CONTAINER_NAME% >nul 2>nul
if not errorlevel 1 (
    docker rm -f %NODE_CONTAINER_NAME% >nul 2>nul
)

echo [lightpanda-node] Starting dedicated Lightpanda container on port %NODE_CDP_PORT%...
docker run -d --name %NODE_CONTAINER_NAME% -p %NODE_CDP_PORT%:9222 %NODE_IMAGE% /bin/lightpanda serve --host 0.0.0.0 --port 9222 --log_level info --timeout 3600 >nul
if errorlevel 1 (
    echo [lightpanda-node] Failed to start %NODE_CONTAINER_NAME% container.
    exit /b 1
)

if not "%OUTPUT_FILE%"=="" (
    set "OUTPUT_ARG=%OUTPUT_FILE%"
    if not "%OUTPUT_FILE:~1,1%"==":" set "OUTPUT_ARG=%CD%\%OUTPUT_FILE%"
    for %%D in ("%OUTPUT_ARG%") do set "OUTPUT_DIR=%%~dpD"
    if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%" >nul 2>nul
)

pushd "%NODE_TOOL_DIR%"
if "%OUTPUT_FILE%"=="" (
    node discover-endpoints-puppeteer.mjs --url "%TARGET_URL%" --wsEndpoint "%NODE_CDP_ENDPOINT%"
) else (
    node discover-endpoints-puppeteer.mjs --url "%TARGET_URL%" --wsEndpoint "%NODE_CDP_ENDPOINT%" --output "%OUTPUT_ARG%"
)
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
