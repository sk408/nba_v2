@echo off
setlocal
cd /d "%~dp0"

where wsl >nul 2>nul
if errorlevel 1 (
    echo [lightpanda-wsl] WSL is not installed.
    echo [lightpanda-wsl] Run: wsl --install
    exit /b 1
)

for /f "delims=" %%i in ('wsl -l -q 2^>nul') do set "HAS_DISTRO=1"
if not defined HAS_DISTRO (
    echo [lightpanda-wsl] No WSL distro found.
    echo [lightpanda-wsl] Run as Administrator, then reboot when prompted:
    echo [lightpanda-wsl]   wsl --install
    echo [lightpanda-wsl]   wsl --install -d Ubuntu
    exit /b 1
)

echo [lightpanda-wsl] Installing/updating Lightpanda inside WSL...
wsl bash -lc "set -euo pipefail; mkdir -p ~/.local/bin; cd ~/.local/bin; ARCH=\$(uname -m); case \"\$ARCH\" in x86_64) URL='https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux' ;; aarch64|arm64) URL='https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-aarch64-linux' ;; *) echo Unsupported architecture: \$ARCH; exit 1 ;; esac; curl -L -o lightpanda \"\$URL\"; chmod +x lightpanda; ./lightpanda --help >/dev/null"
if errorlevel 1 (
    echo [lightpanda-wsl] Failed to install Lightpanda in WSL.
    exit /b 1
)

echo [lightpanda-wsl] Installed successfully in WSL at ~/.local/bin/lightpanda
echo [lightpanda-wsl] To start CDP server from Windows:
echo [lightpanda-wsl]   wsl bash -lc "~/.local/bin/lightpanda serve --host 0.0.0.0 --port 9222"
exit /b 0
