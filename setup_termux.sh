#!/data/data/com.termux/files/usr/bin/bash
# Setup script for NBA Fundamentals V2 web app on Termux
# IMPORTANT: Use Termux from F-Droid, NOT Play Store

set -e

echo "=== NBA Fundamentals V2 — Termux Setup ==="

# Install system dependencies (python-numpy is pre-built; pandas installs via pip)
echo "[1/5] Installing system packages..."
pkg update -y
pkg install -y python git python-numpy

# Create virtual environment with access to system packages (numpy)
echo "[2/5] Creating virtual environment..."
python -m venv --system-site-packages venv

# Install pip packages (pandas compiles fine once numpy is pre-installed)
echo "[3/5] Upgrading pip..."
venv/bin/pip install --upgrade pip

echo "[4/5] Installing Python dependencies..."
venv/bin/pip install flask requests beautifulsoup4 pandas

echo "[5/5] Done!"
echo ""
echo "To run the web app:"
echo "  bash run_termux.sh"
