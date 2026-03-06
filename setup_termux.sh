#!/data/data/com.termux/files/usr/bin/bash
# Setup script for NBA Fundamentals V2 web app on Termux
# IMPORTANT: Use Termux from F-Droid, NOT Play Store

set -e

echo "=== NBA Fundamentals V2 — Termux Setup ==="

# Install system dependencies (includes pre-built numpy/pandas)
echo "[1/4] Installing system packages..."
pkg update -y
pkg install -y python git python-numpy python-pandas

# Create virtual environment with access to system packages
echo "[2/4] Creating virtual environment..."
python -m venv --system-site-packages venv

# Install remaining pip packages
echo "[3/4] Installing Python dependencies..."
venv/bin/pip install --upgrade pip
venv/bin/pip install flask requests beautifulsoup4

echo "[4/4] Done!"
echo ""
echo "To run the web app:"
echo "  bash run_termux.sh"
