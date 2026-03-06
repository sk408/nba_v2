#!/data/data/com.termux/files/usr/bin/bash
# Setup script for NBA Fundamentals V2 web app on Termux

set -e

echo "=== NBA Fundamentals V2 — Termux Setup ==="

# Install system dependencies
echo "[1/4] Installing system packages..."
pkg update -y
pkg install -y python git

# Create virtual environment
echo "[2/4] Creating virtual environment..."
python -m venv venv

# Activate and install pip packages
echo "[3/4] Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install flask requests numpy pandas beautifulsoup4

echo "[4/4] Done!"
echo ""
echo "To run the web app:"
echo "  bash run_termux.sh"
