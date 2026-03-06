#!/data/data/com.termux/files/usr/bin/bash
# Setup script for NBA Fundamentals V2 web app on Termux
# Works with Play Store or F-Droid Termux

set -e

echo "=== NBA Fundamentals V2 — Termux Setup ==="

# ── 1. System packages ──────────────────────────────────────────
echo "[1/4] Installing system packages..."
pkg update -y
pkg install -y python git build-essential cmake ninja libopenblas \
    libandroid-execinfo patchelf binutils-is-llvm clang

# ── 2. Python build helpers ─────────────────────────────────────
echo "[2/4] Installing Python build tools..."
# NOTE: Do NOT upgrade pip — Termux ships a patched version
pip install setuptools wheel packaging pyproject_metadata cython \
    meson-python versioneer setuptools-scm

# ── 3. Compile numpy & pandas ──────────────────────────────────
echo "[3/4] Installing numpy + pandas (compiling — this takes a few minutes)..."
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MATHLIB=m LDFLAGS="-lpython${PYVER}" pip install --no-build-isolation --no-cache-dir numpy==1.26.4
LDFLAGS="-lpython${PYVER}" pip install --no-build-isolation --no-cache-dir pandas

# ── 4. App dependencies ────────────────────────────────────────
echo "[4/4] Installing app dependencies..."
pip install flask requests beautifulsoup4

echo ""
echo "=== Setup complete! ==="
echo "To run:  bash run_termux.sh"
echo "Widget:  cp nba_widget.sh ~/.shortcuts/NBA_Fundamentals.sh"
