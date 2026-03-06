#!/data/data/com.termux/files/usr/bin/bash
# Run the NBA Fundamentals V2 web app on Termux

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run setup_termux.sh first."
    exit 1
fi
source venv/bin/activate

echo "Starting NBA Fundamentals V2 web app..."
echo "Access at http://localhost:5050"
echo "Press Ctrl+C to stop."
echo ""

python web.py
