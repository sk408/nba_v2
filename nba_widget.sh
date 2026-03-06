#!/data/data/com.termux/files/usr/bin/bash
# Termux:Widget launcher for NBA Fundamentals V2
# Copy to ~/.shortcuts/: cp nba_widget.sh ~/.shortcuts/NBA_Fundamentals.sh

PROJECT_DIR="$HOME/nba_v2"
PYTHON="$PROJECT_DIR/venv/bin/python"

cd "$PROJECT_DIR" || { echo "Project not found at $PROJECT_DIR"; exit 1; }

# Kill any existing instance
pkill -f "python web.py" 2>/dev/null
sleep 0.5

# Start server in background — call venv python directly, no activate needed
nohup "$PYTHON" web.py > /dev/null 2>&1 &

# Wait for server to be ready
for i in 1 2 3 4 5; do
    curl -s http://localhost:5050 > /dev/null 2>&1 && break
    sleep 1
done

# Open in browser
am start -a android.intent.action.VIEW -d "http://localhost:5050" 2>/dev/null

echo "NBA Fundamentals running at http://localhost:5050"
