"""NBA Fundamentals V2 -- Web Application."""
import sys
sys.path.insert(0, ".")

from src.bootstrap import setup_logging, bootstrap
setup_logging()
bootstrap()

from src.web.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
