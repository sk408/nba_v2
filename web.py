"""NBA Fundamentals V2 -- Web Application."""
import os
import sys
sys.path.insert(0, ".")

from src.bootstrap import setup_logging, bootstrap, should_bootstrap_for_reloader

DEBUG = os.environ.get("NBA_WEB_DEBUG", "1") == "1"

setup_logging()

from src.web.app import app

if should_bootstrap_for_reloader(DEBUG):
    bootstrap()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=DEBUG)
