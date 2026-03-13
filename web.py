"""NBA Fundamentals V2 -- Web Application."""
import os
import sys
sys.path.insert(0, ".")

from src.bootstrap import setup_logging, bootstrap, should_bootstrap_for_reloader

DEBUG = os.environ.get("NBA_WEB_DEBUG", "1") == "1"
_UNDER_GUNICORN = any("gunicorn" in str(a) for a in sys.argv)

setup_logging()

from src.web.app import app

# Under gunicorn, bootstrap runs in post_fork (see gunicorn.conf.py)
# to avoid fork-after-thread deadlocks with background services.
if not _UNDER_GUNICORN and should_bootstrap_for_reloader(DEBUG):
    bootstrap()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=DEBUG)
