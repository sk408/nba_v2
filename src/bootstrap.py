"""Shared bootstrap logic for web and desktop entry points."""

import atexit
import logging
import os
import sys
import threading
import traceback

_monitor = None
_logger = logging.getLogger(__name__)


def setup_logging():
    """Configure root logging from user config settings."""
    import src.config

    level_name = src.config.get("log_level", "INFO")
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _thread_excepthook(args):
    """Log unhandled exceptions from background threads."""
    if args.exc_type is SystemExit:
        return
    _logger.error(
        "Unhandled exception in thread %s:\n%s",
        args.thread.name if args.thread else "<unknown>",
        "".join(
            traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            )
        ),
    )


def should_bootstrap_for_reloader(debug: bool) -> bool:
    """Return True only in the process that should run background services."""
    if not debug:
        return True
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def bootstrap(status_callback=None):
    """Run shared initialisation sequence.

    Parameters
    ----------
    status_callback : callable, optional
        Called with a status string at each init stage (useful for splash
        screens).

    Returns
    -------
    InjuryMonitor or None
    """
    global _monitor

    threading.excepthook = _thread_excepthook

    def _status(msg, progress=None):
        _logger.info(msg)
        if status_callback:
            status_callback(msg, progress)

    # 1. Header patching
    _status("Patching NBA API headers...", 0.1)
    import src  # noqa: F401

    # 2. Database
    _status("Initializing database...", 0.3)
    from src.database.migrations import init_db

    init_db()

    # 3. Clean old notifications
    _status("Cleaning old notifications...", 0.5)
    try:
        from src.notifications.service import delete_old
        delete_old(days=30)
    except Exception as e:
        _logger.warning("Failed to clean old notifications: %s", e)

    # 4. Injury monitor (use module singleton so all paths share one instance)
    _status("Starting injury monitor...", 0.7)
    try:
        from src.notifications.injury_monitor import get_injury_monitor

        _monitor = get_injury_monitor()
        _monitor.start()
        _logger.info("Injury monitor started (5-min polling)")
    except Exception as e:
        _logger.warning("Injury monitor failed to start: %s", e)

    atexit.register(shutdown)
    return _monitor


def shutdown():
    """Stop background services gracefully."""
    global _monitor
    if _monitor is not None:
        _logger.info("Stopping injury monitor...")
        _monitor.stop()
        _monitor = None
