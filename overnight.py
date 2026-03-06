"""Minimal CLI to run the overnight optimization loop without the desktop UI."""

import sys
import argparse

sys.path.insert(0, ".")

from src.bootstrap import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run overnight optimization")
    parser.add_argument("--hours", type=float, default=8.0,
                        help="Max hours to run (default: 8)")
    parser.add_argument("--reset-weights", action="store_true",
                        help="Reset all weights to defaults before starting")
    args = parser.parse_args()

    setup_logging()

    # Initialize database (runs migrations if needed)
    from src.database.migrations import init_db
    init_db()

    # Run overnight in a thread-local DB context
    from src.database.db import thread_local_db
    from src.analytics.pipeline import run_overnight

    with thread_local_db():
        run_overnight(
            max_hours=args.hours,
            reset_weights=args.reset_weights,
            callback=lambda msg: print(msg, flush=True),
        )


if __name__ == "__main__":
    main()
