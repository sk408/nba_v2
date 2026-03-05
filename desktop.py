"""Minimal entry point for testing database and sync."""
import sys
sys.path.insert(0, ".")

from src.bootstrap import setup_logging, bootstrap, shutdown
setup_logging()
bootstrap()
print("Bootstrap complete — database initialized")

from src.data.sync_service import full_sync
result = full_sync(force=True, callback=lambda msg: print(f"  {msg}"))
print(f"Sync complete: {result}")

shutdown()
