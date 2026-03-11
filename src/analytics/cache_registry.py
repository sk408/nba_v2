"""Central cache invalidation registry + lightweight cache diagnostics."""

import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_last_event_runs: Dict[str, Dict[str, Any]] = {}


def _invalidator_specs() -> Dict[str, List[Tuple[str, Callable[[], None]]]]:
    # Imports are local to avoid import cycles at module import time.
    from src.analytics.backtester import invalidate_backtest_cache
    from src.analytics.prediction import invalidate_precompute_cache, invalidate_results_cache
    from src.analytics.stats_engine import invalidate_stats_caches
    from src.analytics.weight_config import invalidate_weight_cache
    from src.data.gamecast import invalidate_actionnetwork_cache

    return {
        "post_sync": [
            ("results", invalidate_results_cache),
            ("backtest", invalidate_backtest_cache),
            ("stats", invalidate_stats_caches),
            ("actionnetwork", invalidate_actionnetwork_cache),
        ],
        "post_odds_sync": [
            ("results", invalidate_results_cache),
            ("backtest", invalidate_backtest_cache),
            ("actionnetwork", invalidate_actionnetwork_cache),
        ],
        "post_nuke": [
            ("precompute", invalidate_precompute_cache),
            ("results", invalidate_results_cache),
            ("backtest", invalidate_backtest_cache),
            ("stats", invalidate_stats_caches),
            ("weights", invalidate_weight_cache),
            ("actionnetwork", invalidate_actionnetwork_cache),
        ],
        "post_weight_save": [
            ("backtest", invalidate_backtest_cache),
        ],
    }


def invalidate_for_event(event: str, callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Run invalidators registered for an event and return per-cache outcomes."""
    specs = _invalidator_specs()
    entries = specs.get(event, [])
    if not entries:
        return {"event": event, "ran": 0, "results": []}

    results: List[Dict[str, Any]] = []
    for cache_name, invalidator in entries:
        started = time.time()
        ok = True
        error = ""
        try:
            invalidator()
        except Exception as e:
            ok = False
            error = str(e)
            logger.debug("Cache invalidator failed (%s/%s): %s", event, cache_name, e)
        elapsed_ms = int((time.time() - started) * 1000)
        result = {
            "cache": cache_name,
            "ok": ok,
            "error": error,
            "elapsed_ms": elapsed_ms,
        }
        results.append(result)
        if callback:
            verdict = "ok" if ok else f"failed ({error})"
            callback(f"  cache[{cache_name}] {verdict}")

    payload = {
        "event": event,
        "ran": len(results),
        "results": results,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    _last_event_runs[event] = payload
    return payload


def get_cache_registry_state() -> Dict[str, Any]:
    """Return last invalidation timestamps and key cache-file metadata."""
    from src.analytics.backtester import _BACKTEST_CACHE_DIR
    from src.analytics.prediction import _CONTEXT_CACHE_FILE, _PRECOMPUTE_CACHE_FILE

    files = {}
    for key, path in {
        "precompute_games": _PRECOMPUTE_CACHE_FILE,
        "precompute_context": _CONTEXT_CACHE_FILE,
    }.items():
        exists = os.path.exists(path)
        files[key] = {
            "exists": exists,
            "size_bytes": os.path.getsize(path) if exists else 0,
            "mtime": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
            if exists
            else "",
        }

    bt_exists = os.path.isdir(_BACKTEST_CACHE_DIR)
    bt_count = 0
    if bt_exists:
        try:
            bt_count = len([n for n in os.listdir(_BACKTEST_CACHE_DIR) if n.endswith(".pkl")])
        except Exception:
            bt_count = 0
    files["backtest_dir"] = {
        "exists": bt_exists,
        "pickle_files": bt_count,
    }

    return {
        "events_supported": sorted(_invalidator_specs().keys()),
        "last_event_runs": _last_event_runs,
        "files": files,
    }
