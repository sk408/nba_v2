"""10-step pipeline orchestrator.

Steps: backup -> sync -> seed_arenas -> bbref_sync -> referee_sync -> elo_compute
       -> precompute -> optimize_fundamentals -> optimize_sharp -> backtest

No disabled steps, no dead code. Each step receives (callback, is_cancelled) and
returns a result dict. Pipeline state is persisted to data/pipeline_state.json
with timing, result, and status for each step.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.database import db

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────────────────────

PIPELINE_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "pipeline_state.json"
)


def _fmt_step_seconds(seconds: float) -> str:
    """Format step duration with enough precision for fast steps."""
    if seconds < 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds:.1f}s"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types gracefully."""

    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_pipeline_state() -> Dict:
    if os.path.exists(PIPELINE_STATE_PATH):
        try:
            with open(PIPELINE_STATE_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load pipeline state: %s", e)
    return {}


def _save_pipeline_state(state: Dict):
    os.makedirs(os.path.dirname(PIPELINE_STATE_PATH) or ".", exist_ok=True)
    with open(PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, cls=NumpyEncoder)


_SAVE_GATE_DETAIL_EXPORT_KEYS = (
    "weight_delta",
    "min_weight_delta",
    "weight_change_ok",
    "loss_improved",
    "core_guards_ok",
    "winner_guard",
    "favorites_guard",
    "compression_ok",
    "edge_ok",
    "shrunk_upset_lift",
    "roi_lift",
    "candidate_ml_roi_lb95",
    "use_roi_gate",
    "use_hybrid_loss_gate",
    "use_long_dog_tiebreak_gate",
)


def _to_json_scalar(value: Any) -> Any:
    """Convert numpy/object scalars to JSON-safe Python scalars."""
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        logger.debug("Failed scalar conversion for save-gate export", exc_info=True)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _short_reason(reason: Any, max_len: int = 220) -> str:
    """Trim long save-gate reasons to keep state files readable."""
    text = str(reason or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)].rstrip() + "..."


def _extract_save_gate_state(step_result: Any) -> Optional[Dict[str, Any]]:
    """Return compact save-gate metadata from optimizer-style step results."""
    if not isinstance(step_result, dict):
        return None
    reason = step_result.get("save_gate_reason")
    details = step_result.get("save_gate_details")
    if reason is None and not isinstance(details, dict):
        return None

    snapshot: Dict[str, Any] = {}
    if reason is not None:
        snapshot["reason"] = _short_reason(reason)
    if "improved" in step_result:
        snapshot["saved"] = bool(step_result.get("improved"))

    if isinstance(details, dict):
        for key in _SAVE_GATE_DETAIL_EXPORT_KEYS:
            if key not in details:
                continue
            val = _to_json_scalar(details.get(key))
            if val is not None:
                snapshot[key] = val

    return snapshot or None


# ──────────────────────────────────────────────────────────────
# Cancellation
# ──────────────────────────────────────────────────────────────

_cancel_event = threading.Event()


def request_cancel():
    """Request pipeline cancellation (thread-safe)."""
    _cancel_event.set()


def clear_cancel():
    """Clear cancellation flag."""
    _cancel_event.clear()


def is_cancelled() -> bool:
    """Check if cancellation has been requested."""
    return _cancel_event.is_set()


# ──────────────────────────────────────────────────────────────
# Freshness tracking via sync_meta table
# ──────────────────────────────────────────────────────────────

def _is_fresh(step_name: str, max_age_hours: float = 24) -> bool:
    """Check if a step is fresh enough to skip."""
    row = db.fetch_one(
        "SELECT last_synced_at FROM sync_meta WHERE step_name = ?",
        (step_name,),
    )
    if not row or not row["last_synced_at"]:
        return False

    from datetime import datetime

    try:
        last = datetime.fromisoformat(row["last_synced_at"])
        return (datetime.now() - last).total_seconds() < max_age_hours * 3600
    except Exception as e:
        logger.warning("Freshness check parse error for %s: %s", step_name, e)
        return False


def _mark_step_done(step_name: str):
    """Mark a step as completed in sync_meta."""
    row = db.fetch_one(
        "SELECT COUNT(*) as cnt, MAX(game_date) as last_date FROM player_stats"
    )
    count = row["cnt"] if row else 0
    last_date = row["last_date"] or "" if row else ""

    db.execute(
        """
        INSERT INTO sync_meta (step_name, last_synced_at, game_count_at_sync, last_game_date_at_sync)
        VALUES (?, datetime('now'), ?, ?)
        ON CONFLICT(step_name) DO UPDATE SET
            last_synced_at = excluded.last_synced_at,
            game_count_at_sync = excluded.game_count_at_sync,
            last_game_date_at_sync = excluded.last_game_date_at_sync
        """,
        (step_name, count, last_date),
    )


# ──────────────────────────────────────────────────────────────
# Pipeline step functions
# ──────────────────────────────────────────────────────────────

def backup_snapshot(callback=None, is_cancelled=None) -> Dict:
    """Step 1: Snapshot current weights before pipeline modifies them."""
    from src.analytics.weight_config import save_snapshot

    path = save_snapshot("pre_pipeline", notes="Automatic pre-pipeline backup")
    if callback:
        callback(f"Backup saved: {os.path.basename(path)}")
    return {"snapshot_path": path}


def run_data_sync(callback=None, is_cancelled=None) -> Dict:
    """Step 2: Full data sync (teams, players, stats, odds, injuries)."""
    from src.data.sync_service import full_sync

    failures = full_sync(force=False, callback=callback)
    return {"sync_result": "complete" if not failures else "partial", "sync_failures": failures}


def run_seed_arenas(callback=None, is_cancelled=None) -> Dict:
    """Step 2b: Seed arena data (one-time, idempotent)."""
    if callback:
        callback("Seeding arena data...")
    from src.data.arenas import seed_arenas_table
    seed_arenas_table()
    return {"status": "seeded"}


def run_bbref_sync(callback=None, is_cancelled=None) -> Dict:
    """Step 2c: Sync Basketball-Reference advanced stats (daily)."""
    from src.config import get_season
    season = get_season()
    try:
        if callback:
            callback("Syncing Basketball-Reference stats...")
        from src.data.bbref_scraper import sync_bbref_stats
        sync_bbref_stats(season)
        return {"status": "synced"}
    except Exception as e:
        logger.warning("BBRef sync failed (non-fatal): %s", e)
        return {"status": "failed", "error": str(e)}


def run_referee_sync(callback=None, is_cancelled=None) -> Dict:
    """Step 2d: Sync referee tendencies (daily)."""
    from src.config import get_season
    season = get_season()
    try:
        if callback:
            callback("Syncing referee tendencies...")
        from src.data.referee_scraper import sync_referee_stats
        sync_referee_stats(season)
        return {"status": "synced"}
    except Exception as e:
        logger.warning("Referee sync failed (non-fatal): %s", e)
        return {"status": "failed", "error": str(e)}


def run_elo_compute(callback=None, is_cancelled=None) -> Dict:
    """Step 2e: Compute Elo ratings (must run before precompute)."""
    from src.config import get_season
    season = get_season()
    if callback:
        callback("Computing Elo ratings...")
    from src.analytics.elo import compute_all_elo
    compute_all_elo(season)
    return {"status": "computed"}


def run_precompute(callback=None, is_cancelled=None) -> Dict:
    """Step 3: Build GameInput objects for all historical games (disk+memory cached)."""
    from src.analytics.prediction import precompute_all_games

    games = precompute_all_games(callback=callback)
    if callback:
        callback(f"Precomputed {len(games)} games")
    return {"game_count": len(games)}


def run_optimize_fundamentals(callback=None, is_cancelled=None) -> Dict:
    """Step 4: Optimize fundamental-only weights (3000 Optuna trials)."""
    from src.analytics.prediction import precompute_all_games
    from src.analytics.optimizer import optimize_weights

    games = precompute_all_games()  # uses cache from step 3
    result = optimize_weights(
        games,
        n_trials=3000,
        include_sharp=False,
        callback=callback,
        is_cancelled=is_cancelled,
    )
    _mark_step_done("optimize_fundamentals")
    return result


def run_optimize_sharp(callback=None, is_cancelled=None) -> Dict:
    """Step 5: Optimize fundamentals + sharp money weights (3000 Optuna trials)."""
    from src.analytics.prediction import precompute_all_games
    from src.analytics.optimizer import optimize_weights

    games = precompute_all_games()  # uses cache
    result = optimize_weights(
        games,
        n_trials=3000,
        include_sharp=True,
        callback=callback,
        is_cancelled=is_cancelled,
    )
    _mark_step_done("optimize_sharp")
    return result


def run_backtest_and_compare(callback=None, is_cancelled=None) -> Dict:
    """Step 7: Run A/B backtest (fundamentals vs sharp) with fresh data."""
    from src.analytics.backtester import run_backtest, invalidate_backtest_cache

    invalidate_backtest_cache()  # force recompute after optimization
    result = run_backtest(callback=callback)
    _mark_step_done("backtest")
    return result


# ──────────────────────────────────────────────────────────────
# Pipeline definition
# ──────────────────────────────────────────────────────────────

StepFunc = Callable[[Optional[Callable], Optional[Callable]], Dict]

PIPELINE_STEPS: List[Tuple[str, StepFunc]] = [
    ("backup", backup_snapshot),
    ("sync", run_data_sync),
    # ── V2.1 sync steps ──
    ("seed_arenas", run_seed_arenas),
    ("bbref_sync", run_bbref_sync),
    ("referee_sync", run_referee_sync),
    ("elo_compute", run_elo_compute),
    # ──────────────────────
    ("precompute", run_precompute),
    ("optimize_fundamentals", run_optimize_fundamentals),
    ("optimize_sharp", run_optimize_sharp),
    ("backtest", run_backtest_and_compare),
]


# ──────────────────────────────────────────────────────────────
# Main pipeline runner
# ──────────────────────────────────────────────────────────────

def run_pipeline(
    callback: Optional[Callable] = None,
    is_cancelled_fn: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run full pipeline: backup -> sync -> precompute -> optimize x2 -> backtest.

    Args:
        callback: Optional function receiving progress messages (str).
        is_cancelled_fn: Optional function returning True if pipeline should stop.
                         Defaults to the module-level is_cancelled() if not provided.

    Returns:
        Dict with per-step results, timing, and overall summary.
    """
    clear_cancel()
    start_time = time.time()
    state = _load_pipeline_state()
    results: Dict[str, Any] = {}
    step_timings: Dict[str, float] = {}

    cancel_check = is_cancelled_fn or is_cancelled

    def emit(msg: str):
        if callback:
            callback(msg)
        logger.info(msg)

    total_steps = len(PIPELINE_STEPS)
    valid_step_keys = {f"step_{name}" for name, _ in PIPELINE_STEPS}
    for key in list(state.keys()):
        if key.startswith("step_") and key not in valid_step_keys:
            state.pop(key, None)

    try:
        for idx, (step_name, step_func) in enumerate(PIPELINE_STEPS, 1):
            # Check cancellation before each step
            if cancel_check():
                emit("Pipeline cancelled.")
                results["cancelled"] = True
                break

            emit(f"[Step {idx}/{total_steps}] {step_name}...")
            step_start = time.time()

            try:
                step_result = step_func(
                    callback=lambda msg, _sn=step_name: emit(f"  [{_sn}] {msg}"),
                    is_cancelled=cancel_check,
                )
                step_elapsed = time.time() - step_start
                step_timings[step_name] = round(step_elapsed, 3)

                # Store result (strip bulky per_game lists from state file)
                results[step_name] = step_result
                step_state: Dict[str, Any] = {
                    "status": "completed",
                    "elapsed_seconds": round(step_elapsed, 3),
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_gate_state = _extract_save_gate_state(step_result)
                if save_gate_state is not None:
                    step_state["save_gate"] = save_gate_state
                    reason = str(save_gate_state.get("reason", "")).strip()
                    saved = bool(save_gate_state.get("saved", False))
                    if saved:
                        emit(f"  [{step_name}] save gate: pass")
                    elif reason:
                        emit(f"  [{step_name}] save gate: {_short_reason(reason, 180)}")
                state[f"step_{step_name}"] = step_state

                emit(f"  {step_name} completed in {_fmt_step_seconds(step_elapsed)}")

            except Exception as e:
                step_elapsed = time.time() - step_start
                logger.exception("Pipeline step '%s' failed: %s", step_name, e)
                emit(f"  {step_name} FAILED: {e}")

                results[step_name] = {"error": str(e)}
                state[f"step_{step_name}"] = {
                    "status": "error",
                    "error": str(e),
                    "elapsed_seconds": round(step_elapsed, 3),
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                # Continue to next step even on failure

            # Save state after each step
            _save_pipeline_state(state)

        # Overall summary
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = round(elapsed, 1)
        results["step_timings"] = step_timings

        state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["elapsed_seconds"] = round(elapsed, 1)

        # Save a clean summary (exclude bulky data)
        summary = {}
        for k, v in results.items():
            if k in ("sync",):
                continue  # sync result can be huge
            if isinstance(v, dict) and "per_game" in v:
                # Strip per-game lists from backtest results
                summary[k] = {sk: sv for sk, sv in v.items() if sk != "per_game"}
            elif isinstance(v, dict):
                # Strip per_game from nested dicts (backtest has fundamentals/sharp)
                cleaned = {}
                for sk, sv in v.items():
                    if isinstance(sv, dict) and "per_game" in sv:
                        cleaned[sk] = {
                            nk: nv for nk, nv in sv.items() if nk != "per_game"
                        }
                    else:
                        cleaned[sk] = sv
                summary[k] = cleaned
            else:
                summary[k] = v
        state["results_summary"] = summary
        _save_pipeline_state(state)

        emit(f"\nPipeline complete in {elapsed:.0f}s")
        return results

    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        emit(f"Pipeline error: {e}")
        results["error"] = str(e)
        return results


# ──────────────────────────────────────────────────────────────
# Overnight loop
# ──────────────────────────────────────────────────────────────

def run_overnight(
    max_hours: float = 8.0,
    reset_weights: bool = False,
    callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run full pipeline once, then loop optimization steps until time runs out.

    Pass 1: full pipeline (backup, sync, precompute, optimize x2, backtest)
    Pass 2+: optimize_fundamentals -> optimize_sharp -> backtest
    Each pass uses fresh random seeds. Precomputed games are reused across passes.
    """
    clear_cancel()
    overall_start = time.time()
    deadline = overall_start + max_hours * 3600

    def emit(msg: str):
        if callback:
            callback(msg)
        logger.info(msg)

    def time_left():
        return max(0, deadline - time.time())

    def fmt_elapsed(secs):
        h, m = int(secs // 3600), int((secs % 3600) // 60)
        return f"{h}h {m}m" if h else f"{m}m {int(secs % 60)}s"

    from src.config import get as get_setting
    max_no_save_passes = max(0, int(get_setting("overnight_max_no_save_passes", 0)))
    consecutive_no_save_passes = 0

    emit(f"=== Overnight Optimization: {max_hours}h budget ===")
    if max_no_save_passes > 0:
        emit(f"No-save auto-stop enabled: {max_no_save_passes} consecutive passes.")

    if reset_weights:
        from src.analytics.weight_config import clear_all_weights

        clear_all_weights()
        emit("Weights reset to defaults.")

    # Pass 1: Full pipeline
    emit("\n--- Pass 1: Full Pipeline ---")
    pass1_start = time.time()
    results = run_pipeline(callback=callback)

    if results.get("error") or results.get("cancelled") or is_cancelled():
        return results

    pass1_elapsed = time.time() - pass1_start

    # Track best backtest across all passes
    best_bt = results.get("backtest", {})
    pass_num = 1
    attempted_passes = 1
    completed_passes = 1
    save_gate_passes: List[Dict[str, Any]] = []
    pass1_summary = {
        "pass": 1,
        "mode": "full_pipeline",
        "fundamentals": _extract_save_gate_state(results.get("optimize_fundamentals")),
        "sharp": _extract_save_gate_state(results.get("optimize_sharp")),
    }
    pass1_summary["saved_any"] = bool(
        (pass1_summary["fundamentals"] or {}).get("saved", False)
        or (pass1_summary["sharp"] or {}).get("saved", False)
    )
    save_gate_passes.append(pass1_summary)
    for model_name, gate in (
        ("fundamentals", pass1_summary.get("fundamentals")),
        ("sharp", pass1_summary.get("sharp")),
    ):
        if not isinstance(gate, dict):
            continue
        if bool(gate.get("saved", False)):
            emit(f"  Pass 1 {model_name} save gate: pass")
            continue
        reason = str(gate.get("reason", "")).strip()
        if reason:
            emit(f"  Pass 1 {model_name} save gate: {_short_reason(reason, 180)}")

    # Emit Pass 1 backtest results so the TUI can display them
    if best_bt:
        f_met = best_bt.get("fundamentals", {})
        emit(f"  NEW BEST! Winner={f_met.get('winner_pct', 0):.1f}%, "
             f"Upset={f_met.get('upset_accuracy', 0):.0f}% @ "
             f"{f_met.get('upset_rate', 0):.0f}% rate, "
             f"CompDog={f_met.get('competitive_dog_rate', 0):.0f}% "
             f"({f_met.get('competitive_dog_count', 0)}/{f_met.get('upset_count', 0)}), "
             f"OnePosDog={f_met.get('one_possession_dog_rate', 0):.0f}% "
             f"({f_met.get('one_possession_dog_count', 0)}/{f_met.get('upset_count', 0)}), "
             f"LongDog1P={f_met.get('long_dog_onepos_rate', 0):.0f}% "
             f"({f_met.get('long_dog_onepos_count', 0)}/{f_met.get('long_dog_count', 0)}), "
             f"ML ROI={f_met.get('ml_roi', 0):+.1f}%")

    emit(f"Pass 1 complete in {fmt_elapsed(pass1_elapsed)} | "
         f"{fmt_elapsed(time_left())} remaining")
    loop_times: List[float] = []

    # Pass 2+: optimization loops
    while time_left() > 0 and not is_cancelled():
        # Estimate if we have time for another loop
        avg_loop = (
            sum(loop_times) / len(loop_times) if loop_times else pass1_elapsed * 0.6
        )
        if time_left() < avg_loop * 0.5:
            emit(f"\n~{fmt_elapsed(time_left())} remaining, not enough for another pass.")
            break

        pass_num = attempted_passes + 1
        attempted_passes = pass_num
        loop_start = time.time()

        emit(f"\n--- Pass {pass_num}: Optimization Loop "
             f"({fmt_elapsed(time_left())} remaining) ---")

        try:
            pass_saved_any = False

            # Optimize fundamentals
            emit(f"[Loop {pass_num}] Optimizing fundamentals (3000 trials)...")
            fund_result = run_optimize_fundamentals(
                callback=lambda msg: emit(f"  {msg}"),
                is_cancelled=is_cancelled,
            )
            if is_cancelled():
                break
            fund_saved = bool(fund_result.get("improved"))
            pass_saved_any = pass_saved_any or fund_saved
            improved = "IMPROVED" if fund_saved else "no change"
            emit(f"  Fundamentals: {improved}")
            fund_gate = _extract_save_gate_state(fund_result)
            if isinstance(fund_gate, dict):
                if fund_saved:
                    emit("  Fundamentals save gate: pass")
                else:
                    reason = str(fund_gate.get("reason", "")).strip()
                    if reason:
                        emit(f"  Fundamentals save gate: {_short_reason(reason, 180)}")

            # Optimize sharp
            emit(f"[Loop {pass_num}] Optimizing sharp (3000 trials)...")
            sharp_result = run_optimize_sharp(
                callback=lambda msg: emit(f"  {msg}"),
                is_cancelled=is_cancelled,
            )
            if is_cancelled():
                break
            sharp_saved = bool(sharp_result.get("improved"))
            pass_saved_any = pass_saved_any or sharp_saved
            improved = "IMPROVED" if sharp_saved else "no change"
            emit(f"  Sharp: {improved}")
            sharp_gate = _extract_save_gate_state(sharp_result)
            if isinstance(sharp_gate, dict):
                if sharp_saved:
                    emit("  Sharp save gate: pass")
                else:
                    reason = str(sharp_gate.get("reason", "")).strip()
                    if reason:
                        emit(f"  Sharp save gate: {_short_reason(reason, 180)}")

            save_gate_passes.append(
                {
                    "pass": pass_num,
                    "mode": "loop",
                    "saved_any": pass_saved_any,
                    "fundamentals": fund_gate,
                    "sharp": sharp_gate,
                }
            )

            if time_left() <= 0 and not is_cancelled():
                emit(
                    f"  Time budget reached before backtest in pass {pass_num}; "
                    "ending run without counting this pass."
                )
                break

            # Backtest
            emit(f"[Loop {pass_num}] Backtest...")
            bt = run_backtest_and_compare(
                callback=lambda msg: emit(f"  {msg}"),
                is_cancelled=is_cancelled,
            )

            loop_elapsed = time.time() - loop_start
            loop_times.append(loop_elapsed)

            # Compare winner% to best
            fund = bt.get("fundamentals", {})
            cur_winner = fund.get("winner_pct", 0)
            best_winner = best_bt.get("fundamentals", {}).get("winner_pct", 0)
            if cur_winner > best_winner:
                best_bt = bt
                emit(f"  NEW BEST! Winner={cur_winner:.1f}%, "
                     f"Upset={fund.get('upset_accuracy', 0):.0f}% @ "
                     f"{fund.get('upset_rate', 0):.0f}% rate, "
                     f"CompDog={fund.get('competitive_dog_rate', 0):.0f}% "
                     f"({fund.get('competitive_dog_count', 0)}/{fund.get('upset_count', 0)}), "
                     f"OnePosDog={fund.get('one_possession_dog_rate', 0):.0f}% "
                     f"({fund.get('one_possession_dog_count', 0)}/{fund.get('upset_count', 0)}), "
                     f"LongDog1P={fund.get('long_dog_onepos_rate', 0):.0f}% "
                     f"({fund.get('long_dog_onepos_count', 0)}/{fund.get('long_dog_count', 0)}), "
                     f"ML ROI={fund.get('ml_roi', 0):+.1f}%")
            else:
                emit(f"  No improvement ({cur_winner:.1f}% vs best {best_winner:.1f}%)")

            emit(f"  Pass {pass_num} took {fmt_elapsed(loop_elapsed)} | "
                 f"avg {fmt_elapsed(sum(loop_times) / len(loop_times))}/pass")

            completed_passes += 1

            if pass_saved_any:
                consecutive_no_save_passes = 0
            else:
                consecutive_no_save_passes += 1
                if max_no_save_passes > 0:
                    emit("  Save gate: no weights saved this pass "
                         f"({consecutive_no_save_passes}/{max_no_save_passes})")
                    if consecutive_no_save_passes >= max_no_save_passes:
                        emit("Stopping overnight early: reached "
                             f"{consecutive_no_save_passes} consecutive no-save passes.")
                        break
                else:
                    emit("  Save gate: no weights saved this pass")

        except Exception as e:
            logger.exception("Overnight loop %d error: %s", pass_num, e)
            emit(f"  Loop error: {e}")
            continue

    total_elapsed = time.time() - overall_start
    emit(f"\n{'=' * 60}")
    emit(
        f"Overnight complete: {completed_passes} fully evaluated passes "
        f"in {fmt_elapsed(total_elapsed)}"
    )
    if attempted_passes > completed_passes:
        emit(
            f"Attempted passes: {attempted_passes} "
            f"(partial/unscored: {attempted_passes - completed_passes})"
        )
    if best_bt:
        f_met = best_bt.get("fundamentals", {})
        emit(f"Best: Winner={f_met.get('winner_pct', 0):.1f}%, "
             f"Upset acc={f_met.get('upset_accuracy', 0):.0f}% @ "
             f"{f_met.get('upset_rate', 0):.0f}%, "
             f"CompDog={f_met.get('competitive_dog_rate', 0):.0f}% "
             f"({f_met.get('competitive_dog_count', 0)}/{f_met.get('upset_count', 0)}), "
             f"OnePosDog={f_met.get('one_possession_dog_rate', 0):.0f}% "
             f"({f_met.get('one_possession_dog_count', 0)}/{f_met.get('upset_count', 0)}), "
             f"LongDog1P={f_met.get('long_dog_onepos_rate', 0):.0f}% "
             f"({f_met.get('long_dog_onepos_count', 0)}/{f_met.get('long_dog_count', 0)}), "
             f"ML ROI={f_met.get('ml_roi', 0):+.1f}%")
    emit(f"{'=' * 60}")

    try:
        state = _load_pipeline_state()
        state["overnight_last_run"] = {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "passes": completed_passes,
            "attempted_passes": attempted_passes,
            "elapsed_seconds": round(total_elapsed, 1),
            "consecutive_no_save_passes": consecutive_no_save_passes,
            "save_gate_passes": save_gate_passes[-30:],
        }
        _save_pipeline_state(state)
    except Exception as e:
        logger.debug("Failed to persist overnight save-gate summary: %s", e)

    return {
        "passes": completed_passes,
        "attempted_passes": attempted_passes,
        "elapsed_seconds": round(total_elapsed, 1),
        "backtest": best_bt,
        "save_gate_passes": save_gate_passes,
        "consecutive_no_save_passes": consecutive_no_save_passes,
    }
