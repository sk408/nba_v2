"""QObject-based workers for background ops in the desktop UI.

6 sync workers + 12 accuracy/analysis workers.
Each worker runs in a QThread and emits progress/finished signals.
"""

import logging
import threading
from PySide6.QtCore import QObject, Signal, QThread, Qt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base worker
# ---------------------------------------------------------------------------

class BaseWorker(QObject):
    """Base background worker with progress & stop support."""
    progress = Signal(str)
    result = Signal(dict)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def isRunning(self) -> bool:
        if hasattr(self, '_thread_ref') and self._thread_ref is not None:
            try:
                return self._thread_ref.isRunning()
            except RuntimeError:
                # C++ QThread already deleted via deleteLater
                self._thread_ref = None
                return False
        return False

    def _check_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        raise NotImplementedError


_active_workers = set()  # prevent GC until thread actually stops


def _start_worker(worker: BaseWorker, on_progress=None, on_done=None, on_result=None):
    """Launch a worker on a new QThread. Returns the worker for stop().

    Uses QueuedConnection for all cross-thread signal→slot connections
    so that slots always run on the main/GUI thread, even if the
    callable is a lambda or free function (which have no QObject affinity).
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    _QC = Qt.ConnectionType.QueuedConnection
    if on_progress:
        worker.progress.connect(on_progress, _QC)
    if on_result:
        worker.result.connect(on_result, _QC)
    if on_done:
        worker.finished.connect(on_done, _QC)
    worker.finished.connect(thread.quit)
    # Schedule C++ cleanup only after thread has fully stopped
    thread.finished.connect(thread.deleteLater)
    # prevent GC until thread is done — _active_workers holds a strong ref
    # so even if the caller drops its reference (e.g. _on_done sets worker=None),
    # the worker+thread survive until the OS thread actually exits.
    worker._thread_ref = thread
    _active_workers.add(worker)

    def _release_worker():
        _active_workers.discard(worker)

    thread.finished.connect(_release_worker)
    thread.start()
    return worker


# ---------------------------------------------------------------------------
# Sync workers
# ---------------------------------------------------------------------------

class SyncWorker(BaseWorker):
    """Runs one of the sync steps."""
    def __init__(self, step: str, force: bool = False):
        super().__init__()
        self.step = step
        self.force = force

    def run(self):
        try:
            from src.data.sync_service import (
                full_sync, sync_injuries_step, sync_injury_history,
                sync_team_metrics, sync_player_impact,
            )
            from src.data.image_cache import preload_images

            cb = lambda msg: self.progress.emit(msg)

            if self.step == "full":
                full_sync(callback=cb, force=self.force)
            elif self.step == "injuries":
                sync_injuries_step(callback=cb, force=self.force)
            elif self.step == "injury_history":
                sync_injury_history(callback=cb, force=self.force)
            elif self.step == "team_metrics":
                sync_team_metrics(callback=cb, force=self.force)
            elif self.step == "player_impact":
                sync_player_impact(callback=cb, force=self.force)
            elif self.step == "images":
                preload_images(callback=cb)
            else:
                self.progress.emit(f"Unknown step: {self.step}")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class InjuryWorker(BaseWorker):
    """Runs injury scraping."""
    def run(self):
        try:
            from src.data.injury_scraper import scrape_all_injuries
            self.progress.emit("Scraping injuries...")
            injuries = scrape_all_injuries()
            self.progress.emit(f"Found {len(injuries)} injuries")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


def start_sync_worker(step: str, on_progress=None, on_done=None, force: bool = False):
    return _start_worker(SyncWorker(step, force=force), on_progress, on_done)


def start_injury_worker(on_progress=None, on_done=None):
    return _start_worker(InjuryWorker(), on_progress, on_done)


class OddsSyncWorker(BaseWorker):
    """Runs odds backfill."""
    def __init__(self, force: bool = False):
        super().__init__()
        self.force = force

    def run(self):
        try:
            from src.data.odds_sync import backfill_odds
            cb = lambda msg: self.progress.emit(msg)
            mode = "force re-fetch ALL dates" if self.force else "missing dates only"
            self.progress.emit(f"Syncing historical odds ({mode})...")
            count = backfill_odds(callback=cb, force=self.force)
            self.progress.emit(f"Odds sync complete. Saved odds for {count} games.")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class NukeResyncWorker(BaseWorker):
    """Nuke all synced data, then run a full force sync from scratch."""
    def run(self):
        try:
            from src.data.sync_service import nuke_synced_data, full_sync
            cb = lambda msg: self.progress.emit(msg)
            nuke_synced_data(callback=cb)
            self.progress.emit("\nStarting full force resync from scratch...")
            full_sync(callback=cb, force=True)
            self.progress.emit("Nuke & Resync complete!")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


def start_odds_sync_worker(on_progress=None, on_done=None, force: bool = False):
    return _start_worker(OddsSyncWorker(force=force), on_progress, on_done)


def start_nuke_resync_worker(on_progress=None, on_done=None):
    return _start_worker(NukeResyncWorker(), on_progress, on_done)

def start_sensitivity_worker(param: str, steps: int = 100, target: str = "value",
                             on_progress=None, on_done=None):
    return _start_worker(SensitivityWorker(param, steps, target), on_progress, on_done)


def start_coordinate_descent_worker(steps: int = 100, max_rounds: int = 10,
                                    convergence: float = 0.005,
                                    target: str = "value",
                                    apply_results: bool = True,
                                    on_progress=None, on_done=None):
    return _start_worker(
        CoordinateDescentWorker(steps, max_rounds, convergence, target, apply_results),
        on_progress, on_done
    )


# ---------------------------------------------------------------------------
# Accuracy / Analysis workers
# ---------------------------------------------------------------------------

class BacktestWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.backtester import run_backtest
            self.progress.emit("Running backtest...")
            results = run_backtest(
                callback=lambda msg: self.progress.emit(msg)
            )
            self.result.emit(results)
            self.progress.emit("Backtest complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class OptimizerWorker(BaseWorker):
    def __init__(self, continuous: bool = False, target: str = "ats"):
        super().__init__()
        self.continuous = continuous
        self.target = target

    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import optimize_weights
            from src.analytics.weight_config import get_weight_config, save_weight_config, save_snapshot
            import itertools
            
            cb = lambda msg: self.progress.emit(msg)
            is_cancelled = lambda: self._check_stop()
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
                self.finished.emit()
                return

            if self.target == "all":
                targets = ["ats", "roi", "ml"]
            else:
                targets = [self.target]

            if self.continuous:
                target_iter = itertools.cycle(targets)
                n_trials = 2000 if self.target == "all" else 1000000 
            else:
                target_iter = iter(targets)
                n_trials = 2000

            saved_weights = {}

            for t in target_iter:
                if is_cancelled():
                    break
                    
                if t in saved_weights:
                    save_weight_config(saved_weights[t])
                    
                if self.target == "all":
                    self.progress.emit(f"=== Optimizing target: {t.upper()} ===")
                self.progress.emit(f"Optimizing weights ({n_trials} trials, target={t})...")
                
                result = optimize_weights(games, n_trials=n_trials, callback=cb, is_cancelled=is_cancelled, target=t)
                
                saved_weights[t] = get_weight_config()
                
                if result.get("improved", False):
                    snap_name = f"Best_{t.upper()}_Weights"
                    metrics = {
                        "ats_rate": result.get("ats_rate", 0),
                        "ats_roi": result.get("ats_roi", 0),
                        "ml_roi": result.get("ml_roi", 0),
                        "loss": result.get("loss", 0),
                    }
                    save_snapshot(snap_name, notes=f"Auto-saved by optimizer ({t.upper()})", metrics=metrics)
                
            self.progress.emit("Optimization complete")
        except Exception as e:
            import traceback
            import logging
            logging.getLogger(__name__).error("OptimizerWorker failed:\n%s", traceback.format_exc())
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class FeatureImportanceWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import compute_feature_importance
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Computing feature importance...")
                result = compute_feature_importance(games, callback=cb)
                self.result.emit({"feature_importance": result})
                self.progress.emit("Feature importance complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class MLFeatureWorker(BaseWorker):
    """ML/SHAP-based feature importance."""
    def run(self):
        try:
            from src.analytics.ml_model import get_shap_importance
            self.progress.emit("Computing ML feature importance...")
            result = get_shap_importance()
            self.result.emit({"ml_features": result})
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class GroupedFeatureWorker(BaseWorker):
    """Grouped feature importance."""
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import compute_feature_importance
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Computing grouped feature importance...")
                result = compute_feature_importance(games, callback=cb)
                # Group by category
                groups = {}
                for f in result:
                    name = f.get("feature", "")
                    if "factor" in name or "rating" in name:
                        cat = "Defense/Ratings"
                    elif "ff_" in name or "four_factors" in name:
                        cat = "Four Factors"
                    elif "hustle" in name or "clutch" in name:
                        cat = "Hustle/Clutch"
                    elif "fatigue" in name or "pace" in name:
                        cat = "Pace/Fatigue"
                    else:
                        cat = "Other"
                    groups.setdefault(cat, []).append(f)
                self.result.emit({"grouped_features": groups})
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class FFTWorker(BaseWorker):
    """FFT error pattern analysis."""
    def run(self):
        try:
            self.progress.emit("Running FFT analysis...")
            # FFT on backtest errors
            from src.analytics.backtester import run_backtest
            results = run_backtest()
            per_game = results.get("per_game", [])
            import numpy as np
            errors = [g.get("spread_error", 0) for g in per_game]
            if len(errors) > 10:
                fft = np.fft.rfft(errors)
                freqs = np.fft.rfftfreq(len(errors))
                magnitudes = np.abs(fft)
                top_idx = np.argsort(magnitudes)[-5:][::-1]
                fft_result = [
                    {"frequency": float(freqs[i]), "magnitude": float(magnitudes[i])}
                    for i in top_idx
                ]
                self.result.emit({"fft": fft_result})
                self.progress.emit(f"FFT: found {len(fft_result)} dominant frequencies")
            else:
                self.progress.emit("Not enough data for FFT")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class ComboWorker(BaseWorker):
    """Global + per-team optimization."""
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import optimize_weights, per_team_refinement
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Phase 1: Global optimization...")
                optimize_weights(games, callback=cb)
                self.progress.emit("Phase 2: Per-team refinement...")
                per_team_refinement(games, callback=cb)
                self.progress.emit("Combo optimization complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class ContinuousWorker(BaseWorker):
    """Continuous optimization loop until stopped."""
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.weight_optimizer import optimize_weights
            cb = lambda msg: self.progress.emit(msg)
            iteration = 0
            while not self._check_stop():
                iteration += 1
                self.progress.emit(f"Continuous iteration {iteration}...")
                games = precompute_game_data(callback=cb)
                if games:
                    optimize_weights(games, callback=cb)
                if self._check_stop():
                    break
            self.progress.emit(f"Stopped after {iteration} iterations")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class PipelineWorker(BaseWorker):
    """Full 12-step pipeline."""
    def run(self):
        try:
            from src.analytics.pipeline import run_full_pipeline
            self.progress.emit("Starting full pipeline...")
            results = run_full_pipeline(
                callback=lambda msg: self.progress.emit(msg)
            )
            # Emit backtest results so accuracy cards get populated
            bt = results.get("backtest", {})
            if bt and bt.get("total_games", 0) > 0:
                self.result.emit(bt)
            self.progress.emit("Pipeline complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class RetuneWorker(BaseWorker):
    """Skip data sync, force-run autotune + ML + optimize + backtest."""
    def run(self):
        from src.database.db import thread_local_db
        try:
            with thread_local_db():
                from src.analytics.pipeline import run_retune
                self.progress.emit("Starting retune (no data sync)...")
                results = run_retune(
                    callback=lambda msg: self.progress.emit(msg)
                )
                bt = results.get("backtest", {})
                if bt and bt.get("total_games", 0) > 0:
                    self.result.emit(bt)
                self.progress.emit("Retune complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class OvernightWorker(BaseWorker):
    """Runs full pipeline then loops optimization until time runs out."""
    def __init__(self, max_hours: float = 8.0, reset_weights: bool = False):
        super().__init__()
        self.max_hours = max_hours
        self.reset_weights = reset_weights

    def run(self):
        from src.database.db import thread_local_db
        try:
            with thread_local_db():
                from src.analytics.pipeline import run_overnight
                results = run_overnight(
                    max_hours=self.max_hours,
                    reset_weights=self.reset_weights,
                    callback=lambda msg: self.progress.emit(msg)
                )
                bt = results.get("backtest", {})
                if bt and bt.get("total_games", 0) > 0:
                    self.result.emit(bt)
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class MLTrainWorker(BaseWorker):
    def run(self):
        try:
            from src.analytics.prediction import precompute_game_data
            from src.analytics.ml_model import train_models
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Precomputing game data...")
            games = precompute_game_data(callback=cb)
            if not games:
                self.progress.emit("No game data available")
            else:
                self.progress.emit("Training ML models...")
                train_models(games, callback=cb)
                self.progress.emit("ML training complete")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class DiagnosticCSVWorker(BaseWorker):
    """Export diagnostic CSV for worst-performing teams."""
    def __init__(self, backtest_results=None):
        super().__init__()
        self._bt_results = backtest_results

    def run(self):
        try:
            from src.analytics.backtester import export_diagnostic_csv
            from src.database.db import thread_local_db
            cb = lambda msg: self.progress.emit(msg)
            self.progress.emit("Generating diagnostic CSV...")
            with thread_local_db():
                path = export_diagnostic_csv(
                    backtest_results=self._bt_results,
                    callback=cb,
                )
            if path:
                self.result.emit({"csv_path": path})
            self.progress.emit("Diagnostic export complete")
        except Exception as e:
            import traceback
            logger.error("DiagnosticCSVWorker failed:\n%s", traceback.format_exc())
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class SensitivityWorker(BaseWorker):
    """Run sensitivity analysis sweep."""
    def __init__(self, param: str, steps: int = 100, target: str = "ats"):
        super().__init__()
        self.param = param
        self.steps = steps
        self.target = target

    def run(self):
        try:
            from src.analytics.sensitivity import sweep_parameter, EXTREME_RANGES, export_sweep_csv, format_ascii_chart, sweep_all_parameters, run_full_analysis
            from src.database.db import thread_local_db

            cb = lambda msg: self.progress.emit(msg)

            with thread_local_db():
                if self.param == "all":
                    run_full_analysis(steps=self.steps, callback=cb, target=self.target)
                else:
                    self.progress.emit(f"Sweeping {self.param} (target={self.target})...")
                    lo, hi = EXTREME_RANGES.get(self.param, (0, 1))
                    results = sweep_parameter(self.param, lo, hi, steps=self.steps,
                                              callback=cb, target=self.target)
                    path = export_sweep_csv(self.param, results)
                    self.progress.emit(f"\nSaved CSV: {path}\n")

                    # Chart the target-specific ROI metric
                    roi_metric = "dog_roi" if self.target in ("value", "ml") else "ats_roi"
                    chart1 = format_ascii_chart(self.param, results, metric=roi_metric)
                    chart2 = format_ascii_chart(self.param, results, metric="loss")
                    self.progress.emit(chart1 + "\n\n" + chart2)

        except Exception as e:
            self.progress.emit(f"Error: {e}")
        self.finished.emit()


class CoordinateDescentWorker(BaseWorker):
    """Run iterative coordinate descent optimization."""
    def __init__(self, steps: int = 100, max_rounds: int = 10,
                 convergence: float = 0.005, target: str = "ats",
                 apply_results: bool = True):
        super().__init__()
        self.steps = steps
        self.max_rounds = max_rounds
        self.convergence = convergence
        self.target = target
        self.apply_results = apply_results

    def run(self):
        try:
            from src.analytics.sensitivity import coordinate_descent
            from src.analytics.weight_config import save_snapshot
            from src.database.db import thread_local_db

            cb = lambda msg: self.progress.emit(msg)

            with thread_local_db():
                result = coordinate_descent(
                    steps=self.steps,
                    max_rounds=self.max_rounds,
                    convergence_threshold=self.convergence,
                    callback=cb,
                    target=self.target,
                    save=self.apply_results,
                )

                if result.get("improved") and self.apply_results:
                    save_snapshot(
                        f"coord_descent_{self.target}",
                        notes=f"Coordinate descent ({self.target}): "
                              f"val loss {result['initial_val_loss']:.3f} -> {result['final_val_loss']:.3f}",
                        metrics={
                            "dog_roi": result.get("final_dog_roi", 0),
                            "dog_hit_rate": result.get("final_dog_hit_rate", 0),
                            "ml_roi": result.get("final_ml_roi", 0),
                        },
                    )
                elif not self.apply_results:
                    self.progress.emit(f"Dry run complete. Best val loss: {result['final_val_loss']:.3f}")

                # Report top parameter changes
                changes = result.get("changes", {})
                if changes:
                    self.progress.emit(f"\nTop parameter changes ({len(changes)} moved):")
                    sorted_changes = sorted(changes.items(),
                                            key=lambda x: abs(x[1].get("new", 0) - x[1].get("old", 0)),
                                            reverse=True)
                    for name, vals in sorted_changes[:10]:
                        self.progress.emit(f"  {name}: {vals.get('old', 0):.4f} -> {vals.get('new', 0):.4f}")

        except Exception as e:
            import traceback
            self.progress.emit(f"Error: {e}\n{traceback.format_exc()}")
        self.finished.emit()


class OverviewWorker(BaseWorker):
    """Fetch today's scoreboard, odds, and predictions for an overview."""
    def run(self):
        try:
            from datetime import datetime
            from src.data.gamecast import fetch_espn_scoreboard, get_actionnetwork_odds
            from src.analytics.prediction import predict_matchup
            from src.database.db import thread_local_db
            from src.database import db

            self.progress.emit("Fetching today's games from ESPN...")
            games = fetch_espn_scoreboard()

            if not games:
                self.progress.emit("No games found for today.")
                self.finished.emit()
                return

            today_str = datetime.now().strftime("%Y-%m-%d")
            results = []

            # Use thread_local_db so all db reads (including inside
            # predict_matchup) use a private in-memory copy — no concurrent
            # cursor operations on the shared connection.
            with thread_local_db():
                for i, g in enumerate(games):
                    if self._check_stop():
                        break

                    home_abbr = g.get("home_team", "")
                    away_abbr = g.get("away_team", "")
                    self.progress.emit(f"Processing {away_abbr} @ {home_abbr} ({i+1}/{len(games)})...")

                    pred_data = None
                    odds = {}
                    try:
                        home_row = db.fetch_one("SELECT team_id FROM teams WHERE abbreviation = ?", (home_abbr,))
                        away_row = db.fetch_one("SELECT team_id FROM teams WHERE abbreviation = ?", (away_abbr,))

                        if home_row and away_row:
                            pred = predict_matchup(home_row["team_id"], away_row["team_id"], today_str,
                                                   skip_ml=True, skip_espn=True)
                            pred_data = {
                                "spread": getattr(pred, 'predicted_spread', 0.0),
                                "total": getattr(pred, 'predicted_total', 0.0),
                                "winner": pred.winner,
                                "sharp_money_adj": pred.adjustments.get("sharp_money", 0.0) if hasattr(pred, 'adjustments') else 0.0,
                                "sharp_home_public": getattr(pred, 'sharp_home_public', 0),
                                "sharp_home_money": getattr(pred, 'sharp_home_money', 0),
                            }
                    except Exception as e:
                        logger.warning(f"Prediction failed for {away_abbr} @ {home_abbr}: {e}")

                    try:
                        odds = get_actionnetwork_odds(home_abbr, away_abbr)
                    except Exception as oe:
                        logger.warning(f"Odds fetch failed for {away_abbr} @ {home_abbr}: {oe}")

                    g_data = {
                        "espn_id": g.get("espn_id"),
                        "status": g.get("status", ""),
                        "period": g.get("period", 0),
                        "clock": g.get("clock", ""),
                        "home_team": home_abbr,
                        "away_team": away_abbr,
                        "home_score": g.get("home_score", 0),
                        "away_score": g.get("away_score", 0),
                        "prediction": pred_data,
                        "odds": odds
                    }
                    results.append(g_data)

            self.progress.emit(f"Done. {len(results)} games processed.")
            self.result.emit({"games": results})
        except Exception as e:
            import traceback
            logger.error(f"OverviewWorker error:\n{traceback.format_exc()}")
            self.progress.emit(f"Error: {e}")
        self.finished.emit()

# ---------------------------------------------------------------------------
# Factory functions for accuracy workers
# ---------------------------------------------------------------------------

def start_backtest_worker(on_progress=None, on_result=None, on_done=None):
    return _start_worker(BacktestWorker(), on_progress, on_done, on_result)

def start_optimizer_worker(continuous: bool = False, target: str = "ats", on_progress=None, on_done=None):
    return _start_worker(OptimizerWorker(continuous, target), on_progress, on_done)

def start_feature_importance_worker(on_progress=None, on_done=None):
    return _start_worker(FeatureImportanceWorker(), on_progress, on_done)

def start_ml_feature_worker(on_progress=None, on_done=None):
    return _start_worker(MLFeatureWorker(), on_progress, on_done)

def start_grouped_feature_worker(on_progress=None, on_done=None):
    return _start_worker(GroupedFeatureWorker(), on_progress, on_done)

def start_fft_worker(on_progress=None, on_done=None):
    return _start_worker(FFTWorker(), on_progress, on_done)

def start_combo_worker(on_progress=None, on_done=None):
    return _start_worker(ComboWorker(), on_progress, on_done)

def start_continuous_worker(on_progress=None, on_done=None):
    return _start_worker(ContinuousWorker(), on_progress, on_done)

def start_pipeline_worker(on_progress=None, on_result=None, on_done=None):
    return _start_worker(PipelineWorker(), on_progress, on_done, on_result)

def start_retune_worker(on_progress=None, on_result=None, on_done=None):
    return _start_worker(RetuneWorker(), on_progress, on_done, on_result)

def start_overnight_worker(max_hours=8.0, reset_weights=False, on_progress=None, on_result=None, on_done=None):
    return _start_worker(OvernightWorker(max_hours, reset_weights), on_progress, on_done, on_result)

def start_ml_train_worker(on_progress=None, on_done=None):
    return _start_worker(MLTrainWorker(), on_progress, on_done)

def start_overview_worker(on_progress=None, on_result=None, on_done=None):
    return _start_worker(OverviewWorker(), on_progress, on_done, on_result)

def start_diagnostic_csv_worker(backtest_results=None, on_progress=None, on_result=None, on_done=None):
    return _start_worker(DiagnosticCSVWorker(backtest_results), on_progress, on_done, on_result)
