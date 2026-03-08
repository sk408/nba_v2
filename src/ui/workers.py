"""QObject-based workers for background ops in the desktop UI.

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

    Uses QueuedConnection for all cross-thread signal->slot connections
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


class OvernightWorker(BaseWorker):
    """Runs full pipeline then loops optimization until time runs out."""
    def __init__(self, max_hours: float = 8.0, reset_weights: bool = False):
        super().__init__()
        self.max_hours = max_hours
        self.reset_weights = reset_weights

    def run(self):
        try:
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


# ---------------------------------------------------------------------------
# Factory functions for accuracy workers
# ---------------------------------------------------------------------------

def start_backtest_worker(on_progress=None, on_result=None, on_done=None):
    return _start_worker(BacktestWorker(), on_progress, on_done, on_result)

def start_fft_worker(on_progress=None, on_done=None):
    return _start_worker(FFTWorker(), on_progress, on_done)

def start_overnight_worker(max_hours=8.0, reset_weights=False, on_progress=None, on_result=None, on_done=None):
    return _start_worker(OvernightWorker(max_hours, reset_weights), on_progress, on_done, on_result)
