"""Daily automation service for desktop runtime.

Runs the full pipeline on a daily schedule and, if configured, executes:
git add . -> git commit -m "<message>" -> git push
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Optional

from src.analytics.pipeline import run_pipeline
from src.config import get as get_setting

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GIT_TIMEOUT_SECONDS = 120


def _as_bounded_int(value: object, default: int, lower: int, upper: int) -> int:
    """Parse int-like values with bounds and fallback."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, lower), upper)


def _truncate_output(text: str, max_len: int = 1200) -> str:
    """Keep command output readable in logs."""
    clean = (text or "").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max(0, max_len - 3)].rstrip() + "..."


class DailyAutomationService:
    """Background scheduler for daily pipeline + git automation."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._job_lock = threading.Lock()
        self._last_run_date: Optional[str] = None

    def start(self):
        """Start the scheduler thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="daily-automation",
            daemon=True,
        )
        self._thread.start()
        logger.info("Daily automation service started")

    def stop(self):
        """Stop the scheduler thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        logger.info("Daily automation service stopped")

    def _loop(self):
        """Wait for the configured run time and execute jobs."""
        while not self._stop_event.is_set():
            if not bool(get_setting("daily_automation_enabled", True)):
                # Poll config periodically so this can be toggled without restart.
                self._stop_event.wait(60)
                continue

            hour = _as_bounded_int(get_setting("daily_automation_hour", 9), 9, 0, 23)
            minute = _as_bounded_int(
                get_setting("daily_automation_minute", 0), 0, 0, 59
            )
            now = datetime.now()
            run_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if run_at <= now:
                run_at += timedelta(days=1)

            logger.info(
                "Daily automation next run at %s",
                run_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
            wait_seconds = max(0.0, (run_at - now).total_seconds())
            if self._stop_event.wait(wait_seconds):
                return

            run_date = run_at.date().isoformat()
            if self._last_run_date == run_date:
                continue
            self._last_run_date = run_date
            self._run_job(run_date)

    def _run_job(self, run_date: str):
        """Execute one scheduled job."""
        if not self._job_lock.acquire(blocking=False):
            logger.warning("Daily automation skipped: previous job is still running")
            return

        try:
            logger.info("Daily automation job started for %s", run_date)
            result = run_pipeline(
                callback=lambda msg: logger.info("[daily pipeline] %s", msg)
            )
            if result.get("cancelled"):
                logger.warning("Daily automation pipeline cancelled; skipping git step")
                return
            if result.get("error"):
                logger.warning(
                    "Daily automation pipeline error (%s); skipping git step",
                    result.get("error"),
                )
                return

            if not bool(get_setting("daily_automation_git_enabled", True)):
                logger.info("Daily automation git step disabled by settings")
                return

            self._run_git_sequence()
        except Exception:
            logger.exception("Daily automation job failed")
        finally:
            self._job_lock.release()

    def _run_git_sequence(self):
        """Run git add/commit/push when there are staged changes."""
        commit_message = str(get_setting("daily_automation_commit_message", "daily"))
        commit_message = commit_message.strip() or "daily"

        self._run_git(["add", "."])
        staged = self._run_git(["diff", "--cached", "--quiet"], check=False)
        if staged.returncode == 0:
            logger.info("Daily automation: no staged changes; skipping commit/push")
            return
        if staged.returncode != 1:
            raise RuntimeError(
                f"git diff --cached --quiet returned unexpected code {staged.returncode}"
            )

        self._run_git(["commit", "-m", commit_message])
        self._run_git(["push"])
        logger.info("Daily automation git push complete")

    def _run_git(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command from repo root with non-interactive prompt policy."""
        cmd = ["git", *args]
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"

        logger.info("Daily automation running command: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out: {' '.join(cmd)}") from exc

        stdout = _truncate_output(proc.stdout)
        stderr = _truncate_output(proc.stderr)
        if stdout:
            logger.info("[daily git stdout] %s", stdout)
        if stderr:
            logger.info("[daily git stderr] %s", stderr)

        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}): {' '.join(cmd)}"
            )
        return proc


_service_lock = threading.Lock()
_service: Optional[DailyAutomationService] = None


def start_daily_automation() -> DailyAutomationService:
    """Start singleton daily automation service."""
    global _service
    with _service_lock:
        if _service is None:
            _service = DailyAutomationService()
        _service.start()
        return _service


def stop_daily_automation():
    """Stop singleton daily automation service."""
    with _service_lock:
        if _service is not None:
            _service.stop()

