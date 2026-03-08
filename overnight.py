"""CLI to run the overnight optimization loop with rich TUI and graceful shutdown."""

import logging
import re
import signal
import sys
import argparse
import threading
import time

sys.path.insert(0, ".")

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from src.bootstrap import setup_logging


# ── Graceful shutdown ─────────────────────────────────────────

_force_quit = False


def _install_signal_handler(tui: "RichOvernightConsole | None" = None):
    """First Ctrl+C → graceful cancel; second → force exit."""

    def _handle_sigint(signum, frame):
        global _force_quit
        if _force_quit:
            if tui:
                tui.stop()
            console = Console()
            console.print("\n[bold red]Force quit.[/bold red]")
            sys.exit(1)
        _force_quit = True
        from src.analytics.pipeline import request_cancel
        request_cancel()
        if tui:
            tui.stopping = True

    signal.signal(signal.SIGINT, _handle_sigint)


# ── Rich TUI Console ─────────────────────────────────────────

class RichOvernightConsole:
    """Parses pipeline callback messages and renders a structured rich display."""

    # Pipeline steps in execution order
    PIPELINE_STEPS = [
        "backup", "sync", "seed_arenas", "bbref_sync", "referee_sync",
        "elo_compute", "precompute", "optimize_fundamentals",
        "optimize_sharp", "backtest",
    ]

    STEP_LABELS = {
        "backup": "Backup",
        "sync": "Data Sync",
        "seed_arenas": "Seed Arenas",
        "bbref_sync": "BBRef Sync",
        "referee_sync": "Referee Sync",
        "elo_compute": "Elo Ratings",
        "precompute": "Precompute",
        "optimize_fundamentals": "Optimize Fundamentals",
        "optimize_sharp": "Optimize Sharp",
        "backtest": "Backtest",
    }

    def __init__(self, max_hours: float, file=None):
        self.max_hours = max_hours
        self.start_time = time.time()
        self.current_activity = "Initializing..."
        self.pass_results: list[dict] = []
        self.best_result: dict | None = None
        self.trial_current = 0
        self.trial_total = 0
        self.current_pass = 0
        self.pass_start_time = time.time()
        # Trial rate tracking for ETA estimates
        self.opt_start_time: float = 0.0   # when current opt step began
        self.opt_start_trial: int = 0      # first trial number we saw
        self._live: Live | None = None
        self._console = Console(file=file) if file else Console()
        self._ticker_stop = threading.Event()
        self._log_lines: list[str] = []  # rolling log of recent messages
        self._log_max = 10
        self._ticker_thread: threading.Thread | None = None

        # Pipeline step tracking: "pending" | "running" | "done" | "failed"
        self.step_status: dict[str, str] = {s: "pending" for s in self.PIPELINE_STEPS}
        self.step_times: dict[str, float] = {}
        self.step_detail: dict[str, str] = {}  # last detail message per step
        self.current_step: str | None = None
        # Sync sub-step tracking
        self.sync_substeps: list[dict] = []  # {label, status}
        self.sync_current_label: str = ""
        self.in_pipeline = False  # True during Pass 1 full pipeline
        self.stopping = False  # True after Ctrl+C / deadline cancel

    def start(self):
        self._live = Live(
            self._build_display(),
            console=self._console,
            refresh_per_second=2,
            transient=False,
        )
        self._live.start()
        # Background ticker to keep elapsed time updating even when no callbacks arrive
        self._ticker_stop.clear()
        self._ticker_thread = threading.Thread(target=self._tick, daemon=True)
        self._ticker_thread.start()

    def _tick(self):
        """Rebuild the display every second so the elapsed timer stays current."""
        _deadline_cancelled = False
        while not self._ticker_stop.wait(1.0):
            if self._live:
                self._live.update(self._build_display())
            # If past deadline, trigger graceful cancel so current step finishes
            if not _deadline_cancelled:
                elapsed = time.time() - self.start_time
                if elapsed >= self.max_hours * 3600:
                    _deadline_cancelled = True
                    from src.analytics.pipeline import request_cancel
                    request_cancel()
                    self.stopping = True

    def stop(self):
        self._ticker_stop.set()
        if self._live:
            self._live.stop()

    def callback(self, msg: str):
        """Parse a pipeline message and update the display."""
        # Add to rolling log (skip blank lines)
        stripped = msg.strip()
        if stripped:
            self._log_lines.append(stripped)
            if len(self._log_lines) > self._log_max:
                self._log_lines = self._log_lines[-self._log_max:]
        self._parse_message(msg)
        if self._live:
            self._live.update(self._build_display())

    def _parse_message(self, msg: str):
        stripped = msg.strip()

        # Pipeline step start: "[Step 2/10] sync..."
        m = re.match(r"\[Step\s+(\d+)/(\d+)\]\s+(\w+)", stripped)
        if m:
            step_name = m.group(3)
            if step_name in self.step_status:
                # Mark previous step done if it was running
                if self.current_step and self.current_step != step_name:
                    if self.step_status[self.current_step] == "running":
                        self.step_status[self.current_step] = "done"
                self.step_status[step_name] = "running"
                self.current_step = step_name
                label = self.STEP_LABELS.get(step_name, step_name)
                self.current_activity = label
                # Switch from pipeline steps view to results view once optimization starts
                if step_name in ("optimize_fundamentals", "optimize_sharp", "backtest"):
                    self.in_pipeline = False
                else:
                    self.in_pipeline = True
                # Reset sync substeps when starting sync
                if step_name == "sync":
                    self.sync_substeps = []
                # Reset trial progress for optimization steps
                if step_name in ("optimize_fundamentals", "optimize_sharp"):
                    self.trial_current = 0
                    self.trial_total = 0
                    self.opt_start_time = time.time()
                    self.opt_start_trial = 0
            return

        # Step completed: "  backup completed in 0.3s"
        m = re.search(r"(\w+)\s+completed\s+in\s+([\d.]+)s", stripped)
        if m:
            step_name = m.group(1)
            elapsed = float(m.group(2))
            if step_name in self.step_status:
                self.step_status[step_name] = "done"
                self.step_times[step_name] = elapsed
            return

        # Step failed: "  sync FAILED: ..."
        m = re.search(r"(\w+)\s+FAILED:", stripped)
        if m:
            step_name = m.group(1)
            if step_name in self.step_status:
                self.step_status[step_name] = "failed"
            return

        # Sync sub-step: "=== 1/8 Reference data ==="
        m = re.search(r"===\s*(\d+)/(\d+)\s+(.+?)\s*===", stripped)
        if m:
            idx = int(m.group(1))
            label = m.group(3).strip()
            # Mark previous sync substep done
            if self.sync_substeps:
                last = self.sync_substeps[-1]
                if last["status"] == "running":
                    last["status"] = "done"
            self.sync_substeps.append({"label": label, "status": "running"})
            self.sync_current_label = label
            self.step_detail["sync"] = label
            self.current_activity = f"Data Sync - {label}"
            return

        # Sync sub-step detail messages (update the detail text)
        if self.current_step == "sync" and self.sync_substeps:
            # Capture useful detail from sync messages
            for pattern in [
                r"Saved (\d+) teams",
                r"Rosters: (\d+/\d+) teams",
                r"Saved (\d+) game log entries",
                r"Injury sync complete: (\d+) injuries",
                r"Injury history built: (\d+) records",
                r"Fetching (.+?)\.{3}",
                r"On/off: (\d+/\d+) teams",
                r"Odds sync complete: (\d+) games",
                r"(\w.+?is fresh, skipping)",
            ]:
                dm = re.search(pattern, stripped)
                if dm:
                    self.step_detail["sync"] = dm.group(0).strip(".")
                    break
            # "Full data sync complete!" marks sync done
            if "Full data sync complete" in stripped:
                if self.sync_substeps:
                    self.sync_substeps[-1]["status"] = "done"
                return

        # Pipeline complete: "Pipeline complete in Xs"
        if "Pipeline complete" in stripped:
            self.in_pipeline = False
            # Mark any remaining running step as done
            if self.current_step and self.step_status.get(self.current_step) == "running":
                self.step_status[self.current_step] = "done"
            return

        # Pass header: "--- Pass N: ..."
        m = re.search(r"Pass (\d+):", stripped)
        if m:
            self.current_pass = int(m.group(1))
            self.pass_start_time = time.time()
            self.trial_current = 0
            self.trial_total = 0
            if self.current_pass > 1:
                self.in_pipeline = False

        # Trial progress: "Trial 1247/3000" or "trial 1247/3000"
        m = re.search(r"[Tt]rial\s+(\d+)/(\d+)", stripped)
        if m:
            new_current = int(m.group(1))
            new_total = int(m.group(2))
            # Detect new optimization step (total changed or current jumped back)
            if new_total != self.trial_total or new_current < self.trial_current:
                self.opt_start_time = time.time()
                self.opt_start_trial = new_current
            self.trial_current = new_current
            self.trial_total = new_total

        # Step labels in loop passes: [backup], [sync], etc.
        m = re.search(r"\[(backup|sync|seed_arenas|bbref_sync|referee_sync|"
                       r"elo_compute|precompute|optimize_fundamentals|"
                       r"optimize_sharp|backtest)\]", stripped, re.IGNORECASE)
        if m:
            step_name = m.group(1).replace("_", " ").title()
            self.current_activity = step_name

        # Optimizing lines — switch to results view
        if "Optimizing fundamentals" in stripped:
            self.current_activity = "Optimizing Fundamentals"
            self.in_pipeline = False
            self.trial_current = 0
            self.opt_start_time = time.time()
            self.opt_start_trial = 0
        elif "Optimizing sharp" in stripped:
            self.current_activity = "Optimizing Sharp"
            self.trial_current = 0
            self.opt_start_time = time.time()
            self.opt_start_trial = 0
        elif "Backtest" in stripped and "Loop" in stripped:
            self.current_activity = "Backtesting"
            self.trial_current = 0
            self.trial_total = 0

        # Post-optimization phases (clear progress bar, update activity)
        if "Validating top" in stripped:
            self.current_activity = "Validating Top Trials"
            self.trial_current = 0
            self.trial_total = 0
        elif "Top 5 impact parameters" in stripped:
            self.current_activity = "Computing Parameter Importance"
            self.trial_current = 0
            self.trial_total = 0
        elif "Saved" in stripped and "new trials to disk" in stripped:
            self.current_activity = "Saving Trials"
            self.trial_current = 0
            self.trial_total = 0
        elif "Study now has" in stripped:
            self.trial_current = 0
            self.trial_total = 0

        # Precompute progress: "Precomputed 25/150 games"
        m = re.search(r"Precomputed?\s+(\d+)/(\d+)\s+games", stripped)
        if m:
            self.trial_current = int(m.group(1))
            self.trial_total = int(m.group(2))

        # NEW BEST result
        m = re.search(
            r"NEW BEST.*?Winner=([\d.]+)%.*?Upset=([\d.]+)%\s*@\s*([\d.]+)%.*?"
            r"ML ROI=([+\-\d.]+)%",
            stripped,
        )
        if m:
            result = {
                "pass": self.current_pass,
                "duration": time.time() - self.pass_start_time,
                "winner_pct": float(m.group(1)),
                "upset_pct": float(m.group(2)),
                "upset_rate": float(m.group(3)),
                "ml_roi": float(m.group(4)),
                "is_best": True,
            }
            self.pass_results.append(result)
            self.best_result = result
            return

        # No improvement line
        m = re.search(r"No improvement \(([\d.]+)% vs best ([\d.]+)%\)", stripped)
        if m:
            self.pass_results.append({
                "pass": self.current_pass,
                "duration": time.time() - self.pass_start_time,
                "winner_pct": float(m.group(1)),
                "upset_pct": 0,
                "upset_rate": 0,
                "ml_roi": 0,
                "is_best": False,
            })
            return

        # Pass 1 complete (full pipeline results) — update activity
        if "Pass 1 complete" in stripped:
            self.current_activity = "Pass 1 complete"

        # Overnight complete
        if "Overnight complete" in stripped:
            self.current_activity = "Complete"

    def _build_pipeline_steps(self) -> Panel:
        """Build the pipeline steps panel with checkmarks."""
        lines = Text()
        for i, step in enumerate(self.PIPELINE_STEPS):
            status = self.step_status[step]
            label = self.STEP_LABELS.get(step, step)
            elapsed = self.step_times.get(step)
            time_str = f"  {elapsed:.0f}s" if elapsed is not None else ""

            if status == "done":
                icon = Text(" [+] ", style="bold green")
                name = Text(f"{label}{time_str}", style="green")
            elif status == "running":
                icon = Text(" [>] ", style="bold cyan")
                name = Text(label, style="bold cyan")
            elif status == "failed":
                icon = Text(" [!] ", style="bold red")
                name = Text(f"{label}  FAILED", style="red")
            else:
                icon = Text(" [ ] ", style="dim")
                name = Text(label, style="dim")

            lines.append_text(icon)
            lines.append_text(name)

            # Show sync sub-steps inline
            if step == "sync" and self.sync_substeps and status in ("running", "done"):
                for sub in self.sync_substeps:
                    if sub["status"] == "done":
                        sub_icon = Text("   [+] ", style="green")
                        sub_name = Text(sub["label"], style="green")
                    else:
                        sub_icon = Text("   [>] ", style="cyan")
                        detail = self.step_detail.get("sync", "")
                        sub_name = Text(
                            f"{sub['label']}"
                            + (f"  ({detail})" if detail and detail != sub["label"] else ""),
                            style="cyan",
                        )
                    lines.append("\n")
                    lines.append_text(sub_icon)
                    lines.append_text(sub_name)

            if i < len(self.PIPELINE_STEPS) - 1:
                lines.append("\n")

        return Panel(lines, title="Pipeline Steps", border_style="blue")

    def _build_display(self) -> Layout:
        layout = Layout()

        # Log height: panel border (2) + lines
        log_height = min(len(self._log_lines), self._log_max) + 2
        log_height = max(log_height, 4)  # at least a small panel

        # Show pipeline steps panel during Pass 1, results panel during later passes
        if self.in_pipeline:
            # Dynamic height: 10 steps + 2 for panel border + sync sub-steps
            steps_height = len(self.PIPELINE_STEPS) + 2 + len(self.sync_substeps)
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="activity", size=5),
                Layout(name="steps", size=steps_height),
                Layout(name="log", size=log_height),
                Layout(name="footer", size=3),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="activity", size=5),
                Layout(name="results", minimum_size=6),
                Layout(name="log", size=log_height),
                Layout(name="footer", size=3),
            )

        # Header
        elapsed = time.time() - self.start_time
        remaining = max(0, self.max_hours * 3600 - elapsed)
        elapsed_str = self._fmt_duration(elapsed)
        remaining_str = self._fmt_duration(remaining)
        header_text = Text.assemble(
            ("Overnight Optimization", "bold cyan"),
            "  |  ",
            (f"Elapsed: {elapsed_str}", "green"),
            "  |  ",
            (f"Remaining: {remaining_str}", "yellow"),
        )
        layout["header"].update(Panel(header_text, style="blue"))

        # Current activity + progress bar
        activity_parts = []
        if self.current_pass > 0:
            activity_parts.append(
                Text(f"Pass {self.current_pass}", style="bold magenta")
            )
        activity_parts.append(Text(f"  {self.current_activity}", style="bold white"))

        activity_text = Text()
        for part in activity_parts:
            activity_text.append_text(part)

        if self.trial_total > 0 and self.trial_current > 0:
            # Progress is based on NEW trials done this run, not absolute position
            new_done = self.trial_current - self.opt_start_trial
            new_done = max(0, new_done)
            pct = min(100.0, new_done / self.trial_total * 100)
            bar_width = 30
            filled = min(bar_width, int(bar_width * new_done / self.trial_total))
            bar = f"[{'=' * filled}{' ' * (bar_width - filled)}]"

            # ETA based on observed trial rate
            eta_str = ""
            if new_done > 5 and self.opt_start_time > 0:
                elapsed_opt = time.time() - self.opt_start_time
                rate = new_done / elapsed_opt if elapsed_opt > 0 else 0
                if rate > 0:
                    trials_left = max(0, self.trial_total - new_done)
                    eta_secs = trials_left / rate
                    eta_str = f"  ETA {self._fmt_duration(eta_secs)}"

            # Note prior trials if resuming
            prior_str = ""
            if self.opt_start_trial > 0:
                prior_str = f"  (resuming from {self.opt_start_trial:,} prior)"

            progress_line = Text.assemble(
                "\n  ",
                (f"Trial {new_done:,}/{self.trial_total:,}", "cyan"),
                (prior_str, "dim") if prior_str else "",
                f"  {bar}  ",
                (f"{pct:.0f}%", "bold green"),
                (eta_str, "yellow") if eta_str else "",
            )
            activity_text.append_text(progress_line)

        layout["activity"].update(
            Panel(activity_text, title="Current Activity", border_style="cyan")
        )

        # Pipeline steps panel (during Pass 1)
        if self.in_pipeline:
            layout["steps"].update(self._build_pipeline_steps())
        else:
            # Results table (during optimization passes)
            table = Table(
                title="Pass Results",
                show_header=True,
                header_style="bold",
                expand=True,
            )
            table.add_column("Pass", justify="center", style="dim", width=6)
            table.add_column("Duration", justify="center", width=10)
            table.add_column("Winner%", justify="center", width=10)
            table.add_column("Upset%", justify="center", width=10)
            table.add_column("Upset Rate", justify="center", width=10)
            table.add_column("ML ROI", justify="center", width=10)
            table.add_column("Status", justify="center", width=12)

            for r in self.pass_results:
                style = "bold green" if r["is_best"] else ""
                status = "[bold green]NEW BEST[/bold green]" if r["is_best"] else "no change"
                table.add_row(
                    str(r["pass"]),
                    self._fmt_duration(r["duration"]),
                    f"{r['winner_pct']:.1f}%",
                    f"{r['upset_pct']:.0f}%" if r["upset_pct"] else "-",
                    f"{r['upset_rate']:.0f}%" if r["upset_rate"] else "-",
                    f"{r['ml_roi']:+.1f}%" if r["ml_roi"] else "-",
                    status,
                    style=style,
                )

            if not self.pass_results:
                table.add_row("-", "-", "-", "-", "-", "-", "awaiting first pass")

            # Best result highlight
            results_layout = Layout()
            if self.best_result:
                best_text = Text.assemble(
                    ("Best: ", "bold"),
                    (f"Winner={self.best_result['winner_pct']:.1f}%", "bold green"),
                    "  ",
                    (f"Upset={self.best_result['upset_pct']:.0f}%", "cyan"),
                    f" @ {self.best_result['upset_rate']:.0f}%  ",
                    (f"ML ROI={self.best_result['ml_roi']:+.1f}%", "yellow"),
                    f"  (Pass {self.best_result['pass']})",
                )
                results_layout.split_column(
                    Layout(table, ratio=3),
                    Layout(Panel(best_text, style="green"), size=3),
                )
            else:
                results_layout.update(table)

            layout["results"].update(results_layout)

        # Live log
        log_text = Text()
        for i, line in enumerate(self._log_lines):
            if i > 0:
                log_text.append("\n")
            log_text.append(line, style="dim")
        layout["log"].update(Panel(log_text, title="Log", border_style="dim"))

        # Footer
        if self.stopping:
            footer_text = Text(
                "Stopping - finishing current step... (Ctrl+C again to force quit)",
                style="bold yellow",
            )
            layout["footer"].update(Panel(footer_text, style="yellow"))
        else:
            footer_text = Text("Press Ctrl+C to stop after current step", style="dim")
            layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    @staticmethod
    def _fmt_duration(secs: float) -> str:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        return f"{m}m {s:02d}s"


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run overnight optimization")
    parser.add_argument("--hours", type=float, default=8.0,
                        help="Max hours to run (default: 8)")
    parser.add_argument("--reset-weights", action="store_true",
                        help="Reset all weights to defaults before starting")
    parser.add_argument("--plain", action="store_true",
                        help="Use plain text output instead of rich TUI")
    args = parser.parse_args()

    setup_logging()

    # Initialize database (runs migrations if needed)
    from src.database.migrations import init_db
    init_db()

    from src.analytics.pipeline import run_overnight

    if args.plain:
        _install_signal_handler()
        run_overnight(
            max_hours=args.hours,
            reset_weights=args.reset_weights,
            callback=lambda msg: print(msg, flush=True),
        )
    else:
        # Redirect logging and stdout so only the rich Live display is visible.
        # Logging goes to data/overnight.log; stdout is silenced.
        import io
        import os

        log_path = os.path.join("data", "overnight.log")
        os.makedirs("data", exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        root_logger = logging.getLogger()
        # Replace console handlers with file handler
        for h in root_logger.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root_logger.removeHandler(h)
        root_logger.addHandler(file_handler)

        # Silence stray print() calls by redirecting stdout.
        # Give Rich the real stdout before we swap it out.
        _real_stdout = sys.stdout
        tui = RichOvernightConsole(max_hours=args.hours, file=_real_stdout)
        sys.stdout = io.StringIO()
        _install_signal_handler(tui)
        tui.start()
        try:
            run_overnight(
                max_hours=args.hours,
                reset_weights=args.reset_weights,
                callback=tui.callback,
            )
        finally:
            tui.stop()
            sys.stdout = _real_stdout


if __name__ == "__main__":
    main()
