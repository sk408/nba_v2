# Coordinate Descent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add coordinate descent (CD) as an on-demand weight refinement tool, triggered from a dedicated section in the Settings view.

**Architecture:** CD grid-searches each parameter individually using wider `CD_RANGES`, accepting improvements on training loss only. Save gate uses the same validation winner_pct check as Optuna. UI in Settings view with mode selector, steps/rounds controls, progress log, and results table.

**Tech Stack:** Python, NumPy, PySide6 (QThread workers, QGroupBox, QTextEdit for log)

**Design doc:** `docs/plans/2026-03-05-coordinate-descent-design.md`

---

### Task 1: Add CD_RANGES to weight_config.py

**Files:**
- Modify: `src/analytics/weight_config.py:153` (after SHARP_MODE_RANGES)

**Step 1: Add CD_RANGES dict**

Insert after line 153 (`SHARP_MODE_RANGES = {**OPTIMIZER_RANGES, **SHARP_RANGES}`):

```python
# Coordinate Descent ranges — wider than OPTIMIZER_RANGES for broader exploration.
# CD grid-searches every value so it can safely explore beyond Optuna's TPE bounds.
CD_RANGES = {
    "def_factor_dampening": (0.05, 12.0),
    "turnover_margin_mult": (0.0, 12.0),
    "rebound_diff_mult": (0.0, 5.0),
    "rating_matchup_mult": (0.0, 6.0),
    "four_factors_scale": (0.5, 120.0),
    "ff_efg_weight": (0.0, 18.0),
    "ff_tov_weight": (0.0, 18.0),
    "ff_oreb_weight": (0.0, 18.0),
    "ff_fta_weight": (0.0, 18.0),
    "opp_ff_efg_weight": (0.0, 18.0),
    "opp_ff_tov_weight": (0.0, 18.0),
    "opp_ff_oreb_weight": (0.0, 18.0),
    "opp_ff_fta_weight": (0.0, 18.0),
    "clutch_scale": (0.0, 3.0),
    "hustle_effort_mult": (0.0, 3.0),
    "hustle_contested_wt": (0.0, 2.0),
    "steals_penalty": (0.0, 5.0),
    "blocks_penalty": (0.0, 5.0),
    "rest_advantage_mult": (0.0, 5.0),
    "altitude_b2b_penalty": (0.0, 12.0),
    "fatigue_b2b": (0.0, 12.0),
    "fatigue_3in4": (0.0, 12.0),
    "fatigue_4in6": (0.0, 6.0),
    "fatigue_same_day": (0.0, 8.0),
    "fatigue_rest_bonus": (0.0, 5.0),
    "fatigue_total_mult": (0.0, 3.0),
}

# CD ranges with sharp money parameter included
CD_SHARP_RANGES = {**CD_RANGES, "sharp_ml_weight": (0.0, 20.0)}
```

**Step 2: Update the import list in optimizer.py**

In `optimizer.py` line 18-21, add `CD_RANGES` and `CD_SHARP_RANGES` to the import:

```python
from src.analytics.weight_config import (
    WeightConfig, get_weight_config, save_weight_config,
    OPTIMIZER_RANGES, SHARP_MODE_RANGES, invalidate_weight_cache,
    CD_RANGES, CD_SHARP_RANGES,
)
```

**Step 3: Verify**

```bash
cd nba_v2 && python -c "
from src.analytics.weight_config import CD_RANGES, CD_SHARP_RANGES, OPTIMIZER_RANGES
print(f'OPTIMIZER_RANGES: {len(OPTIMIZER_RANGES)} params')
print(f'CD_RANGES: {len(CD_RANGES)} params')
print(f'CD_SHARP_RANGES: {len(CD_SHARP_RANGES)} params')
# Verify all CD keys are a superset of OPTIMIZER keys
assert set(CD_RANGES.keys()) == set(OPTIMIZER_RANGES.keys()), 'Key mismatch!'
# Verify CD ranges are >= OPTIMIZER ranges
for k in CD_RANGES:
    cd_lo, cd_hi = CD_RANGES[k]
    opt_lo, opt_hi = OPTIMIZER_RANGES[k]
    assert cd_lo <= opt_lo and cd_hi >= opt_hi, f'{k}: CD range not wider!'
print('All checks passed')
"
```

Expected: All checks pass.

**Step 4: Commit**

```bash
git add src/analytics/weight_config.py src/analytics/optimizer.py
git commit -m "feat: add CD_RANGES (wider exploration bounds for coordinate descent)"
```

---

### Task 2: Add coordinate_descent() to optimizer.py

**Files:**
- Modify: `src/analytics/optimizer.py` (append after `compare_modes()` function, ~line 691)

**Step 1: Add the coordinate_descent function**

Append after the `compare_modes()` function:

```python
def coordinate_descent(
    games: List[GameInput],
    params: Optional[List[str]] = None,
    steps: int = 100,
    max_rounds: int = 10,
    convergence_threshold: float = 0.005,
    include_sharp: bool = False,
    callback: Optional[Callable] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Grid-search refinement of individual parameters after Optuna TPE.

    For each parameter, evaluates `steps` equally-spaced values across the
    CD_RANGES bounds.  Accepts a new value only when it improves training loss.
    Repeats for up to `max_rounds` until convergence.

    Save gate: validation winner_pct must beat both favorites_pct and the
    baseline winner_pct (same gate as optimize_weights).
    """
    # Walk-forward split (same as Optuna)
    sorted_games = sorted(games, key=lambda g: g.game_date)
    split_idx = int(len(sorted_games) * WALK_FORWARD_SPLIT)
    train_games = sorted_games[:split_idx]
    val_games = sorted_games[split_idx:]

    if not train_games or not val_games:
        if callback:
            callback("Not enough games for walk-forward split")
        return {"improved": False, "rounds": 0}

    vg_train = VectorizedGames(train_games)
    vg_val = VectorizedGames(val_games)

    if callback:
        callback(f"CD: {len(train_games)} train, {len(val_games)} validation")

    # Select ranges and parameters
    ranges = CD_SHARP_RANGES if include_sharp else CD_RANGES
    if params is None:
        params = list(ranges.keys())

    if callback:
        callback(f"CD: {len(params)} parameters, {steps} steps/param, "
                 f"max {max_rounds} rounds")

    # Load current weights as starting point
    w = get_weight_config()
    w_dict = w.to_dict()

    # Baseline evaluation
    baseline_train = vg_train.evaluate(w, include_sharp=include_sharp)
    baseline_val = vg_val.evaluate(w, include_sharp=include_sharp)
    current_train_loss = baseline_train["loss"]

    if callback:
        callback(f"CD baseline (train): Winner={baseline_train['winner_pct']:.1f}%, "
                 f"Loss={baseline_train['loss']:.3f}")
        callback(f"CD baseline (valid): Winner={baseline_val['winner_pct']:.1f}%, "
                 f"Favorites={baseline_val['favorites_pct']:.1f}%")

    best_w_dict = w_dict.copy()
    history = []
    all_changes = {}
    prev_val_loss = baseline_val["loss"]

    for round_num in range(1, max_rounds + 1):
        if is_cancelled and is_cancelled():
            if callback:
                callback("CD cancelled by user.")
            break

        accepted_count = 0
        round_start_loss = current_train_loss

        if callback:
            callback(f"--- Round {round_num}/{max_rounds} ---")

        for p_idx, param_name in enumerate(params):
            if is_cancelled and is_cancelled():
                break

            lo, hi = ranges.get(param_name, (0, 1))
            grid = np.linspace(lo, hi, steps)

            best_param_loss = current_train_loss
            best_param_val = w_dict[param_name]

            for val in grid:
                test_dict = {**w_dict, param_name: float(val)}
                test_w = WeightConfig.from_dict(test_dict)
                result = vg_train.evaluate(test_w, include_sharp=include_sharp)
                if result["loss"] < best_param_loss:
                    best_param_loss = result["loss"]
                    best_param_val = float(val)

            # Accept if training loss improved
            if best_param_loss < current_train_loss:
                old_val = w_dict[param_name]
                w_dict[param_name] = best_param_val
                current_train_loss = best_param_loss
                best_w_dict = w_dict.copy()
                accepted_count += 1
                all_changes[param_name] = {
                    "before": old_val,
                    "after": best_param_val,
                }
                if callback:
                    callback(f"  {param_name}: {old_val:.4f} -> "
                             f"{best_param_val:.4f} "
                             f"(loss {current_train_loss:.4f}) KEPT")
            else:
                if callback and p_idx % 5 == 0:
                    callback(f"  [{p_idx + 1}/{len(params)}] {param_name}: no improvement")

        # End-of-round: evaluate on validation
        round_w = WeightConfig.from_dict(best_w_dict)
        round_val = vg_val.evaluate(round_w, include_sharp=include_sharp)
        round_train = vg_train.evaluate(round_w, include_sharp=include_sharp)

        round_info = {
            "round": round_num,
            "accepted": accepted_count,
            "train_loss": round_train["loss"],
            "train_winner_pct": round_train["winner_pct"],
            "val_loss": round_val["loss"],
            "val_winner_pct": round_val["winner_pct"],
            "val_favorites_pct": round_val["favorites_pct"],
        }
        history.append(round_info)

        if callback:
            callback(f"Round {round_num} summary: "
                     f"{accepted_count}/{len(params)} params accepted, "
                     f"val Winner={round_val['winner_pct']:.1f}% "
                     f"(fav={round_val['favorites_pct']:.1f}%), "
                     f"train_loss={round_train['loss']:.3f}")

        # Convergence checks
        if accepted_count == 0:
            if callback:
                callback("No parameters improved this round. Stopping.")
            break

        val_improvement = abs(prev_val_loss - round_val["loss"])
        if round_num >= 2 and val_improvement < convergence_threshold:
            if callback:
                callback(f"Converged (improvement {val_improvement:.4f} "
                         f"< threshold {convergence_threshold}). Stopping.")
            break

        prev_val_loss = round_val["loss"]

    # Final evaluation
    final_w = WeightConfig.from_dict(best_w_dict)
    final_val = vg_val.evaluate(final_w, include_sharp=include_sharp)
    final_train = vg_train.evaluate(final_w, include_sharp=include_sharp)

    # Save gate: same as optimize_weights
    baseline_winner_pct = baseline_val.get("winner_pct", 0)
    favorites_pct = final_val.get("favorites_pct", 0)
    best_winner_pct = final_val.get("winner_pct", 0)
    save_ok = best_winner_pct > max(favorites_pct, baseline_winner_pct)

    if callback:
        callback(f"--- CD Final ---")
        callback(f"  Train:  Winner={final_train['winner_pct']:.1f}%, "
                 f"Loss={final_train['loss']:.3f}")
        callback(f"  Valid:  Winner={best_winner_pct:.1f}% "
                 f"(was {baseline_winner_pct:.1f}%), "
                 f"Favorites={favorites_pct:.1f}%")

    if save and save_ok:
        save_weight_config(final_w)
        invalidate_weight_cache()
        if callback:
            callback(f"CD saved improved weights "
                     f"({baseline_winner_pct:.1f}% -> {best_winner_pct:.1f}%)")
    elif save:
        reason = ""
        if best_winner_pct <= favorites_pct:
            reason = f" (winner {best_winner_pct:.1f}% <= favorites {favorites_pct:.1f}%)"
        elif best_winner_pct <= baseline_winner_pct:
            reason = f" (winner {best_winner_pct:.1f}% <= baseline {baseline_winner_pct:.1f}%)"
        if callback:
            callback(f"CD: validation winner% did not improve{reason} "
                     f"- keeping current weights")

    return {
        "weights": best_w_dict,
        "history": history,
        "changes": all_changes,
        "rounds": len(history),
        "improved": save_ok,
        "initial_winner_pct": baseline_winner_pct,
        "final_winner_pct": best_winner_pct,
        "initial_loss": baseline_val["loss"],
        "final_loss": final_val["loss"],
        "favorites_pct": favorites_pct,
        **final_val,
    }
```

**Step 2: Verify the function imports and runs**

```bash
cd nba_v2 && python -c "
from src.analytics.optimizer import coordinate_descent
print('coordinate_descent imported OK')
import inspect
sig = inspect.signature(coordinate_descent)
print(f'Parameters: {list(sig.parameters.keys())}')
"
```

Expected: Import OK with correct parameter list.

**Step 3: Commit**

```bash
git add src/analytics/optimizer.py
git commit -m "feat: add coordinate_descent() with grid-search refinement"
```

---

### Task 3: Add CD section to Settings view

**Files:**
- Modify: `src/ui/views/settings_view.py`

**Step 1: Add imports**

At the top of `settings_view.py`, add to the PySide6 imports (line 10-13):

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QScrollArea, QSlider, QSpinBox,
    QRadioButton, QButtonGroup, QGroupBox, QSizePolicy,
    QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
```

**Step 2: Add `_CDWorker` class**

Add after the module-level constants (after line 28):

```python
class _CDWorker(QObject):
    """Runs coordinate descent on a background thread."""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, include_sharp: bool, steps: int, max_rounds: int):
        super().__init__()
        self.include_sharp = include_sharp
        self.steps = steps
        self.max_rounds = max_rounds
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from src.database import thread_local_db
            thread_local_db()
            from src.analytics.prediction import precompute_all_games
            from src.analytics.optimizer import coordinate_descent

            self.progress.emit("Loading precomputed games...")
            games = precompute_all_games(
                callback=lambda msg: self.progress.emit(msg),
            )
            if not games:
                self.error.emit("No games available for CD")
                return

            self.progress.emit(f"Starting CD: {len(games)} games, "
                              f"steps={self.steps}, rounds={self.max_rounds}")

            result = coordinate_descent(
                games=games,
                steps=self.steps,
                max_rounds=self.max_rounds,
                include_sharp=self.include_sharp,
                callback=lambda msg: self.progress.emit(msg),
                is_cancelled=lambda: self._cancelled,
                save=True,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
```

**Step 3: Add `_build_cd_section` method to SettingsView**

Add this method to the SettingsView class, and call it from `__init__` between the theme and weight management sections. In `__init__`, insert between lines 71-74:

```python
        # ---- Section 4: Theme ----
        self._build_theme_section(layout)

        # ---- Section 5: Coordinate Descent ----
        self._build_cd_section(layout)

        # ---- Section 6: Weight Management ----
        self._build_weight_management(layout)
```

Then add the method body:

```python
    def _build_cd_section(self, parent_layout: QVBoxLayout):
        """Build coordinate descent refinement section."""
        group = QGroupBox("Coordinate Descent Refinement")
        gl = QVBoxLayout(group)
        gl.setSpacing(10)

        desc = QLabel(
            "Grid-search refinement of individual parameters after Optuna optimization. "
            "CD fine-tunes each weight by testing many values across wider ranges, "
            "accepting only improvements. Run after the pipeline optimizer for best results."
        )
        desc.setProperty("class", "text-secondary")
        desc.setWordWrap(True)
        gl.addWidget(desc)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        mode_row.addWidget(mode_lbl)
        self._cd_mode_group = QButtonGroup(self)
        self._cd_mode_fund = QRadioButton("Fundamentals Only")
        self._cd_mode_fund.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        self._cd_mode_fund.setChecked(True)
        self._cd_mode_group.addButton(self._cd_mode_fund, 0)
        mode_row.addWidget(self._cd_mode_fund)
        self._cd_mode_sharp = QRadioButton("Fundamentals + Sharp")
        self._cd_mode_sharp.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        self._cd_mode_group.addButton(self._cd_mode_sharp, 1)
        mode_row.addWidget(self._cd_mode_sharp)
        mode_row.addStretch()
        gl.addLayout(mode_row)

        # Controls row: steps, max rounds, buttons
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)

        steps_lbl = QLabel("Steps:")
        steps_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        ctrl_row.addWidget(steps_lbl)
        self._cd_steps_spin = QSpinBox()
        self._cd_steps_spin.setRange(50, 2000)
        self._cd_steps_spin.setValue(100)
        self._cd_steps_spin.setSingleStep(50)
        self._cd_steps_spin.setFixedWidth(80)
        ctrl_row.addWidget(self._cd_steps_spin)

        rounds_lbl = QLabel("Max Rounds:")
        rounds_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        ctrl_row.addWidget(rounds_lbl)
        self._cd_rounds_spin = QSpinBox()
        self._cd_rounds_spin.setRange(1, 20)
        self._cd_rounds_spin.setValue(10)
        self._cd_rounds_spin.setFixedWidth(60)
        ctrl_row.addWidget(self._cd_rounds_spin)

        self._cd_run_btn = QPushButton("Run CD")
        self._cd_run_btn.setProperty("class", "success")
        self._cd_run_btn.setFixedHeight(36)
        self._cd_run_btn.setMinimumWidth(100)
        self._cd_run_btn.clicked.connect(self._on_cd_run)
        ctrl_row.addWidget(self._cd_run_btn)

        self._cd_cancel_btn = QPushButton("Cancel")
        self._cd_cancel_btn.setProperty("class", "danger")
        self._cd_cancel_btn.setFixedHeight(36)
        self._cd_cancel_btn.setMinimumWidth(80)
        self._cd_cancel_btn.setVisible(False)
        self._cd_cancel_btn.clicked.connect(self._on_cd_cancel)
        ctrl_row.addWidget(self._cd_cancel_btn)

        ctrl_row.addStretch()
        gl.addLayout(ctrl_row)

        # Progress log
        self._cd_log = QTextEdit()
        self._cd_log.setReadOnly(True)
        self._cd_log.setFixedHeight(200)
        self._cd_log.setStyleSheet(
            "QTextEdit { background: #0a0e14; color: #22c55e; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; "
            "border: 1px solid rgba(255,255,255,0.08); border-radius: 4px; "
            "padding: 8px; }"
        )
        gl.addWidget(self._cd_log)

        # Results table (hidden until CD completes)
        self._cd_results_lbl = QLabel("")
        self._cd_results_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 13px; font-weight: 600;"
        )
        gl.addWidget(self._cd_results_lbl)

        self._cd_changes_table = QTableWidget()
        self._cd_changes_table.setColumnCount(4)
        self._cd_changes_table.setHorizontalHeaderLabels(
            ["Parameter", "Before", "After", "Delta"]
        )
        self._cd_changes_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for col in range(1, 4):
            self._cd_changes_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._cd_changes_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._cd_changes_table.verticalHeader().setVisible(False)
        self._cd_changes_table.setAlternatingRowColors(True)
        self._cd_changes_table.setMaximumHeight(200)
        self._cd_changes_table.setVisible(False)
        gl.addWidget(self._cd_changes_table)

        # Worker state
        self._cd_worker = None
        self._cd_thread = None

        apply_card_shadow(group)
        parent_layout.addWidget(group)
```

**Step 4: Add CD event handlers**

Add these methods to the SettingsView class (before `_notify`):

```python
    def _on_cd_run(self):
        """Start coordinate descent on a background thread."""
        include_sharp = self._cd_mode_group.checkedId() == 1
        steps = self._cd_steps_spin.value()
        max_rounds = self._cd_rounds_spin.value()

        self._cd_log.clear()
        self._cd_results_lbl.setText("")
        self._cd_changes_table.setVisible(False)
        self._cd_run_btn.setEnabled(False)
        self._cd_cancel_btn.setVisible(True)

        mode_str = "Fundamentals + Sharp" if include_sharp else "Fundamentals Only"
        self._cd_log.append(f"Starting CD ({mode_str}, {steps} steps, {max_rounds} rounds)...")

        self._cd_worker = _CDWorker(include_sharp, steps, max_rounds)
        self._cd_thread = QThread()
        self._cd_worker.moveToThread(self._cd_thread)
        self._cd_thread.started.connect(self._cd_worker.run)

        _QC = Qt.ConnectionType.QueuedConnection
        self._cd_worker.progress.connect(self._on_cd_progress, _QC)
        self._cd_worker.finished.connect(self._on_cd_finished, _QC)
        self._cd_worker.error.connect(self._on_cd_error, _QC)
        self._cd_worker.finished.connect(self._cd_thread.quit)
        self._cd_worker.error.connect(self._cd_thread.quit)
        self._cd_thread.finished.connect(self._cd_cleanup)

        self._cd_thread.start()

    def _on_cd_cancel(self):
        """Cancel a running CD."""
        if self._cd_worker:
            self._cd_worker.cancel()
            self._cd_log.append("Cancelling...")

    def _on_cd_progress(self, msg: str):
        """Append progress message to the CD log."""
        self._cd_log.append(msg)
        # Auto-scroll to bottom
        sb = self._cd_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_cd_finished(self, result: dict):
        """CD completed -- show results."""
        self._cd_run_btn.setEnabled(True)
        self._cd_cancel_btn.setVisible(False)

        improved = result.get("improved", False)
        rounds = result.get("rounds", 0)
        changes = result.get("changes", {})
        initial_wp = result.get("initial_winner_pct", 0)
        final_wp = result.get("final_winner_pct", 0)

        if improved:
            self._cd_results_lbl.setText(
                f"CD improved: {initial_wp:.1f}% -> {final_wp:.1f}% winner "
                f"({rounds} rounds, {len(changes)} params changed)"
            )
            self._cd_results_lbl.setStyleSheet(
                f"color: {GREEN}; font-size: 13px; font-weight: 600;"
            )
        else:
            self._cd_results_lbl.setText(
                f"CD did not improve validation winner% "
                f"({rounds} rounds, {len(changes)} params tested)"
            )
            self._cd_results_lbl.setStyleSheet(
                f"color: {AMBER}; font-size: 13px; font-weight: 600;"
            )

        # Populate changes table
        if changes:
            sorted_changes = sorted(
                changes.items(),
                key=lambda x: abs(x[1]["after"] - x[1]["before"]),
                reverse=True,
            )
            self._cd_changes_table.setRowCount(len(sorted_changes))
            for r, (param, vals) in enumerate(sorted_changes):
                before = vals["before"]
                after = vals["after"]
                delta = after - before
                self._cd_changes_table.setItem(
                    r, 0, QTableWidgetItem(param)
                )
                self._cd_changes_table.setItem(
                    r, 1, QTableWidgetItem(f"{before:.4f}")
                )
                self._cd_changes_table.setItem(
                    r, 2, QTableWidgetItem(f"{after:.4f}")
                )
                delta_item = QTableWidgetItem(f"{delta:+.4f}")
                delta_item.setForeground(
                    QColor(GREEN if delta > 0 else RED)
                )
                self._cd_changes_table.setItem(r, 3, delta_item)
            self._cd_changes_table.setVisible(True)

        self._update_weight_summary()
        self._notify(
            f"CD complete: {'improved' if improved else 'no improvement'} "
            f"({rounds} rounds)"
        )

    def _on_cd_error(self, msg: str):
        """CD failed -- show error."""
        self._cd_run_btn.setEnabled(True)
        self._cd_cancel_btn.setVisible(False)
        self._cd_log.append(f"ERROR: {msg}")
        self._cd_results_lbl.setText("CD failed")
        self._cd_results_lbl.setStyleSheet(
            f"color: {RED}; font-size: 13px; font-weight: 600;"
        )

    def _cd_cleanup(self):
        """Clean up CD thread resources."""
        if self._cd_thread:
            self._cd_thread.deleteLater()
        if self._cd_worker:
            self._cd_worker.deleteLater()
        self._cd_thread = None
        self._cd_worker = None
```

**Step 5: Add QColor import**

Add to the Qt imports at the top of settings_view.py:

```python
from PySide6.QtGui import QColor
```

**Step 6: Verify compilation**

```bash
cd nba_v2 && python -m py_compile src/ui/views/settings_view.py && echo "OK"
```

**Step 7: Verify import chain works**

```bash
cd nba_v2 && python -c "
from src.ui.views.settings_view import SettingsView, _CDWorker
print('SettingsView with CD section: OK')
print('_CDWorker: OK')
"
```

**Step 8: Commit**

```bash
git add src/ui/views/settings_view.py
git commit -m "feat: add coordinate descent UI in Settings view"
```

---

### Task 4: End-to-end verification and push

**Step 1: Verify all modules compile**

```bash
cd nba_v2 && python -m py_compile src/analytics/weight_config.py && echo "weight_config: OK" \
  && python -m py_compile src/analytics/optimizer.py && echo "optimizer: OK" \
  && python -m py_compile src/ui/views/settings_view.py && echo "settings_view: OK" \
  && python -m py_compile src/ui/main_window.py && echo "main_window: OK"
```

**Step 2: Verify CD function runs with small data**

```bash
cd nba_v2 && python -c "
from src.analytics.weight_config import CD_RANGES, CD_SHARP_RANGES
from src.analytics.optimizer import coordinate_descent, VectorizedGames
from src.analytics.prediction import GameInput
import numpy as np

# Create minimal test games
games = []
for i in range(100):
    gi = GameInput(
        game_date=f'2024-{(i%12)+1:02d}-{(i%28)+1:02d}',
        season='2024-25',
        home_team_id=1610612747, away_team_id=1610612738,
        home_proj={'points': 110+np.random.randn()*5, 'turnovers': 14, 'rebounds': 44, 'steals': 7, 'blocks': 5},
        away_proj={'points': 112+np.random.randn()*5, 'turnovers': 13, 'rebounds': 45, 'steals': 8, 'blocks': 6},
        home_ff={'efg': 0.53, 'tov': 0.13, 'oreb': 0.28, 'fta': 0.25, 'opp_efg': 0.51, 'opp_tov': 0.14, 'opp_oreb': 0.26, 'opp_fta': 0.23},
        away_ff={'efg': 0.55, 'tov': 0.12, 'oreb': 0.30, 'fta': 0.27, 'opp_efg': 0.50, 'opp_tov': 0.135, 'opp_oreb': 0.25, 'opp_fta': 0.22},
        home_clutch={'net_rating': 2.0}, away_clutch={'net_rating': 3.0},
        home_hustle={'deflections': 12, 'contested': 30, 'effort': 5},
        away_hustle={'deflections': 14, 'contested': 32, 'effort': 6},
        actual_home_score=105+np.random.randn()*10,
        actual_away_score=108+np.random.randn()*10,
        vegas_spread=-3.5, vegas_home_ml=-150, vegas_away_ml=130,
    )
    games.append(gi)

# Run CD with minimal settings
result = coordinate_descent(
    games=games,
    params=['def_factor_dampening', 'turnover_margin_mult'],
    steps=10,
    max_rounds=2,
    save=False,
    callback=print,
)
print(f'Rounds: {result[\"rounds\"]}')
print(f'Changes: {list(result[\"changes\"].keys())}')
print(f'Improved: {result[\"improved\"]}')
print('CD function test PASSED')
"
```

**Step 3: Push**

```bash
cd nba_v2 && git push
```
