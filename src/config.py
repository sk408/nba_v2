"""JSON-backed application settings at data/app_settings.json.

Runtime source of truth is ``data/app_settings.json``.
This module keeps code-side defaults only as bootstrap/fallback values when the
settings file is missing/corrupt, and auto-hydrates missing keys back to disk.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Optional dependency for local env overrides.
    logger.debug("python-dotenv load skipped", exc_info=True)

_SETTINGS_PATH = Path("data") / "app_settings.json"
_settings_lock = threading.Lock()

# MAINTAINER NOTE:
# - Runtime source-of-truth is data/app_settings.json.
# - _DEFAULTS below are bootstrap/fallback values (missing/corrupt file, new keys).
# - If you intentionally tune runtime behavior, update app_settings.json in the
#   same change so troubleshooting sees the effective values in one place.

_DEFAULTS: Dict[str, Any] = {
    "db_path": "data/nba_analytics.db",
    "season": "2025-26",
    "season_year": "2025",
    "historical_seasons": ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"],
    "theme": "dark",
    "auto_sync_interval_minutes": 60,
    "daily_automation_enabled": True,
    "daily_automation_hour": 9,
    "daily_automation_minute": 0,
    "daily_automation_git_enabled": True,
    "daily_automation_commit_message": "daily",
    "notification_webhook_url": "",
    "notification_ntfy_topic": "",
    "enable_toast_notifications": True,
    "log_level": "INFO",
    "worker_threads": max(1, (os.cpu_count() or 4) - 2),
    "timezone": "US/Pacific",
    "oled_mode": False,
    "sync_freshness_hours": 4,
    "precompute_progress_log_every": 200,
    "nba_api_min_interval_seconds": 1.6,
    "nba_api_on_off_timeout_seconds": 12.0,
    "nba_api_on_off_retries": 1,
    "nba_api_on_off_abort_after_failures": 0,  # 0 = disabled
    "optimizer_log_interval": 300,
    "optimizer_deterministic": False,
    "optimizer_deterministic_seed": 42,
    "optimizer_cmaes_suppress_independent_sampling_warn": True,
    "optimizer_tuning_mode": "blocked",  # "classic" or "blocked"
    "prediction_mode": "fundamentals_sharp",  # "fundamentals" or "fundamentals_sharp"
    "upset_bonus_mult": 0.5,  # optimizer upset reward multiplier
    "upset_bonus_mult_max": 5.0,  # hard cap for upset bonus tuning
    # Optimizer objective tuning (coverage + tier-A quality)
    "optimizer_objective_target_upset_coverage_pct": 20.0,
    "optimizer_objective_target_tier_a_hit_rate": 56.0,
    "optimizer_objective_coverage_shortfall_mult": 0.20,
    "optimizer_objective_tier_a_shortfall_mult": 0.12,
    "optimizer_objective_tier_a_prior_weight": 12.0,
    "optimizer_objective_tier_a_min_confidence": 70.0,
    "optimizer_objective_tier_b_min_confidence": 55.0,
    # Anti-overfit objective probe: penalize train-only winners that regress on
    # a deterministic slice of validation games during trial scoring.
    "optimizer_objective_val_probe_enabled": True,
    "optimizer_objective_val_probe_sample_size": 480,
    "optimizer_objective_val_probe_slices": 3,
    "optimizer_objective_val_probe_loss_mult": 1.0,
    "optimizer_objective_val_probe_winner_drop_mult": 0.35,
    # Dominance penalties for feature families (soft regularization, not clamps)
    "optimizer_objective_use_family_dominance_penalty": True,
    "optimizer_objective_family_penalty_mult": 0.05,
    "optimizer_objective_ff_p95_cap": 95.0,
    "optimizer_objective_onoff_p95_cap": 25.0,
    # Optional prior pull toward current champion to discourage gratuitous drift.
    "optimizer_objective_l2_prior_mult": 0.02,
    # Underdog confidence calibration for tiering/frontier analytics.
    # Uses raw model edge to avoid confidence saturation at 100.
    "underdog_confidence_use_edge_logistic": True,
    "underdog_confidence_edge_scale": 70.0,
    # Optional user-controlled study namespace suffix for clean retunes.
    "optimizer_study_tag": "",
    "optimizer_blocked_cycle_tag": "",
    "optimizer_blocked_auto_cycle_tag": True,
    "optimizer_blocked_cycle_hours": 24,
    # Blocked-pathway optimizer stage settings.
    "optimizer_blocked_core_fraction": 0.35,
    "optimizer_blocked_ff_fraction": 0.25,
    "optimizer_blocked_onoff_fraction": 0.15,
    "optimizer_blocked_joint_fraction": 0.25,
    "optimizer_blocked_min_stage_trials": 250,
    "optimizer_blocked_joint_radius_fraction": 0.18,
    "optimizer_blocked_stage_verbose": False,
    # Diagnostic threshold: dog pick counted "competitive" if it loses by <= this margin
    "optimizer_competitive_dog_margin": 7.5,
    # Moneyline filter for optimizer ROI diagnostics (1.50 == risk 100 to return 150 total)
    "optimizer_min_ml_payout": 1.50,
    # On/off signal shaping (feature engineering, not hard clamps)
    "optimizer_onoff_player_minutes_smoothing": 800.0,
    "optimizer_onoff_team_reliability_slots": 12.0,
    # Optimizer anti-gaming save gate settings
    "optimizer_save_loss_margin": 0.01,
    "optimizer_save_min_weight_delta": 0.0001,
    "optimizer_save_max_winner_drop": 0.35,
    "optimizer_save_favorites_slack": 0.25,
    "optimizer_save_compression_floor": 0.55,
    "optimizer_save_min_upset_count": 0,  # 0 = auto from validation sample size
    "optimizer_save_min_upset_rate": 8.0,
    "optimizer_save_max_upset_rate": 55.0,
    "optimizer_save_upset_prior_weight": 25.0,
    "optimizer_save_min_shrunk_upset_lift": 0.40,
    "optimizer_save_min_ml_bets": 0,  # 0 = auto from validation sample size
    "optimizer_save_min_roi_lift": 0.15,
    "optimizer_save_roi_lb95_slack": 0.35,
    "optimizer_save_use_roi_gate": False,  # False = ROI diagnostics only (not a hard save gate)
    "optimizer_save_use_hybrid_loss_gate": True,
    "optimizer_save_hybrid_val_weight": 0.70,
    "optimizer_save_hybrid_margin": 0.003,
    "optimizer_save_max_val_loss_regress": 0.020,
    # Long-dog one-possession tiebreak (diagnostic/secondary gate)
    "optimizer_save_use_long_dog_tiebreak_gate": True,
    "optimizer_save_long_dog_min_payout": 3.0,
    "optimizer_save_long_dog_onepos_margin": 3.0,
    "optimizer_save_long_dog_min_count": 40,
    "optimizer_save_long_dog_prior_weight": 25.0,
    "optimizer_save_min_long_dog_onepos_lift": 0.75,
    "optimizer_save_long_dog_tiebreak_loss_window": 0.010,
    # Score realism calibrator (post-prediction only; does not change winner picks)
    "score_calibration_enabled": True,
    "score_calibration_apply_to_display": True,
    "score_calibration_train_window_games": 2500,  # 0 = all games
    "score_calibration_min_games": 500,
    "score_calibration_bins": 15,
    "score_calibration_strict_sign_lock": True,
    "score_calibration_sign_epsilon": 0.05,
    "score_calibration_min_abs_spread": 0.10,
    "score_calibration_near_spread_enabled": False,
    "score_calibration_near_spread_identity_band": 1.5,
    "score_calibration_near_spread_deadband": 4.0,
    "score_calibration_near_spread_raw_weight": 0.85,
    "score_calibration_spread_cap": 34.0,
    "score_calibration_total_min": 160.0,
    "score_calibration_total_max": 275.0,
    "score_calibration_point_floor": 65.0,
    "score_calibration_point_ceiling": 175.0,
    "score_calibration_tail_margin_threshold": 20.0,
    "score_calibration_team_residual_enabled": False,
    "score_calibration_team_min_games": 30,
    "score_calibration_team_shrinkage": 50.0,
    "score_calibration_team_max_abs_correction": 5.0,
    "score_calibration_team_range_enabled": True,
    "score_calibration_team_range_min_games": 35,
    "score_calibration_team_range_quantile_low": 0.03,
    "score_calibration_team_range_quantile_high": 0.97,
    "score_calibration_team_range_padding": 6.0,
    # Optuna controls
    "optuna_top_n_validation": 120,
    "optimizer_seed_disk_trials_enabled": True,
    "optimizer_seed_disk_trials_max": 600,
    "optimizer_seed_disk_trials_top_fraction": 0.6,
    "optimizer_use_wide_ranges": False,
    "optuna_stagnation_threshold": 500,
    "optuna_early_stop_trials": 2000,
    "optuna_min_trials_before_stop": 500,
    # Rolling walk-forward CV objective (robustness-aware)
    "optimizer_rolling_cv_enabled": True,
    "optimizer_rolling_cv_folds": 4,
    "optimizer_rolling_cv_min_train_games": 320,
    "optimizer_rolling_cv_val_games": 160,
    "optimizer_rolling_cv_worst_fold_mult": 0.40,
    # Optional ML upset scorer promotion gate (feature-flagged)
    "optimizer_ml_underdog_scorer_enabled": False,
    "optimizer_ml_underdog_scorer_lr": 0.05,
    "optimizer_ml_underdog_scorer_l2": 1.0,
    "optimizer_ml_underdog_scorer_max_iter": 220,
    "optimizer_ml_underdog_scorer_min_train_samples": 140,
    "optimizer_ml_underdog_scorer_min_val_samples": 60,
    "optimizer_ml_underdog_scorer_min_brier_lift": 0.0025,
    # Overnight loop controls
    "overnight_max_no_save_passes": 3,  # 0 = disabled
    # Weekly frontier/drift reporting
    "weekly_frontier_report_dir": "data/reports",
    "drift_tier_a_min_hit_rate": 56.0,
    "drift_tier_b_min_hit_rate": 50.0,
    "drift_tier_c_min_hit_rate": 44.0,
    "drift_tier_min_samples": 12,
    "drift_band_2_00_2_99_min_roi": -2.0,
    "drift_band_3_00_3_99_min_roi": -4.0,
    "drift_band_4_plus_min_roi": -8.0,
    "drift_band_min_bets": 15,
    # Phase acceptance guardrails
    "phase_gate_min_winner_pct": 60.0,
    "phase_gate_min_upset_coverage_pct": 20.0,
    "phase_gate_upset_coverage_tolerance_pp": 0.5,
    "phase_gate_min_tier_a_hit_rate": 56.0,
    "phase_gate_min_ml_roi": 0.0,
    "phase_gate_max_fund_drift_alerts": 1,
}

_cache: Dict[str, Any] | None = None


def _ensure_dir():
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    """Load settings and keep app_settings.json synced with effective values."""
    global _cache
    with _settings_lock:
        if _cache is not None:
            return _cache
        _ensure_dir()
        if _SETTINGS_PATH.exists():
            try:
                with open(_SETTINGS_PATH, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}
        merged = {**_DEFAULTS, **data}
        override_keys = [
            key for key, value in data.items()
            if key in _DEFAULTS and _DEFAULTS[key] != value
        ]
        if override_keys:
            sample = ", ".join(sorted(override_keys)[:6])
            if len(override_keys) > 6:
                sample += ", ..."
            logger.info(
                "Runtime settings source is %s: %d keys override defaults (%s)",
                _SETTINGS_PATH.as_posix(),
                len(override_keys),
                sample,
            )
        # Keep a single visible source-of-truth on disk: if defaults filled in
        # any missing keys (or recovered from a corrupt/empty file), persist the
        # merged settings immediately so future edits only need one file.
        if data != merged:
            with open(_SETTINGS_PATH, "w") as f:
                json.dump(merged, f, indent=2)
        _cache = merged
        return merged


def save_settings(settings: Dict[str, Any] | None = None):
    """Persist current settings to disk."""
    global _cache
    with _settings_lock:
        if settings is not None:
            _cache = settings
        if _cache is None:
            _cache = dict(_DEFAULTS)
        _ensure_dir()
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(_cache, f, indent=2)


def get(key: str, default: Any = None) -> Any:
    env_key = f"NBA_{key.upper()}"
    env_raw = os.environ.get(env_key)
    if env_raw is not None:
        ref = _DEFAULTS.get(key, default)
        if isinstance(ref, bool):
            return env_raw.strip().lower() in ("1", "true", "yes", "on", "y")
        if isinstance(ref, int) and not isinstance(ref, bool):
            try:
                return int(env_raw)
            except ValueError:
                return ref
        if isinstance(ref, float):
            try:
                return float(env_raw)
            except ValueError:
                return ref
        if isinstance(ref, list):
            return [v.strip() for v in env_raw.split(",") if v.strip()]
        return env_raw

    s = load_settings()
    return s.get(key, default)


def set_value(key: str, value: Any):
    s = load_settings()
    if key == "upset_bonus_mult":
        try:
            upset_bonus = float(value)
        except (TypeError, ValueError):
            upset_bonus = float(_DEFAULTS["upset_bonus_mult"])
        try:
            upset_bonus_max = float(
                s.get("upset_bonus_mult_max", _DEFAULTS["upset_bonus_mult_max"])
            )
        except (TypeError, ValueError):
            upset_bonus_max = float(_DEFAULTS["upset_bonus_mult_max"])
        upset_bonus_max = max(0.0, upset_bonus_max)
        value = min(max(0.0, upset_bonus), upset_bonus_max)
    elif key == "timezone":
        from zoneinfo import ZoneInfo
        try:
            ZoneInfo(str(value))
        except (KeyError, Exception):
            value = _DEFAULTS["timezone"]
    elif key == "score_calibration_bins":
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = int(_DEFAULTS["score_calibration_bins"])
        value = max(5, min(61, value))
    elif key in (
        "score_calibration_train_window_games",
        "score_calibration_min_games",
        "score_calibration_team_min_games",
        "score_calibration_team_range_min_games",
    ):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = int(_DEFAULTS.get(key, 0))
        value = max(0, value)
    elif key == "score_calibration_near_spread_raw_weight":
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(_DEFAULTS["score_calibration_near_spread_raw_weight"])
        value = min(max(0.0, value), 1.0)
    elif key == "score_calibration_team_range_quantile_low":
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(_DEFAULTS["score_calibration_team_range_quantile_low"])
        value = min(max(0.0, value), 0.49)
    elif key == "score_calibration_team_range_quantile_high":
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(_DEFAULTS["score_calibration_team_range_quantile_high"])
        value = min(max(0.51, value), 1.0)
    elif key in (
        "score_calibration_sign_epsilon",
        "score_calibration_min_abs_spread",
        "score_calibration_near_spread_identity_band",
        "score_calibration_near_spread_deadband",
        "score_calibration_spread_cap",
        "score_calibration_total_min",
        "score_calibration_total_max",
        "score_calibration_point_floor",
        "score_calibration_point_ceiling",
        "score_calibration_tail_margin_threshold",
        "score_calibration_team_shrinkage",
        "score_calibration_team_max_abs_correction",
        "score_calibration_team_range_padding",
    ):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(_DEFAULTS.get(key, 0.0))
        if key in ("score_calibration_sign_epsilon", "score_calibration_min_abs_spread"):
            value = max(0.0, value)
        elif key in ("score_calibration_spread_cap", "score_calibration_tail_margin_threshold"):
            value = max(1.0, value)
        elif key in ("score_calibration_total_min", "score_calibration_total_max"):
            value = max(100.0, value)
        elif key in ("score_calibration_point_floor", "score_calibration_point_ceiling"):
            value = max(40.0, value)
        else:
            value = max(0.0, value)
    s[key] = value
    save_settings(s)


def get_db_path() -> str:
    return get("db_path", _DEFAULTS["db_path"])


def get_season() -> str:
    return get("season", _DEFAULTS["season"])


def get_season_year() -> str:
    return get("season_year", _DEFAULTS["season_year"])


def invalidate_cache():
    global _cache
    with _settings_lock:
        _cache = None


def get_historical_seasons() -> list:
    return get("historical_seasons", [])


def get_config() -> Dict[str, Any]:
    """Return the full settings dict (alias for load_settings)."""
    return load_settings()
