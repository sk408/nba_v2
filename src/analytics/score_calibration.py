"""Post-prediction score realism calibrator.

This module is intentionally independent from WeightConfig optimization.
It learns a mapping from raw model outputs -> realistic NBA score outputs and
can be applied after predict() without changing winner picks or core weights.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.database import db
from src.utils.settings_helpers import (
    safe_bool_setting as _safe_bool_setting,
    safe_float_setting as _safe_float_setting,
    safe_int_setting as _safe_int_setting,
)

logger = logging.getLogger(__name__)

_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "score_calibration.json",
)
_CALIBRATOR_VERSION = 1

_cache_lock = threading.Lock()
_cache_payload: Optional[Dict[str, Any]] = None
_cache_signature: str = ""


def _settings_snapshot() -> Dict[str, Any]:
    """Return calibrator settings used for fit/apply."""
    return {
        "enabled": _safe_bool_setting("score_calibration_enabled", True),
        "apply_to_display": _safe_bool_setting("score_calibration_apply_to_display", True),
        "train_window_games": max(0, _safe_int_setting("score_calibration_train_window_games", 2500)),
        "min_games": max(25, _safe_int_setting("score_calibration_min_games", 500)),
        "bins": int(np.clip(_safe_int_setting("score_calibration_bins", 15), 5, 61)),
        "strict_sign_lock": _safe_bool_setting("score_calibration_strict_sign_lock", True),
        "sign_epsilon": max(0.0, _safe_float_setting("score_calibration_sign_epsilon", 0.05)),
        "min_abs_spread": max(0.0, _safe_float_setting("score_calibration_min_abs_spread", 0.10)),
        "near_spread_enabled": _safe_bool_setting("score_calibration_near_spread_enabled", False),
        "near_spread_identity_band": max(
            0.0,
            _safe_float_setting("score_calibration_near_spread_identity_band", 1.5),
        ),
        "near_spread_deadband": max(
            0.0,
            _safe_float_setting("score_calibration_near_spread_deadband", 4.0),
        ),
        "near_spread_raw_weight": float(
            np.clip(_safe_float_setting("score_calibration_near_spread_raw_weight", 0.85), 0.0, 1.0)
        ),
        "spread_cap": max(1.0, _safe_float_setting("score_calibration_spread_cap", 28.0)),
        "total_min": _safe_float_setting("score_calibration_total_min", 165.0),
        "total_max": _safe_float_setting("score_calibration_total_max", 270.0),
        "point_floor": _safe_float_setting("score_calibration_point_floor", 70.0),
        "point_ceiling": _safe_float_setting("score_calibration_point_ceiling", 170.0),
        "tail_margin_threshold": max(1.0, _safe_float_setting("score_calibration_tail_margin_threshold", 20.0)),
        "team_residual_enabled": _safe_bool_setting("score_calibration_team_residual_enabled", False),
        "team_min_games": max(5, _safe_int_setting("score_calibration_team_min_games", 30)),
        "team_shrinkage": max(0.0, _safe_float_setting("score_calibration_team_shrinkage", 50.0)),
        "team_max_abs_correction": max(
            0.0,
            _safe_float_setting("score_calibration_team_max_abs_correction", 5.0),
        ),
        "team_range_enabled": _safe_bool_setting("score_calibration_team_range_enabled", True),
        "team_range_min_games": max(5, _safe_int_setting("score_calibration_team_range_min_games", 35)),
        "team_range_quantile_low": float(
            np.clip(_safe_float_setting("score_calibration_team_range_quantile_low", 0.03), 0.0, 0.49)
        ),
        "team_range_quantile_high": float(
            np.clip(_safe_float_setting("score_calibration_team_range_quantile_high", 0.97), 0.51, 1.0)
        ),
        "team_range_padding": max(0.0, _safe_float_setting("score_calibration_team_range_padding", 6.0)),
    }


def _make_signature(payload: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _fit_piecewise_map(
    raw: np.ndarray,
    actual: np.ndarray,
    bins: int,
) -> Dict[str, List[float]]:
    """Fit a monotone piecewise map raw -> actual."""
    raw = np.asarray(raw, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    mask = np.isfinite(raw) & np.isfinite(actual)
    raw = raw[mask]
    actual = actual[mask]
    if raw.size == 0:
        return {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}

    q = np.linspace(0.0, 1.0, max(3, bins))
    edges = np.quantile(raw, q)
    edges = np.unique(edges)

    x_pts: List[float] = []
    y_pts: List[float] = []

    if edges.size >= 2:
        for idx in range(edges.size - 1):
            lo = edges[idx]
            hi = edges[idx + 1]
            if idx == edges.size - 2:
                m = (raw >= lo) & (raw <= hi)
            else:
                m = (raw >= lo) & (raw < hi)
            if int(np.sum(m)) < 3:
                continue
            x_pts.append(float(np.mean(raw[m])))
            y_pts.append(float(np.mean(actual[m])))

    if len(x_pts) < 2:
        mu_x = float(np.mean(raw))
        mu_y = float(np.mean(actual))
        var_x = float(np.var(raw))
        if var_x > 1e-9:
            cov = float(np.mean((raw - mu_x) * (actual - mu_y)))
            slope = cov / var_x
        else:
            slope = 1.0
        intercept = mu_y - slope * mu_x
        x_lo = mu_x - 1.0
        x_hi = mu_x + 1.0
        x_pts = [x_lo, x_hi]
        y_pts = [slope * x_lo + intercept, slope * x_hi + intercept]

    order = np.argsort(np.asarray(x_pts))
    x = np.asarray(x_pts)[order]
    y = np.asarray(y_pts)[order]

    x_clean: List[float] = []
    y_clean: List[float] = []
    for xi, yi in zip(x.tolist(), y.tolist()):
        if not x_clean or abs(xi - x_clean[-1]) > 1e-6:
            x_clean.append(float(xi))
            y_clean.append(float(yi))
        else:
            y_clean[-1] = float((y_clean[-1] + yi) / 2.0)

    if len(x_clean) < 2:
        x_clean = [x_clean[0] - 1.0, x_clean[0] + 1.0]
        y_clean = [y_clean[0] - 1.0, y_clean[0] + 1.0]

    y_arr = np.asarray(y_clean, dtype=np.float64)
    # enforce monotone non-decreasing map (stability + realism)
    y_arr = np.maximum.accumulate(y_arr)
    return {
        "x": [float(v) for v in x_clean],
        "y": [float(v) for v in y_arr.tolist()],
    }


def _interp_piecewise(x: float, mapping: Dict[str, Any]) -> float:
    xp = np.asarray(mapping.get("x", []), dtype=np.float64)
    yp = np.asarray(mapping.get("y", []), dtype=np.float64)
    if xp.size < 2 or yp.size < 2:
        return float(x)

    if x <= xp[0]:
        den = max(1e-6, xp[1] - xp[0])
        slope = (yp[1] - yp[0]) / den
        return float(yp[0] + slope * (x - xp[0]))
    if x >= xp[-1]:
        den = max(1e-6, xp[-1] - xp[-2])
        slope = (yp[-1] - yp[-2]) / den
        return float(yp[-1] + slope * (x - xp[-1]))
    return float(np.interp(x, xp, yp))


def _apply_sign_lock(
    raw_spread: np.ndarray,
    calibrated_spread: np.ndarray,
    strict: bool,
    eps: float,
    min_abs: float,
) -> np.ndarray:
    out = np.asarray(calibrated_spread, dtype=np.float64).copy()
    if not strict:
        return out

    pos_mask = raw_spread > eps
    neg_mask = raw_spread < -eps

    if np.any(pos_mask):
        out[pos_mask] = np.where(
            out[pos_mask] <= eps,
            np.maximum(np.abs(out[pos_mask]), min_abs),
            out[pos_mask],
        )
    if np.any(neg_mask):
        out[neg_mask] = np.where(
            out[neg_mask] >= -eps,
            -np.maximum(np.abs(out[neg_mask]), min_abs),
            out[neg_mask],
        )
    return out


def _fit_team_offsets(
    home_ids: np.ndarray,
    away_ids: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    base_home: np.ndarray,
    base_away: np.ndarray,
    min_games: int,
    shrinkage: float,
    max_abs: float,
) -> Dict[str, Dict[str, float]]:
    home_sum: Dict[int, float] = {}
    away_sum: Dict[int, float] = {}
    home_cnt: Dict[int, int] = {}
    away_cnt: Dict[int, int] = {}

    for idx in range(len(home_ids)):
        hid = int(home_ids[idx])
        aid = int(away_ids[idx])
        hr = float(actual_home[idx] - base_home[idx])
        ar = float(actual_away[idx] - base_away[idx])

        home_sum[hid] = home_sum.get(hid, 0.0) + hr
        home_cnt[hid] = home_cnt.get(hid, 0) + 1
        away_sum[aid] = away_sum.get(aid, 0.0) + ar
        away_cnt[aid] = away_cnt.get(aid, 0) + 1

    offsets: Dict[str, Dict[str, float]] = {}
    all_team_ids = set(home_sum.keys()) | set(away_sum.keys())
    for tid in all_team_ids:
        h_games = int(home_cnt.get(tid, 0))
        a_games = int(away_cnt.get(tid, 0))

        h_mean = float(home_sum.get(tid, 0.0) / max(1, h_games))
        a_mean = float(away_sum.get(tid, 0.0) / max(1, a_games))

        if h_games >= min_games:
            h_corr = h_mean * (h_games / (h_games + shrinkage))
        else:
            h_corr = 0.0
        if a_games >= min_games:
            a_corr = a_mean * (a_games / (a_games + shrinkage))
        else:
            a_corr = 0.0

        h_corr = float(np.clip(h_corr, -max_abs, max_abs))
        a_corr = float(np.clip(a_corr, -max_abs, max_abs))
        if abs(h_corr) < 1e-6 and abs(a_corr) < 1e-6:
            continue

        offsets[str(tid)] = {
            "home_pts_correction": h_corr,
            "away_pts_correction": a_corr,
            "home_games": h_games,
            "away_games": a_games,
        }
    return offsets


def _apply_team_offsets_arrays(
    home_ids: np.ndarray,
    away_ids: np.ndarray,
    home_pts: np.ndarray,
    away_pts: np.ndarray,
    team_offsets: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    out_home = np.asarray(home_pts, dtype=np.float64).copy()
    out_away = np.asarray(away_pts, dtype=np.float64).copy()
    if not team_offsets:
        return out_home, out_away

    for idx in range(len(home_ids)):
        hid = str(int(home_ids[idx]))
        aid = str(int(away_ids[idx]))
        h_off = float(team_offsets.get(hid, {}).get("home_pts_correction", 0.0))
        a_off = float(team_offsets.get(aid, {}).get("away_pts_correction", 0.0))
        out_home[idx] += h_off
        out_away[idx] += a_off
    return out_home, out_away


def _blend_near_spread(
    raw_spread: np.ndarray,
    calibrated_spread: np.ndarray,
    identity_band: float,
    deadband: float,
    raw_weight: float,
    enabled: bool = False,
) -> np.ndarray:
    """Blend toward raw spread near pick'em to avoid over-correction noise."""
    out = np.asarray(calibrated_spread, dtype=np.float64).copy()
    if not enabled:
        return out

    raw = np.asarray(raw_spread, dtype=np.float64)
    abs_raw = np.abs(raw)
    identity_band = max(0.0, float(identity_band))
    deadband = max(identity_band, float(deadband))
    raw_weight = float(np.clip(raw_weight, 0.0, 1.0))
    if raw_weight <= 0.0:
        return out

    if deadband <= identity_band + 1e-9:
        w = np.where(abs_raw <= identity_band, raw_weight, 0.0)
    else:
        taper = (deadband - abs_raw) / max(1e-6, deadband - identity_band)
        taper = np.clip(taper, 0.0, 1.0)
        w = np.where(abs_raw <= identity_band, raw_weight, raw_weight * taper)

    return (out * (1.0 - w)) + (raw * w)


def _fit_team_point_ranges(
    home_ids: np.ndarray,
    away_ids: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    min_games: int,
    q_low: float,
    q_high: float,
    padding: float,
) -> Dict[str, Dict[str, float]]:
    """Learn per-team scoring ranges from historical outcomes."""
    team_scores: Dict[int, List[float]] = {}
    for idx in range(len(home_ids)):
        hid = int(home_ids[idx])
        aid = int(away_ids[idx])
        team_scores.setdefault(hid, []).append(float(actual_home[idx]))
        team_scores.setdefault(aid, []).append(float(actual_away[idx]))

    ranges: Dict[str, Dict[str, float]] = {}
    q_low = float(np.clip(q_low, 0.0, 0.49))
    q_high = float(np.clip(q_high, 0.51, 1.0))
    pad = max(0.0, float(padding))
    for team_id, scores in team_scores.items():
        if len(scores) < int(min_games):
            continue
        arr = np.asarray(scores, dtype=np.float64)
        lo = float(np.quantile(arr, q_low) - pad)
        hi = float(np.quantile(arr, q_high) + pad)
        if hi <= lo:
            continue
        ranges[str(team_id)] = {
            "point_floor": lo,
            "point_ceiling": hi,
            "games": int(len(scores)),
        }
    return ranges


def _apply_team_point_ranges_arrays(
    home_ids: np.ndarray,
    away_ids: np.ndarray,
    home_pts: np.ndarray,
    away_pts: np.ndarray,
    team_point_ranges: Dict[str, Dict[str, float]],
    global_floor: float,
    global_ceiling: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply per-team point floor/ceiling constraints."""
    out_home = np.asarray(home_pts, dtype=np.float64).copy()
    out_away = np.asarray(away_pts, dtype=np.float64).copy()
    if not team_point_ranges:
        return (
            np.clip(out_home, global_floor, global_ceiling),
            np.clip(out_away, global_floor, global_ceiling),
        )

    for idx in range(len(home_ids)):
        hid = str(int(home_ids[idx]))
        aid = str(int(away_ids[idx]))
        h_cfg = team_point_ranges.get(hid, {})
        a_cfg = team_point_ranges.get(aid, {})
        h_floor = float(h_cfg.get("point_floor", global_floor))
        h_ceiling = float(h_cfg.get("point_ceiling", global_ceiling))
        a_floor = float(a_cfg.get("point_floor", global_floor))
        a_ceiling = float(a_cfg.get("point_ceiling", global_ceiling))
        out_home[idx] = float(np.clip(out_home[idx], h_floor, h_ceiling))
        out_away[idx] = float(np.clip(out_away[idx], a_floor, a_ceiling))
    return out_home, out_away


def _mode_key(include_sharp: bool) -> str:
    return "sharp" if include_sharp else "fundamentals"


def _fit_mode_payload(
    games: List[Any],
    include_sharp: bool,
    settings: Dict[str, Any],
    callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    from src.analytics.weight_config import get_weight_config
    from src.analytics.prediction import predict

    mode = _mode_key(include_sharp)
    w = get_weight_config()

    train_window = int(settings["train_window_games"])
    fit_games = games[-train_window:] if train_window > 0 else list(games)
    if len(fit_games) < int(settings["min_games"]):
        return {
            "mode": mode,
            "include_sharp": include_sharp,
            "spread_map": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
            "total_map": {"x": [180.0, 240.0], "y": [180.0, 240.0]},
            "team_offsets": {},
            "metrics": {
                "sample_size": len(fit_games),
                "status": "insufficient_sample",
            },
        }

    raw_spread: List[float] = []
    raw_total: List[float] = []
    actual_spread: List[float] = []
    actual_total: List[float] = []
    actual_home: List[float] = []
    actual_away: List[float] = []
    home_ids: List[int] = []
    away_ids: List[int] = []

    n_games = len(fit_games)
    log_every = max(1, n_games // 8)
    for idx, g in enumerate(fit_games):
        if is_cancelled and is_cancelled():
            break
        pred = predict(g, w, include_sharp=include_sharp)
        rs = float(pred.game_score)
        rt = float(pred.projected_home_pts + pred.projected_away_pts)
        ah = float(g.actual_home_score)
        aa = float(g.actual_away_score)

        raw_spread.append(rs)
        raw_total.append(rt)
        actual_home.append(ah)
        actual_away.append(aa)
        actual_spread.append(ah - aa)
        actual_total.append(ah + aa)
        home_ids.append(int(g.home_team_id))
        away_ids.append(int(g.away_team_id))

        if callback and ((idx + 1) % log_every == 0 or (idx + 1) == n_games):
            callback(f"[score_cal] {mode}: processed {idx + 1}/{n_games}")

    rs_arr = np.asarray(raw_spread, dtype=np.float64)
    rt_arr = np.asarray(raw_total, dtype=np.float64)
    as_arr = np.asarray(actual_spread, dtype=np.float64)
    at_arr = np.asarray(actual_total, dtype=np.float64)
    ah_arr = np.asarray(actual_home, dtype=np.float64)
    aa_arr = np.asarray(actual_away, dtype=np.float64)
    hid_arr = np.asarray(home_ids, dtype=np.int32)
    aid_arr = np.asarray(away_ids, dtype=np.int32)

    spread_map = _fit_piecewise_map(rs_arr, as_arr, int(settings["bins"]))
    total_map = _fit_piecewise_map(rt_arr, at_arr, int(settings["bins"]))

    cal_spread = np.asarray([_interp_piecewise(v, spread_map) for v in rs_arr], dtype=np.float64)
    cal_total = np.asarray([_interp_piecewise(v, total_map) for v in rt_arr], dtype=np.float64)

    cal_spread = np.clip(cal_spread, -float(settings["spread_cap"]), float(settings["spread_cap"]))
    cal_total = np.clip(cal_total, float(settings["total_min"]), float(settings["total_max"]))
    cal_spread = _apply_sign_lock(
        rs_arr,
        cal_spread,
        bool(settings["strict_sign_lock"]),
        float(settings["sign_epsilon"]),
        float(settings["min_abs_spread"]),
    )

    home_pts = (cal_total + cal_spread) / 2.0
    away_pts = (cal_total - cal_spread) / 2.0

    team_offsets: Dict[str, Dict[str, float]] = {}
    if bool(settings["team_residual_enabled"]):
        team_offsets = _fit_team_offsets(
            home_ids=hid_arr,
            away_ids=aid_arr,
            actual_home=ah_arr,
            actual_away=aa_arr,
            base_home=home_pts,
            base_away=away_pts,
            min_games=int(settings["team_min_games"]),
            shrinkage=float(settings["team_shrinkage"]),
            max_abs=float(settings["team_max_abs_correction"]),
        )
        home_pts, away_pts = _apply_team_offsets_arrays(
            home_ids=hid_arr,
            away_ids=aid_arr,
            home_pts=home_pts,
            away_pts=away_pts,
            team_offsets=team_offsets,
        )

    team_point_ranges: Dict[str, Dict[str, float]] = {}
    if bool(settings["team_range_enabled"]):
        team_point_ranges = _fit_team_point_ranges(
            home_ids=hid_arr,
            away_ids=aid_arr,
            actual_home=ah_arr,
            actual_away=aa_arr,
            min_games=int(settings["team_range_min_games"]),
            q_low=float(settings["team_range_quantile_low"]),
            q_high=float(settings["team_range_quantile_high"]),
            padding=float(settings["team_range_padding"]),
        )

    home_pts = np.clip(home_pts, float(settings["point_floor"]), float(settings["point_ceiling"]))
    away_pts = np.clip(away_pts, float(settings["point_floor"]), float(settings["point_ceiling"]))

    final_total = home_pts + away_pts
    final_spread = home_pts - away_pts
    final_total = np.clip(final_total, float(settings["total_min"]), float(settings["total_max"]))
    final_spread = np.clip(final_spread, -float(settings["spread_cap"]), float(settings["spread_cap"]))
    final_spread = _apply_sign_lock(
        rs_arr,
        final_spread,
        bool(settings["strict_sign_lock"]),
        float(settings["sign_epsilon"]),
        float(settings["min_abs_spread"]),
    )
    final_spread = _blend_near_spread(
        rs_arr,
        final_spread,
        float(settings["near_spread_identity_band"]),
        float(settings["near_spread_deadband"]),
        float(settings["near_spread_raw_weight"]),
        enabled=bool(settings["near_spread_enabled"]),
    )
    final_home = (final_total + final_spread) / 2.0
    final_away = (final_total - final_spread) / 2.0
    if team_point_ranges:
        final_home, final_away = _apply_team_point_ranges_arrays(
            home_ids=hid_arr,
            away_ids=aid_arr,
            home_pts=final_home,
            away_pts=final_away,
            team_point_ranges=team_point_ranges,
            global_floor=float(settings["point_floor"]),
            global_ceiling=float(settings["point_ceiling"]),
        )
    final_home = np.clip(final_home, float(settings["point_floor"]), float(settings["point_ceiling"]))
    final_away = np.clip(final_away, float(settings["point_floor"]), float(settings["point_ceiling"]))
    final_total = np.clip(
        final_home + final_away,
        float(settings["total_min"]),
        float(settings["total_max"]),
    )
    final_spread = np.clip(
        final_home - final_away,
        -float(settings["spread_cap"]),
        float(settings["spread_cap"]),
    )
    final_spread = _apply_sign_lock(
        rs_arr,
        final_spread,
        bool(settings["strict_sign_lock"]),
        float(settings["sign_epsilon"]),
        float(settings["min_abs_spread"]),
    )
    final_home = (final_total + final_spread) / 2.0
    final_away = (final_total - final_spread) / 2.0

    raw_spread_mae = float(np.mean(np.abs(rs_arr - as_arr)))
    cal_spread_mae = float(np.mean(np.abs(final_spread - as_arr)))
    raw_total_mae = float(np.mean(np.abs(rt_arr - at_arr)))
    cal_total_mae = float(np.mean(np.abs(final_total - at_arr)))

    tail = float(settings["tail_margin_threshold"])
    raw_tail_rate = float(np.mean(np.abs(rs_arr) >= tail) * 100.0)
    cal_tail_rate = float(np.mean(np.abs(final_spread) >= tail) * 100.0)

    eps = float(settings["sign_epsilon"])
    stable_mask = np.abs(rs_arr) > eps
    if np.any(stable_mask):
        flips = int(np.sum(np.sign(rs_arr[stable_mask]) != np.sign(final_spread[stable_mask])))
        winner_invariance_pct = float((1.0 - (flips / max(1, int(np.sum(stable_mask))))) * 100.0)
    else:
        flips = 0
        winner_invariance_pct = 100.0

    return {
        "mode": mode,
        "include_sharp": include_sharp,
        "spread_map": spread_map,
        "total_map": total_map,
        "team_offsets": team_offsets,
        "team_point_ranges": team_point_ranges,
        "metrics": {
            "status": "ok",
            "sample_size": int(len(rs_arr)),
            "raw_spread_mae": raw_spread_mae,
            "cal_spread_mae": cal_spread_mae,
            "raw_total_mae": raw_total_mae,
            "cal_total_mae": cal_total_mae,
            "raw_tail_margin_rate": raw_tail_rate,
            "cal_tail_margin_rate": cal_tail_rate,
            "winner_flip_count": flips,
            "winner_invariance_pct": winner_invariance_pct,
            "team_offsets_count": int(len(team_offsets)),
        },
    }


def invalidate_score_calibration_cache():
    global _cache_payload, _cache_signature
    with _cache_lock:
        _cache_payload = None
        _cache_signature = ""


def load_score_calibration(force_reload: bool = False) -> Optional[Dict[str, Any]]:
    """Load calibration artifact from disk (cached)."""
    global _cache_payload, _cache_signature
    with _cache_lock:
        if not force_reload and _cache_payload is not None:
            return _cache_payload

        if not os.path.exists(_CALIBRATION_PATH):
            _cache_payload = None
            _cache_signature = ""
            return None
        try:
            with open(_CALIBRATION_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload_sig = str(payload.get("artifact_signature", ""))
            if not payload_sig:
                payload_sig = _make_signature(payload)
                payload["artifact_signature"] = payload_sig
            _cache_payload = payload
            _cache_signature = payload_sig
            return payload
        except Exception as e:
            logger.warning("Failed loading score calibration: %s", e)
            _cache_payload = None
            _cache_signature = ""
            return None


def get_score_calibration_signature() -> str:
    """Return stable signature for cache keys and diagnostics."""
    if _cache_signature:
        return _cache_signature
    payload = load_score_calibration(force_reload=False)
    if payload is None:
        return ""
    return str(payload.get("artifact_signature", ""))


def _save_team_tuning_for_mode(mode_payload: Dict[str, Any]):
    """Optionally persist team offsets into team_tuning for visibility."""
    offsets = mode_payload.get("team_offsets", {})
    metrics = mode_payload.get("metrics", {})
    if not offsets:
        return

    mode = str(mode_payload.get("mode", "fundamentals"))
    tuning_mode = f"score_realism_{mode}"
    tuning_version = f"v{_CALIBRATOR_VERSION}_{mode}"
    now_iso = datetime.now().isoformat(timespec="seconds")
    avg_spread_error = float(metrics.get("raw_spread_mae", 0.0))
    avg_total_error = float(metrics.get("raw_total_mae", 0.0))

    for tid_str, vals in offsets.items():
        try:
            team_id = int(tid_str)
        except (TypeError, ValueError):
            continue
        h_corr = float(vals.get("home_pts_correction", 0.0))
        a_corr = float(vals.get("away_pts_correction", 0.0))
        h_games = int(vals.get("home_games", 0))
        a_games = int(vals.get("away_games", 0))
        n_games = h_games + a_games

        db.execute(
            """
            INSERT INTO team_tuning
            (team_id, home_pts_correction, away_pts_correction, games_analyzed,
             avg_spread_error_before, avg_total_error_before, last_tuned_at,
             tuning_mode, tuning_version, tuning_sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                home_pts_correction = excluded.home_pts_correction,
                away_pts_correction = excluded.away_pts_correction,
                games_analyzed = excluded.games_analyzed,
                avg_spread_error_before = excluded.avg_spread_error_before,
                avg_total_error_before = excluded.avg_total_error_before,
                last_tuned_at = excluded.last_tuned_at,
                tuning_mode = excluded.tuning_mode,
                tuning_version = excluded.tuning_version,
                tuning_sample_size = excluded.tuning_sample_size
            """,
            (
                team_id,
                h_corr,
                a_corr,
                n_games,
                avg_spread_error,
                avg_total_error,
                now_iso,
                tuning_mode,
                tuning_version,
                n_games,
            ),
        )


def optimize_score_realism(
    games: Optional[List[Any]] = None,
    callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Train and persist score realism calibration (separate from weight optimizer)."""
    from src.analytics.prediction import precompute_all_games
    from src.config import get as get_setting

    settings = _settings_snapshot()
    if not settings["enabled"]:
        return {"status": "disabled", "saved": False}

    if games is None:
        games = precompute_all_games(callback=callback)
    if not games:
        return {"status": "no_games", "saved": False}

    sorted_games = sorted(games, key=lambda g: g.game_date)
    if len(sorted_games) < int(settings["min_games"]):
        return {
            "status": "insufficient_sample",
            "saved": False,
            "sample_size": len(sorted_games),
            "min_games": int(settings["min_games"]),
        }

    if callback:
        callback(f"[score_cal] training on {len(sorted_games)} games...")

    fund_payload = _fit_mode_payload(
        sorted_games,
        include_sharp=False,
        settings=settings,
        callback=callback,
        is_cancelled=is_cancelled,
    )
    if is_cancelled and is_cancelled():
        return {"status": "cancelled", "saved": False}

    sharp_payload = _fit_mode_payload(
        sorted_games,
        include_sharp=True,
        settings=settings,
        callback=callback,
        is_cancelled=is_cancelled,
    )
    if is_cancelled and is_cancelled():
        return {"status": "cancelled", "saved": False}

    payload: Dict[str, Any] = {
        "version": _CALIBRATOR_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "sample_size": len(sorted_games),
        "modes": {
            "fundamentals": fund_payload,
            "sharp": sharp_payload,
        },
    }
    payload["artifact_signature"] = _make_signature(
        {"version": payload["version"], "settings": payload["settings"], "modes": payload["modes"]}
    )

    prior = load_score_calibration(force_reload=True)
    prior_sig = str(prior.get("artifact_signature", "")) if prior else ""
    unchanged = prior_sig == payload["artifact_signature"]
    if unchanged:
        return {
            "status": "unchanged",
            "saved": False,
            "artifact_signature": payload["artifact_signature"],
            "fundamentals_metrics": fund_payload.get("metrics", {}),
            "sharp_metrics": sharp_payload.get("metrics", {}),
        }

    os.makedirs(os.path.dirname(_CALIBRATION_PATH) or ".", exist_ok=True)
    with open(_CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Optional visibility in team_tuning for whichever prediction mode is active.
    active_mode = str(get_setting("prediction_mode", "fundamentals"))
    selected = "sharp" if active_mode == "fundamentals_sharp" else "fundamentals"
    _save_team_tuning_for_mode(payload["modes"].get(selected, {}))

    invalidate_score_calibration_cache()
    load_score_calibration(force_reload=True)

    if callback:
        f_m = fund_payload.get("metrics", {})
        s_m = sharp_payload.get("metrics", {})
        callback(
            "[score_cal] saved "
            f"(fund spread MAE {f_m.get('raw_spread_mae', 0.0):.3f}->{f_m.get('cal_spread_mae', 0.0):.3f}, "
            f"sharp spread MAE {s_m.get('raw_spread_mae', 0.0):.3f}->{s_m.get('cal_spread_mae', 0.0):.3f})"
        )

    return {
        "status": "saved",
        "saved": True,
        "artifact_path": _CALIBRATION_PATH,
        "artifact_signature": payload["artifact_signature"],
        "fundamentals_metrics": fund_payload.get("metrics", {}),
        "sharp_metrics": sharp_payload.get("metrics", {}),
    }


def apply_score_calibration(pred: Any, game: Any, include_sharp: bool = False):
    """Apply trained score calibration to a Prediction-like object.

    This mutates only calibrated_* fields and never changes pick/game_score.
    """
    raw_spread = float(getattr(pred, "game_score", 0.0))
    raw_home = float(getattr(pred, "projected_home_pts", 0.0))
    raw_away = float(getattr(pred, "projected_away_pts", 0.0))
    raw_total = raw_home + raw_away

    setattr(pred, "calibrated_spread", raw_spread)
    setattr(pred, "calibrated_total", raw_total)
    setattr(pred, "calibrated_home_pts", raw_home)
    setattr(pred, "calibrated_away_pts", raw_away)
    setattr(pred, "score_calibrated", False)
    setattr(pred, "score_calibration_mode", _mode_key(include_sharp))

    settings = _settings_snapshot()
    if not settings["enabled"] or not settings["apply_to_display"]:
        return

    payload = load_score_calibration(force_reload=False)
    if not payload:
        return

    mode_key = _mode_key(include_sharp)
    mode_payload = payload.get("modes", {}).get(mode_key)
    if not mode_payload:
        mode_payload = payload.get("modes", {}).get("fundamentals")
    if not mode_payload:
        return

    spread_map = mode_payload.get("spread_map", {})
    total_map = mode_payload.get("total_map", {})
    team_offsets = mode_payload.get("team_offsets", {})
    team_point_ranges = mode_payload.get("team_point_ranges", {})

    cal_spread = _interp_piecewise(raw_spread, spread_map)
    cal_total = _interp_piecewise(raw_total, total_map)
    cal_spread = float(np.clip(cal_spread, -settings["spread_cap"], settings["spread_cap"]))
    cal_total = float(np.clip(cal_total, settings["total_min"], settings["total_max"]))
    cal_spread = float(
        _apply_sign_lock(
            np.asarray([raw_spread], dtype=np.float64),
            np.asarray([cal_spread], dtype=np.float64),
            bool(settings["strict_sign_lock"]),
            float(settings["sign_epsilon"]),
            float(settings["min_abs_spread"]),
        )[0]
    )
    cal_spread = float(
        _blend_near_spread(
            np.asarray([raw_spread], dtype=np.float64),
            np.asarray([cal_spread], dtype=np.float64),
            float(settings["near_spread_identity_band"]),
            float(settings["near_spread_deadband"]),
            float(settings["near_spread_raw_weight"]),
            enabled=bool(settings["near_spread_enabled"]),
        )[0]
    )

    home_pts = (cal_total + cal_spread) / 2.0
    away_pts = (cal_total - cal_spread) / 2.0

    if team_offsets:
        hid = str(int(getattr(game, "home_team_id", 0)))
        aid = str(int(getattr(game, "away_team_id", 0)))
        home_pts += float(team_offsets.get(hid, {}).get("home_pts_correction", 0.0))
        away_pts += float(team_offsets.get(aid, {}).get("away_pts_correction", 0.0))

    if team_point_ranges:
        adj_home, adj_away = _apply_team_point_ranges_arrays(
            home_ids=np.asarray([int(getattr(game, "home_team_id", 0))], dtype=np.int32),
            away_ids=np.asarray([int(getattr(game, "away_team_id", 0))], dtype=np.int32),
            home_pts=np.asarray([home_pts], dtype=np.float64),
            away_pts=np.asarray([away_pts], dtype=np.float64),
            team_point_ranges=team_point_ranges,
            global_floor=float(settings["point_floor"]),
            global_ceiling=float(settings["point_ceiling"]),
        )
        home_pts = float(adj_home[0])
        away_pts = float(adj_away[0])
    else:
        home_pts = float(np.clip(home_pts, settings["point_floor"], settings["point_ceiling"]))
        away_pts = float(np.clip(away_pts, settings["point_floor"], settings["point_ceiling"]))
    cal_total = float(np.clip(home_pts + away_pts, settings["total_min"], settings["total_max"]))
    cal_spread = float(np.clip(home_pts - away_pts, -settings["spread_cap"], settings["spread_cap"]))
    cal_spread = float(
        _apply_sign_lock(
            np.asarray([raw_spread], dtype=np.float64),
            np.asarray([cal_spread], dtype=np.float64),
            bool(settings["strict_sign_lock"]),
            float(settings["sign_epsilon"]),
            float(settings["min_abs_spread"]),
        )[0]
    )
    home_pts = (cal_total + cal_spread) / 2.0
    away_pts = (cal_total - cal_spread) / 2.0

    setattr(pred, "calibrated_spread", float(cal_spread))
    setattr(pred, "calibrated_total", float(cal_total))
    setattr(pred, "calibrated_home_pts", float(home_pts))
    setattr(pred, "calibrated_away_pts", float(away_pts))
    setattr(pred, "score_calibrated", True)
    setattr(pred, "score_calibration_mode", mode_key)
