from src.analytics.optimizer import (
    _allocate_blocked_stage_trials,
    _build_trust_region_ranges,
    _ranges_signature_blob,
    _select_optimizer_ranges,
)
from src.analytics.weight_config import WeightConfig


def test_select_optimizer_ranges_excludes_legacy_four_factors_scale():
    ranges, _ = _select_optimizer_ranges(include_sharp=False)
    assert "four_factors_scale" not in ranges
    assert "onoff_reliability_lambda" in ranges


def test_allocate_blocked_stage_trials_sums_to_total():
    total = 3000
    alloc = _allocate_blocked_stage_trials(total)
    assert set(alloc.keys()) == {"core", "ff", "onoff", "joint_refine"}
    assert sum(alloc.values()) == total
    assert alloc["joint_refine"] > 0


def test_build_trust_region_ranges_respects_global_bounds():
    center = WeightConfig()
    base_ranges = {
        "turnover_margin_mult": (0.0, 10.0),
        "onoff_impact_mult": (0.0, 3.0),
        "onoff_reliability_lambda": (0.0, 2.5),
    }
    trust = _build_trust_region_ranges(center, base_ranges, radius_fraction=0.2)

    for key, (lo, hi) in trust.items():
        base_lo, base_hi = base_ranges[key]
        center_value = float(center.to_dict()[key])
        assert base_lo <= lo <= hi <= base_hi
        assert lo <= center_value <= hi


def test_ranges_signature_changes_when_bounds_change():
    base = {
        "a": (0.0, 1.0),
        "b": (2.0, 3.0),
    }
    shifted = {
        "a": (0.0, 1.1),
        "b": (2.0, 3.0),
    }
    sig_base = _ranges_signature_blob(base)
    sig_shifted = _ranges_signature_blob(shifted)
    assert sig_base != sig_shifted
