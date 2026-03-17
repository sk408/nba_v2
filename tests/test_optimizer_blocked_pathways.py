from src.analytics.optimizer import (
    _allocate_blocked_stage_trials,
    _blocked_candidate_snapshot,
    _build_trust_region_ranges,
    _get_stage_champion_entries,
    _persist_stage_champion_candidate,
    _rank_blocked_candidates,
    _ranges_signature_blob,
    _select_optimizer_ranges,
)
from src.analytics import optimizer as optimizer_module
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


def test_blocked_candidate_snapshot_extracts_metrics():
    stage_result = {
        "best_weights": WeightConfig().to_dict(),
        "objective_selected_loss": 41.25,
        "best_loss": 42.5,
        "best_winner_pct": 61.3,
    }
    snap = _blocked_candidate_snapshot(
        stage_name="core",
        stage_id="fund_cycle_core",
        trials=900,
        stage_result=stage_result,
    )
    assert snap is not None
    assert snap["stage"] == "core"
    assert snap["stage_id"] == "fund_cycle_core"
    assert snap["trials"] == 900
    assert snap["objective_selected_loss"] == 41.25
    assert snap["best_loss"] == 42.5
    assert snap["best_winner_pct"] == 61.3


def test_rank_blocked_candidates_uses_objective_then_val_then_winner():
    ranked = _rank_blocked_candidates(
        [
            {
                "stage": "ff",
                "objective_selected_loss": 10.0,
                "best_loss": 8.1,
                "best_winner_pct": 60.0,
            },
            {
                "stage": "core",
                "objective_selected_loss": 10.0,
                "best_loss": 8.1,
                "best_winner_pct": 61.0,
            },
            {
                "stage": "joint_refine",
                "objective_selected_loss": 9.9,
                "best_loss": 8.5,
                "best_winner_pct": 59.0,
            },
        ]
    )
    assert [c["stage"] for c in ranked] == ["joint_refine", "core", "ff"]
    assert [c["rank"] for c in ranked] == [1, 2, 3]


def test_persist_stage_champion_candidate_keeps_top_k(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "_stage_champion_bank_path",
        str(tmp_path / "stage_champion_bank.json"),
    )
    top_k = 2
    base = WeightConfig().to_dict()

    for idx, objective in enumerate([9.0, 8.5, 9.5], start=1):
        payload = {
            "stage_id": f"fund_core_{idx}",
            "trials": 100 * idx,
            "objective_selected_loss": objective,
            "best_loss": objective + 0.2,
            "best_winner_pct": 60.0 + idx,
            "best_weights": {**base, "turnover_margin_mult": 2.0 + idx * 0.01},
        }
        _persist_stage_champion_candidate(
            mode_key="fundamentals",
            stage_name="core",
            candidate=payload,
            top_k=top_k,
        )

    champions = _get_stage_champion_entries("fundamentals", "core", 10)
    assert len(champions) == top_k
    assert champions[0]["objective_selected_loss"] <= champions[1]["objective_selected_loss"]
    assert champions[0]["objective_selected_loss"] == 8.5
    assert champions[1]["objective_selected_loss"] == 9.0


def test_persist_stage_champion_candidate_updates_existing_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "_stage_champion_bank_path",
        str(tmp_path / "stage_champion_bank.json"),
    )
    weights = WeightConfig().to_dict()
    first = {
        "stage_id": "fund_ff_1",
        "trials": 200,
        "objective_selected_loss": 9.0,
        "best_loss": 9.2,
        "best_winner_pct": 60.0,
        "best_weights": weights,
    }
    improved = {
        "stage_id": "fund_ff_2",
        "trials": 240,
        "objective_selected_loss": 8.7,
        "best_loss": 8.9,
        "best_winner_pct": 60.5,
        "best_weights": weights,
    }
    _persist_stage_champion_candidate(
        mode_key="fundamentals",
        stage_name="ff",
        candidate=first,
        top_k=100,
    )
    _persist_stage_champion_candidate(
        mode_key="fundamentals",
        stage_name="ff",
        candidate=improved,
        top_k=100,
    )
    champions = _get_stage_champion_entries("fundamentals", "ff", 10)
    assert len(champions) == 1
    assert champions[0]["objective_selected_loss"] == 8.7
    assert champions[0]["stage_id"] == "fund_ff_2"
