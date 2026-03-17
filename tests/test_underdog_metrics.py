import pytest

from src.analytics.underdog_metrics import (
    quality_tier_for_confidence,
    summarize_underdog_quality,
)


def test_quality_tier_for_confidence_thresholds():
    assert quality_tier_for_confidence(80.0) == "A"
    assert quality_tier_for_confidence(70.0) == "A"
    assert quality_tier_for_confidence(69.9) == "B"
    assert quality_tier_for_confidence(55.0) == "B"
    assert quality_tier_for_confidence(54.9) == "C"


def test_summarize_underdog_quality_builds_tiers_bands_and_frontier():
    samples = [
        {"confidence": 82.0, "upset_correct": True, "ml_profit": 1.4, "ml_payout": 2.4},
        {"confidence": 76.0, "upset_correct": False, "ml_profit": -1.0, "ml_payout": 3.2},
        {"confidence": 62.0, "upset_correct": True, "ml_profit": 0.8, "ml_payout": 1.8},
        {"confidence": 58.0, "upset_correct": False, "ml_profit": -1.0, "ml_payout": 2.1},
        {"confidence": 40.0, "upset_correct": True, "ml_profit": 2.5, "ml_payout": 3.5},
    ]

    summary = summarize_underdog_quality(
        samples,
        total_games=20,
        frontier_coverage_pcts=(20, 40, 60, 80, 100),
    )

    assert summary["upset_pick_count"] == 5
    assert summary["coverage_pct"] == pytest.approx(25.0)

    tiers = summary["tier_metrics"]
    assert tiers["A"]["count"] == 2.0
    assert tiers["A"]["hit_rate"] == pytest.approx(50.0)
    assert tiers["B"]["count"] == 2.0
    assert tiers["B"]["hit_rate"] == pytest.approx(50.0)
    assert tiers["C"]["count"] == 1.0
    assert tiers["C"]["hit_rate"] == pytest.approx(100.0)

    roi_bands = summary["roi_by_odds_band"]
    assert roi_bands["3_00_3_99"]["bets"] == 2
    assert roi_bands["3_00_3_99"]["ml_roi"] == pytest.approx(75.0)
    assert roi_bands["1_50_1_99"]["bets"] == 1
    assert roi_bands["1_50_1_99"]["hit_rate"] == pytest.approx(100.0)

    frontier = summary["quality_frontier"]
    assert len(frontier) == 5
    assert frontier[0]["picks"] == 1.0
    assert frontier[0]["hit_rate"] == pytest.approx(100.0)
    assert frontier[-1]["upset_coverage_pct"] == pytest.approx(100.0)
    assert frontier[-1]["hit_rate"] == pytest.approx(60.0)

    obs = summary["hit_rate_quality_observation"]
    assert "tier hit A/B/C=" in obs
    assert "frontier top" in obs


def test_summarize_underdog_quality_empty_samples():
    summary = summarize_underdog_quality([], total_games=50)
    assert summary["upset_pick_count"] == 0
    assert summary["coverage_pct"] == 0.0
    assert summary["quality_frontier"] == []
    assert summary["hit_rate_quality_observation"] == (
        "Hit/quality observation: no upset picks."
    )
