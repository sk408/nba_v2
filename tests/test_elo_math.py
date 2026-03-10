from src.analytics import elo


def test_elo_expected_is_monotonic_and_symmetric():
    high_vs_low = elo._expected(1600.0, 1500.0)
    low_vs_high = elo._expected(1500.0, 1600.0)

    assert high_vs_low > 0.5
    assert low_vs_high < 0.5
    assert abs((high_vs_low + low_vs_high) - 1.0) < 1e-9
