def test_cache_invalidation_functions_are_importable():
    from src.analytics.prediction import invalidate_precompute_cache, invalidate_results_cache
    from src.analytics.backtester import invalidate_backtest_cache
    from src.analytics.stats_engine import invalidate_stats_caches
    from src.analytics.weight_config import invalidate_weight_cache
    from src.data.gamecast import invalidate_actionnetwork_cache

    assert callable(invalidate_precompute_cache)
    assert callable(invalidate_results_cache)
    assert callable(invalidate_backtest_cache)
    assert callable(invalidate_stats_caches)
    assert callable(invalidate_weight_cache)
    assert callable(invalidate_actionnetwork_cache)
