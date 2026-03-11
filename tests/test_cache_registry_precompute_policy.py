def _event_cache_names(event: str) -> list[str]:
    from src.analytics.cache_registry import _invalidator_specs

    return [name for name, _ in _invalidator_specs().get(event, [])]


def test_precompute_not_invalidated_on_sync_events():
    post_sync = _event_cache_names("post_sync")
    post_odds_sync = _event_cache_names("post_odds_sync")

    assert "precompute" not in post_sync
    assert "precompute" not in post_odds_sync


def test_precompute_still_invalidated_on_nuke():
    post_nuke = _event_cache_names("post_nuke")

    assert "precompute" in post_nuke
