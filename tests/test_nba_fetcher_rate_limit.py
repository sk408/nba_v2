import pytest


def test_pace_nba_api_request_enforces_min_interval(monkeypatch):
    from src.data import nba_fetcher

    monkeypatch.setattr(nba_fetcher, "_next_nba_api_request_at", 0.0)
    monkeypatch.setattr(nba_fetcher, "_nba_api_min_interval_seconds", lambda: 0.5)

    monotonic_values = iter([10.0, 10.1])
    sleep_calls = []
    monkeypatch.setattr(nba_fetcher.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(nba_fetcher.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    nba_fetcher._pace_nba_api_request()
    nba_fetcher._pace_nba_api_request()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(0.4, rel=1e-6, abs=1e-6)


def test_safe_get_applies_pacing_on_each_retry_attempt(monkeypatch):
    from src.data import nba_fetcher

    pace_calls = []
    monkeypatch.setattr(nba_fetcher, "_pace_nba_api_request", lambda: pace_calls.append(1))

    def fake_retry_call(callable_fn, *args, **kwargs):
        try:
            callable_fn(*args)
        except RuntimeError:
            pass
        return callable_fn(*args)

    monkeypatch.setattr(nba_fetcher, "retry_call", fake_retry_call)

    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")
        return "ok"

    result = nba_fetcher._safe_get(flaky_call, retries=2)
    assert result == "ok"
    assert len(pace_calls) == 2


def test_fetch_player_on_off_uses_tuned_timeout_retry_and_status(monkeypatch):
    from src.data import nba_fetcher

    class _Frame:
        def __init__(self, rows):
            self._rows = rows

        def to_dict(self, orient):
            assert orient == "records"
            return list(self._rows)

    class _Result:
        def get_data_frames(self):
            return [
                _Frame([]),
                _Frame([{"VS_PLAYER_ID": 1, "NET_RATING": 2.0}]),
                _Frame([{"VS_PLAYER_ID": 1, "NET_RATING": -1.0}]),
            ]

    captured = {}

    def fake_safe_get(func, *args, **kwargs):
        captured.update(kwargs)
        return _Result()

    def fake_get_setting(key, default=None):
        if key == "nba_api_on_off_timeout_seconds":
            return 11.0
        if key == "nba_api_on_off_retries":
            return 2
        return default

    monkeypatch.setattr(nba_fetcher, "_safe_get", fake_safe_get)
    monkeypatch.setattr(nba_fetcher, "get_setting", fake_get_setting)
    monkeypatch.setattr(nba_fetcher, "get_season", lambda: "2025-26")

    out = nba_fetcher.fetch_player_on_off(1610612747)
    assert out["_ok"] is True
    assert len(out["on"]) == 1
    assert len(out["off"]) == 1
    assert captured["timeout"] == 11.0
    assert captured["retries"] == 2
    assert "team_id=1610612747" in captured["log_label"]


def test_fetch_player_on_off_marks_failure_when_safe_get_returns_none(monkeypatch):
    from src.data import nba_fetcher

    monkeypatch.setattr(nba_fetcher, "_safe_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(nba_fetcher, "get_season", lambda: "2025-26")

    out = nba_fetcher.fetch_player_on_off(1610612738)
    assert out["_ok"] is False
    assert out["on"] == []
    assert out["off"] == []
