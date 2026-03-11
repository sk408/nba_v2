import pytest


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_get_json_retries_transient_status(monkeypatch):
    from src.data import http_client

    calls = {"n": 0}
    responses = [
        _FakeResponse(status_code=503, payload={"error": "busy"}, text="busy"),
        _FakeResponse(status_code=200, payload={"ok": True}, text='{"ok": true}'),
    ]

    def _fake_request(**kwargs):
        idx = calls["n"]
        calls["n"] += 1
        return responses[idx]

    monkeypatch.setattr(http_client.requests, "request", _fake_request)
    monkeypatch.setattr(http_client.time, "sleep", lambda *_args, **_kwargs: None)

    data = http_client.get_json("https://example.test/api", retries=2, backoff_base=0.0)
    assert data["ok"] is True
    assert calls["n"] == 2


def test_get_json_decode_error(monkeypatch):
    from src.data import http_client

    def _fake_request(**kwargs):
        return _FakeResponse(status_code=200, payload=ValueError("bad json"), text="{")

    monkeypatch.setattr(http_client.requests, "request", _fake_request)

    with pytest.raises(http_client.HttpDecodeError):
        http_client.get_json("https://example.test/api")


def test_retry_call_retries_then_succeeds(monkeypatch):
    from src.data import http_client

    monkeypatch.setattr(http_client.time, "sleep", lambda *_args, **_kwargs: None)
    attempts = {"n": 0}

    def _flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("try again")
        return "ok"

    result = http_client.retry_call(_flaky, retries=3, backoff_base=0.0)
    assert result == "ok"
    assert attempts["n"] == 3
