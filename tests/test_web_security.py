from src.web.app import app


def test_security_headers_are_set():
    with app.test_client() as client:
        resp = client.get("/gamecast")
    assert resp.status_code == 200
    assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" in resp.headers


def test_predict_requires_csrf_token():
    with app.test_client() as client:
        resp = client.post(
            "/api/predict",
            json={"home_id": 1, "away_id": 2, "date": "2025-03-01"},
        )
    assert resp.status_code == 403


def test_predict_requires_json_content_type_even_with_csrf():
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["_csrf_token"] = "token-123"
        resp = client.post(
            "/api/predict",
            data="{}",
            content_type="text/plain",
            headers={"X-CSRF-Token": "token-123"},
        )
    assert resp.status_code == 415


def test_odds_today_sync_requires_csrf_token():
    with app.test_client() as client:
        resp = client.post("/api/sync/odds-today")
    assert resp.status_code == 403


def test_api_responses_are_not_cacheable():
    with app.test_client() as client:
        resp = client.get("/api/sync/status")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "no-store"


def test_static_assets_have_cache_header():
    with app.test_client() as client:
        resp = client.get("/static/style.css")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "public, max-age=3600"
