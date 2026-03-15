import src.web.app as web_app
from src.utils.timezone_utils import nba_today
from src.web.app import app


def _sample_predictions():
    return [
        {
            "home_team_id": 1,
            "away_team_id": 2,
            "home_team": "LAL",
            "away_team": "BOS",
            "home_name": "Lakers",
            "away_name": "Celtics",
            "game_date": "2025-03-01",
            "pick": "AWAY",
            "confidence": 74.0,
            "is_dog_pick": True,
            "is_value_zone": True,
            "dog_payout": 2.5,
            "game_score": -5.2,
            "vegas_spread": -6.5,
            "ml_sharp_home_public": 62,
            "ml_sharp_home_money": 47,
            "sharp_agrees": False,
            "adjustments": {
                "fatigue": -1.4,
                "turnover": -0.8,
                "home_court": 0.5,
                "pace": -0.3,
            },
            "start_utc": "2025-03-01T00:30Z",
        },
        {
            "home_team_id": 3,
            "away_team_id": 4,
            "home_team": "MIA",
            "away_team": "NYK",
            "home_name": "Heat",
            "away_name": "Knicks",
            "game_date": "2025-03-01",
            "pick": "HOME",
            "confidence": 58.0,
            "is_dog_pick": True,
            "is_value_zone": False,
            "dog_payout": 1.9,
            "game_score": 2.7,
            "vegas_spread": 4.0,
            "ml_sharp_home_public": 49,
            "ml_sharp_home_money": 51,
            "sharp_agrees": True,
            "adjustments": {"rebound": 0.4},
            "start_utc": "2025-03-01T02:00Z",
        },
        {
            "home_team_id": 5,
            "away_team_id": 6,
            "home_team": "DEN",
            "away_team": "SAC",
            "home_name": "Nuggets",
            "away_name": "Kings",
            "game_date": "2025-03-01",
            "pick": "HOME",
            "confidence": 61.0,
            "is_dog_pick": False,
            "is_value_zone": False,
            "dog_payout": 0.0,
            "game_score": 3.0,
            "adjustments": {},
            "start_utc": "2025-03-01T03:00Z",
        },
    ]


def test_api_underdogs_filters_and_uses_include_sharp(monkeypatch):
    call_args = {}

    def _fake_get_games(game_date, include_sharp=False):
        call_args["game_date"] = game_date
        call_args["include_sharp"] = include_sharp
        return _sample_predictions()

    monkeypatch.setattr(web_app, "_get_games_for_date", _fake_get_games)

    with app.test_client() as client:
        resp = client.get(
            "/api/underdogs"
            "?date=2025-03-01"
            "&tier=A"
            "&min_confidence=65"
            "&min_payout=2.0"
            "&include_sharp=1"
            "&limit=10"
        )

    assert resp.status_code == 200
    assert call_args["game_date"] == "2025-03-01"
    assert call_args["include_sharp"] is True
    payload = resp.get_json()
    assert payload["count"] == 1
    assert payload["total_candidates"] == 2
    top = payload["underdogs"][0]
    assert top["tier"] == "A"
    assert top["is_value_zone"] is True
    assert top["rank"] == 1
    assert "why_pick" in top
    assert top["why_pick"]["drivers"]
    assert top["why_pick"]["summary"]
    assert top["caution_flags"]
    assert any(flag["code"] == "SHARP_DISAGREEMENT" for flag in top["caution_flags"])
    assert payload["alerts"]
    assert payload["alert_digest"]["alert_count"] >= 1


def test_api_underdogs_invalid_date_falls_back_to_today(monkeypatch):
    call_args = {}

    def _fake_get_games(game_date, include_sharp=False):
        call_args["game_date"] = game_date
        return []

    monkeypatch.setattr(web_app, "_get_games_for_date", _fake_get_games)

    with app.test_client() as client:
        resp = client.get("/api/underdogs?date=not-a-date")

    assert resp.status_code == 200
    assert call_args["game_date"] == nba_today()


def test_api_underdogs_honors_sorting(monkeypatch):
    monkeypatch.setattr(web_app, "_get_games_for_date", lambda *_args, **_kwargs: _sample_predictions())

    with app.test_client() as client:
        resp = client.get(
            "/api/underdogs"
            "?date=2025-03-01"
            "&tier=ALL"
            "&sort_by=confidence"
            "&sort_dir=asc"
            "&limit=10"
        )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["count"] == 2
    underdogs = payload["underdogs"]
    assert underdogs[0]["confidence"] <= underdogs[1]["confidence"]
    assert payload["summary"]["count"] == 2
    assert "tier_counts" in payload["summary"]


def test_underdogs_page_renders_results(monkeypatch):
    monkeypatch.setattr(web_app, "_get_games_for_date", lambda *_args, **_kwargs: _sample_predictions())

    with app.test_client() as client:
        resp = client.get("/underdogs?date=2025-03-01&tier=ALL")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Underdog Workbench" in html
    assert "High Quality" in html
    assert "VALUE UPSET" in html
    assert "Lakers" in html
    assert "Why this pick?" in html
    assert "Sharp money disagrees" in html
    assert "CSV" in html
    assert "Alert Preview" in html


def test_api_underdogs_is_not_cacheable(monkeypatch):
    monkeypatch.setattr(web_app, "_get_games_for_date", lambda *_args, **_kwargs: [])
    with app.test_client() as client:
        resp = client.get("/api/underdogs")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "no-store"


def test_api_underdogs_csv_export(monkeypatch):
    monkeypatch.setattr(web_app, "_get_games_for_date", lambda *_args, **_kwargs: _sample_predictions())
    with app.test_client() as client:
        resp = client.get("/api/underdogs/export.csv?date=2025-03-01&tier=ALL")

    assert resp.status_code == 200
    assert resp.headers.get("Content-Type", "").startswith("text/csv")
    disposition = resp.headers.get("Content-Disposition", "")
    assert "underdogs_2025-03-01.csv" in disposition
    body = resp.get_data(as_text=True)
    assert "home_team,away_team,pick" in body
    assert "LAL,BOS,AWAY" in body


def test_api_underdogs_alerts_endpoint(monkeypatch):
    monkeypatch.setattr(web_app, "_get_games_for_date", lambda *_args, **_kwargs: _sample_predictions())
    with app.test_client() as client:
        resp = client.get("/api/underdogs/alerts?date=2025-03-01&tier=ALL")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["date"] == "2025-03-01"
    assert "alerts" in payload
    assert "alert_digest" in payload
    assert "scope_key" in payload
    if payload["alerts"]:
        assert "signal_key" in payload["alerts"][0]


def test_api_underdogs_digest_endpoint(monkeypatch):
    monkeypatch.setattr(web_app, "_get_games_for_date", lambda *_args, **_kwargs: _sample_predictions())
    with app.test_client() as client:
        resp = client.get("/api/underdogs/digest?date=2025-03-01&tier=ALL")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["date"] == "2025-03-01"
    assert "digest_text" in payload
    assert "Top underdogs" in payload["digest_text"]


def test_api_underdogs_alert_dispatch_is_idempotent(monkeypatch, isolated_db):
    monkeypatch.setattr(web_app, "_get_games_for_date", lambda *_args, **_kwargs: _sample_predictions())
    created = []

    def _fake_create_notification(category, severity, title, message, data=None):
        created.append(
            {
                "category": category,
                "severity": severity,
                "title": title,
                "message": message,
                "data": data,
            }
        )
        return len(created)

    monkeypatch.setattr(web_app, "create_notification", _fake_create_notification)

    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["_csrf_token"] = "dispatch-token"

        resp_one = client.post(
            "/api/underdogs/alerts/dispatch?date=2025-03-01&tier=ALL",
            headers={"X-CSRF-Token": "dispatch-token"},
        )
        assert resp_one.status_code == 200
        payload_one = resp_one.get_json()
        assert payload_one["state"]["new_count"] >= 1
        assert payload_one["snapshot"]["stored_count"] == payload_one["count"]
        assert len(created) == payload_one["state"]["new_count"]

        resp_two = client.post(
            "/api/underdogs/alerts/dispatch?date=2025-03-01&tier=ALL",
            headers={"X-CSRF-Token": "dispatch-token"},
        )
        assert resp_two.status_code == 200
        payload_two = resp_two.get_json()
        assert payload_two["state"]["new_count"] == 0
        assert payload_two["state"]["resolved_count"] == 0
        assert len(created) == payload_one["state"]["new_count"]
