"""ESPN integration: odds, play-by-play, box score, predictor, WebSocket."""

import logging
import json
import threading
from typing import Dict, Any, Optional, List

from src.data.http_client import get_json, HttpClientError

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
}

from src.utils.team_mapper import normalize_espn_abbr, normalize_action_abbr


def _extract_total_record(competitor: Dict[str, Any]) -> str:
    """Return overall team record from ESPN competitor record list."""
    records = competitor.get("records", [])
    if not isinstance(records, list):
        return ""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("type") == "total":
            return str(rec.get("summary", "") or rec.get("displayValue", "")).strip()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("name", "")).lower() == "overall":
            return str(rec.get("summary", "") or rec.get("displayValue", "")).strip()
    return ""


def fetch_espn_scoreboard() -> List[Dict[str, Any]]:
    """Fetch today's ESPN scoreboard with retry on transient errors."""
    def _on_retry(attempt: int, total: int, exc: Exception):
        logger.warning("ESPN scoreboard retry %d/%d: %s", attempt, total, exc)

    try:
        data = get_json(
            ESPN_SCOREBOARD_URL,
            headers=_HEADERS,
            timeout=10,
            retries=3,
            backoff_base=1.0,
            on_retry=_on_retry,
        )
        games = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            games.append({
                "espn_id": event.get("id", ""),
                "name": event.get("name", ""),
                "date": event.get("date", ""),
                "status": event.get("status", {}).get("type", {}).get("description", ""),
                "short_detail": event.get("status", {}).get("type", {}).get("shortDetail", ""),
                "period": event.get("status", {}).get("period", 0),
                "clock": event.get("status", {}).get("displayClock", ""),
                "state": event.get("status", {}).get("type", {}).get("state", ""),
                "home_team": normalize_espn_abbr(home.get("team", {}).get("abbreviation", "")),
                "away_team": normalize_espn_abbr(away.get("team", {}).get("abbreviation", "")),
                "home_team_id": home.get("team", {}).get("id", ""),
                "away_team_id": away.get("team", {}).get("id", ""),
                "home_record": _extract_total_record(home),
                "away_record": _extract_total_record(away),
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
            })
        return games
    except HttpClientError as e:
        logger.error("ESPN scoreboard error after retries: %s", e)
        logger.debug("ESPN scoreboard stacktrace", exc_info=True)
        return []


def fetch_espn_game_summary(game_id: str) -> Dict[str, Any]:
    """Fetch full game summary from ESPN."""
    try:
        return get_json(
            ESPN_SUMMARY_URL,
            params={"event": game_id},
            headers=_HEADERS,
            timeout=10,
            retries=2,
            backoff_base=0.8,
        )
    except HttpClientError as e:
        logger.error("ESPN summary error for %s: %s", game_id, e)
        logger.debug("ESPN summary stacktrace", exc_info=True)
        return {}


def get_espn_odds(game_id: str) -> Dict[str, Any]:
    """Extract odds from ESPN game summary (pickcenter section)."""
    summary = fetch_espn_game_summary(game_id)
    pickcenter = summary.get("pickcenter", [])
    if not pickcenter:
        return {}
    odds_data = pickcenter[0] if pickcenter else {}
    return {
        "spread": odds_data.get("details", ""),
        "over_under": odds_data.get("overUnder", None),
        "home_moneyline": odds_data.get("homeTeamOdds", {}).get("moneyLine", None),
        "away_moneyline": odds_data.get("awayTeamOdds", {}).get("moneyLine", None),
        "provider": odds_data.get("provider", {}).get("name", ""),
    }


def get_espn_predictor(game_id: str) -> Dict[str, float]:
    """Extract ESPN predictor win probabilities."""
    summary = fetch_espn_game_summary(game_id)
    predictor = summary.get("predictor", {})
    home_pct = float(predictor.get("homeTeam", {}).get("gameProjection", 50.0))
    away_pct = float(predictor.get("awayTeam", {}).get("gameProjection", 50.0))
    return {"home_win_pct": home_pct, "away_win_pct": away_pct}


def get_espn_win_probability(game_id: str) -> List[Dict[str, Any]]:
    """Extract live win probability data."""
    summary = fetch_espn_game_summary(game_id)
    return summary.get("winprobability", [])


def get_espn_plays(game_id: str) -> List[Dict[str, Any]]:
    """Extract play-by-play from summary."""
    summary = fetch_espn_game_summary(game_id)
    return summary.get("plays", [])


def get_espn_boxscore(game_id: str) -> Dict[str, Any]:
    """Extract box score from summary."""
    summary = fetch_espn_game_summary(game_id)
    return summary.get("boxscore", {})


def get_espn_linescores(game_id: str) -> List[Dict[str, Any]]:
    """Extract quarter-by-quarter line scores."""
    summary = fetch_espn_game_summary(game_id)
    header = summary.get("header", {})
    competitions = header.get("competitions", [{}])
    if not competitions:
        return []
    competitors = competitions[0].get("competitors", [])
    scores = []
    for c in competitors:
        team = c.get("team", {})
        linescores = c.get("linescores", [])
        scores.append({
            "team": team.get("abbreviation", ""),
            "team_id": team.get("id", ""),
            "is_home": c.get("homeAway") == "home",
            "quarters": [int(q.get("displayValue", 0) or 0) for q in linescores],
            "score": int(c.get("score", 0) or 0),
        })
    return scores


FASTCAST_HOST_URL = "https://fastcast.semfs.engsvc.go.com/public/websockethost"
FASTCAST_CHANNEL_PREFIX = "gp-basketball-nba-"


class FastcastWebSocket:
    """Real-time ESPN Fastcast WebSocket for live game updates.

    Connects to ESPN's Fastcast service in a background thread and fires
    a callback when game data changes.  Consumers should refetch full
    data when notified — this keeps parsing logic in one place and the
    WebSocket layer thin.

    Auto-reconnects on disconnect.  Falls back silently if websocket-client
    is not installed.
    """

    def __init__(self, game_id: str, on_data_changed=None, on_error=None):
        self.game_id = game_id
        self._on_data_changed = on_data_changed
        self._on_error = on_error
        self._ws = None
        self._thread = None
        self._running = False
        self._channel = FASTCAST_CHANNEL_PREFIX + game_id
        self._last_notify = 0.0
        self._debounce_sec = 1.5  # min seconds between notifications
        self._connected = False
        self._reconnect_attempt = 0

    def start(self):
        """Connect and start receiving in a daemon background thread."""
        if self._running:
            return
        self._running = True
        import threading
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"fastcast-{self.game_id}")
        self._thread.start()

    def stop(self, timeout_sec: float = 3.0):
        """Disconnect and stop the background thread.

        Returns once the thread exits or timeout is reached.
        """
        self._running = False
        self._connected = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                logger.debug("Fastcast websocket close failed", exc_info=True)
        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(max(0.0, float(timeout_sec)))
            if thread.is_alive():
                logger.warning("Fastcast thread did not stop within %.1fs", timeout_sec)
            else:
                self._thread = None

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_running(self):
        return self._running

    def _run(self):
        """Background thread: connect → subscribe → receive → auto-reconnect."""
        try:
            import websocket
        except ImportError:
            logger.warning("websocket-client not installed — Fastcast unavailable")
            self._running = False
            return

        import time as _time

        while self._running:
            try:
                info = get_json(
                    FASTCAST_HOST_URL,
                    headers=_HEADERS,
                    timeout=10,
                    retries=3,
                    backoff_base=1.0,
                )
                ip = info.get("ip")
                secure_port = info.get("securePort")
                token = info.get("token")
                if not ip or not secure_port or not token:
                    raise ValueError("Fastcast host payload missing connection fields")
                ws_url = (
                    f"wss://{ip}:{secure_port}/"
                    f"FastcastService/pubsub/profiles/12000?"
                    f"TrafficManager-Token={token}"
                )

                self._ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_close,
                )
                # Avoid websocket-client ping-thread shutdown races while tabs switch games.
                self._ws.run_forever()

            except (HttpClientError, ValueError) as e:
                logger.error(f"Fastcast connect error: {e}")
                if self._on_error:
                    try:
                        self._on_error(str(e))
                    except Exception:
                        logger.debug("Fastcast on_error callback failed", exc_info=True)

            self._connected = False
            if self._running:
                delay = min(60, 2 ** self._reconnect_attempt)
                logger.info("Fastcast reconnecting in %ds (attempt %d)...",
                            delay, self._reconnect_attempt + 1)
                _time.sleep(delay)
                self._reconnect_attempt += 1

    # -- WebSocket callbacks (run on WS background thread) --

    def _on_open(self, ws):
        logger.info(f"Fastcast connected for game {self.game_id}")
        self._reconnect_attempt = 0
        try:
            ws.send(json.dumps({"op": "C"}))
        except Exception:
            # Can happen if caller stops websocket during connect race.
            logger.debug("Fastcast open handshake send skipped", exc_info=True)
            self._connected = False

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        op = msg.get("op", "")

        if op == "C":
            sid = msg.get("sid", "")
            try:
                ws.send(json.dumps({"op": "S", "sid": sid, "tc": self._channel}))
                logger.info(f"Fastcast subscribed to {self._channel}")
                self._connected = True
            except Exception:
                logger.debug("Fastcast subscribe send skipped", exc_info=True)
                self._connected = False

        elif "pl" in msg:
            tc = msg.get("tc", "")
            pl = msg.get("pl", "0")
            if pl == "0" or tc != self._channel:
                return
            if isinstance(pl, str) and pl.startswith("http"):
                return  # checkpoint URL

            # Relevant data arrived — notify consumer (debounced)
            import time as _time
            now = _time.time()
            if now - self._last_notify >= self._debounce_sec:
                self._last_notify = now
                if self._on_data_changed:
                    try:
                        self._on_data_changed()
                    except Exception:
                        logger.debug("Fastcast on_data_changed callback failed", exc_info=True)

    def _on_ws_error(self, ws, error):
        text = str(error)
        if ("NoneType' object has no attribute 'sock'" in text
                or "Connection is already closed" in text):
            logger.debug("Fastcast WS close-race: %s", text)
        else:
            logger.warning("Fastcast WS error: %s", text)
        self._connected = False

    def _on_close(self, ws, code, reason):
        logger.info(f"Fastcast closed: code={code}, reason={reason}")
        self._connected = False

_an_odds_cache = {}
_an_last_fetch = 0.0
_an_odds_lock = threading.Lock()


def invalidate_actionnetwork_cache() -> None:
    """Clear cached Action Network scoreboard payload."""
    global _an_odds_cache, _an_last_fetch
    with _an_odds_lock:
        _an_odds_cache = {}
        _an_last_fetch = 0.0


def get_actionnetwork_odds(home_abbr: str, away_abbr: str) -> Dict[str, Any]:
    """Fetch live odds from Action Network API.
    Matches based on team abbreviations.
    Caches the full scoreboard request for 15 seconds to prevent spam.
    """
    import time
    global _an_odds_cache, _an_last_fetch

    home_query = normalize_action_abbr(home_abbr)
    away_query = normalize_action_abbr(away_abbr)

    now = time.time()
    with _an_odds_lock:
        if not _an_odds_cache or (now - _an_last_fetch) > 10.0:
            try:
                data = get_json(
                    "https://api.actionnetwork.com/web/v1/scoreboard/nba",
                    headers=_HEADERS,
                    timeout=10,
                    retries=3,
                    backoff_base=0.8,
                )
                _an_odds_cache = data
                _an_last_fetch = now
            except HttpClientError as e:
                logger.warning(f"ActionNetwork fetch failed: {e}")

    if not _an_odds_cache:
        return {}
        
    games = _an_odds_cache.get("games", [])
    
    for g in games:
        teams = g.get("teams", [])
        if len(teams) < 2:
            continue
            
        t1_abbr = normalize_action_abbr(teams[0].get("abbr", ""))
        t2_abbr = normalize_action_abbr(teams[1].get("abbr", ""))

        # Check if this game matches our teams
        match = (t1_abbr == home_query and t2_abbr == away_query) or \
                (t1_abbr == away_query and t2_abbr == home_query)
                
        if match:
            odds_list = g.get("odds", [])
            if odds_list:
                # Prefer live odds if available, then fallback to pre-game
                live_odds = [o for o in odds_list if o.get("type") == "live"]
                game_odds = [o for o in odds_list if o.get("type") == "game"]
                
                o = live_odds[0] if live_odds else (game_odds[0] if game_odds else odds_list[0])
                
                # Figure out which spread goes to which team based on IDs
                home_team_id = g.get("home_team_id")
                away_team_id = g.get("away_team_id")
                
                # Make sure we map the home spread correctly
                # Action Network provides "spread_home" directly!
                spread_val = o.get("spread_home")
                spread_str = f"{spread_val:+.1f}" if spread_val is not None else ""
                
                # Sharp money: public bet % vs actual money %
                # Use pre-game odds for sharp money (not live in-game)
                sharp_src = game_odds[0] if game_odds else o
                return {
                    "spread": spread_str,
                    "over_under": o.get("total"),
                    "home_moneyline": o.get("ml_home"),
                    "away_moneyline": o.get("ml_away"),
                    "provider": "Action Network" + (" (Live)" if live_odds else ""),
                    # Sharp money data
                    "spread_home_public": sharp_src.get("spread_home_public"),
                    "spread_away_public": sharp_src.get("spread_away_public"),
                    "spread_home_money": sharp_src.get("spread_home_money"),
                    "spread_away_money": sharp_src.get("spread_away_money"),
                    "ml_home_public": sharp_src.get("ml_home_public"),
                    "ml_away_public": sharp_src.get("ml_away_public"),
                    "ml_home_money": sharp_src.get("ml_home_money"),
                    "ml_away_money": sharp_src.get("ml_away_money"),
                }
            
    return {}
