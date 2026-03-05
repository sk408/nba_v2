"""Data models (dataclasses) for the NBA analytics system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Team:
    team_id: int
    name: str
    abbreviation: str
    conference: str = ""


@dataclass
class Player:
    player_id: int
    name: str
    team_id: int
    position: str = ""
    is_injured: bool = False
    injury_note: str = ""
    height: str = ""
    weight: str = ""
    age: int = 0
    experience: int = 0


@dataclass
class PlayerStat:
    id: int = 0
    player_id: int = 0
    opponent_team_id: int = 0
    is_home: int = 0
    game_date: str = ""
    game_id: str = ""
    points: float = 0.0
    rebounds: float = 0.0
    assists: float = 0.0
    minutes: float = 0.0
    steals: float = 0.0
    blocks: float = 0.0
    turnovers: float = 0.0
    fg_made: int = 0
    fg_attempted: int = 0
    fg3_made: int = 0
    fg3_attempted: int = 0
    ft_made: int = 0
    ft_attempted: int = 0
    oreb: float = 0.0
    dreb: float = 0.0
    plus_minus: float = 0.0
    win_loss: str = ""
    personal_fouls: float = 0.0


@dataclass
class Prediction:
    id: int = 0
    home_team_id: int = 0
    away_team_id: int = 0
    game_date: str = ""
    predicted_spread: float = 0.0
    predicted_total: float = 0.0
    actual_spread: Optional[float] = None
    actual_total: Optional[float] = None


@dataclass
class LiveGame:
    game_id: str = ""
    home_team_id: int = 0
    away_team_id: int = 0
    start_time_utc: str = ""
    status: str = ""
    period: int = 0
    clock: str = ""
    home_score: int = 0
    away_score: int = 0
    last_updated: str = ""


@dataclass
class TeamMetrics:
    team_id: int = 0
    season: str = ""
    gp: int = 0
    w: int = 0
    l: int = 0
    w_pct: float = 0.0
    e_off_rating: float = 0.0
    e_def_rating: float = 0.0
    e_net_rating: float = 0.0
    e_pace: float = 0.0
    e_ast_ratio: float = 0.0
    e_oreb_pct: float = 0.0
    e_dreb_pct: float = 0.0
    e_reb_pct: float = 0.0
    e_tm_tov_pct: float = 0.0
    off_rating: float = 0.0
    def_rating: float = 0.0
    net_rating: float = 0.0
    pace: float = 0.0
    efg_pct: float = 0.0
    ts_pct: float = 0.0
    ast_ratio: float = 0.0
    ast_to: float = 0.0
    oreb_pct: float = 0.0
    dreb_pct: float = 0.0
    reb_pct: float = 0.0
    tm_tov_pct: float = 0.0
    pie: float = 0.0
    ff_efg_pct: float = 0.0
    ff_fta_rate: float = 0.0
    ff_tm_tov_pct: float = 0.0
    ff_oreb_pct: float = 0.0
    opp_efg_pct: float = 0.0
    opp_fta_rate: float = 0.0
    opp_tm_tov_pct: float = 0.0
    opp_oreb_pct: float = 0.0
    opp_pts: float = 0.0
    opp_fg_pct: float = 0.0
    opp_fg3_pct: float = 0.0
    opp_ft_pct: float = 0.0
    clutch_gp: int = 0
    clutch_w: int = 0
    clutch_l: int = 0
    clutch_net_rating: float = 0.0
    clutch_efg_pct: float = 0.0
    clutch_ts_pct: float = 0.0
    deflections: float = 0.0
    loose_balls_recovered: float = 0.0
    contested_shots: float = 0.0
    charges_drawn: float = 0.0
    screen_assists: float = 0.0
    home_gp: int = 0
    home_w: int = 0
    home_l: int = 0
    home_pts: float = 0.0
    home_opp_pts: float = 0.0
    road_gp: int = 0
    road_w: int = 0
    road_l: int = 0
    road_pts: float = 0.0
    road_opp_pts: float = 0.0
    last_synced_at: str = ""


@dataclass
class PlayerImpact:
    player_id: int = 0
    team_id: int = 0
    season: str = ""
    on_court_off_rating: float = 0.0
    on_court_def_rating: float = 0.0
    on_court_net_rating: float = 0.0
    off_court_off_rating: float = 0.0
    off_court_def_rating: float = 0.0
    off_court_net_rating: float = 0.0
    net_rating_diff: float = 0.0
    on_court_minutes: float = 0.0
    e_usg_pct: float = 0.0
    e_off_rating: float = 0.0
    e_def_rating: float = 0.0
    e_net_rating: float = 0.0
    e_pace: float = 0.0
    e_ast_ratio: float = 0.0
    e_oreb_pct: float = 0.0
    e_dreb_pct: float = 0.0
    last_synced_at: str = ""


@dataclass
class InjuryHistory:
    id: int = 0
    player_id: int = 0
    team_id: int = 0
    game_date: str = ""
    was_out: int = 1
    avg_minutes: float = 0.0
    reason: str = ""


@dataclass
class InjuryStatusLog:
    id: int = 0
    player_id: int = 0
    team_id: int = 0
    log_date: str = ""
    status_level: str = ""
    injury_keyword: str = ""
    injury_detail: str = ""
    next_game_date: str = ""
    did_play: Optional[int] = None


@dataclass
class TeamTuning:
    team_id: int = 0
    home_pts_correction: float = 0.0
    away_pts_correction: float = 0.0
    games_analyzed: int = 0
    avg_spread_error_before: float = 0.0
    avg_total_error_before: float = 0.0
    last_tuned_at: str = ""
    tuning_mode: str = "classic"
    tuning_version: str = "v1_classic"
    tuning_sample_size: int = 0


@dataclass
class Notification:
    id: int = 0
    category: str = "info"
    severity: str = "info"
    title: str = ""
    message: str = ""
    data: str = "{}"
    created_at: str = ""
    read: int = 0


@dataclass
class GameQuarterScore:
    game_id: str = ""
    team_id: int = 0
    q1: int = 0
    q2: int = 0
    q3: int = 0
    q4: int = 0
    ot: int = 0
    final_score: int = 0
    game_date: str = ""
    is_home: int = 0
