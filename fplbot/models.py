"""Domain types. All money is integer tenths of a million."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from fplbot.config import POSITION_BY_ELEMENT_TYPE


def _f(value) -> float:
    """The FPL API returns many numeric fields as strings."""
    return float(value) if value not in (None, "") else 0.0


@dataclass(frozen=True)
class Team:
    id: int
    name: str
    short_name: str
    strength_defence_home: int
    strength_defence_away: int


@dataclass(frozen=True)
class Player:
    id: int
    web_name: str
    position: str
    team_id: int
    now_cost: int
    status: str
    chance_of_playing: int | None
    minutes: int
    starts: int
    form: float
    selected_by_percent: float
    expected_goals: float
    expected_assists: float
    expected_goals_conceded: float
    defensive_contribution_per_90: float
    ict_index: float
    bps: int

    @property
    def price_m(self) -> float:
        return self.now_cost / 10

    @property
    def availability(self) -> float:
        """Fraction of a chance this player is available.

        `chance_of_playing_next_round` of None means no injury news has been
        filed, which is good news. Only trust the number when it is present.
        """
        if self.chance_of_playing is not None:
            return self.chance_of_playing / 100
        return 1.0 if self.status == "a" else 0.0

    @classmethod
    def from_api(cls, raw: dict) -> "Player":
        return cls(
            id=raw["id"],
            web_name=raw["web_name"],
            position=POSITION_BY_ELEMENT_TYPE[raw["element_type"]],
            team_id=raw["team"],
            now_cost=raw["now_cost"],
            status=raw["status"],
            chance_of_playing=raw.get("chance_of_playing_next_round"),
            minutes=raw.get("minutes", 0),
            starts=raw.get("starts", 0),
            form=_f(raw.get("form")),
            selected_by_percent=_f(raw.get("selected_by_percent")),
            expected_goals=_f(raw.get("expected_goals")),
            expected_assists=_f(raw.get("expected_assists")),
            expected_goals_conceded=_f(raw.get("expected_goals_conceded")),
            defensive_contribution_per_90=_f(raw.get("defensive_contribution_per_90")),
            ict_index=_f(raw.get("ict_index")),
            bps=raw.get("bps", 0),
        )


@dataclass(frozen=True)
class Fixture:
    id: int
    event: int | None
    team_h: int
    team_a: int
    team_h_difficulty: int
    team_a_difficulty: int
    kickoff_time: datetime | None


@dataclass(frozen=True)
class Gameweek:
    id: int
    name: str
    deadline_time: datetime
    is_next: bool


@dataclass(frozen=True)
class NewsItem:
    source: str
    url: str
    published_at: datetime
    title: str
    body: str
    score: int


@dataclass(frozen=True)
class Mention:
    player_id: int
    item: NewsItem
    matched_text: str


@dataclass(frozen=True)
class MentionScore:
    player_id: int
    sentiment: float
    category: str
    confidence: float


@dataclass(frozen=True)
class PlayerSentiment:
    player_id: int
    availability_signal: float
    form_signal: float
    volume: int
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Projection:
    player_id: int
    per_gameweek: list[float]
    total: float
    sentiment_applied: float
    explanation: str


@dataclass
class Squad:
    players: list[int]
    starting_xi: list[int]
    captain_id: int
    bank: int

    def __post_init__(self):
        if len(self.players) not in (0, 15):
            raise ValueError(f"squad must hold 15 players, got {len(self.players)}")
        if not self.players:
            raise ValueError("squad must not be empty")
