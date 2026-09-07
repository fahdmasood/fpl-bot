"""Settings and the FPL scoring table."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Points awarded per event, by position.
#
# The FPL API does NOT expose these values, so they are transcribed from the
# official rules. Verify against https://fantasy.premierleague.com/help/rules
# before trusting a projection: a wrong value here corrupts every number
# downstream without raising anything.
SCORING = {
    "appearance_under_60": 1,
    "appearance_60_plus": 2,
    "goal": {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": {"GKP": 3, "DEF": 3, "MID": 3, "FWD": 3},
    "clean_sheet": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    # Defenders: 2 points at 10+ clearances, blocks, interceptions and tackles.
    # Midfielders and forwards: 2 points at 12+ of those plus recoveries.
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
    "defensive_threshold": {"GKP": 99, "DEF": 10, "MID": 12, "FWD": 12},
    "saves_per_point": 3,
    "goals_conceded_per_penalty": 2,
    "yellow_card": -1,
    "red_card": -3,
    "own_goal": -2,
    "penalty_miss": -2,
    "penalty_save": 5,
}

POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def user_agent(username: str) -> str:
    """Reddit mandates this exact shape. Generic UAs get heavily throttled."""
    return f"python:fpl-bot:v1.0.0 (by /u/{username})"


@dataclass
class Settings:
    cache_dir: Path
    horizon: int = 3
    decay_base: float = 0.84
    sentiment_cap: float = 0.15
    comment_cap: int = 600
    comment_ceiling: int = 1000
    max_threads: int = 8
    score_floor: int = 3
    max_age_hours: int = 36
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_username: str | None = None
    anthropic_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            cache_dir=Path(os.environ.get("FPLBOT_CACHE", "~/.cache/fplbot")).expanduser(),
            reddit_client_id=os.environ.get("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
            reddit_username=os.environ.get("REDDIT_USERNAME"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    @property
    def decay_weights(self) -> list[float]:
        return [self.decay_base**i for i in range(self.horizon)]
