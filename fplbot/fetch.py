"""FPL API client. Public endpoints only; no credentials anywhere."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from fplbot.config import POSITION_BY_ELEMENT_TYPE, Settings
from fplbot.models import Fixture, Gameweek, Player, Team

log = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fplbot/0.1)"}


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


class FplClient:
    def __init__(self, settings: Settings, session=None):
        self.settings = settings
        self.session = session or requests.Session()
        self.cache_dir = Path(settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict] = {}

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def cache_age_seconds(self, key: str) -> float | None:
        path = self._cache_path(key)
        return time.time() - path.stat().st_mtime if path.exists() else None

    def _get(self, key: str, url: str) -> dict:
        if key in self._memory:
            return self._memory[key]

        try:
            response = self.session.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            payload = response.json()
            self._cache_path(key).write_text(json.dumps(payload))
        except Exception as exc:
            # A stale answer beats no answer. If the network is down at the
            # deadline, an old projection is still actionable.
            path = self._cache_path(key)
            if not path.exists():
                raise
            age = self.cache_age_seconds(key)
            log.warning("fetch %s failed (%s); using cache %.0fs old", key, exc, age)
            payload = json.loads(path.read_text())

        self._memory[key] = payload
        return payload

    def bootstrap(self) -> dict:
        return self._get("bootstrap-static", f"{BASE}/bootstrap-static/")

    def players(self) -> list[Player]:
        return [Player.from_api(e) for e in self.bootstrap()["elements"]]

    def teams(self) -> list[Team]:
        return [
            Team(
                id=t["id"],
                name=t["name"],
                short_name=t["short_name"],
                strength_defence_home=t["strength_defence_home"],
                strength_defence_away=t["strength_defence_away"],
            )
            for t in self.bootstrap()["teams"]
        ]

    def gameweeks(self) -> list[Gameweek]:
        return [
            Gameweek(
                id=e["id"],
                name=e["name"],
                deadline_time=_dt(e["deadline_time"]),
                is_next=e.get("is_next", False),
            )
            for e in self.bootstrap()["events"]
        ]

    def next_gameweek(self) -> Gameweek:
        weeks = self.gameweeks()
        for gw in weeks:
            if gw.is_next:
                return gw
        # Past the final deadline the API sets no is_next; fall back to the
        # first gameweek whose deadline is still ahead of us.
        now = datetime.now(tz=weeks[0].deadline_time.tzinfo)
        future = [gw for gw in weeks if gw.deadline_time > now]
        if future:
            return future[0]
        return weeks[-1]

    def fixtures(self, event: int | None = None) -> list[Fixture]:
        raw = self._get("fixtures", f"{BASE}/fixtures/")
        items = [
            Fixture(
                id=f["id"],
                event=f["event"],
                team_h=f["team_h"],
                team_a=f["team_a"],
                team_h_difficulty=f["team_h_difficulty"],
                team_a_difficulty=f["team_a_difficulty"],
                kickoff_time=_dt(f.get("kickoff_time")),
            )
            for f in raw
        ]
        return [f for f in items if event is None or f.event == event]

    def squad_rules(self) -> dict:
        boot = self.bootstrap()
        gs = boot["game_settings"]
        formation = {
            POSITION_BY_ELEMENT_TYPE[t["id"]]: (t["squad_min_play"], t["squad_max_play"])
            for t in boot["element_types"]
        }
        squad_quota = {
            POSITION_BY_ELEMENT_TYPE[t["id"]]: t["squad_select"]
            for t in boot["element_types"]
        }
        return {
            "budget": gs["squad_total_spend"],
            "squad_size": gs["squad_squadsize"],
            "xi_size": gs["squad_squadplay"],
            "team_limit": gs["squad_team_limit"],
            "formation": formation,
            "quota": squad_quota,
        }

    def entry_picks(self, entry_id: int, event: int) -> list[int]:
        key = f"entry-{entry_id}-{event}"
        payload = self._get(key, f"{BASE}/entry/{entry_id}/event/{event}/picks/")
        return [p["element"] for p in payload["picks"]]
