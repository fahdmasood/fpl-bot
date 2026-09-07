# FPL Sentiment Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Fantasy Premier League advisory bot that combines the official FPL statistical feed with Reddit/news sentiment to produce an optimal 15-man squad, published to a report before each deadline.

**Architecture:** A deterministic core (fetch → project → optimize → report) written as pure functions over cached API snapshots, with a language model confined to one job: turning football prose into per-player sentiment scores. Sentiment enters the projection only as a bounded ±15% modifier on the next gameweek's term, so it can surface early injury news without overriding the statistics.

**Tech Stack:** Python 3.13, `requests`, `feedparser`, `praw` (Reddit OAuth), `pulp` (ILP solver, bundles CBC), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-07-fpl-sentiment-bot-design.md`

**Research:** `docs/research/2026-09-07-fpl-horizon-and-sentiment-volume.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python >= 3.13.** Available at `~/.pyenv/shims/python3` (3.13.5).
- **No FPL account credentials anywhere in this system.** The bot reads public endpoints only and never writes to a user's team.
- **Horizon default 3 gameweeks, `decay_base = 0.84`** — weights 1.00 / 0.84 / 0.71.
- **Sentiment modifier capped at ±15%, applied to the GW+1 term only**, never to GW+2 or GW+3, and never directly to a points total.
- **Item collection: 600 per run, hard ceiling 1000**, score floor >= 3, age <= 36h, deduplicated on normalised text. Reddit, when enabled, draws from at most 8 threads.
- **News RSS is the primary sentiment source; Reddit is optional and gated on approved Data API access.** A run without Reddit is the expected default, not a degraded state.
- **Never store a comment author.** `NewsItem` has no `author` field and must not gain one — Reddit policy forbids inferring characteristics about users.
- **Reddit User-Agent must be exactly** `python:fpl-bot:v1.0.0 (by /u/<username>)`. Generic UAs are heavily throttled by Reddit.
- **One Reddit client id.** Registering multiple accounts or apps for the same use case violates Reddit's Responsible Builder Policy and risks a permanent block.
- **Sentiment model:** `claude-haiku-4-5`.
- **Squad constraints are read from the API**, never hardcoded: `game_settings.squad_total_spend` (1000 = £100.0m), `squad_squadsize` (15), `squad_squadplay` (11), `squad_team_limit` (3), and `element_types[].squad_min_play`/`squad_max_play`.
- **Prices are integers in tenths of a million.** Never use floats for money; `now_cost: 55` means £5.5m. Float arithmetic on budgets causes off-by-one-tenth infeasibility that is miserable to debug.
- **Tests never hit the network.** Every test runs against cached JSON fixtures.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, package metadata, pytest config |
| `.env.example` | Documents required env vars; never contains real secrets |
| `fplbot/config.py` | Settings dataclass, env loading, FPL scoring table |
| `fplbot/models.py` | Dataclasses: `Player`, `Team`, `Fixture`, `Gameweek`, `NewsItem`, `Mention`, `PlayerSentiment`, `Projection`, `Squad` |
| `fplbot/fetch.py` | FPL API client with disk cache |
| `fplbot/news.py` | Reddit + RSS collectors and filters |
| `fplbot/sentiment.py` | Mention resolution, `Scorer` implementations, aggregation |
| `fplbot/projection.py` | Expected-points model |
| `fplbot/optimize.py` | ILP squad selection, XI, captain |
| `fplbot/report.py` | JSON + HTML rendering, transfer diff |
| `fplbot/cli.py` | Command-line entry point wiring the pipeline |
| `tests/fixtures/` | Cached API JSON snapshots |
| `tests/conftest.py` | Shared pytest fixtures loading the snapshots |

---

## Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml`, `.env.example`, `fplbot/__init__.py`, `fplbot/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `Settings` dataclass with fields `cache_dir: Path`, `horizon: int = 3`, `decay_base: float = 0.84`, `sentiment_cap: float = 0.15`, `comment_cap: int = 600`, `comment_ceiling: int = 1000`, `max_threads: int = 8`, `score_floor: int = 3`, `max_age_hours: int = 36`, `reddit_client_id: str | None`, `reddit_client_secret: str | None`, `reddit_username: str | None`, `anthropic_api_key: str | None`; classmethod `Settings.from_env() -> Settings`; module constant `SCORING: dict`; function `user_agent(username: str) -> str`.

**Background the implementer needs:** FPL awards points by position. The API does *not* expose the points values, so they live in a hardcoded `SCORING` table. A wrong value here silently corrupts every projection downstream, so it is worth checking against the official rules at `fantasy.premierleague.com/help/rules` before trusting it. Note that `defensive_contribution` is a scoring category in the current season — defenders earn points above a clearances/blocks/interceptions threshold, midfielders and forwards on a combined tackles-and-recoveries measure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from fplbot.config import Settings, SCORING, user_agent


def test_defaults_match_spec():
    s = Settings(cache_dir="/tmp/x")
    assert s.horizon == 3
    assert s.decay_base == 0.84
    assert s.sentiment_cap == 0.15
    assert s.comment_cap == 600
    assert s.score_floor == 3


def test_user_agent_matches_reddit_mandated_format():
    assert user_agent("someone") == "python:fpl-bot:v1.0.0 (by /u/someone)"


def test_scoring_table_covers_all_four_positions():
    for pos in ("GKP", "DEF", "MID", "FWD"):
        assert pos in SCORING["goal"]
        assert pos in SCORING["clean_sheet"]


def test_from_env_reads_reddit_credentials(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "abc")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "def")
    monkeypatch.setenv("REDDIT_USERNAME", "ghi")
    s = Settings.from_env()
    assert s.reddit_client_id == "abc"
    assert s.reddit_client_secret == "def"


def test_from_env_tolerates_missing_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    s = Settings.from_env()
    assert s.reddit_client_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot'`

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "fplbot"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "requests>=2.32",
    "feedparser>=6.0",
    "praw>=7.7",
    "pulp>=2.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
fplbot = "fplbot.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 4: Write `fplbot/config.py`**

```python
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
```

- [ ] **Step 5: Write `.env.example` and `fplbot/__init__.py`**

```bash
touch fplbot/__init__.py
cat > .env.example <<'EOF'
# Create a "script" app at https://reddit.com/prefs/apps
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USERNAME=

# Only needed to run the CLI standalone. The scheduled path scores
# sentiment inside the Claude session and needs no key.
ANTHROPIC_API_KEY=

FPLBOT_CACHE=~/.cache/fplbot
EOF
```

- [ ] **Step 6: Install and run tests**

Run: `python3 -m pip install -e '.[dev]' && python3 -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example fplbot/ tests/
git commit -m "feat: project scaffold, settings, and FPL scoring table"
```

---

## Task 2: Domain models

**Files:**
- Create: `fplbot/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `POSITION_BY_ELEMENT_TYPE` from `fplbot.config`
- Produces: frozen dataclasses `Team`, `Player`, `Fixture`, `Gameweek`, `NewsItem`, `Mention`, `MentionScore`, `PlayerSentiment`, `Projection`, `Squad`, plus `Player.from_api(raw: dict) -> Player` and `Player.price_m -> float`.

**Background the implementer needs:** `now_cost` is an integer in tenths of a million; keep it an integer everywhere and only convert for display. `status` is a single character: `a` available, `d` doubtful, `i` injured, `s` suspended, `u` unavailable, `n` on loan / not in squad. `chance_of_playing_next_round` is an integer percentage or `None` — and `None` means "no news", which is good news, not missing data. Getting that backwards benches every fit player.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from fplbot.models import Player, Squad


RAW = {
    "id": 1, "web_name": "Raya", "element_type": 1, "team": 1,
    "now_cost": 55, "status": "a", "chance_of_playing_next_round": None,
    "minutes": 900, "form": "4.5", "selected_by_percent": "12.3",
    "expected_goals": "0.0", "expected_assists": "0.1",
    "expected_goals_conceded": "8.2", "defensive_contribution_per_90": "0.0",
    "ict_index": "40.0", "bps": 300, "starts": 10,
}


def test_from_api_maps_position_and_price():
    p = Player.from_api(RAW)
    assert p.position == "GKP"
    assert p.now_cost == 55
    assert p.price_m == 5.5


def test_none_chance_of_playing_means_available_not_missing():
    p = Player.from_api(RAW)
    assert p.availability == 1.0


def test_doubtful_player_uses_reported_chance():
    p = Player.from_api({**RAW, "status": "d", "chance_of_playing_next_round": 25})
    assert p.availability == 0.25


def test_injured_player_is_unavailable():
    p = Player.from_api({**RAW, "status": "i", "chance_of_playing_next_round": 0})
    assert p.availability == 0.0


def test_squad_rejects_wrong_size():
    import pytest
    with pytest.raises(ValueError):
        Squad(players=[], starting_xi=[], captain_id=1, bank=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'Player'`

- [ ] **Step 3: Write `fplbot/models.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_models.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add fplbot/models.py tests/test_models.py
git commit -m "feat: domain models with availability semantics"
```

---

## Task 3: FPL API client with disk cache, and test snapshots

**Files:**
- Create: `fplbot/fetch.py`, `tests/conftest.py`
- Create: `tests/fixtures/bootstrap-static.json`, `tests/fixtures/fixtures.json`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `Settings` from `fplbot.config`; `Player`, `Team`, `Fixture`, `Gameweek` from `fplbot.models`
- Produces: `FplClient(settings, session=None)` with methods `bootstrap() -> dict`, `fixtures(event: int | None = None) -> list[Fixture]`, `players() -> list[Player]`, `teams() -> list[Team]`, `gameweeks() -> list[Gameweek]`, `next_gameweek() -> Gameweek`, `entry_picks(entry_id: int, event: int) -> list[int]`, `squad_rules() -> dict`. Also `cache_age_seconds(key: str) -> float | None`.

**Background the implementer needs:** The base URL is `https://fantasy.premierleague.com/api/`. No authentication. All four endpoints used here are public; `entry_picks` reads another manager's public team and needs no login. Requests without a browser-like `User-Agent` are sometimes rejected, so set one. A stale cache is explicitly preferred over a failed run — if the network is down at 11:00 on a Friday, a slightly old projection is far more useful than a traceback.

`squad_rules()` extracts constraints from the live API rather than hardcoding them: `game_settings.squad_total_spend`, `squad_squadsize`, `squad_squadplay`, `squad_team_limit`, and each `element_types` entry's `squad_min_play` / `squad_max_play`.

- [ ] **Step 1: Capture the test snapshots**

These are real API responses, committed so every later test runs offline.

```bash
mkdir -p tests/fixtures
curl -s -H 'User-Agent: Mozilla/5.0' \
  https://fantasy.premierleague.com/api/bootstrap-static/ \
  -o tests/fixtures/bootstrap-static.json
curl -s -H 'User-Agent: Mozilla/5.0' \
  https://fantasy.premierleague.com/api/fixtures/ \
  -o tests/fixtures/fixtures.json
python3 -c "
import json
b = json.load(open('tests/fixtures/bootstrap-static.json'))
f = json.load(open('tests/fixtures/fixtures.json'))
assert len(b['elements']) > 400, 'bootstrap looks truncated'
assert len(b['teams']) == 20
assert len(f) > 300, 'fixtures look truncated'
print('snapshots OK:', len(b['elements']), 'players,', len(f), 'fixtures')
"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/conftest.py
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def bootstrap():
    return json.loads((FIXTURES / "bootstrap-static.json").read_text())


@pytest.fixture
def raw_fixtures():
    return json.loads((FIXTURES / "fixtures.json").read_text())


@pytest.fixture
def client(tmp_path, bootstrap, raw_fixtures):
    from fplbot.config import Settings
    from fplbot.fetch import FplClient

    class FakeSession:
        """Serves the snapshots; fails loudly if a test reaches the network."""

        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(url)
            if "bootstrap-static" in url:
                payload = bootstrap
            elif "fixtures" in url:
                payload = raw_fixtures
            else:
                raise AssertionError(f"unexpected network call: {url}")

            class R:
                status_code = 200

                @staticmethod
                def json():
                    return payload

                @staticmethod
                def raise_for_status():
                    return None

            return R()

    return FplClient(Settings(cache_dir=tmp_path), session=FakeSession())
```

```python
# tests/test_fetch.py
def test_players_parse_from_snapshot(client):
    players = client.players()
    assert len(players) > 400
    assert all(p.position in ("GKP", "DEF", "MID", "FWD") for p in players)


def test_next_gameweek_is_identified(client):
    gw = client.next_gameweek()
    assert gw.id >= 1
    assert gw.deadline_time is not None


def test_squad_rules_come_from_the_api_not_constants(client):
    rules = client.squad_rules()
    assert rules["budget"] == 1000
    assert rules["squad_size"] == 15
    assert rules["team_limit"] == 3
    assert rules["xi_size"] == 11
    assert rules["formation"]["GKP"] == (1, 1)
    assert rules["formation"]["DEF"] == (3, 5)


def test_second_call_is_served_from_cache(client):
    client.players()
    calls_after_first = len(client.session.calls)
    client.players()
    assert len(client.session.calls) == calls_after_first


def test_stale_cache_is_preferred_over_a_failed_run(client, tmp_path):
    client.bootstrap()

    class BrokenSession:
        calls = []

        def get(self, url, **kwargs):
            raise ConnectionError("network down")

    client.session = BrokenSession()
    client._memory.clear()
    assert len(client.players()) > 400
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot.fetch'`

- [ ] **Step 4: Write `fplbot/fetch.py`**

```python
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
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_fetch.py -v`
Expected: 5 passed

If `test_squad_rules_come_from_the_api_not_constants` fails on `quota`, print `boot["element_types"][0].keys()` and use whichever key holds the per-position squad count. The field name is asserted here rather than assumed precisely because it is easy to get wrong.

- [ ] **Step 6: Commit**

```bash
git add fplbot/fetch.py tests/
git commit -m "feat: FPL API client with stale-cache fallback"
```

---

## Task 4: Expected-points projection

**Files:**
- Create: `fplbot/projection.py`
- Test: `tests/test_projection.py`

**Interfaces:**
- Consumes: `Player`, `Team`, `Fixture`, `Projection`, `PlayerSentiment` from `fplbot.models`; `SCORING` and `Settings` from `fplbot.config`; `FplClient` from `fplbot.fetch`
- Produces: `project_all(players, teams, fixtures, next_gw_id, settings, sentiment=None) -> dict[int, Projection]`; `minutes_probability(player, sentiment=None, cap=0.15) -> float`; `expected_points_one_gw(player, team, fixture, minutes_prob) -> float`

**Background the implementer needs:** This is the model at the heart of the bot, and its structure matters more than its precision. Expected points for one gameweek is roughly:

```
appearance + (xG90 * goal_points) + (xA90 * 3) + clean_sheet_prob * cs_points
          + defensive_contribution_points + bonus_estimate
```

all scaled by the probability the player is on the pitch. Per-90 rates come from dividing season totals by `minutes / 90`; a player with very few minutes has an unstable rate, so require a minutes floor before trusting it and fall back to the position's median otherwise.

Two constraints from the spec that are easy to violate:

1. **Sentiment applies to the GW+1 term only.** With horizon 3 and decay 0.84, GW+1 holds about 39% of the objective. Applying the ±15% modifier across all three weeks would dilute the guardrail to roughly ±6%, which is not what the spec intends.
2. **Freeze lag features at the deadline.** When projecting GW+2 and GW+3, the model may not use anything that happens in GW+1 — those results do not exist yet at decision time. Every gameweek in the horizon reuses the same player-form inputs. Violating this makes a backtest look excellent and the live bot perform badly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_projection.py
import pytest

from fplbot.config import Settings
from fplbot.models import PlayerSentiment
from fplbot.projection import minutes_probability, project_all


@pytest.fixture
def projections(client):
    settings = Settings(cache_dir="/tmp/x")
    return project_all(
        players=client.players(),
        teams=client.teams(),
        fixtures=client.fixtures(),
        next_gw_id=client.next_gameweek().id,
        settings=settings,
    )


def test_every_player_gets_a_projection(projections, client):
    assert len(projections) == len(client.players())


def test_projections_are_non_negative_and_bounded(projections):
    for p in projections.values():
        assert 0 <= p.total < 60, f"implausible projection: {p}"


def test_horizon_produces_one_value_per_gameweek(projections):
    assert all(len(p.per_gameweek) == 3 for p in projections.values())


def test_injured_player_projects_near_zero(client, projections):
    injured = [p for p in client.players() if p.status == "i"]
    if not injured:
        pytest.skip("no injured players in this snapshot")
    assert projections[injured[0].id].total < 1.0


def test_positive_sentiment_never_lowers_a_projection(client):
    settings = Settings(cache_dir="/tmp/x")
    target = max(client.players(), key=lambda p: p.minutes)
    common = dict(
        players=client.players(), teams=client.teams(),
        fixtures=client.fixtures(), next_gw_id=client.next_gameweek().id,
        settings=settings,
    )
    base = project_all(**common)
    boosted = project_all(**common, sentiment={
        target.id: PlayerSentiment(target.id, availability_signal=1.0,
                                   form_signal=1.0, volume=10)
    })
    assert boosted[target.id].total >= base[target.id].total


def test_sentiment_modifier_respects_the_cap():
    from fplbot.models import Player
    raw = {
        "id": 1, "web_name": "X", "element_type": 3, "team": 1, "now_cost": 100,
        "status": "a", "chance_of_playing_next_round": None, "minutes": 900,
        "form": "5.0", "selected_by_percent": "10", "expected_goals": "5.0",
        "expected_assists": "3.0", "expected_goals_conceded": "10.0",
        "defensive_contribution_per_90": "1.0", "ict_index": "100", "bps": 400,
        "starts": 10,
    }
    player = Player.from_api(raw)
    neutral = minutes_probability(player, sentiment=None)
    best = minutes_probability(
        player, PlayerSentiment(1, availability_signal=1.0, form_signal=0.0, volume=5))
    worst = minutes_probability(
        player, PlayerSentiment(1, availability_signal=-1.0, form_signal=0.0, volume=5))
    assert best <= neutral * 1.15 + 1e-9
    assert worst >= neutral * 0.85 - 1e-9


def test_sentiment_cannot_rescue_a_flagged_player():
    from fplbot.models import Player
    raw = {
        "id": 2, "web_name": "Y", "element_type": 3, "team": 1, "now_cost": 100,
        "status": "i", "chance_of_playing_next_round": 0, "minutes": 900,
        "form": "5.0", "selected_by_percent": "10", "expected_goals": "5.0",
        "expected_assists": "3.0", "expected_goals_conceded": "10.0",
        "defensive_contribution_per_90": "1.0", "ict_index": "100", "bps": 400,
        "starts": 10,
    }
    player = Player.from_api(raw)
    hyped = minutes_probability(
        player, PlayerSentiment(2, availability_signal=1.0, form_signal=1.0, volume=50))
    assert hyped == 0.0


def test_a_strong_defence_beats_a_weak_one_on_clean_sheets():
    """The bug this replaced ranked Ipswich above Manchester City because it
    only read the fixture-difficulty digit, never the clubs' actual defences."""
    from fplbot.projection import clean_sheet_probability
    strong = clean_sheet_probability(team_xgc90=0.9, difficulty=4)   # elite D, hard game
    weak = clean_sheet_probability(team_xgc90=2.0, difficulty=3)     # poor D, easier game
    assert strong > weak


def test_clean_sheet_probability_falls_as_difficulty_rises():
    from fplbot.projection import clean_sheet_probability
    probs = [clean_sheet_probability(1.2, d) for d in (1, 2, 3, 4, 5)]
    assert probs == sorted(probs, reverse=True)


def test_bonus_uses_bps_not_just_ict(client):
    """The spec names BPS; an ICT-only bonus term ignores half the signal."""
    from fplbot.projection import bonus_estimate, position_medians
    from dataclasses import replace
    med = position_medians(client.players())
    base = max(client.players(), key=lambda p: p.minutes)
    prior = med[base.position]["bps"]
    high_bps = replace(base, bps=base.bps * 3)
    assert bonus_estimate(high_bps, prior) > bonus_estimate(base, prior)


def test_form_signal_changes_a_projection(client):
    """form/hype sentiment must actually move something. It previously did not."""
    settings = Settings(cache_dir="/tmp/x")
    target = max(client.players(), key=lambda p: p.expected_goals)
    common = dict(players=client.players(), teams=client.teams(),
                  fixtures=client.fixtures(), next_gw_id=client.next_gameweek().id,
                  settings=settings)
    base = project_all(**common)
    hyped = project_all(**common, sentiment={
        target.id: PlayerSentiment(target.id, availability_signal=0.0,
                                   form_signal=1.0, volume=5)})
    assert hyped[target.id].total > base[target.id].total


def test_low_minutes_player_falls_back_to_position_median(client):
    """Returning 0.0 for a thin sample silently zeroes every new signing."""
    from fplbot.projection import position_medians, _per_90
    med = position_medians(client.players())
    assert med["FWD"]["xg"] > 0
    # A 40-minute cameo is dominated by the prior, not by its own rate.
    assert _per_90(5.0, minutes=0, prior=med["FWD"]["xg"]) == med["FWD"]["xg"]
    thin = _per_90(5.0, minutes=40, prior=med["FWD"]["xg"])
    assert abs(thin - med["FWD"]["xg"]) < abs(thin - (5.0 / (40 / 90)))


def test_no_clean_sheet_probability_is_physically_implausible(client):
    """No Premier League defence keeps a clean sheet 55% of the time. A model
    that says otherwise is reading a three-match sample as settled fact."""
    from fplbot.projection import clean_sheet_probability, team_xgc_per_90
    xgc = team_xgc_per_90(client.players())
    worst = max((clean_sheet_probability(v, 1), tid) for tid, v in xgc.items())
    assert worst[0] <= 0.55, f"team {worst[1]} projects a {worst[0]:.0%} clean sheet"


def test_defensive_contribution_is_a_probability_not_a_ratio():
    """A player averaging exactly the threshold clears it about half the time.
    The bug this replaced credited him ~100% of the award."""
    from fplbot.projection import poisson_at_least
    at_threshold = poisson_at_least(12, 12.0)
    assert 0.3 < at_threshold < 0.6, at_threshold
    # And the ratio proxy it replaced would have said 1.0.
    assert at_threshold < 1.0


def test_defensive_contribution_probability_rises_with_the_rate():
    from fplbot.projection import poisson_at_least
    probs = [poisson_at_least(12, m) for m in (6.0, 9.0, 12.0, 15.0, 18.0)]
    assert probs == sorted(probs)
    assert probs[0] < 0.05 and probs[-1] > 0.9


def test_a_player_who_cannot_have_cleared_the_threshold_is_not_credited_as_if_he_did(client):
    """Janelt: 35 defensive actions across 3 matches, threshold 12. He cannot
    have cleared it more than twice, so he must not be credited near-fully."""
    from fplbot.projection import poisson_at_least, position_medians, _shrink
    med = position_medians(client.players())
    janelt = [p for p in client.players() if p.web_name == "Janelt"]
    if not janelt:
        import pytest
        pytest.skip("Janelt not in this snapshot")
    p = janelt[0]
    dc90 = _shrink(p.defensive_contribution_per_90, med["MID"]["dc"], p.minutes / 90)
    assert poisson_at_least(12, dc90) < 0.6


def test_shrinkage_pulls_a_small_sample_toward_the_prior():
    from fplbot.projection import _shrink
    # An extreme rate seen over 3 matches should land nearer the prior than
    # the observation; over 30 matches it should barely move.
    early = _shrink(observed=0.3, prior=1.5, matches=3)
    late = _shrink(observed=0.3, prior=1.5, matches=30)
    assert 0.3 < late < early < 1.5
    assert abs(late - 0.3) < abs(early - 0.3)


def test_no_budget_defender_outranks_the_premium_attackers(client):
    """The original bug put £4.0m Ipswich defenders alongside Haaland. This
    targets that shape directly, and has real margin unlike a bare count."""
    settings = Settings(cache_dir="/tmp/x")
    proj = project_all(players=client.players(), teams=client.teams(),
                       fixtures=client.fixtures(),
                       next_gw_id=client.next_gameweek().id, settings=settings)
    by_id = {p.id: p for p in client.players()}
    top10 = sorted(proj.values(), key=lambda x: x.total, reverse=True)[:10]
    cheap = [by_id[t.player_id].web_name for t in top10
             if by_id[t.player_id].position == "DEF" and by_id[t.player_id].now_cost <= 45]
    assert not cheap, f"budget defenders in the top 10: {cheap}"


def test_defenders_do_not_dominate_the_top_of_the_board(client):
    """Coarse net beneath the sharper guards above."""
    settings = Settings(cache_dir="/tmp/x")
    proj = project_all(players=client.players(), teams=client.teams(),
                       fixtures=client.fixtures(),
                       next_gw_id=client.next_gameweek().id, settings=settings)
    by_id = {p.id: p for p in client.players()}
    top20 = sorted(proj.values(), key=lambda x: x.total, reverse=True)[:20]
    defenders = sum(1 for t in top20 if by_id[t.player_id].position == "DEF")
    assert defenders <= 12, f"{defenders}/20 of the top projections are defenders"


def test_later_gameweeks_are_decayed(projections):
    """Pick a player scoring in both weeks; a blank gameweek is a legitimate
    zero and would make a naive comparison flake."""
    both = [p for p in projections.values()
            if p.per_gameweek[0] > 0 and p.per_gameweek[1] > 0]
    assert both, "no player projects points in both of the first two gameweeks"
    p = max(both, key=lambda x: x.total)
    assert p.per_gameweek[1] < p.per_gameweek[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_projection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot.projection'`

- [ ] **Step 3: Write `fplbot/projection.py`**

```python
"""Expected-points model.

Sentiment is a bounded nudge on availability and form, never a direct
addition to points, and it applies only to the next gameweek.
"""
from __future__ import annotations

import math
import statistics

from fplbot.config import SCORING, Settings
from fplbot.models import Fixture, Player, PlayerSentiment, Projection, Team

MINUTES_FLOOR = 270  # three full matches; used for selection share, not rates

# Prior weight in matches when regressing a rate toward its cohort mean.
# Chosen by measurement, not feel: K=2 lets a clean-sheet probability breach
# the 55% plausibility ceiling; K=6 leaves ~9 points of unused headroom under
# it while halving the league's defensive spread. K=4 keeps a 5-point margin
# and preserves noticeably more of the real signal.
SHRINKAGE_K = 4.0

LEAGUE_DEFAULT_XGC = 1.4  # goals per 90, used only when a club has no data

# Multiplies a club's own expected goals conceded for this fixture. Difficulty
# runs 1 (easiest) to 5 (hardest).
DIFFICULTY_MULTIPLIER = {1: 0.70, 2: 0.85, 3: 1.00, 4: 1.20, 5: 1.45}


def _shrink(observed: float, prior: float, matches: float) -> float:
    """Regress a rate toward a prior, weighted by how much football we have seen.

    Three matches into a season a raw per-90 rate is mostly noise. Arsenal's
    observed 0.31 xGC/90 implies a 74% clean-sheet chance; no real defence
    achieves that. Shrinking toward the cohort mean stops early-season
    extremes from driving squad selection, and fades out on its own as
    matches accumulate.
    """
    if matches <= 0:
        return prior
    return (matches * observed + SHRINKAGE_K * prior) / (matches + SHRINKAGE_K)


def _per_90(total: float, minutes: int, prior: float) -> float:
    """Shrunk per-90 rate. A continuous curve, not a cliff at a minutes floor."""
    matches = minutes / 90
    if matches <= 0:
        return prior
    return _shrink(total / matches, prior, matches)


def position_medians(players: list[Player]) -> dict[str, dict[str, float]]:
    """Median per-90 rates by position, for the low-minutes fallback."""
    out: dict[str, dict[str, float]] = {}
    for position in ("GKP", "DEF", "MID", "FWD"):
        pool = [p for p in players
                if p.position == position and p.minutes >= MINUTES_FLOOR]
        if not pool:
            out[position] = {"xg": 0.0, "xa": 0.0, "bps": 0.0, "dc": 0.0}
            continue
        out[position] = {
            "xg": statistics.median(p.expected_goals / (p.minutes / 90) for p in pool),
            "xa": statistics.median(p.expected_assists / (p.minutes / 90) for p in pool),
            "bps": statistics.median(p.bps / (p.minutes / 90) for p in pool),
            "dc": statistics.median(p.defensive_contribution_per_90 for p in pool),
        }
    return out


def team_xgc_per_90(players: list[Player]) -> dict[int, float]:
    """Each club's expected goals conceded per 90.

    Taken from the club's most-played goalkeeper: they are on the pitch for
    every goal conceded, so their expected_goals_conceded IS the club's. This
    is what the spec means by "the club's expected_goals_conceded" — using a
    fixture-difficulty digit alone cannot tell a good defence from a bad one.
    """
    raw: dict[int, tuple[float, float]] = {}  # team_id -> (rate, matches)
    for team_id in {p.team_id for p in players}:
        keepers = [p for p in players
                   if p.team_id == team_id and p.position == "GKP" and p.minutes >= 180]
        if keepers:
            gk = max(keepers, key=lambda p: p.minutes)
            matches = gk.minutes / 90
            raw[team_id] = (gk.expected_goals_conceded / matches, matches)
        else:
            pool = [(p.expected_goals_conceded / (p.minutes / 90), p.minutes / 90)
                    for p in players
                    if p.team_id == team_id and p.minutes >= 180]
            raw[team_id] = (
                (statistics.median(r for r, _ in pool), max(m for _, m in pool))
                if pool else (LEAGUE_DEFAULT_XGC, 0.0)
            )

    league_mean = statistics.mean(r for r, _ in raw.values()) if raw else LEAGUE_DEFAULT_XGC
    # Shrink toward the league mean: an elite defence three matches in is
    # partly real and partly a small sample, and the model should say so.
    return {tid: _shrink(rate, league_mean, matches) for tid, (rate, matches) in raw.items()}


def minutes_probability(
    player: Player, sentiment: PlayerSentiment | None = None, cap: float = 0.15
) -> float:
    """Probability the player is on the pitch, in [0, 1].

    The API's own injury flag outranks the internet: a player who is not
    available cannot be talked back onto the pitch by sentiment.
    """
    base = player.availability
    if base == 0.0:
        return 0.0

    if player.minutes >= MINUTES_FLOOR:
        share = min(1.0, player.minutes / (player.starts * 90) if player.starts else 0.7)
    else:
        share = 0.5
    base *= share

    if sentiment is not None:
        base *= 1 + cap * max(-1.0, min(1.0, sentiment.availability_signal))

    return max(0.0, min(1.0, base))


def poisson_at_least(threshold: int, mean: float) -> float:
    """P(X >= threshold) for X ~ Poisson(mean).

    Defensive contributions are counts, and the 2 points are awarded per match
    for clearing a threshold — so what matters is the PROBABILITY of clearing
    it, not the ratio of the average to it. A player averaging 11.67 actions
    against a threshold of 12 clears it roughly half the time; treating
    11.67/12 as a 97% share credits him nearly full points every match.
    """
    if mean <= 0:
        return 0.0
    term = math.exp(-mean)
    cumulative = term
    for k in range(1, threshold):
        term *= mean / k
        cumulative += term
    return max(0.0, min(1.0, 1.0 - cumulative))


def clean_sheet_probability(team_xgc90: float, difficulty: int) -> float:
    """Poisson probability of conceding zero, given the club's own xGC.

    P(0 goals) = exp(-expected_goals_conceded). Scaling the club's own rate by
    fixture difficulty means a weak defence facing an easy fixture and a strong
    defence facing a hard one are ranked on their actual quality, not on the
    difficulty digit alone.
    """
    adjusted = team_xgc90 * DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
    return math.exp(-max(0.05, adjusted))


def bonus_estimate(player: Player, prior_bps90: float) -> float:
    """Expected bonus points per appearance, from BPS rate and ICT.

    Bonus is awarded to the top three BPS scorers in a match. A player
    averaging well above the ~25 BPS mark earns bonus regularly; below it,
    rarely. ICT is blended in as a secondary signal of involvement.
    """
    bps90 = _per_90(player.bps, player.minutes, prior_bps90)
    from_bps = max(0.0, (bps90 - 18.0) / 12.0)
    from_ict = player.ict_index / 200
    return min(1.5, 0.7 * from_bps + 0.3 * from_ict)


def expected_points_one_gw(
    player: Player,
    team_xgc90: float,
    fixture: Fixture,
    minutes_prob: float,
    medians: dict[str, dict[str, float]],
    form_multiplier: float = 1.0,
) -> float:
    if minutes_prob <= 0 or fixture is None:
        return 0.0

    pos = player.position
    appearance = SCORING["appearance_60_plus"] * minutes_prob

    med = medians.get(pos, {"xg": 0.0, "xa": 0.0, "bps": 0.0})
    xg90 = _per_90(player.expected_goals, player.minutes, med["xg"])
    xa90 = _per_90(player.expected_assists, player.minutes, med["xa"])
    attacking = xg90 * SCORING["goal"][pos] + xa90 * SCORING["assist"][pos]

    at_home = fixture.team_h == player.team_id
    difficulty = fixture.team_h_difficulty if at_home else fixture.team_a_difficulty
    cs_points = SCORING["clean_sheet"][pos]
    clean_sheet = (
        clean_sheet_probability(team_xgc90, difficulty) * cs_points if cs_points else 0.0
    )

    threshold = SCORING["defensive_threshold"][pos]
    if threshold < 99:
        # Shrunk like every other rate, then converted to a probability of
        # clearing the threshold rather than a ratio against it.
        dc90 = _shrink(player.defensive_contribution_per_90, med["dc"],
                       player.minutes / 90)
        dc = SCORING["defensive_contribution"][pos] * poisson_at_least(threshold, dc90)
    else:
        dc = 0.0

    # The form channel moves attacking output and bonus — the parts of a
    # projection that genuine form talk is about. It does not touch clean
    # sheets or appearance, which are team and selection properties.
    scored = ((attacking + bonus_estimate(player, med["bps"])) * form_multiplier
              + clean_sheet + dc)
    return minutes_prob * scored + appearance


def project_all(
    players: list[Player],
    teams: list[Team],
    fixtures: list[Fixture],
    next_gw_id: int,
    settings: Settings,
    sentiment: dict[int, PlayerSentiment] | None = None,
) -> dict[int, Projection]:
    sentiment = sentiment or {}
    weights = settings.decay_weights
    medians = position_medians(players)
    team_xgc = team_xgc_per_90(players)

    fixtures_by_gw: dict[int, list[Fixture]] = {}
    for f in fixtures:
        if f.event is not None:
            fixtures_by_gw.setdefault(f.event, []).append(f)

    out: dict[int, Projection] = {}
    for player in players:
        sig = sentiment.get(player.id)
        per_gw: list[float] = []

        for offset, weight in enumerate(weights):
            gw_id = next_gw_id + offset
            # Sentiment applies to GW+1 only. Spreading it across the decayed
            # horizon would dilute the +/-15% guardrail to roughly +/-6%.
            sig_for_gw = sig if offset == 0 else None
            mp = minutes_probability(player, sig_for_gw, settings.sentiment_cap)
            form_mult = 1.0
            if sig_for_gw is not None:
                form_mult = 1 + settings.sentiment_cap * max(
                    -1.0, min(1.0, sig_for_gw.form_signal))

            gw_fixtures = [
                f for f in fixtures_by_gw.get(gw_id, [])
                if player.team_id in (f.team_h, f.team_a)
            ]
            # Blanks score nothing; double gameweeks score twice.
            points = sum(
                expected_points_one_gw(
                    player, team_xgc.get(player.team_id, 1.4), f, mp, medians, form_mult)
                for f in gw_fixtures
            )
            per_gw.append(points * weight)

        applied = settings.sentiment_cap * sig.availability_signal if sig else 0.0
        out[player.id] = Projection(
            player_id=player.id,
            per_gameweek=per_gw,
            total=sum(per_gw),
            sentiment_applied=applied,
            explanation=(
                f"{player.web_name} ({player.position}, £{player.price_m}m): "
                f"{sum(per_gw):.2f} pts over {settings.horizon} GW"
                + (f", sentiment {applied:+.1%}" if sig else "")
            ),
        )
    return out
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_projection.py -v`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add fplbot/projection.py tests/test_projection.py
git commit -m "feat: expected-points model with capped GW+1 sentiment modifier"
```

---

## Task 5: ILP squad optimizer

**Files:**
- Create: `fplbot/optimize.py`
- Test: `tests/test_optimize.py`

**Interfaces:**
- Consumes: `Player`, `Projection`, `Squad` from `fplbot.models`; `squad_rules()` output from `fplbot.fetch`
- Produces: `pick_squad(players, projections, rules) -> Squad`; `pick_xi(squad_player_ids, projections, rules, players) -> tuple[list[int], int]`; exception `InfeasibleSquad(Exception)`

**Background the implementer needs:** This is a binary integer program, and PuLP with its bundled CBC solver handles it in well under a second at this size. Maximise the sum of projected points over a 0/1 selection variable per player, subject to budget, squad size, per-position quota, and the per-club limit.

Two things that bite:

- **Keep money in integers.** `now_cost` is tenths of a million and the budget is 1000. Introducing floats here produces infeasibility at the last tenth for no visible reason.
- **Solve the XI separately.** The 15-man squad and the starting XI have different constraints — the squad quota is 2/5/5/3, while the XI needs 1 GKP and a legal outfield formation. Solving them as one program is possible but harder to read and debug; two small programs are clearer.

When the model is infeasible, report which constraint bound rather than returning a partial squad. An infeasible FPL squad almost always means the projections are broken (for example every player projecting zero), not that the constraints are wrong.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_optimize.py
import pytest

from fplbot.config import Settings
from fplbot.optimize import InfeasibleSquad, pick_squad, pick_xi
from fplbot.projection import project_all


@pytest.fixture
def setup(client):
    players = client.players()
    rules = client.squad_rules()
    projections = project_all(
        players=players, teams=client.teams(), fixtures=client.fixtures(),
        next_gw_id=client.next_gameweek().id, settings=Settings(cache_dir="/tmp/x"),
    )
    return players, projections, rules


def test_squad_has_fifteen_players(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    assert len(squad.players) == 15
    assert len(set(squad.players)) == 15


def test_squad_respects_budget(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    assert sum(by_id[i].now_cost for i in squad.players) <= rules["budget"]


def test_squad_respects_position_quota(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    counts = {}
    for i in squad.players:
        counts[by_id[i].position] = counts.get(by_id[i].position, 0) + 1
    assert counts == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_squad_respects_three_per_club_limit(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    counts = {}
    for i in squad.players:
        counts[by_id[i].team_id] = counts.get(by_id[i].team_id, 0) + 1
    assert max(counts.values()) <= rules["team_limit"]


def test_xi_is_eleven_with_a_legal_formation(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    xi, captain = pick_xi(squad.players, projections, rules, players)
    assert len(xi) == 11
    assert set(xi) <= set(squad.players)
    counts = {}
    for i in xi:
        counts[by_id[i].position] = counts.get(by_id[i].position, 0) + 1
    assert counts["GKP"] == 1
    assert 3 <= counts["DEF"] <= 5
    assert 1 <= counts["FWD"] <= 3


def test_captain_is_the_highest_projected_starter(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    xi, captain = pick_xi(squad.players, projections, rules, players)
    assert captain in xi
    assert projections[captain].total == max(projections[i].total for i in xi)


def test_optimum_beats_a_naive_greedy_pick(setup):
    """The whole reason for using an ILP rather than sorting by points.

    The greedy walk reserves enough budget to fill its remaining slots at the
    cheapest available price, so it always completes a legal 15. An earlier
    version of this test let greedy stall at 13 players and then skipped its
    only assertion — it passed while proving nothing.
    """
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    optimal = sum(projections[i].total for i in squad.starting_xi)

    budget = rules["budget"]
    remaining = dict(rules["quota"])
    cheapest = {pos: min(p.now_cost for p in players if p.position == pos)
                for pos in remaining}
    greedy, clubs = [], {}

    for p in sorted(players, key=lambda p: projections[p.id].total, reverse=True):
        if remaining.get(p.position, 0) == 0:
            continue
        if clubs.get(p.team_id, 0) >= rules["team_limit"]:
            continue
        reserve = sum(cheapest[q] * (remaining[q] - (1 if q == p.position else 0))
                      for q in remaining)
        if p.now_cost + reserve > budget:
            continue
        greedy.append(p.id)
        budget -= p.now_cost
        remaining[p.position] -= 1
        clubs[p.team_id] = clubs.get(p.team_id, 0) + 1

    assert len(greedy) == 15, f"greedy built only {len(greedy)} players; test is vacuous"
    greedy_xi, _ = pick_xi(greedy, projections, rules, players)
    assert optimal >= sum(projections[i].total for i in greedy_xi)


def test_bench_is_discounted_but_still_playable(setup):
    """Bench points are worth a fraction of starting points, but a bench of
    players with no chance of appearing is worthless when a starter is ruled
    out before the deadline."""
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    bench = [i for i in squad.players if i not in set(squad.starting_xi)]
    assert len(bench) == 4
    unplayable = [by_id[i].web_name for i in bench if by_id[i].availability <= 0]
    assert not unplayable, f"bench players who cannot play: {unplayable}"


def test_starting_xi_is_chosen_jointly_with_the_squad(setup):
    """pick_squad returns its own XI; it does not defer to a second solve."""
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    assert len(squad.starting_xi) == rules["xi_size"]
    assert set(squad.starting_xi) <= set(squad.players)


def test_infeasible_projections_raise_a_clear_error(setup):
    players, projections, rules = setup
    with pytest.raises(InfeasibleSquad):
        pick_squad(players, projections, {**rules, "budget": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_optimize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot.optimize'`

- [ ] **Step 3: Write `fplbot/optimize.py`**

```python
"""Squad selection as a binary integer program.

An ILP finds the true optimum. Sorting by points and taking the best
affordable player at each position does not, and the gap is worth real
points across a season.
"""
from __future__ import annotations

import pulp

from fplbot.models import Player, Projection, Squad


# A benched player scores only if an auto-substitution fires, so bench points
# are worth a fraction of starting points. Weighting them equally spends real
# budget on players who mostly never play; weighting them zero produces a
# bench of non-starters that is useless when someone is ruled out on the
# morning of a deadline. Measured on the snapshot: 0.1 recovers the full XI
# gain of a pure XI-max objective while keeping a playable bench.
BENCH_WEIGHT = 0.1


class InfeasibleSquad(Exception):
    """No legal squad exists under these constraints."""


def pick_squad(
    players: list[Player], projections: dict[int, Projection], rules: dict
) -> Squad:
    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    # Squad and starting-XI membership are solved together. Choosing the 15
    # first and the XI afterwards optimises the wrong quantity: it values a
    # bench upgrade as highly as a starter upgrade.
    pick = {p.id: pulp.LpVariable(f"sq{p.id}", cat="Binary") for p in players}
    start = {p.id: pulp.LpVariable(f"st{p.id}", cat="Binary") for p in players}

    problem += pulp.lpSum(
        projections[p.id].total
        * (start[p.id] + BENCH_WEIGHT * (pick[p.id] - start[p.id]))
        for p in players
    )

    problem += pulp.lpSum(pick.values()) == rules["squad_size"]
    problem += pulp.lpSum(start.values()) == rules["xi_size"]
    # Costs are integer tenths of a million; keep them integers throughout.
    problem += pulp.lpSum(p.now_cost * pick[p.id] for p in players) <= rules["budget"]

    for p in players:
        problem += start[p.id] <= pick[p.id]

    for position, count in rules["quota"].items():
        problem += (
            pulp.lpSum(pick[p.id] for p in players if p.position == position) == count
        )

    for position, (low, high) in rules["formation"].items():
        in_position = [start[p.id] for p in players if p.position == position]
        problem += pulp.lpSum(in_position) >= low
        problem += pulp.lpSum(in_position) <= high

    for team_id in {p.team_id for p in players}:
        problem += (
            pulp.lpSum(pick[p.id] for p in players if p.team_id == team_id)
            <= rules["team_limit"]
        )

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleSquad(
            f"solver returned {pulp.LpStatus[status]}. "
            f"Budget {rules['budget']}, cheapest legal squad costs "
            f"{_cheapest_possible(players, rules)}. "
            "If the budget looks fine, check that projections are non-zero."
        )

    chosen = [p.id for p in players if pick[p.id].value() == 1]
    xi = [p.id for p in players if start[p.id].value() == 1]
    by_id = {p.id: p for p in players}
    spend = sum(by_id[i].now_cost for i in chosen)
    captain = max(xi, key=lambda i: projections[i].total)
    return Squad(
        players=chosen, starting_xi=xi, captain_id=captain,
        bank=rules["budget"] - spend,
    )


def _cheapest_possible(players: list[Player], rules: dict) -> int:
    total = 0
    for position, count in rules["quota"].items():
        costs = sorted(p.now_cost for p in players if p.position == position)
        total += sum(costs[:count])
    return total


def pick_xi(
    squad_player_ids: list[int],
    projections: dict[int, Projection],
    rules: dict,
    players: list[Player],
) -> tuple[list[int], int]:
    by_id = {p.id: p for p in players if p.id in set(squad_player_ids)}

    problem = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    start = {i: pulp.LpVariable(f"s{i}", cat="Binary") for i in squad_player_ids}

    problem += pulp.lpSum(projections[i].total * start[i] for i in squad_player_ids)
    problem += pulp.lpSum(start.values()) == rules["xi_size"]

    for position, (low, high) in rules["formation"].items():
        in_position = [start[i] for i in squad_player_ids if by_id[i].position == position]
        problem += pulp.lpSum(in_position) >= low
        problem += pulp.lpSum(in_position) <= high

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleSquad(f"no legal XI: solver returned {pulp.LpStatus[status]}")

    xi = [i for i in squad_player_ids if start[i].value() == 1]
    captain = max(xi, key=lambda i: projections[i].total)
    return xi, captain
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_optimize.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add fplbot/optimize.py tests/test_optimize.py
git commit -m "feat: ILP squad and XI selection"
```

---

## Task 6: News and comment collection

**Files:**
- Create: `fplbot/news.py`
- Test: `tests/test_news.py`

**Interfaces:**
- Consumes: `Settings` from `fplbot.config`; `NewsItem` from `fplbot.models`
- Produces: `collect(settings, reddit=None, now=None) -> tuple[list[NewsItem], dict]` returning items and a stats dict with keys `items_fetched`, `items_after_filter`, `sources_used`, `sources_absent`, `reddit_available`, `reddit_status`; `filter_items(items, settings, now) -> list[NewsItem]`; `normalise(text) -> str`; module constants `FEEDS`, `SOURCE_TRUST`; classes `RssCollector`, `RedditCollector`

**Background the implementer needs:** The signal worth having is early availability news — injuries, knocks, rotation hints — reaching us before the statistics absorb it. Six news feeds are the primary source, all verified returning items on 2026-09-07. ESPN's soccer feed returns an empty document and Football365's 404s; both are deliberately excluded, so do not add them back.

Sources carry a **trust tier**, exported as `SOURCE_TRUST`, which Task 7 uses to weight a mention's confidence. BBC, Guardian and Sky are 1.0; talkSPORT, Metro and Mirror are 0.6. The tabloids break real team news often enough to be worth reading and speculate often enough to be worth discounting.

**Reddit is optional and gated on approval.** Reddit's Responsible Builder Policy requires explicit approved access before using their Data API; creating a script app is not sufficient. So `RedditCollector` runs only when credentials are present, and its absence is a normal state reported in `sources_absent` — not an error and not a fallback. Do not add an unauthenticated Reddit RSS path: a run either has approved API access or reports Reddit as absent.

Two policy constraints bind this file: never store a comment author (`NewsItem` has no `author` field — do not add one), and never retain collected text beyond the run.

Reddit rate limits are not a real constraint (100 QPM per client, ~15-25 calls per run) so do not build throttling. `/comments/{id}` is not a listing and has no `after` cursor — use `sort="new"` plus a bounded `replace_more`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_news.py
from datetime import datetime, timedelta, timezone

from fplbot.config import Settings
from fplbot.models import NewsItem
from fplbot.news import FEEDS, SOURCE_TRUST, collect, filter_items, normalise

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def item(**kw):
    base = dict(source="bbc", url="u", published_at=NOW - timedelta(hours=1),
                title="t", body="Saka looked sharp", score=10)
    return NewsItem(**{**base, **kw})


def test_every_feed_has_a_trust_tier():
    for name, _url in FEEDS:
        assert name in SOURCE_TRUST, f"{name} has no trust tier"
    assert SOURCE_TRUST["bbc"] > SOURCE_TRUST["mirror"]


def test_excluded_feeds_are_not_present():
    """ESPN returns an empty document and Football365 404s (verified 2026-09-07)."""
    urls = " ".join(u for _n, u in FEEDS)
    assert "espn" not in urls
    assert "football365" not in urls


def test_drops_low_scoring_items():
    kept = filter_items([item(score=1), item(score=5)], Settings(cache_dir="/tmp/x"), NOW)
    assert len(kept) == 1


def test_drops_stale_items():
    old = item(published_at=NOW - timedelta(hours=72))
    assert filter_items([old], Settings(cache_dir="/tmp/x"), NOW) == []


def test_deduplicates_on_normalised_text():
    """The same claim is syndicated across outlets and would otherwise count
    as several independent pieces of evidence."""
    a = item(body="Saka is a doubt for Saturday!!")
    b = item(source="mirror", body="saka is a doubt for saturday")
    assert len(filter_items([a, b], Settings(cache_dir="/tmp/x"), NOW)) == 1


def test_respects_the_item_cap():
    settings = Settings(cache_dir="/tmp/x", comment_cap=5)
    many = [item(body=f"unique text {i}") for i in range(50)]
    assert len(filter_items(many, settings, NOW)) == 5


def test_keeps_newest_first_when_capping():
    settings = Settings(cache_dir="/tmp/x", comment_cap=1)
    older = item(body="older", published_at=NOW - timedelta(hours=10))
    newer = item(body="newer", published_at=NOW - timedelta(hours=1))
    assert filter_items([older, newer], settings, NOW)[0].body == "newer"


def test_normalise_strips_case_and_punctuation():
    assert normalise("Saka  is  OUT!!") == normalise("saka is out")


def test_news_item_has_no_author_field():
    """Reddit policy forbids inferring characteristics about users. The
    pipeline must not carry an author at all."""
    assert not hasattr(item(), "author")


def test_collect_without_reddit_reports_it_absent_not_failed(monkeypatch):
    import fplbot.news as news
    monkeypatch.setattr(news.RssCollector, "collect", lambda self: [item()])
    items, stats = collect(Settings(cache_dir="/tmp/x"), now=NOW)
    assert stats["reddit_available"] is False
    assert "reddit" in stats["sources_absent"]
    assert stats["reddit_status"] == "not_configured"
    assert items, "news-only run must still produce items"


def test_a_broken_reddit_is_distinguishable_from_an_absent_one(monkeypatch):
    """Both end up in sources_absent, but "never set up" and "set up and
    broken" need different responses from whoever reads the run."""
    import fplbot.news as news

    class Boom:
        def subreddit(self, name):
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(news.RssCollector, "collect", lambda self: [item()])
    items, stats = collect(Settings(cache_dir="/tmp/x"), reddit=Boom(), now=NOW)
    assert stats["reddit_available"] is False
    assert stats["reddit_status"].startswith("failed:")
    assert items, "a broken Reddit must not lose the news items"


def test_distinct_stories_sharing_an_opening_are_not_merged():
    """Syndicated copy shares lead sentences. Keying dedupe on a prefix would
    merge two different stories and throw away real evidence."""
    lead = "The Premier League returns this weekend after the international break. "
    a = item(body=lead + "Saka is expected to start against Chelsea.")
    b = item(source="sky", body=lead + "Haaland has been ruled out with a knock.")
    kept = filter_items([a, b], Settings(cache_dir="/tmp/x"), NOW)
    assert len(kept) == 2, "two different stories were merged as duplicates"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_news.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot.news'`

- [ ] **Step 3: Write `fplbot/news.py`**

```python
"""Collect football text from news feeds, and optionally Reddit.

News is the primary source: the signal worth having is early availability
news, which reaches the press before it reaches the statistics. Reddit is
optional and requires approved Data API access.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import feedparser

from fplbot.config import Settings, user_agent
from fplbot.models import NewsItem

log = logging.getLogger(__name__)

SUBREDDIT = "FantasyPL"

# Verified returning items on 2026-09-07. ESPN's soccer feed returns an empty
# document and Football365's 404s; both are excluded deliberately.
FEEDS = [
    ("bbc", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
    ("guardian", "https://www.theguardian.com/football/rss"),
    ("sky", "https://www.skysports.com/rss/12040"),
    ("talksport", "https://talksport.com/football/feed/"),
    ("metro", "https://metro.co.uk/sport/football/feed/"),
    ("mirror", "https://www.mirror.co.uk/sport/football/?service=rss"),
]

# Multiplies a mention's confidence during aggregation (Task 7). The tabloids
# break real team news often enough to read and speculate often enough to
# discount.
SOURCE_TRUST = {
    "bbc": 1.0, "guardian": 1.0, "sky": 1.0,
    "talksport": 0.6, "metro": 0.6, "mirror": 0.6,
    "reddit-post": 0.7, "reddit-comment": 0.6,
}

# These threads carry the most comments and the least decision-relevant text.
EXCLUDED_TITLE_PATTERNS = re.compile(
    r"(match thread|live thread|bonus point|rate my team|who to captain\?)", re.I
)


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def filter_items(items: list[NewsItem], settings: Settings, now: datetime) -> list[NewsItem]:
    cutoff = now - timedelta(hours=settings.max_age_hours)
    seen: set[str] = set()
    kept: list[NewsItem] = []

    for it in sorted(items, key=lambda i: i.published_at, reverse=True):
        if it.score < settings.score_floor:
            continue
        if it.published_at < cutoff:
            continue
        # Key on the whole normalised body, not a prefix. Truncating to a
        # fixed prefix merges distinct stories that share a syndicated opening
        # sentence, and silently discarding real evidence is worse than
        # counting one claim twice.
        key = normalise(it.body)
        if key in seen:
            continue
        seen.add(key)
        kept.append(it)
        if len(kept) >= settings.comment_cap:
            break

    return kept


class RssCollector:
    def __init__(self, feeds=FEEDS, default_score: int = 5):
        self.feeds = feeds
        self.default_score = default_score

    def collect(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        for name, url in self.feeds:
            try:
                parsed = feedparser.parse(url)
            except Exception as exc:
                log.warning("feed %s failed: %s", name, exc)
                continue
            for entry in parsed.entries:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if not published:
                    continue
                items.append(
                    NewsItem(
                        source=name,
                        url=entry.get("link", ""),
                        published_at=datetime(*published[:6], tzinfo=timezone.utc),
                        title=entry.get("title", ""),
                        body=entry.get("summary", entry.get("title", "")),
                        # RSS carries no score; pass the floor so these are kept.
                        score=self.default_score,
                    )
                )
        return items


class RedditCollector:
    """Optional. Requires approved Reddit Data API access."""

    def __init__(self, settings: Settings, reddit=None):
        self.settings = settings
        self.reddit = reddit or self._connect()

    def _connect(self):
        import praw

        return praw.Reddit(
            client_id=self.settings.reddit_client_id,
            client_secret=self.settings.reddit_client_secret,
            user_agent=user_agent(self.settings.reddit_username or "fplbot"),
        )

    def collect(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        subreddit = self.reddit.subreddit(SUBREDDIT)

        threads = [
            s for s in subreddit.hot(limit=25)
            if not EXCLUDED_TITLE_PATTERNS.search(s.title)
        ][: self.settings.max_threads]

        for submission in threads:
            items.append(
                NewsItem(
                    source="reddit-post",
                    url=f"https://reddit.com{submission.permalink}",
                    published_at=datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc),
                    title=submission.title,
                    body=submission.selftext or submission.title,
                    score=submission.score,
                )
            )
            submission.comment_sort = "new"
            # /comments/{id} has no cursor; depth and replace_more are the
            # only levers on volume.
            submission.comments.replace_more(limit=2)
            for comment in submission.comments.list():
                # No author is recorded, deliberately: Reddit policy forbids
                # inferring characteristics about users.
                items.append(
                    NewsItem(
                        source="reddit-comment",
                        url=f"https://reddit.com{comment.permalink}",
                        published_at=datetime.fromtimestamp(
                            comment.created_utc, tz=timezone.utc),
                        title=submission.title,
                        body=comment.body,
                        score=comment.score,
                    )
                )
                if len(items) >= self.settings.comment_ceiling:
                    return items
        return items


def collect(settings: Settings, reddit=None, now: datetime | None = None):
    now = now or datetime.now(tz=timezone.utc)
    raw: list[NewsItem] = []
    sources_used: list[str] = []
    sources_absent: list[str] = []

    try:
        raw += RssCollector().collect()
        sources_used.append("news-rss")
    except Exception as exc:
        log.warning("news feeds failed: %s", exc)
        sources_absent.append("news-rss")

    have_credentials = bool(settings.reddit_client_id and settings.reddit_client_secret)
    reddit_available = False
    # "Not configured" and "configured but broken" are different situations:
    # the first is the expected default, the second is worth alerting on.
    # Collapsing both into one absent-source string hides that from anyone
    # reading the run's stats afterwards.
    if have_credentials or reddit is not None:
        try:
            raw += RedditCollector(settings, reddit).collect()
            sources_used.append("reddit")
            reddit_available = True
            reddit_status = "ok"
        except Exception as exc:
            log.warning("Reddit collection failed: %s", exc)
            sources_absent.append("reddit")
            reddit_status = f"failed: {type(exc).__name__}"
    else:
        # Expected default: Reddit requires approved Data API access.
        log.info("no Reddit credentials; running on news feeds alone")
        sources_absent.append("reddit")
        reddit_status = "not_configured"

    kept = filter_items(raw, settings, now)
    stats = {
        "items_fetched": len(raw),
        "items_after_filter": len(kept),
        "sources_used": sources_used,
        "sources_absent": sources_absent,
        "reddit_available": reddit_available,
        "reddit_status": reddit_status,
    }
    return kept, stats
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_news.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add fplbot/news.py tests/test_news.py
git commit -m "feat: news-first collection with trust tiers and optional Reddit"
```

---

## Task 7: Mention resolution and sentiment scoring

**Files:**
- Create: `fplbot/sentiment.py`
- Test: `tests/test_sentiment.py`

**Interfaces:**
- Consumes: `Player`, `NewsItem`, `Mention`, `MentionScore`, `PlayerSentiment` from `fplbot.models`; `Settings` from `fplbot.config`; `SOURCE_TRUST` from `fplbot.news`
- Produces: `resolve_mentions(items, players) -> list[Mention]`; `Scorer` protocol with `score(mentions) -> dict[int, MentionScore]` keyed by index into `mentions`; `NullScorer`, `ApiScorer(api_key, model="claude-haiku-4-5")`, `SessionScorer(batch_path, scores_path)`; `aggregate(mentions, scores: dict[int, MentionScore], now, half_life_days=7) -> dict[int, PlayerSentiment]`

**Background the implementer needs:** Resolution is deterministic string matching, not a model call. Match on `web_name` and on surname, case-insensitively, at word boundaries. Ambiguity is real and common — several Premier League squads contain players who share a surname — so when a surname matches more than one player, resolve using club context in the same text and **drop the mention if that fails**. A wrongly attributed injury rumour is worse than a missing one.

Aggregation collapses many mentions into two signals per player. `injury` and `rotation` categories feed `availability_signal`; `form` and `hype` feed `form_signal`. Weight each mention by its confidence and by recency, using a 7-day half-life: `weight = confidence * 0.5 ** (age_days / 7)`. Mention volume is recorded but never fed into the model — it measures how popular a player is, not how good, and letting it into the projection would systematically favour already-owned players.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sentiment.py
from datetime import datetime, timedelta, timezone

from fplbot.models import Mention, MentionScore, NewsItem, Player
from fplbot.sentiment import NullScorer, aggregate, resolve_mentions

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def make_player(pid, name, team=1):
    return Player.from_api({
        "id": pid, "web_name": name, "element_type": 3, "team": team,
        "now_cost": 50, "status": "a", "chance_of_playing_next_round": None,
        "minutes": 900, "form": "3.0", "selected_by_percent": "5",
        "expected_goals": "1.0", "expected_assists": "1.0",
        "expected_goals_conceded": "10.0", "defensive_contribution_per_90": "1.0",
        "ict_index": "50", "bps": 200, "starts": 10,
    })


def item(body, hours_ago=1):
    return NewsItem(source="reddit-comment", url="u",
                    published_at=NOW - timedelta(hours=hours_ago),
                    title="t", body=body, score=10)


def test_resolves_a_plain_mention():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka looked sharp today")], players)
    assert len(mentions) == 1
    assert mentions[0].player_id == 1


def test_matching_is_case_insensitive_and_word_bounded():
    players = [make_player(1, "Saka")]
    assert len(resolve_mentions([item("SAKA is fit")], players)) == 1
    assert resolve_mentions([item("Sakamoto scored")], players) == []


def test_ambiguous_surname_without_club_context_is_dropped():
    """A wrongly attributed injury rumour is worse than a missing one."""
    players = [make_player(1, "Silva", team=1), make_player(2, "Silva", team=2)]
    assert resolve_mentions([item("Silva is injured")], players) == []


def test_ambiguous_surname_resolved_by_club_context():
    players = [make_player(1, "Silva", team=1), make_player(2, "Silva", team=2)]
    teams = {1: "Arsenal", 2: "Chelsea"}
    mentions = resolve_mentions([item("Chelsea's Silva is injured")], players, teams)
    assert [m.player_id for m in mentions] == [2]


def test_null_scorer_returns_nothing():
    assert NullScorer().score([1, 2, 3]) == {}


def test_scores_are_matched_by_index_not_position():
    """A discarded batch must not shift scores onto the wrong player."""
    players = [make_player(1, "Saka"), make_player(2, "Odegaard")]
    mentions = resolve_mentions(
        [item("Saka has a knock"), item("Odegaard has a knock")], players)
    assert [m.player_id for m in mentions] == [1, 2]
    # Only the second mention scored; the first was dropped as malformed.
    agg = aggregate(mentions, {1: MentionScore(2, -0.8, "injury", 0.9)}, NOW)
    assert 1 not in agg
    assert agg[2].availability_signal < 0


def test_aggregate_splits_availability_and_form_signals():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka has a knock")], players)
    scores = {0: MentionScore(1, sentiment=-0.8, category="injury", confidence=0.9)}
    agg = aggregate(mentions, scores, NOW)
    assert agg[1].availability_signal < 0
    assert agg[1].form_signal == 0


def test_aggregate_decays_old_mentions():
    players = [make_player(1, "Saka")]
    recent = resolve_mentions([item("Saka has a knock", hours_ago=1)], players)
    old = resolve_mentions([item("Saka has a knock", hours_ago=24 * 14)], players)
    score = {0: MentionScore(1, sentiment=-1.0, category="injury", confidence=1.0)}
    fresh = aggregate(recent, score, NOW)[1].availability_signal
    stale = aggregate(old, score, NOW)[1].availability_signal
    assert abs(stale) < abs(fresh)


def test_aggregate_signals_stay_in_range():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka has a knock")] * 1, players)
    scores = {0: MentionScore(1, sentiment=-5.0, category="injury", confidence=2.0)}
    agg = aggregate(mentions, scores, NOW)
    assert -1.0 <= agg[1].availability_signal <= 1.0


def test_volume_is_recorded_but_separate():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka good"), item("Saka great")], players)
    scores = {0: MentionScore(1, 0.5, "form", 0.8), 1: MentionScore(1, 0.5, "form", 0.8)}
    agg = aggregate(mentions, scores, NOW)
    assert agg[1].volume == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sentiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot.sentiment'`

- [ ] **Step 3: Write `fplbot/sentiment.py`**

```python
"""Resolve player mentions in text, score them, and aggregate per player."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Protocol

from fplbot.models import Mention, MentionScore, NewsItem, Player, PlayerSentiment
from fplbot.news import SOURCE_TRUST

log = logging.getLogger(__name__)

AVAILABILITY_CATEGORIES = {"injury", "rotation"}
FORM_CATEGORIES = {"form", "hype"}
VALID_CATEGORIES = AVAILABILITY_CATEGORIES | FORM_CATEGORIES

# Names the plain matcher gets wrong. Extend as mismatches show up in runs.
ALIASES: dict[str, str] = {
    "kdb": "De Bruyne",
    "trent": "Alexander-Arnold",
    "vvd": "van Dijk",
}


def _build_index(players: list[Player]) -> dict[str, list[Player]]:
    index: dict[str, list[Player]] = {}
    for p in players:
        for key in {p.web_name.lower(), p.web_name.split()[-1].lower()}:
            index.setdefault(key, []).append(p)
    return index


def resolve_mentions(
    items: list[NewsItem],
    players: list[Player],
    team_names: dict[int, str] | None = None,
) -> list[Mention]:
    index = _build_index(players)
    team_names = team_names or {}
    mentions: list[Mention] = []

    for item in items:
        text = f"{item.title} {item.body}"
        lowered = text.lower()
        for alias, real in ALIASES.items():
            lowered = re.sub(rf"\b{re.escape(alias)}\b", real.lower(), lowered)

        for key, matches in index.items():
            if not re.search(rf"\b{re.escape(key)}\b", lowered):
                continue

            if len(matches) == 1:
                mentions.append(Mention(matches[0].id, item, key))
                continue

            # Ambiguous surname. Resolve by club named in the same text, or
            # drop it: a misattributed injury rumour is worse than none.
            disambiguated = [
                p for p in matches
                if team_names.get(p.team_id, "\0").lower() in lowered
            ]
            if len(disambiguated) == 1:
                mentions.append(Mention(disambiguated[0].id, item, key))
            else:
                log.debug("dropped ambiguous mention %r", key)

    return mentions


class Scorer(Protocol):
    def score(self, mentions: list[Mention]) -> dict[int, MentionScore]:
        """Return scores keyed by index into `mentions`.

        Keyed rather than positional on purpose: a discarded malformed batch
        must not shift every later score onto the wrong player.
        """
        ...


class NullScorer:
    """Stats-only runs. Implemented as a scorer so no branch is needed
    anywhere else in the pipeline."""

    def score(self, mentions) -> dict[int, MentionScore]:
        return {}


PROMPT = """You are scoring football text for Fantasy Premier League decisions.

For each numbered item, return one JSON object with:
  index      - the item number given
  player_id  - the integer given
  sentiment  - float -1.0 (very bad news) to 1.0 (very good news)
  category   - one of: injury, rotation, form, hype
  confidence - float 0.0 to 1.0

Judge the implication for the player's next match. "Has a knock" is strongly
negative even with no negative words. Banter and jokes get low confidence.

Return a JSON array only, no prose.

Items:
"""


class ApiScorer:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5", batch_size: int = 40):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size

    def score(self, mentions: list[Mention]) -> dict[int, MentionScore]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        out: dict[int, MentionScore] = {}

        for start in range(0, len(mentions), self.batch_size):
            batch = mentions[start : start + self.batch_size]
            listing = "\n".join(
                f"{start + i}. player_id={m.player_id}: {m.item.body[:300]}"
                for i, m in enumerate(batch)
            )
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": PROMPT + listing}],
            )
            out.update(_parse_scores(response.content[0].text))

        return out


class SessionScorer:
    """Writes a batch for a Claude session to score, reads the result back.

    Used by the scheduled routine so no API key is needed.
    """

    def __init__(self, batch_path: Path, scores_path: Path):
        self.batch_path = Path(batch_path)
        self.scores_path = Path(scores_path)

    def score(self, mentions: list[Mention]) -> dict[int, MentionScore]:
        self.batch_path.write_text(json.dumps([
            {"index": i, "player_id": m.player_id, "text": m.item.body[:300]}
            for i, m in enumerate(mentions)
        ], indent=2))
        if not self.scores_path.exists():
            log.warning("no scores at %s; treating run as stats-only", self.scores_path)
            return {}
        return _parse_scores(self.scores_path.read_text())


def _parse_scores(text: str) -> dict[int, MentionScore]:
    """Discard a malformed batch rather than guessing at scores."""
    try:
        start, end = text.index("["), text.rindex("]") + 1
        rows = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("discarding malformed score batch: %s", exc)
        return {}

    out: dict[int, MentionScore] = {}
    for row in rows:
        try:
            category = row["category"]
            if category not in VALID_CATEGORIES:
                continue
            out[int(row["index"])] = MentionScore(
                player_id=int(row["player_id"]),
                sentiment=max(-1.0, min(1.0, float(row["sentiment"]))),
                category=category,
                confidence=max(0.0, min(1.0, float(row["confidence"]))),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def aggregate(
    mentions: list[Mention],
    scores: dict[int, MentionScore],
    now: datetime,
    half_life_days: float = 7.0,
) -> dict[int, PlayerSentiment]:
    by_player: dict[int, dict] = {}

    for index, mention in enumerate(mentions):
        score = scores.get(index)
        if score is None:
            continue
        age_days = (now - mention.item.published_at).total_seconds() / 86400
        # Trust tier discounts sources that speculate; see news.SOURCE_TRUST.
        trust = SOURCE_TRUST.get(mention.item.source, 0.6)
        weight = (min(1.0, max(0.0, score.confidence)) * trust
                  * 0.5 ** (age_days / half_life_days))
        bucket = by_player.setdefault(
            mention.player_id,
            {"avail": 0.0, "avail_w": 0.0, "form": 0.0, "form_w": 0.0,
             "volume": 0, "evidence": []},
        )
        bucket["volume"] += 1
        clamped = max(-1.0, min(1.0, score.sentiment))

        if score.category in AVAILABILITY_CATEGORIES:
            bucket["avail"] += clamped * weight
            bucket["avail_w"] += weight
        else:
            bucket["form"] += clamped * weight
            bucket["form_w"] += weight

        if len(bucket["evidence"]) < 3:
            bucket["evidence"].append(mention.item.body[:160])

    out: dict[int, PlayerSentiment] = {}
    for player_id, b in by_player.items():
        out[player_id] = PlayerSentiment(
            player_id=player_id,
            availability_signal=b["avail"] / b["avail_w"] if b["avail_w"] else 0.0,
            form_signal=b["form"] / b["form_w"] if b["form_w"] else 0.0,
            # Volume is reported, never modelled: it measures popularity,
            # not quality, and would bias toward already-owned players.
            volume=b["volume"],
            evidence=b["evidence"],
        )
    return out
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_sentiment.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add fplbot/sentiment.py tests/test_sentiment.py
git commit -m "feat: mention resolution, scorer implementations, and aggregation"
```

---

## Task 8: Report rendering and transfer diff

**Files:**
- Create: `fplbot/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Player`, `Projection`, `Squad`, `PlayerSentiment`, `Gameweek` from `fplbot.models`
- Produces: `build_report(squad, players, projections, sentiment, stats, gameweek, current_squad=None) -> dict`; `transfer_diff(current_ids, target_ids, players, projections) -> list[dict]`; `render_html(report) -> str`

**Background the implementer needs:** The JSON is the source of truth and the HTML is a view of it. Keep every number in the JSON so a later change of reporting destination needs no new computation.

The report must state its own degradation. If the run had no comment-level Reddit access, or used a stale cache, that belongs at the top of the report where it cannot be missed — a confident-looking squad built on a degraded run is the failure mode worth guarding against.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from fplbot.config import Settings
from fplbot.optimize import pick_squad
from fplbot.projection import project_all
from fplbot.report import build_report, render_html, transfer_diff


def _report(client):
    players = client.players()
    rules = client.squad_rules()
    projections = project_all(
        players=players, teams=client.teams(), fixtures=client.fixtures(),
        next_gw_id=client.next_gameweek().id, settings=Settings(cache_dir="/tmp/x"))
    squad = pick_squad(players, projections, rules)
    stats = {"comments_fetched": 0, "comments_after_filter": 0,
             "sources_used": [], "degraded": True}
    return build_report(squad, players, projections, {}, stats,
                        client.next_gameweek()), players, projections, squad


def test_report_lists_fifteen_players_with_reasons(client):
    report, *_ = _report(client)
    assert len(report["squad"]) == 15
    assert all(p["explanation"] for p in report["squad"])


def test_report_surfaces_degradation_prominently(client):
    report, *_ = _report(client)
    assert report["degraded"] is True
    assert "comment" in report["degraded_reason"].lower()


def test_report_totals_are_consistent(client):
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}
    assert report["total_cost"] == sum(by_id[i].now_cost for i in squad.players)
    assert report["captain"]["id"] == squad.captain_id


def test_transfer_diff_identifies_swaps(client):
    report, players, projections, squad = _report(client)
    current = list(squad.players)
    replacement = next(p.id for p in players if p.id not in current)
    current[0] = replacement
    moves = transfer_diff(current, squad.players, players, projections)
    assert len(moves) == 1
    assert moves[0]["out"]["id"] == replacement
    assert moves[0]["in"]["id"] == squad.players[0]


def test_identical_squads_produce_no_transfers(client):
    report, players, projections, squad = _report(client)
    assert transfer_diff(squad.players, squad.players, players, projections) == []


def test_html_renders_and_escapes(client):
    report, *_ = _report(client)
    html = render_html(report)
    assert "<html" in html.lower()
    assert "<script>" not in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot.report'`

- [ ] **Step 3: Write `fplbot/report.py`**

```python
"""Render a run into JSON (the source of truth) and HTML (a view of it)."""
from __future__ import annotations

import html
from datetime import datetime, timezone

from fplbot.models import Gameweek, Player, Projection, PlayerSentiment, Squad


def _player_row(player: Player, projection: Projection,
                sentiment: PlayerSentiment | None) -> dict:
    return {
        "id": player.id,
        "name": player.web_name,
        "position": player.position,
        "team_id": player.team_id,
        "cost": player.now_cost,
        "price_m": player.price_m,
        "projected_points": round(projection.total, 2),
        "per_gameweek": [round(x, 2) for x in projection.per_gameweek],
        "explanation": projection.explanation,
        "sentiment": (
            {
                "availability_signal": round(sentiment.availability_signal, 3),
                "form_signal": round(sentiment.form_signal, 3),
                "volume": sentiment.volume,
                "evidence": sentiment.evidence,
            }
            if sentiment
            else None
        ),
    }


def transfer_diff(current_ids, target_ids, players, projections) -> list[dict]:
    by_id = {p.id: p for p in players}
    out_ids = [i for i in current_ids if i not in set(target_ids)]
    in_ids = [i for i in target_ids if i not in set(current_ids)]

    moves = []
    for out_id, in_id in zip(out_ids, in_ids):
        moves.append({
            "out": {"id": out_id, "name": by_id[out_id].web_name,
                    "projected_points": round(projections[out_id].total, 2)},
            "in": {"id": in_id, "name": by_id[in_id].web_name,
                   "projected_points": round(projections[in_id].total, 2)},
            "gain": round(projections[in_id].total - projections[out_id].total, 2),
            "cost_change": by_id[in_id].now_cost - by_id[out_id].now_cost,
        })
    return sorted(moves, key=lambda m: m["gain"], reverse=True)


def build_report(squad: Squad, players, projections, sentiment, stats,
                 gameweek: Gameweek, current_squad=None) -> dict:
    by_id = {p.id: p for p in players}

    reasons = []
    if stats.get("degraded"):
        reasons.append(
            "No comment-level Reddit access this run; sentiment came from "
            "post titles only and is materially weaker."
        )
    if stats.get("cache_age_seconds", 0) > 6 * 3600:
        reasons.append(
            f"FPL data served from a cache "
            f"{stats['cache_age_seconds'] / 3600:.1f}h old."
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "gameweek": {"id": gameweek.id, "name": gameweek.name,
                     "deadline": gameweek.deadline_time.isoformat()},
        "degraded": bool(reasons),
        "degraded_reason": " ".join(reasons),
        "stats": stats,
        "total_cost": sum(by_id[i].now_cost for i in squad.players),
        "bank": squad.bank,
        "projected_total": round(
            sum(projections[i].total for i in squad.starting_xi), 2),
        "captain": _player_row(by_id[squad.captain_id],
                               projections[squad.captain_id],
                               sentiment.get(squad.captain_id)),
        "squad": [_player_row(by_id[i], projections[i], sentiment.get(i))
                  for i in squad.players],
        "starting_xi": squad.starting_xi,
        "transfers": (
            transfer_diff(current_squad, squad.players, players, projections)
            if current_squad else []
        ),
    }


def render_html(report: dict) -> str:
    def esc(value) -> str:
        return html.escape(str(value))

    banner = (
        f'<p class="warn">Reduced coverage: {esc(report["degraded_reason"])}</p>'
        if report["degraded"] else ""
    )
    rows = "".join(
        f"<tr><td>{esc(p['position'])}</td><td>{esc(p['name'])}</td>"
        f"<td>£{esc(p['price_m'])}m</td><td>{esc(p['projected_points'])}</td></tr>"
        for p in report["squad"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FPL squad — {esc(report['gameweek']['name'])}</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 2rem auto; max-width: 46rem; }}
.warn {{ background: #fde68a; padding: .75rem; border-radius: .375rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #e5e7eb; padding: .4rem .6rem; text-align: left; }}
</style></head><body>
<h1>{esc(report['gameweek']['name'])}</h1>
<p>Deadline {esc(report['gameweek']['deadline'])} ·
   Projected XI total {esc(report['projected_total'])} pts ·
   Captain {esc(report['captain']['name'])}</p>
{banner}
<table><tr><th>Pos</th><th>Player</th><th>Price</th><th>xP</th></tr>{rows}</table>
</body></html>"""
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add fplbot/report.py tests/test_report.py
git commit -m "feat: JSON and HTML reporting with transfer diff"
```

---

## Task 9: CLI and end-to-end pipeline

**Files:**
- Create: `fplbot/cli.py`, `README.md`
- Test: `tests/test_e2e.py`

**Interfaces:**
- Consumes: every module above
- Produces: `main(argv=None) -> int`; `run(settings, client, scorer, entry_id=None, now=None) -> dict`

**Background the implementer needs:** The CLI wires the pipeline and chooses a scorer: `SessionScorer` when `--batch` is given, `ApiScorer` when `ANTHROPIC_API_KEY` is set, otherwise `NullScorer`. Choosing `NullScorer` silently is correct behaviour — a stats-only squad is still useful — but the report must say so, which it does via the degraded banner.

The end-to-end test is the one that catches integration mistakes the unit tests cannot: type mismatches between modules, and the lag-freezing requirement from the spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e.py
import copy
import json

import pytest

from fplbot.config import Settings
from fplbot.sentiment import NullScorer


def test_full_pipeline_produces_a_legal_squad_offline(client, tmp_path):
    from fplbot.cli import run

    report = run(Settings(cache_dir=tmp_path), client, NullScorer())
    assert len(report["squad"]) == 15
    assert report["total_cost"] <= 1000
    assert report["captain"]["id"] in report["starting_xi"]
    json.dumps(report)  # must be serialisable


def test_pipeline_is_deterministic(client, tmp_path):
    from fplbot.cli import run

    a = run(Settings(cache_dir=tmp_path), client, NullScorer())
    b = run(Settings(cache_dir=tmp_path), client, NullScorer())
    assert [p["id"] for p in a["squad"]] == [p["id"] for p in b["squad"]]


def test_later_gameweek_projections_do_not_use_earlier_results(client, tmp_path):
    """Lag features must be frozen at the deadline.

    A projection for GW+3 may not change when GW+1 results are mutated,
    because at decision time those results do not exist. Getting this wrong
    makes a backtest look excellent and the live bot perform badly.
    """
    from fplbot.projection import project_all

    settings = Settings(cache_dir=tmp_path)
    players = client.players()
    next_gw = client.next_gameweek().id
    common = dict(teams=client.teams(), fixtures=client.fixtures(),
                  next_gw_id=next_gw, settings=settings)

    base = project_all(players=players, **common)

    mutated = []
    for p in players:
        # Simulate GW+1 having happened differently.
        mutated.append(copy.replace(p, form=p.form + 5.0) if p.minutes else p)

    after = project_all(players=mutated, **common)

    changed = [pid for pid in base
               if abs(base[pid].per_gameweek[2] - after[pid].per_gameweek[2]) > 1e-9]
    assert not changed, (
        f"{len(changed)} GW+3 projections moved when GW+1 form changed; "
        "lag features are leaking future results"
    )


def test_cli_writes_json_and_html(client, tmp_path, monkeypatch):
    from fplbot import cli

    monkeypatch.setattr(cli, "_build_client", lambda settings: client)
    monkeypatch.setenv("FPLBOT_CACHE", str(tmp_path))
    out = tmp_path / "report"
    assert cli.main(["--output", str(out)]) == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
```

Note on the third test: `copy.replace` requires Python 3.13, which this project already mandates. If `project_all` currently reads `form` for later gameweeks, this test will fail — that is the point. Fix it by computing form-derived inputs once, before the horizon loop, rather than per gameweek.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_e2e.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fplbot.cli'`

- [ ] **Step 3: Write `fplbot/cli.py`**

```python
"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fplbot.config import Settings
from fplbot.fetch import FplClient
from fplbot.news import collect
from fplbot.optimize import pick_squad
from fplbot.projection import project_all
from fplbot.report import build_report, render_html
from fplbot.sentiment import (ApiScorer, NullScorer, SessionScorer, aggregate,
                              resolve_mentions)

log = logging.getLogger(__name__)


def _build_client(settings: Settings) -> FplClient:
    return FplClient(settings)


def run(settings: Settings, client: FplClient, scorer, entry_id: int | None = None,
        now: datetime | None = None) -> dict:
    now = now or datetime.now(tz=timezone.utc)

    players = client.players()
    teams = client.teams()
    fixtures = client.fixtures()
    gameweek = client.next_gameweek()
    rules = client.squad_rules()

    if isinstance(scorer, NullScorer):
        items, stats = [], {"comments_fetched": 0, "comments_after_filter": 0,
                            "sources_used": [], "degraded": True}
    else:
        items, stats = collect(settings, now=now)

    team_names = {t.id: t.name for t in teams}
    mentions = resolve_mentions(items, players, team_names)
    scores = scorer.score(mentions)
    sentiment = aggregate(mentions, scores, now)

    stats["mentions_resolved"] = len(mentions)
    stats["unique_players_touched"] = len(sentiment)
    stats["cache_age_seconds"] = client.cache_age_seconds("bootstrap-static") or 0

    projections = project_all(players=players, teams=teams, fixtures=fixtures,
                              next_gw_id=gameweek.id, settings=settings,
                              sentiment=sentiment)
    squad = pick_squad(players, projections, rules)

    current = None
    if entry_id and gameweek.id > 1:
        try:
            current = client.entry_picks(entry_id, gameweek.id - 1)
        except Exception as exc:
            # GW1 has no prior squad, and a private or wrong id 404s. Neither
            # is worth aborting an otherwise good run for.
            log.warning("could not read entry %s: %s", entry_id, exc)
    return build_report(squad, players, projections, sentiment, stats,
                        gameweek, current)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fplbot")
    parser.add_argument("--team-id", type=int, default=None,
                        help="public FPL entry id, to show a transfer diff")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--output", default="report",
                        help="path prefix; writes .json and .html")
    parser.add_argument("--batch", default=None,
                        help="score sentiment via files (for a Claude session)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    settings = Settings.from_env()
    if args.horizon:
        settings.horizon = args.horizon

    if args.batch:
        scorer = SessionScorer(Path(f"{args.batch}.batch.json"),
                               Path(f"{args.batch}.scores.json"))
    elif settings.anthropic_api_key:
        scorer = ApiScorer(settings.anthropic_api_key)
    else:
        log.warning("no scorer configured; producing a stats-only squad")
        scorer = NullScorer()

    report = run(settings, _build_client(settings), scorer, args.team_id)

    out = Path(args.output)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    out.with_suffix(".html").write_text(render_html(report))

    print(f"GW{report['gameweek']['id']}: "
          f"{report['projected_total']} projected pts, "
          f"captain {report['captain']['name']}")
    if report["degraded"]:
        print(f"WARNING reduced coverage: {report['degraded_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full suite**

Run: `python3 -m pytest -v`
Expected: all tests pass across every module

- [ ] **Step 5: Run it for real once**

```bash
python3 -m fplbot.cli --output /tmp/fpl-report -v
```

Expected: prints a gameweek line and a reduced-coverage warning (no Reddit credentials yet), and writes `/tmp/fpl-report.json` and `.html`. Open the HTML and sanity-check the squad against your own football judgement — an optimizer producing a squad full of players you have never heard of usually means the projection is rewarding a stat it should not.

- [ ] **Step 6: Write `README.md`**

```bash
cat > README.md <<'EOF'
# fplbot

Picks a Fantasy Premier League squad from the official FPL statistics plus
sentiment mined from r/FantasyPL and football news.

Advisory only. It holds no FPL credentials and never touches your team.

## Setup

    python3 -m pip install -e '.[dev]'
    cp .env.example .env    # then fill in the Reddit credentials

Reddit credentials come from a free "script" app at
<https://reddit.com/prefs/apps>. Without them the bot still runs, but sees
post titles only and marks its report as reduced coverage.

## Use

    fplbot --output report                 # stats + sentiment
    fplbot --team-id 1234567               # also show a transfer diff
    fplbot --horizon 1                     # next gameweek only

## Design

- `docs/superpowers/specs/` — the design and why each parameter is what it is
- `docs/research/` — the sourced research behind the horizon and comment cap
EOF
```

- [ ] **Step 7: Commit**

```bash
git add fplbot/cli.py tests/test_e2e.py README.md
git commit -m "feat: CLI and end-to-end pipeline"
```

---

## Task 10: Scheduled runs

**Files:**
- Create: `docs/scheduling.md`

**Interfaces:**
- Consumes: the `fplbot` CLI from Task 9
- Produces: documentation only; no code

**Background the implementer needs:** The spec calls for two runs per gameweek, at T-48h and T-3h relative to `deadline_time`. Deadlines move week to week, so a fixed weekly cron is wrong — the schedule must be derived from the API.

- [ ] **Step 1: Verify the deadline query works**

```bash
python3 -c "
from fplbot.config import Settings
from fplbot.fetch import FplClient
gw = FplClient(Settings.from_env()).next_gameweek()
print(gw.name, gw.deadline_time)
"
```

Expected: prints the next gameweek and its deadline.

- [ ] **Step 2: Write `docs/scheduling.md`**

```bash
cat > docs/scheduling.md <<'EOF'
# Scheduling

Two runs per gameweek, both derived from the API's `deadline_time`:

- **T-48h** — early look, while prices are still moving.
- **T-3h** — the run that matters, after Friday press conferences.

Deadlines shift week to week, so do not use a fixed weekly cron. Query the
next deadline and schedule against it:

    python3 -c "from fplbot.config import Settings; from fplbot.fetch import FplClient; print(FplClient(Settings.from_env()).next_gameweek().deadline_time)"

## Claude routine (default)

A scheduled Claude routine runs the CLI with `--batch`, scores the written
batch inside its own session, re-runs to pick up the scores, and publishes the
report as an Artifact that updates in place at one URL. No API key needed.

## Standalone alternative

Set `ANTHROPIC_API_KEY` and run the CLI directly from any scheduler. The
package holds no knowledge of where reports go, so pointing it at a Discord or
Slack webhook is a change to the caller, not to `fplbot`.
EOF
```

- [ ] **Step 3: Commit**

```bash
git add docs/scheduling.md
git commit -m "docs: deadline-derived scheduling"
```

---

## Self-Review Notes

Checked against the spec:

| Spec requirement | Task |
|---|---|
| Legal 15-man squad, constraints read from API | 3, 5 |
| Reasons a human can check | 4, 8 |
| Transfer diff from a team ID | 3, 8 |
| Two runs per gameweek | 10 |
| Graceful degradation, labelled | 6, 8, 9 |
| FPL client with stale-cache fallback | 3 |
| Reddit comments, filters, dedupe, cap | 6 |
| Mention resolution with ambiguity dropped | 7 |
| `Scorer` interface, three implementations | 7 |
| Confidence-weighted, recency-decayed aggregation | 7 |
| Volume reported but never modelled | 7 |
| xP model incl. defensive contribution | 4 |
| Sentiment capped at ±15%, GW+1 only | 4 |
| API injury flag outranks sentiment | 4 |
| ILP squad, XI, captain | 5 |
| Infeasibility reports the binding constraint | 5 |
| Lag features frozen at deadline | 9 |
| Tests offline from snapshots | 3 and onward |
