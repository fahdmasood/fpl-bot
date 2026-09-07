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
            {"avail": 0.0, "avail_w": 0.0, "avail_max": 0.0,
             "form": 0.0, "form_w": 0.0, "form_max": 0.0,
             "volume": 0, "evidence": []},
        )
        bucket["volume"] += 1
        clamped = max(-1.0, min(1.0, score.sentiment))

        if score.category in AVAILABILITY_CATEGORIES:
            bucket["avail"] += clamped * weight
            bucket["avail_w"] += weight
            bucket["avail_max"] = max(bucket["avail_max"], weight)
        else:
            bucket["form"] += clamped * weight
            bucket["form_w"] += weight
            bucket["form_max"] = max(bucket["form_max"], weight)

        if len(bucket["evidence"]) < 3:
            bucket["evidence"].append(mention.item.body[:160])

    def _signal(total: float, weight_sum: float, best_weight: float) -> float:
        """Weighted mean, attenuated by the strength of the best evidence.

        The mean alone gives direction without letting volume matter: ten
        mentions of the same claim say no more than one. But a mean also
        cancels the decay for a lone mention — (s x w) / w = s — so a
        fortnight-old "he has a knock" would move the projection exactly as
        much as this morning's. Multiplying by the single best weight (its
        confidence x source trust x recency) restores that attenuation
        without reintroducing volume: adding more mentions cannot push the
        signal past what the strongest one supports.
        """
        if weight_sum <= 0:
            return 0.0
        return (total / weight_sum) * min(1.0, best_weight)

    out: dict[int, PlayerSentiment] = {}
    for player_id, b in by_player.items():
        out[player_id] = PlayerSentiment(
            player_id=player_id,
            availability_signal=_signal(b["avail"], b["avail_w"], b["avail_max"]),
            form_signal=_signal(b["form"], b["form_w"], b["form_max"]),
            # Volume is reported, never modelled: it measures popularity,
            # not quality, and would bias toward already-owned players.
            volume=b["volume"],
            evidence=b["evidence"],
        )
    return out
