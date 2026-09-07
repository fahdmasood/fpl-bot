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

# A mention is only allowed to influence the signal if its weight is within
# this fraction of the best evidence available for that player. Without it,
# a large enough pile of weak chatter drowns out one confident report.
RELEVANCE_FLOOR = 0.25

AVAILABILITY_CATEGORIES = {"injury", "rotation"}
FORM_CATEGORIES = {"form", "hype"}
VALID_CATEGORIES = AVAILABILITY_CATEGORIES | FORM_CATEGORIES

# Names the plain matcher gets wrong. Extend as mismatches show up in runs.
ALIASES: dict[str, str] = {
    "kdb": "De Bruyne",
    "trent": "Alexander-Arnold",
    "vvd": "van Dijk",
}

# Player names that are also ordinary English words. Matching one of these on
# lowercased text alone produced real false positives in live runs:
#
#     "Chelsea paid cash for the deal"           -> Matty Cash, Aston Villa
#     "Manchester United will mount a challenge" -> Mason Mount
#     "Rice and beans is the king"               -> Rice, and King
#
# These are not harmless. RELEVANCE_FLOOR scales to the best evidence
# available for a player, so a player whose only mention is spurious takes
# his entire signal from it.
#
# MAINTENANCE: this set tracks the names in the current squads and goes stale
# every transfer window. Recheck it each season, and whenever a signing
# arrives whose name is also a common word.
COMMON_WORD_NAMES = frozenset({
    "cash", "rice", "mount", "wood", "white", "hall", "king", "sels", "eze",
    "tel", "reed", "obi", "cho", "pau", "vaz", "young", "moore", "may",
    "price", "banks", "france", "sanchez",
})


def _build_index(players: list[Player]) -> dict[str, list[Player]]:
    index: dict[str, list[Player]] = {}
    for p in players:
        for key in {p.web_name.lower(), p.web_name.split()[-1].lower()}:
            index.setdefault(key, []).append(p)
    return index


def _capitalised_away_from_a_sentence_start(key: str, fields: list[str]) -> bool:
    """True if `key` appears capitalised somewhere its capital means something.

    Every sentence opens with a capital, so a leading "Cash" says nothing
    about whether the writer meant the player or the money. A capital in the
    middle of a sentence is a deliberate proper noun.

    The title and the body are checked as separate texts: a headline is its
    own sentence, so the first word of the body is a sentence opening even
    though the two are concatenated before matching.
    """
    pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
    for field in fields:
        for match in pattern.finditer(field):
            if not field[match.start()].isupper():
                continue
            before = field[: match.start()].rstrip()
            # A colon or a dash still reads as mid sentence, and headlines
            # use both ("Villa team news: Cash is fit").
            if before and before[-1] not in ".!?":
                return True
    return False


def _corroborated(key: str, player: Player, lowered: str, fields: list[str],
                  team_names: dict[int, str]) -> bool:
    """Whether a name that is also an ordinary word really means the player.

    Accepted on either of two independent signals: the player's own club is
    named in the same text, or the name is capitalised where a capital is
    informative. Naming some other club is not corroboration, which is what
    made "Chelsea paid cash for the deal" resolve to an Aston Villa defender.
    """
    club = team_names.get(player.team_id)
    if club and club.lower() in lowered:
        return True
    return _capitalised_away_from_a_sentence_start(key, fields)


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
        # The original casing is kept alongside the lowercased text: it is
        # the only evidence that separates a player called Cash from money.
        original_fields = [item.title, item.body]
        for alias, real in ALIASES.items():
            lowered = re.sub(rf"\b{re.escape(alias)}\b", real.lower(), lowered)

        # One comment yields at most one mention per player. A player whose
        # web name contains a space, such as "De Bruyne", is indexed under
        # both the full name and the surname, so a single mention would
        # otherwise match twice and count as two independent pieces of
        # evidence.
        matched: dict[int, str] = {}

        for key, matches in index.items():
            if not re.search(rf"\b{re.escape(key)}\b", lowered):
                continue

            if len(matches) == 1:
                # A name that is also an ordinary English word needs
                # corroboration even when it is unambiguous as a name.
                if key in COMMON_WORD_NAMES and not _corroborated(
                        key, matches[0], lowered, original_fields, team_names):
                    log.debug("dropped uncorroborated common word %r", key)
                    continue
                matched.setdefault(matches[0].id, key)
                continue

            # Ambiguous surname. Resolve by club named in the same text, or
            # drop it: a misattributed injury rumour is worse than none.
            disambiguated = [
                p for p in matches
                if team_names.get(p.team_id, "\0").lower() in lowered
            ]
            if len(disambiguated) == 1:
                matched.setdefault(disambiguated[0].id, key)
            else:
                log.debug("dropped ambiguous mention %r", key)

        for player_id, key in matched.items():
            mentions.append(Mention(player_id, item, key))

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
    buckets: dict[int, dict] = {}

    for index, mention in enumerate(mentions):
        score = scores.get(index)
        if score is None:
            continue
        # The scorer is given the player_id and echoes it back. If it comes
        # back different, the index and the mention have desynchronised, and
        # applying the score would attribute a claim to the wrong player.
        if score.player_id != mention.player_id:
            log.debug("score at index %d claims player %s but the mention is "
                      "about player %s; skipping",
                      index, score.player_id, mention.player_id)
            continue

        age_days = (now - mention.item.published_at).total_seconds() / 86400
        # Trust tier discounts sources that speculate; see news.SOURCE_TRUST.
        trust = SOURCE_TRUST.get(mention.item.source, 0.6)
        weight = (min(1.0, max(0.0, score.confidence)) * trust
                  * 0.5 ** (age_days / half_life_days))
        clamped = max(-1.0, min(1.0, score.sentiment))

        bucket = buckets.setdefault(
            mention.player_id,
            {"avail": [], "form": [], "volume": 0, "evidence": []},
        )
        bucket["volume"] += 1
        channel = "avail" if score.category in AVAILABILITY_CATEGORIES else "form"
        bucket[channel].append((clamped, weight))

        if len(bucket["evidence"]) < 3:
            bucket["evidence"].append(mention.item.body[:160])

    def _signal(rows: list[tuple[float, float]]) -> float:
        """Direction from the strong evidence, attenuated by how strong it is.

        Three properties have to hold together:

        1. Repeating a claim adds nothing. Ten copies of one comment say no
           more than one, so the mean is taken rather than the sum.
        2. Stale or unconfident evidence moves the projection less, which the
           mean alone cannot express: for a single mention (s * w) / w = s
           cancels the decay entirely. Multiplying by the best weight
           restores it.
        3. A swarm of weak mentions cannot outvote one strong one. Averaging
           over everything let 600 low confidence comments saying "he is
           fine" overturn a confident wire service injury report. Only
           mentions within RELEVANCE_FLOOR of the best evidence take part.
        """
        if not rows:
            return 0.0
        best = max(w for _, w in rows)
        if best <= 0:
            return 0.0

        strong = [(s, w) for s, w in rows if w >= RELEVANCE_FLOOR * best]
        total = sum(w for _, w in strong)
        if total <= 0:
            return 0.0
        mean = sum(s * w for s, w in strong) / total
        return max(-1.0, min(1.0, mean * min(1.0, best)))

    out: dict[int, PlayerSentiment] = {}
    for player_id, b in buckets.items():
        out[player_id] = PlayerSentiment(
            player_id=player_id,
            availability_signal=_signal(b["avail"]),
            form_signal=_signal(b["form"]),
            # Volume is reported, never modelled: it measures popularity,
            # not quality, and would bias toward already-owned players.
            volume=b["volume"],
            evidence=b["evidence"],
        )
    return out
