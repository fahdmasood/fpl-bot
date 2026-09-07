"""Expected-points model.

Sentiment is a bounded nudge on availability and form, never a direct
addition to points, and it applies only to the next gameweek.
"""
from __future__ import annotations

import statistics

from fplbot.config import SCORING, Settings
from fplbot.models import Fixture, Player, PlayerSentiment, Projection, Team

MINUTES_FLOOR = 270  # three full matches before per-90 rates are trusted


def _per_90(total: float, minutes: int) -> float:
    return total / (minutes / 90) if minutes >= MINUTES_FLOOR else 0.0


def minutes_probability(
    player: Player, sentiment: PlayerSentiment | None = None, cap: float = 0.15
) -> float:
    """Probability the player is on the pitch, in [0, 1].

    The API's own injury flag outranks the internet: a player who is not
    `status == "a"` cannot be talked back onto the pitch by sentiment.
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


def _clean_sheet_probability(team: Team, fixture: Fixture, at_home: bool) -> float:
    difficulty = fixture.team_h_difficulty if at_home else fixture.team_a_difficulty
    # Difficulty runs 1 (easiest) to 5 (hardest). Map to a plausible band.
    return {1: 0.50, 2: 0.42, 3: 0.32, 4: 0.22, 5: 0.14}.get(difficulty, 0.28)


def expected_points_one_gw(
    player: Player, team: Team | None, fixture: Fixture | None, minutes_prob: float
) -> float:
    if minutes_prob <= 0 or fixture is None:
        return 0.0

    pos = player.position
    appearance = SCORING["appearance_60_plus"] * minutes_prob

    xg90 = _per_90(player.expected_goals, player.minutes)
    xa90 = _per_90(player.expected_assists, player.minutes)
    attacking = xg90 * SCORING["goal"][pos] + xa90 * SCORING["assist"][pos]

    at_home = fixture.team_h == player.team_id
    cs_points = SCORING["clean_sheet"][pos]
    clean_sheet = (
        _clean_sheet_probability(team, fixture, at_home) * cs_points if cs_points else 0.0
    )

    dc90 = player.defensive_contribution_per_90
    threshold = SCORING["defensive_threshold"][pos]
    dc = SCORING["defensive_contribution"][pos] * min(1.0, dc90 / threshold) if threshold < 99 else 0.0

    bonus = min(1.2, player.ict_index / 200)

    return minutes_prob * (attacking + clean_sheet + dc + bonus) + appearance


def project_all(
    players: list[Player],
    teams: list[Team],
    fixtures: list[Fixture],
    next_gw_id: int,
    settings: Settings,
    sentiment: dict[int, PlayerSentiment] | None = None,
) -> dict[int, Projection]:
    sentiment = sentiment or {}
    team_by_id = {t.id: t for t in teams}
    weights = settings.decay_weights

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

            gw_fixtures = [
                f for f in fixtures_by_gw.get(gw_id, [])
                if player.team_id in (f.team_h, f.team_a)
            ]
            # Blanks score nothing; double gameweeks score twice.
            points = sum(
                expected_points_one_gw(player, team_by_id.get(player.team_id), f, mp)
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
