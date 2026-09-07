"""Expected-points model.

Sentiment is a bounded nudge on availability and form, never a direct
addition to points, and it applies only to the next gameweek.
"""
from __future__ import annotations

import math
import statistics

from fplbot.config import SCORING, Settings
from fplbot.models import Fixture, Player, PlayerSentiment, Projection, Team

MINUTES_FLOOR = 270  # three full matches before per-90 rates are trusted

# Multiplies a club's own expected goals conceded for this fixture. Difficulty
# runs 1 (easiest) to 5 (hardest).
DIFFICULTY_MULTIPLIER = {1: 0.70, 2: 0.85, 3: 1.00, 4: 1.20, 5: 1.45}


def _per_90(total: float, minutes: int, fallback: float = 0.0) -> float:
    """Per-90 rate, or the position's median when the sample is too small.

    A player with 40 minutes has an unstable rate; returning their raw
    extrapolation would let a single cameo goal project like a season of form.
    """
    if minutes < MINUTES_FLOOR:
        return fallback
    return total / (minutes / 90)


def position_medians(players: list[Player]) -> dict[str, dict[str, float]]:
    """Median per-90 rates by position, for the low-minutes fallback."""
    out: dict[str, dict[str, float]] = {}
    for position in ("GKP", "DEF", "MID", "FWD"):
        pool = [p for p in players
                if p.position == position and p.minutes >= MINUTES_FLOOR]
        if not pool:
            out[position] = {"xg": 0.0, "xa": 0.0}
            continue
        out[position] = {
            "xg": statistics.median(p.expected_goals / (p.minutes / 90) for p in pool),
            "xa": statistics.median(p.expected_assists / (p.minutes / 90) for p in pool),
        }
    return out


def team_xgc_per_90(players: list[Player]) -> dict[int, float]:
    """Each club's expected goals conceded per 90.

    Taken from the club's most-played goalkeeper: they are on the pitch for
    every goal conceded, so their expected_goals_conceded IS the club's. This
    is what the spec means by "the club's expected_goals_conceded" — using a
    fixture-difficulty digit alone cannot tell a good defence from a bad one.
    """
    by_team: dict[int, float] = {}
    for team_id in {p.team_id for p in players}:
        keepers = [p for p in players
                   if p.team_id == team_id and p.position == "GKP" and p.minutes >= 180]
        if keepers:
            gk = max(keepers, key=lambda p: p.minutes)
            by_team[team_id] = gk.expected_goals_conceded / (gk.minutes / 90)
        else:
            outfield = [p.expected_goals_conceded / (p.minutes / 90) for p in players
                        if p.team_id == team_id and p.minutes >= MINUTES_FLOOR]
            by_team[team_id] = statistics.median(outfield) if outfield else 1.4
    return by_team


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


def clean_sheet_probability(team_xgc90: float, difficulty: int) -> float:
    """Poisson probability of conceding zero, given the club's own xGC.

    P(0 goals) = exp(-expected_goals_conceded). Scaling the club's own rate by
    fixture difficulty means a weak defence facing an easy fixture and a strong
    defence facing a hard one are ranked on their actual quality, not on the
    difficulty digit alone.
    """
    adjusted = team_xgc90 * DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
    return math.exp(-max(0.05, adjusted))


def bonus_estimate(player: Player) -> float:
    """Expected bonus points per appearance, from BPS rate and ICT.

    Bonus is awarded to the top three BPS scorers in a match. A player
    averaging well above the ~25 BPS mark earns bonus regularly; below it,
    rarely. ICT is blended in as a secondary signal of involvement.
    """
    if player.minutes < MINUTES_FLOOR:
        return 0.0
    bps90 = player.bps / (player.minutes / 90)
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

    med = medians.get(pos, {"xg": 0.0, "xa": 0.0})
    xg90 = _per_90(player.expected_goals, player.minutes, med["xg"])
    xa90 = _per_90(player.expected_assists, player.minutes, med["xa"])
    attacking = xg90 * SCORING["goal"][pos] + xa90 * SCORING["assist"][pos]

    at_home = fixture.team_h == player.team_id
    difficulty = fixture.team_h_difficulty if at_home else fixture.team_a_difficulty
    cs_points = SCORING["clean_sheet"][pos]
    clean_sheet = (
        clean_sheet_probability(team_xgc90, difficulty) * cs_points if cs_points else 0.0
    )

    dc90 = player.defensive_contribution_per_90
    threshold = SCORING["defensive_threshold"][pos]
    dc = (SCORING["defensive_contribution"][pos] * min(1.0, dc90 / threshold)
          if threshold < 99 else 0.0)

    # The form channel moves attacking output and bonus — the parts of a
    # projection that genuine form talk is about. It does not touch clean
    # sheets or appearance, which are team and selection properties.
    scored = (attacking + bonus_estimate(player)) * form_multiplier + clean_sheet + dc
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
