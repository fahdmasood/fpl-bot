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
