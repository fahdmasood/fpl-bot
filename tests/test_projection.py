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
    from fplbot.projection import bonus_estimate
    from dataclasses import replace
    base = max(client.players(), key=lambda p: p.minutes)
    high_bps = replace(base, bps=base.bps * 3)
    assert bonus_estimate(high_bps) > bonus_estimate(base)


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
    assert _per_90(5.0, minutes=40, fallback=med["FWD"]["xg"]) == med["FWD"]["xg"]


def test_defenders_do_not_dominate_the_top_of_the_board(client):
    """Sanity check against football, not just against the code. Budget
    defenders on weak teams previously projected alongside Haaland."""
    settings = Settings(cache_dir="/tmp/x")
    proj = project_all(players=client.players(), teams=client.teams(),
                       fixtures=client.fixtures(),
                       next_gw_id=client.next_gameweek().id, settings=settings)
    by_id = {p.id: p for p in client.players()}
    top20 = sorted(proj.values(), key=lambda x: x.total, reverse=True)[:20]
    defenders = sum(1 for t in top20 if by_id[t.player_id].position == "DEF")
    assert defenders <= 10, f"{defenders}/20 of the top projections are defenders"


def test_later_gameweeks_are_decayed(projections):
    """Pick a player scoring in both weeks; a blank gameweek is a legitimate
    zero and would make a naive comparison flake."""
    both = [p for p in projections.values()
            if p.per_gameweek[0] > 0 and p.per_gameweek[1] > 0]
    assert both, "no player projects points in both of the first two gameweeks"
    p = max(both, key=lambda x: x.total)
    assert p.per_gameweek[1] < p.per_gameweek[0]
