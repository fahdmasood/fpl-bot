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


def test_later_gameweeks_are_decayed(projections):
    """Pick a player scoring in both weeks; a blank gameweek is a legitimate
    zero and would make a naive comparison flake."""
    both = [p for p in projections.values()
            if p.per_gameweek[0] > 0 and p.per_gameweek[1] > 0]
    assert both, "no player projects points in both of the first two gameweeks"
    p = max(both, key=lambda x: x.total)
    assert p.per_gameweek[1] < p.per_gameweek[0]
