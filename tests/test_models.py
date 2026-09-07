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


def test_squad_rejects_wrong_size_non_empty():
    import pytest
    with pytest.raises(ValueError):
        Squad(players=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], starting_xi=[], captain_id=1, bank=0)
