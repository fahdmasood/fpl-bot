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
