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


def test_cache_age_seconds_returns_none_before_fetch(client):
    age = client.cache_age_seconds("bootstrap-static")
    assert age is None


def test_cache_age_seconds_returns_float_after_fetch(client):
    client.bootstrap()
    age = client.cache_age_seconds("bootstrap-static")
    assert isinstance(age, float)
    assert age >= 0


def test_next_gameweek_returns_earliest_future_when_no_is_next(client):
    from datetime import datetime, timezone
    from fplbot.models import Gameweek

    # Create a list where no gameweek has is_next=True, but some have future deadlines
    now = datetime(2025, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    past = datetime(2025, 9, 1, 18, 0, 0, tzinfo=timezone.utc)
    future_1 = datetime(2025, 9, 14, 18, 0, 0, tzinfo=timezone.utc)
    future_2 = datetime(2025, 9, 21, 18, 0, 0, tzinfo=timezone.utc)

    weeks = [
        Gameweek(id=1, name="GW 1", deadline_time=past, is_next=False),
        Gameweek(id=2, name="GW 2", deadline_time=future_1, is_next=False),
        Gameweek(id=3, name="GW 3", deadline_time=future_2, is_next=False),
    ]

    from fplbot.fetch import _select_next_gameweek

    result = _select_next_gameweek(weeks, now)
    assert result.id == 2


def test_next_gameweek_returns_last_when_all_deadlines_past(client):
    from datetime import datetime, timezone
    from fplbot.models import Gameweek

    # Create a list where no gameweek has is_next=True and all deadlines are in the past
    now = datetime(2025, 9, 30, 12, 0, 0, tzinfo=timezone.utc)
    past_1 = datetime(2025, 9, 1, 18, 0, 0, tzinfo=timezone.utc)
    past_2 = datetime(2025, 9, 7, 18, 0, 0, tzinfo=timezone.utc)
    past_3 = datetime(2025, 9, 14, 18, 0, 0, tzinfo=timezone.utc)

    weeks = [
        Gameweek(id=1, name="GW 1", deadline_time=past_1, is_next=False),
        Gameweek(id=2, name="GW 2", deadline_time=past_2, is_next=False),
        Gameweek(id=3, name="GW 3", deadline_time=past_3, is_next=False),
    ]

    from fplbot.fetch import _select_next_gameweek

    result = _select_next_gameweek(weeks, now)
    assert result.id == 3
