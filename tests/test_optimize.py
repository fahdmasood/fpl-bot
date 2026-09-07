import pytest

from fplbot.config import Settings
from fplbot.optimize import InfeasibleSquad, pick_squad, pick_xi
from fplbot.projection import project_all


@pytest.fixture
def setup(client):
    players = client.players()
    rules = client.squad_rules()
    projections = project_all(
        players=players, teams=client.teams(), fixtures=client.fixtures(),
        next_gw_id=client.next_gameweek().id, settings=Settings(cache_dir="/tmp/x"),
    )
    return players, projections, rules


def test_squad_has_fifteen_players(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    assert len(squad.players) == 15
    assert len(set(squad.players)) == 15


def test_squad_respects_budget(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    assert sum(by_id[i].now_cost for i in squad.players) <= rules["budget"]


def test_squad_respects_position_quota(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    counts = {}
    for i in squad.players:
        counts[by_id[i].position] = counts.get(by_id[i].position, 0) + 1
    assert counts == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_squad_respects_three_per_club_limit(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    counts = {}
    for i in squad.players:
        counts[by_id[i].team_id] = counts.get(by_id[i].team_id, 0) + 1
    assert max(counts.values()) <= rules["team_limit"]


def test_xi_is_eleven_with_a_legal_formation(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    xi, captain = pick_xi(squad.players, projections, rules, players)
    assert len(xi) == 11
    assert set(xi) <= set(squad.players)
    counts = {}
    for i in xi:
        counts[by_id[i].position] = counts.get(by_id[i].position, 0) + 1
    assert counts["GKP"] == 1
    assert 3 <= counts["DEF"] <= 5
    assert 1 <= counts["FWD"] <= 3


def test_captain_is_the_highest_projected_starter(setup):
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    xi, captain = pick_xi(squad.players, projections, rules, players)
    assert captain in xi
    assert projections[captain].total == max(projections[i].total for i in xi)


def test_optimum_beats_a_naive_greedy_pick(setup):
    """The whole reason for using an ILP rather than sorting by points."""
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    optimal = sum(projections[i].total for i in squad.players)

    by_id = {p.id: p for p in players}
    greedy, spend, counts, clubs = [], 0, {}, {}
    quota = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for p in sorted(players, key=lambda p: projections[p.id].total, reverse=True):
        if (counts.get(p.position, 0) < quota[p.position]
                and spend + p.now_cost <= rules["budget"]
                and clubs.get(p.team_id, 0) < rules["team_limit"]):
            greedy.append(p.id)
            spend += p.now_cost
            counts[p.position] = counts.get(p.position, 0) + 1
            clubs[p.team_id] = clubs.get(p.team_id, 0) + 1
    if len(greedy) == 15:
        assert optimal >= sum(projections[i].total for i in greedy)


def test_infeasible_projections_raise_a_clear_error(setup):
    players, projections, rules = setup
    with pytest.raises(InfeasibleSquad):
        pick_squad(players, projections, {**rules, "budget": 1})
