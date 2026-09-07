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
    """The whole reason for using an ILP rather than sorting by points.

    The greedy walk reserves enough budget to fill its remaining slots at the
    cheapest available price, so it always completes a legal 15. An earlier
    version of this test let greedy stall at 13 players and then skipped its
    only assertion, it passed while proving nothing.
    """
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    optimal = sum(projections[i].total for i in squad.starting_xi)

    budget = rules["budget"]
    remaining = dict(rules["quota"])
    cheapest = {pos: min(p.now_cost for p in players if p.position == pos)
                for pos in remaining}
    greedy, clubs = [], {}

    for p in sorted(players, key=lambda p: projections[p.id].total, reverse=True):
        if remaining.get(p.position, 0) == 0:
            continue
        if clubs.get(p.team_id, 0) >= rules["team_limit"]:
            continue
        reserve = sum(cheapest[q] * (remaining[q] - (1 if q == p.position else 0))
                      for q in remaining)
        if p.now_cost + reserve > budget:
            continue
        greedy.append(p.id)
        budget -= p.now_cost
        remaining[p.position] -= 1
        clubs[p.team_id] = clubs.get(p.team_id, 0) + 1

    assert len(greedy) == 15, f"greedy built only {len(greedy)} players; test is vacuous"
    greedy_xi, _ = pick_xi(greedy, projections, rules, players)
    assert optimal >= sum(projections[i].total for i in greedy_xi)


def test_bench_is_discounted_but_still_playable(setup):
    """Bench points are worth a fraction of starting points, but a bench of
    players with no chance of appearing is worthless when a starter is ruled
    out before the deadline."""
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    by_id = {p.id: p for p in players}
    bench = [i for i in squad.players if i not in set(squad.starting_xi)]
    assert len(bench) == 4
    unplayable = [by_id[i].web_name for i in bench if by_id[i].availability <= 0]
    assert not unplayable, f"bench players who cannot play: {unplayable}"


def test_starting_xi_is_chosen_jointly_with_the_squad(setup):
    """pick_squad returns its own XI; it does not defer to a second solve."""
    players, projections, rules = setup
    squad = pick_squad(players, projections, rules)
    assert len(squad.starting_xi) == rules["xi_size"]
    assert set(squad.starting_xi) <= set(squad.players)


def test_infeasible_projections_raise_a_clear_error(setup):
    players, projections, rules = setup
    with pytest.raises(InfeasibleSquad):
        pick_squad(players, projections, {**rules, "budget": 1})
