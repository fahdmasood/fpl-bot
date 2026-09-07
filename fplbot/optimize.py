"""Squad selection as a binary integer program.

An ILP finds the true optimum. Sorting by points and taking the best
affordable player at each position does not, and the gap is worth real
points across a season.
"""
from __future__ import annotations

import pulp

from fplbot.models import Player, Projection, Squad


class InfeasibleSquad(Exception):
    """No legal squad exists under these constraints."""


def pick_squad(
    players: list[Player], projections: dict[int, Projection], rules: dict
) -> Squad:
    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    pick = {p.id: pulp.LpVariable(f"p{p.id}", cat="Binary") for p in players}

    problem += pulp.lpSum(projections[p.id].total * pick[p.id] for p in players)

    problem += pulp.lpSum(pick.values()) == rules["squad_size"]
    # Costs are integer tenths of a million; keep them integers throughout.
    problem += pulp.lpSum(p.now_cost * pick[p.id] for p in players) <= rules["budget"]

    for position, count in rules["quota"].items():
        problem += (
            pulp.lpSum(pick[p.id] for p in players if p.position == position) == count
        )

    for team_id in {p.team_id for p in players}:
        problem += (
            pulp.lpSum(pick[p.id] for p in players if p.team_id == team_id)
            <= rules["team_limit"]
        )

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleSquad(
            f"solver returned {pulp.LpStatus[status]}. "
            f"Budget {rules['budget']}, cheapest legal squad costs "
            f"{_cheapest_possible(players, rules)}. "
            "If the budget looks fine, check that projections are non-zero."
        )

    chosen = [p.id for p in players if pick[p.id].value() == 1]
    by_id = {p.id: p for p in players}
    spend = sum(by_id[i].now_cost for i in chosen)
    xi, captain = pick_xi(chosen, projections, rules, players)
    return Squad(
        players=chosen, starting_xi=xi, captain_id=captain,
        bank=rules["budget"] - spend,
    )


def _cheapest_possible(players: list[Player], rules: dict) -> int:
    total = 0
    for position, count in rules["quota"].items():
        costs = sorted(p.now_cost for p in players if p.position == position)
        total += sum(costs[:count])
    return total


def pick_xi(
    squad_player_ids: list[int],
    projections: dict[int, Projection],
    rules: dict,
    players: list[Player],
) -> tuple[list[int], int]:
    by_id = {p.id: p for p in players if p.id in set(squad_player_ids)}

    problem = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    start = {i: pulp.LpVariable(f"s{i}", cat="Binary") for i in squad_player_ids}

    problem += pulp.lpSum(projections[i].total * start[i] for i in squad_player_ids)
    problem += pulp.lpSum(start.values()) == rules["xi_size"]

    for position, (low, high) in rules["formation"].items():
        in_position = [start[i] for i in squad_player_ids if by_id[i].position == position]
        problem += pulp.lpSum(in_position) >= low
        problem += pulp.lpSum(in_position) <= high

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleSquad(f"no legal XI: solver returned {pulp.LpStatus[status]}")

    xi = [i for i in squad_player_ids if start[i].value() == 1]
    captain = max(xi, key=lambda i: projections[i].total)
    return xi, captain
