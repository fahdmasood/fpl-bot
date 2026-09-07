import copy
import json

import pytest

from fplbot.config import Settings
from fplbot.sentiment import NullScorer


def test_full_pipeline_produces_a_legal_squad_offline(client, tmp_path):
    from fplbot.cli import run

    report = run(Settings(cache_dir=tmp_path), client, NullScorer())
    assert len(report["squad"]) == 15
    assert report["total_cost"] <= 1000
    assert report["captain"]["id"] in report["starting_xi"]
    json.dumps(report)  # must be serialisable


def test_pipeline_is_deterministic(client, tmp_path):
    from fplbot.cli import run

    a = run(Settings(cache_dir=tmp_path), client, NullScorer())
    b = run(Settings(cache_dir=tmp_path), client, NullScorer())
    assert [p["id"] for p in a["squad"]] == [p["id"] for p in b["squad"]]


def test_the_projection_harness_can_detect_a_change_at_all(client, tmp_path):
    """Positive control for the freeze lag test below.

    That test asserts a projection does NOT move. An assertion of absence is
    worthless until you have shown the harness can see a change at all. An
    earlier version mutated `form`, which the model never reads, so it could
    not have failed no matter how badly the model leaked.
    """
    from fplbot.projection import project_all

    settings = Settings(cache_dir=tmp_path)
    players = client.players()
    common = dict(teams=client.teams(), fixtures=client.fixtures(),
                  next_gw_id=client.next_gameweek().id, settings=settings)

    base = project_all(players=players, **common)
    target = max(players, key=lambda p: p.expected_goals)
    bumped = [copy.replace(p, expected_goals=p.expected_goals + 5.0)
              if p.id == target.id else p for p in players]
    after = project_all(players=bumped, **common)

    assert after[target.id].total > base[target.id].total, (
        "mutating a field the model reads did not change the projection, so "
        "the freeze lag test below cannot be trusted"
    )


def test_every_gameweek_in_the_horizon_uses_the_same_frozen_inputs(client, tmp_path):
    """Lag features are frozen at the deadline.

    At decision time no gameweek in the horizon has been played, so all of
    them must be projected from the same season to date inputs. If a later
    gameweek responded differently to a change than the first one, something
    would be recomputing from results that do not exist yet, and a backtest
    would look excellent while the live bot underperformed.

    Checked by ratio rather than by equality, because the raw values differ
    legitimately across gameweeks through decay and fixture difficulty.
    """
    from fplbot.projection import project_all

    settings = Settings(cache_dir=tmp_path)
    players = client.players()
    common = dict(teams=client.teams(), fixtures=client.fixtures(),
                  next_gw_id=client.next_gameweek().id, settings=settings)

    base = project_all(players=players, **common)
    target = max(players, key=lambda p: p.expected_goals)
    bumped = [copy.replace(p, expected_goals=p.expected_goals + 5.0)
              if p.id == target.id else p for p in players]
    after = project_all(players=bumped, **common)

    ratios = [
        a / b
        for b, a in zip(base[target.id].per_gameweek, after[target.id].per_gameweek)
        if b > 0
    ]
    assert len(ratios) >= 2, "need at least two scoring gameweeks to compare"
    assert max(ratios) - min(ratios) < 1e-9, (
        f"gameweeks responded differently to the same input change: {ratios}"
    )


def test_an_entry_with_no_picks_yet_does_not_break_the_run(client, tmp_path):
    """A team registered mid season has no picks until it has played a
    gameweek, and that endpoint 404s. A squad must still be produced."""
    from fplbot.cli import run

    class NoPicks:
        def __getattr__(self, name):
            return getattr(client, name)

        def entry_picks(self, entry_id, event):
            raise RuntimeError("404 Not Found")

    report = run(Settings(cache_dir=tmp_path), NoPicks(), NullScorer(),
                 entry_id=10541438)
    assert len(report["squad"]) == 15
    assert report["transfers"] == []


def test_free_transfers_limit_what_is_recommended(client, tmp_path):
    """Listing every difference from the optimum is a wish list. Only moves
    covered by a free transfer, or gaining more than the four point hit,
    should be recommended."""
    from fplbot.report import annotate_transfer_plan

    moves = [
        {"position": "MID", "gain": 9.0, "affordable": True,
         "out": {"name": "a"}, "in": {"name": "b"}, "cost_change": 0},
        {"position": "MID", "gain": 5.0, "affordable": True,
         "out": {"name": "c"}, "in": {"name": "d"}, "cost_change": 0},
        {"position": "DEF", "gain": 2.0, "affordable": True,
         "out": {"name": "e"}, "in": {"name": "f"}, "cost_change": 0},
    ]
    planned = annotate_transfer_plan(moves, free_transfers=1)

    assert planned[0]["uses_free_transfer"] is True
    assert planned[0]["recommended"] is True
    # Second move gains 5 against a 4 point hit, so it is narrowly worth it.
    assert planned[1]["uses_free_transfer"] is False
    assert planned[1]["net_gain"] == 1.0
    assert planned[1]["recommended"] is True
    # Third gains 2 against a 4 point hit, so it is not.
    assert planned[2]["net_gain"] == -2.0
    assert planned[2]["recommended"] is False


def test_an_unaffordable_move_is_never_recommended(client):
    from fplbot.report import annotate_transfer_plan
    moves = [{"position": "FWD", "gain": 20.0, "affordable": False,
              "out": {"name": "a"}, "in": {"name": "b"}, "cost_change": 50}]
    planned = annotate_transfer_plan(moves, free_transfers=1)
    assert planned[0]["recommended"] is False


def test_cli_writes_json_and_html(client, tmp_path, monkeypatch):
    from fplbot import cli

    monkeypatch.setattr(cli, "_build_client", lambda settings: client)
    monkeypatch.setenv("FPLBOT_CACHE", str(tmp_path))
    out = tmp_path / "report"
    assert cli.main(["--output", str(out)]) == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
