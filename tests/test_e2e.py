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


def test_later_gameweek_projections_do_not_use_earlier_results(client, tmp_path):
    """Lag features must be frozen at the deadline.

    A projection for GW+3 may not change when GW+1 results are mutated,
    because at decision time those results do not exist. Getting this wrong
    makes a backtest look excellent and the live bot perform badly.
    """
    from fplbot.projection import project_all

    settings = Settings(cache_dir=tmp_path)
    players = client.players()
    next_gw = client.next_gameweek().id
    common = dict(teams=client.teams(), fixtures=client.fixtures(),
                  next_gw_id=next_gw, settings=settings)

    base = project_all(players=players, **common)

    mutated = []
    for p in players:
        # Simulate GW+1 having happened differently.
        mutated.append(copy.replace(p, form=p.form + 5.0) if p.minutes else p)

    after = project_all(players=mutated, **common)

    changed = [pid for pid in base
               if abs(base[pid].per_gameweek[2] - after[pid].per_gameweek[2]) > 1e-9]
    assert not changed, (
        f"{len(changed)} GW+3 projections moved when GW+1 form changed; "
        "lag features are leaking future results"
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
