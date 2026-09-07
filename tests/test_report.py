# tests/test_report.py
from fplbot.config import Settings
from fplbot.optimize import pick_squad
from fplbot.projection import project_all
from fplbot.report import build_report, render_html, transfer_diff


def _report(client):
    players = client.players()
    rules = client.squad_rules()
    projections = project_all(
        players=players, teams=client.teams(), fixtures=client.fixtures(),
        next_gw_id=client.next_gameweek().id, settings=Settings(cache_dir="/tmp/x"))
    squad = pick_squad(players, projections, rules)
    stats = {"comments_fetched": 0, "comments_after_filter": 0,
             "sources_used": [], "degraded": True}
    return build_report(squad, players, projections, {}, stats,
                        client.next_gameweek()), players, projections, squad


def test_report_lists_fifteen_players_with_reasons(client):
    report, *_ = _report(client)
    assert len(report["squad"]) == 15
    assert all(p["explanation"] for p in report["squad"])


def test_report_surfaces_degradation_prominently(client):
    report, *_ = _report(client)
    assert report["degraded"] is True
    assert "comment" in report["degraded_reason"].lower()


def test_report_totals_are_consistent(client):
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}
    assert report["total_cost"] == sum(by_id[i].now_cost for i in squad.players)
    assert report["captain"]["id"] == squad.captain_id


def test_transfer_diff_identifies_swaps(client):
    report, players, projections, squad = _report(client)
    current = list(squad.players)
    replacement = next(p.id for p in players if p.id not in current)
    current[0] = replacement
    moves = transfer_diff(current, squad.players, players, projections)
    assert len(moves) == 1
    assert moves[0]["out"]["id"] == replacement
    assert moves[0]["in"]["id"] == squad.players[0]


def test_identical_squads_produce_no_transfers(client):
    report, players, projections, squad = _report(client)
    assert transfer_diff(squad.players, squad.players, players, projections) == []


def test_html_renders_and_escapes(client):
    report, *_ = _report(client)
    html = render_html(report)
    assert "<html" in html.lower()
    assert "<script>" not in html
