# tests/test_report.py
from fplbot.config import Settings
from fplbot.optimize import pick_squad
from fplbot.projection import project_all
from fplbot.report import annotate_transfer_plan, build_report, render_html, transfer_diff


def _report(client, stats=None):
    players = client.players()
    rules = client.squad_rules()
    projections = project_all(
        players=players, teams=client.teams(), fixtures=client.fixtures(),
        next_gw_id=client.next_gameweek().id, settings=Settings(cache_dir="/tmp/x"))
    squad = pick_squad(players, projections, rules)
    if stats is None:
        # The shape news.collect() actually returns, with the news feeds down.
        stats = {"items_fetched": 0, "items_after_filter": 0,
                 "sources_used": [], "sources_absent": ["news-rss", "reddit"],
                 "reddit_available": False, "reddit_status": "not_configured"}
    return build_report(squad, players, projections, {}, stats,
                        client.next_gameweek()), players, projections, squad


def test_report_lists_fifteen_players_with_reasons(client):
    report, *_ = _report(client)
    assert len(report["squad"]) == 15
    assert all(p["explanation"] for p in report["squad"])


def test_report_surfaces_degradation_prominently(client):
    report, *_ = _report(client)
    assert report["degraded"] is True
    assert "news feeds" in report["degraded_reason"].lower()


def test_build_report_consumes_the_stats_news_collect_actually_returns(client, monkeypatch):
    """Guards the contract between news.collect and build_report.

    These two were specified at different times and drifted: the report read
    a "degraded" key that collect never produced, so the warning banner could
    never fire in the real pipeline while every unit test passed against a
    hand built dict. This test feeds the real thing through.
    """
    from datetime import datetime, timezone
    import fplbot.news as news
    from fplbot.models import NewsItem

    sample = NewsItem(source="bbc", url="u",
                      published_at=datetime(2026, 9, 7, 11, tzinfo=timezone.utc),
                      title="t", body="Saka trained fully today", score=10)
    monkeypatch.setattr(news.RssCollector, "collect", lambda self: [sample])
    _items, stats = news.collect(Settings(cache_dir="/tmp/x"),
                                 now=datetime(2026, 9, 7, 12, tzinfo=timezone.utc))

    report, *_ = _report(client, stats=stats)
    # A healthy news only run is NOT degraded: Reddit is optional by design.
    assert report["degraded"] is False, report["degraded_reason"]
    assert "news-rss" in report["sources_used"]
    assert "reddit" in report["sources_absent"]


def test_a_configured_but_broken_reddit_is_reported_as_degradation(client):
    stats = {"items_fetched": 5, "items_after_filter": 5,
             "sources_used": ["news-rss"], "sources_absent": ["reddit"],
             "reddit_available": False, "reddit_status": "failed: RuntimeError"}
    report, *_ = _report(client, stats=stats)
    assert report["degraded"] is True
    assert "reddit" in report["degraded_reason"].lower()


def test_a_stale_cache_is_reported(client):
    stats = {"items_fetched": 5, "items_after_filter": 5,
             "sources_used": ["news-rss"], "sources_absent": [],
             "reddit_available": False, "reddit_status": "not_configured",
             "cache_age_seconds": 12 * 3600}
    report, *_ = _report(client, stats=stats)
    assert report["degraded"] is True
    assert "cache" in report["degraded_reason"].lower()


def test_report_totals_are_consistent(client):
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}
    assert report["total_cost"] == sum(by_id[i].now_cost for i in squad.players)
    assert report["captain"]["id"] == squad.captain_id


def test_transfer_diff_identifies_swaps(client):
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}
    current = list(squad.players)
    target = current[0]
    replacement = next(p.id for p in players
                       if p.id not in current
                       and by_id[p.id].position == by_id[target].position)
    current[0] = replacement
    moves = transfer_diff(current, squad.players, players, projections)
    assert len(moves) == 1
    assert moves[0]["out"]["id"] == replacement
    assert moves[0]["in"]["id"] == target


def test_transfers_are_always_like_for_like_by_position(client):
    """FPL will not let you sell a goalkeeper to buy a midfielder. Pairing the
    departures and arrivals in incidental list order produced exactly that,
    and presented it as advice."""
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}

    # Swap out one player of each position, ordering the current squad so the
    # naive pairing would cross positions.
    current = list(squad.players)
    for position in ("FWD", "GKP", "MID", "DEF"):
        victim = next(i for i in current if by_id[i].position == position)
        replacement = next(p.id for p in players
                           if p.id not in current and p.position == position)
        current[current.index(victim)] = replacement
    current.reverse()

    moves = transfer_diff(current, squad.players, players, projections)
    assert moves, "expected some transfers"
    for m in moves:
        out_pos = by_id[m["out"]["id"]].position
        in_pos = by_id[m["in"]["id"]].position
        assert out_pos == in_pos, f"illegal transfer suggested: {out_pos} for {in_pos}"
        assert m["position"] == out_pos


def test_unaffordable_transfers_are_flagged_not_hidden(client):
    """Knowing a move is out of reach beats not being told about it."""
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}
    current = list(squad.players)
    target = max((i for i in current if by_id[i].position == "MID"),
                 key=lambda i: by_id[i].now_cost)
    cheap = min((p for p in players
                 if p.id not in current and p.position == "MID"),
                key=lambda p: p.now_cost)
    current[current.index(target)] = cheap.id

    moves = transfer_diff(current, squad.players, players, projections, bank=0)
    assert len(moves) == 1
    # Buying back a dearer player with nothing in the bank is not affordable.
    if moves[0]["cost_change"] > 0:
        assert moves[0]["affordable"] is False


def test_html_shows_reasoning_and_transfers(client):
    """The spec requires the rendered page to carry the reasoning, not just
    the names and numbers."""
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}
    current = list(squad.players)
    target = current[0]
    replacement = next(p.id for p in players
                       if p.id not in current
                       and by_id[p.id].position == by_id[target].position)
    current[0] = replacement
    moves = transfer_diff(current, squad.players, players, projections)
    report["transfers"] = annotate_transfer_plan(moves, free_transfers=1)
    report["transfers_note"] = "test"

    html_out = render_html(report)
    assert "Suggested transfers" in html_out
    assert report["squad"][0]["explanation"][:20] in html_out
    assert "Sources used" in html_out


def test_identical_squads_produce_no_transfers(client):
    report, players, projections, squad = _report(client)
    assert transfer_diff(squad.players, squad.players, players, projections) == []


def test_html_renders_and_escapes(client):
    report, *_ = _report(client)
    html = render_html(report)
    assert "<html" in html.lower()
    assert "<script>" not in html


def test_money_in_the_bank_makes_an_upgrade_recommendable(client):
    """The bank was dropped on the floor between fetch and the report, so
    every upgrade priced above the player it replaced looked unaffordable and
    nothing was ever recommended."""
    report, players, projections, squad = _report(client)
    by_id = {p.id: p for p in players}
    current = list(squad.players)
    # Sell the dearest midfielder in the optimal squad and hold the cheapest
    # one available instead, so buying him back costs real money.
    target = max((i for i in current if by_id[i].position == "MID"),
                 key=lambda i: by_id[i].now_cost)
    cheap = min((p for p in players
                 if p.id not in current and p.position == "MID"),
                key=lambda p: p.now_cost)
    current[current.index(target)] = cheap.id
    shortfall = by_id[target].now_cost - cheap.now_cost
    assert shortfall > 0, "fixture must require money to fund the move"

    stats = {"items_fetched": 0, "items_after_filter": 0, "sources_used": [],
             "sources_absent": ["news-rss", "reddit"], "reddit_available": False,
             "reddit_status": "not_configured"}
    from fplbot.report import build_report

    broke = build_report(squad, players, projections, {}, stats,
                         client.next_gameweek(), current, bank=0)
    assert broke["transfers_recommended"] == [], "nothing is affordable on nothing"

    funded = build_report(squad, players, projections, {}, stats,
                          client.next_gameweek(), current, bank=shortfall)
    assert funded["transfers_recommended"], (
        "money in the bank must make the upgrade recommendable"
    )


def test_an_unaffordable_move_does_not_burn_a_free_transfer():
    """Free transfers were allotted by rank before affordability was checked,
    so the one free transfer was spent on a move that would never be made and
    a smaller affordable move was left to look like a 4 point hit."""
    moves = [
        {"position": "FWD", "gain": 9.0, "affordable": False, "cost_change": 40,
         "out": {"name": "a"}, "in": {"name": "b"}},
        {"position": "DEF", "gain": 2.0, "affordable": True, "cost_change": 0,
         "out": {"name": "c"}, "in": {"name": "d"}},
    ]
    planned = annotate_transfer_plan(moves, free_transfers=1)

    assert planned[0]["uses_free_transfer"] is False
    assert planned[0]["recommended"] is False
    # The free transfer falls through to the affordable move, which is only
    # worth making because it is free: a 2 point gain loses to a 4 point hit.
    assert planned[1]["uses_free_transfer"] is True
    assert planned[1]["hit_cost"] == 0
    assert planned[1]["recommended"] is True
