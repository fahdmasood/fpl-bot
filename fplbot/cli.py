"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fplbot.config import Settings
from fplbot.fetch import FplClient
from fplbot.news import collect
from fplbot.optimize import pick_squad
from fplbot.projection import project_all
from fplbot.report import build_report, render_html
from fplbot.sentiment import (ApiScorer, NullScorer, SessionScorer, aggregate,
                              resolve_mentions)

log = logging.getLogger(__name__)


def _build_client(settings: Settings) -> FplClient:
    return FplClient(settings)


def run(settings: Settings, client: FplClient, scorer, entry_id: int | None = None,
        now: datetime | None = None, free_transfers: int = 1) -> dict:
    now = now or datetime.now(tz=timezone.utc)

    players = client.players()
    teams = client.teams()
    fixtures = client.fixtures()
    gameweek = client.next_gameweek()
    rules = client.squad_rules()

    if isinstance(scorer, NullScorer):
        # Match the shape news.collect() returns, so the report's degradation
        # logic reads the same keys either way.
        items, stats = [], {
            "items_fetched": 0, "items_after_filter": 0,
            "sources_used": [], "sources_absent": ["news-rss", "reddit"],
            "reddit_available": False, "reddit_status": "not_configured",
        }
    else:
        items, stats = collect(settings, now=now)

    team_names = {t.id: t.name for t in teams}
    mentions = resolve_mentions(items, players, team_names)
    scores = scorer.score(mentions)
    sentiment = aggregate(mentions, scores, now)

    stats["mentions_resolved"] = len(mentions)
    stats["unique_players_touched"] = len(sentiment)
    stats["cache_age_seconds"] = client.cache_age_seconds("bootstrap-static") or 0

    projections = project_all(players=players, teams=teams, fixtures=fixtures,
                              next_gw_id=gameweek.id, settings=settings,
                              sentiment=sentiment)
    squad = pick_squad(players, projections, rules)

    current = None
    if entry_id and gameweek.id > 1:
        try:
            current = client.entry_picks(entry_id, gameweek.id - 1)
        except Exception as exc:
            # GW1 has no prior squad, and a private or wrong id 404s. Neither
            # is worth aborting an otherwise good run for.
            log.warning("could not read entry %s: %s", entry_id, exc)
    return build_report(squad, players, projections, sentiment, stats,
                        gameweek, current, free_transfers=free_transfers)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fplbot")
    parser.add_argument("--team-id", type=int, default=None,
                        help="public FPL entry id, to show a transfer diff")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--free-transfers", type=int, default=1,
                        help="free transfers available, banked up to 5. "
                             "Moves beyond these cost 4 points each")
    parser.add_argument("--output", default="report",
                        help="path prefix; writes .json and .html")
    parser.add_argument("--batch", default=None,
                        help="score sentiment via files (for a Claude session)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    settings = Settings.from_env()
    if args.horizon:
        settings.horizon = args.horizon

    if args.batch:
        scorer = SessionScorer(Path(f"{args.batch}.batch.json"),
                               Path(f"{args.batch}.scores.json"))
    elif settings.anthropic_api_key:
        scorer = ApiScorer(settings.anthropic_api_key)
    else:
        log.warning("no scorer configured; producing a stats-only squad")
        scorer = NullScorer()

    report = run(settings, _build_client(settings), scorer, args.team_id,
                 free_transfers=args.free_transfers)

    out = Path(args.output)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    out.with_suffix(".html").write_text(render_html(report))

    print(f"GW{report['gameweek']['id']}: "
          f"{report['projected_total']} projected pts, "
          f"captain {report['captain']['name']}")

    recommended = report.get("transfers_recommended", [])
    if recommended:
        print(f"{len(recommended)} transfer(s) worth making:")
        for move in recommended:
            cost = "free" if move["uses_free_transfer"] else f"-{move['hit_cost']} pts"
            print(f"  {move['position']}: {move['out']['name']} out, "
                  f"{move['in']['name']} in "
                  f"({move['net_gain']:+.2f} net, {cost})")
    elif report.get("transfers"):
        print("No transfer is worth making this week.")

    if report["degraded"]:
        print(f"WARNING reduced coverage: {report['degraded_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
