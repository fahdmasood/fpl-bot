"""Render a run into JSON (the source of truth) and HTML (a view of it)."""
from __future__ import annotations

import html
from datetime import datetime, timezone

from fplbot.models import Gameweek, Player, Projection, PlayerSentiment, Squad


def _player_row(player: Player, projection: Projection,
                sentiment: PlayerSentiment | None) -> dict:
    return {
        "id": player.id,
        "name": player.web_name,
        "position": player.position,
        "team_id": player.team_id,
        "cost": player.now_cost,
        "price_m": player.price_m,
        "projected_points": round(projection.total, 2),
        "per_gameweek": [round(x, 2) for x in projection.per_gameweek],
        "explanation": projection.explanation,
        "sentiment": (
            {
                "availability_signal": round(sentiment.availability_signal, 3),
                "form_signal": round(sentiment.form_signal, 3),
                "volume": sentiment.volume,
                "evidence": sentiment.evidence,
            }
            if sentiment
            else None
        ),
    }


def transfer_diff(current_ids, target_ids, players, projections,
                  bank: int = 0) -> list[dict]:
    """Pair departures with arrivals, within position, cheapest funding first.

    FPL transfers are like for like: a goalkeeper sold funds a goalkeeper
    bought. Pairing the two lists in incidental order can suggest selling a
    keeper to buy a midfielder, which the game will simply reject, and makes
    the reported points gain meaningless because it compares two different
    scoring scales.

    `bank` is the money available on top of the sales. It defaults to zero,
    which means every move must fund itself, because the bot does not know
    the real bank balance of a squad it did not build.
    """
    by_id = {p.id: p for p in players}
    target = set(target_ids)
    current = set(current_ids)
    out_ids = [i for i in current_ids if i not in target]
    in_ids = [i for i in target_ids if i not in current]

    moves: list[dict] = []
    for position in ("GKP", "DEF", "MID", "FWD"):
        # Sell the weakest, buy the strongest, so each pairing is the best
        # available swap for that position rather than an arbitrary one.
        going = sorted((i for i in out_ids if by_id[i].position == position),
                       key=lambda i: projections[i].total)
        coming = sorted((i for i in in_ids if by_id[i].position == position),
                        key=lambda i: projections[i].total, reverse=True)
        for out_id, in_id in zip(going, coming):
            moves.append({
                "position": position,
                "out": {"id": out_id, "name": by_id[out_id].web_name,
                        "price_m": by_id[out_id].price_m,
                        "projected_points": round(projections[out_id].total, 2)},
                "in": {"id": in_id, "name": by_id[in_id].web_name,
                       "price_m": by_id[in_id].price_m,
                       "projected_points": round(projections[in_id].total, 2)},
                "gain": round(projections[in_id].total - projections[out_id].total, 2),
                "cost_change": by_id[in_id].now_cost - by_id[out_id].now_cost,
            })

    # Take the best gains first and mark where the money runs out, rather
    # than hiding the unaffordable ones. Knowing a move is out of reach is
    # more useful than not being told about it.
    ordered = sorted(moves, key=lambda m: m["gain"], reverse=True)
    spent = 0
    for move in ordered:
        if spent + move["cost_change"] <= bank:
            move["affordable"] = True
            spent += move["cost_change"]
        else:
            move["affordable"] = False
    return ordered


def _degradation_notes(stats: dict) -> list[str]:
    """Reasons this run is worth less than a normal one.

    Reads the stats dict that news.collect() actually returns. An absent
    Reddit is NOT degradation: Reddit requires approved Data API access and
    running without it is the expected default. A Reddit that is configured
    and then fails IS worth reporting, because something broke rather than
    was never set up.
    """
    notes: list[str] = []
    absent = stats.get("sources_absent", [])

    if not stats.get("collection_attempted", True):
        # Nothing was collected because nothing would have scored it. Saying
        # the feeds "could not be read" would send someone debugging feeds
        # that are working perfectly well.
        notes.append(
            "No sentiment scorer is configured, so no news was collected and "
            "the squad was built from statistics alone."
        )
    elif "news-rss" in absent:
        notes.append(
            "The news feeds could not be read this run, so the squad was "
            "built from statistics alone with no sentiment applied."
        )
    elif stats.get("items_after_filter", 0) == 0:
        notes.append(
            "No news items survived filtering, so no sentiment was applied "
            "this run."
        )

    status = stats.get("reddit_status", "not_configured")
    if isinstance(status, str) and status.startswith("failed:"):
        notes.append(
            f"Reddit is configured but failed this run ({status}), so "
            "comment level signal is missing."
        )

    age = stats.get("cache_age_seconds") or 0
    if age > 6 * 3600:
        notes.append(
            f"FPL data was served from a cache {age / 3600:.1f} hours old."
        )

    return notes


HIT_COST = 4  # points charged for each transfer beyond your free ones


def annotate_transfer_plan(moves: list[dict], free_transfers: int) -> list[dict]:
    """Mark which moves your free transfers cover and which are worth a hit.

    A squad differs from the optimum by however many players it differs by,
    often ten or more. Listing all of them is a wish list, not advice. You get
    one free transfer per gameweek, banked up to five, and every transfer
    beyond that costs four points.

    Moves are already sorted by projected gain, so the free transfers go to
    the best ones you can actually make. Affordability is checked BEFORE the
    allocation: a free transfer handed to a move the money does not cover is
    a free transfer thrown away, and it makes the affordable move below it
    look like a four point hit when it would in fact be free.

    Each move past the free ones is judged on its own: a gain larger than the
    four point hit is worth taking, a smaller one is not.

    Note the comparison is slightly generous to taking hits, because the gain
    is spread across the projection horizon while the hit is charged once.
    """
    planned = []
    free_used = 0
    for move in moves:
        affordable = bool(move["affordable"])
        free = affordable and free_used < free_transfers
        if free:
            free_used += 1
        hit = 0 if free else HIT_COST
        move = {
            **move,
            "uses_free_transfer": free,
            "hit_cost": hit,
            "net_gain": round(move["gain"] - hit, 2),
        }
        move["recommended"] = affordable and (free or move["net_gain"] > 0)
        planned.append(move)
    return planned


def build_report(squad: Squad, players, projections, sentiment, stats,
                 gameweek: Gameweek, current_squad=None,
                 free_transfers: int = 1, bank: int = 0) -> dict:
    """`bank` is the manager's own money, in integer tenths of a million,
    read from the entry's picks payload. Without it every upgrade that costs
    more than the player it replaces is reported as unaffordable."""
    by_id = {p.id: p for p in players}

    transfers = (
        annotate_transfer_plan(
            transfer_diff(current_squad, squad.players, players, projections,
                          bank=bank),
            free_transfers,
        )
        if current_squad else []
    )

    reasons = _degradation_notes(stats)

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "gameweek": {"id": gameweek.id, "name": gameweek.name,
                     "deadline": gameweek.deadline_time.isoformat()},
        "degraded": bool(reasons),
        "degraded_reason": " ".join(reasons),
        "stats": stats,
        "sources_used": stats.get("sources_used", []),
        "sources_absent": stats.get("sources_absent", []),
        "total_cost": sum(by_id[i].now_cost for i in squad.players),
        "bank": squad.bank,
        "projected_total": round(
            sum(projections[i].total for i in squad.starting_xi), 2),
        "captain": _player_row(by_id[squad.captain_id],
                               projections[squad.captain_id],
                               sentiment.get(squad.captain_id)),
        "squad": [_player_row(by_id[i], projections[i], sentiment.get(i))
                  for i in squad.players],
        "starting_xi": squad.starting_xi,
        "transfers": transfers,
        "transfers_recommended": [m for m in transfers if m["recommended"]],
        "free_transfers": free_transfers,
        "transfers_note": (
            f"You have {free_transfers} free transfer(s) and £{bank / 10}m in "
            f"the bank. Moves beyond your free transfers cost {HIT_COST} "
            "points each and are only recommended when the projected gain "
            "exceeds the hit. Pairing is within position, and a move you "
            "cannot fund is shown but never recommended and never given a "
            "free transfer."
            if current_squad else ""
        ),
    }


def render_html(report: dict) -> str:
    def esc(value) -> str:
        return html.escape(str(value))

    banner = (
        f'<p class="warn">Reduced coverage: {esc(report["degraded_reason"])}</p>'
        if report["degraded"] else ""
    )
    starting = set(report["starting_xi"])

    def _row(p: dict) -> str:
        marks = []
        if p["id"] in starting:
            marks.append("XI")
        if p["id"] == report["captain"]["id"]:
            marks.append("C")
        evidence = ""
        if p.get("sentiment") and p["sentiment"]["evidence"]:
            quotes = "".join(
                f"<li>{esc(q)}</li>" for q in p["sentiment"]["evidence"])
            evidence = (
                f"<details><summary>sentiment "
                f"{p['sentiment']['availability_signal']:+.2f} availability, "
                f"{p['sentiment']['form_signal']:+.2f} form, "
                f"{p['sentiment']['volume']} mentions</summary>"
                f"<ul>{quotes}</ul></details>"
            )
        return (
            f"<tr><td>{esc(' '.join(marks))}</td><td>{esc(p['position'])}</td>"
            f"<td>{esc(p['name'])}</td><td>£{esc(p['price_m'])}m</td>"
            f"<td>{esc(p['projected_points'])}</td>"
            f"<td class='why'>{esc(p['explanation'])}{evidence}</td></tr>"
        )

    rows = "".join(_row(p) for p in report["squad"])

    transfers = ""
    if report.get("transfers"):
        moves = "".join(
            f"<tr><td>{esc(m['position'])}</td>"
            f"<td>{esc(m['out']['name'])} (£{esc(m['out']['price_m'])}m, "
            f"{esc(m['out']['projected_points'])})</td>"
            f"<td>{esc(m['in']['name'])} (£{esc(m['in']['price_m'])}m, "
            f"{esc(m['in']['projected_points'])})</td>"
            f"<td>{esc(m['gain'])}</td>"
            f"<td>{'free' if m['uses_free_transfer'] else str(m['hit_cost']) + ' pts'}</td>"
            f"<td>{esc(m['net_gain'])}</td>"
            f"<td>{'yes' if m['recommended'] else 'no'}</td></tr>"
            for m in report["transfers"]
        )
        transfers = (
            "<h2>Suggested transfers</h2>"
            f"<p>{esc(report.get('transfers_note', ''))}</p>"
            "<table><tr><th>Pos</th><th>Out</th><th>In</th><th>Gain</th>"
            "<th>Cost</th><th>Net</th><th>Do it</th>"
            f"</tr>{moves}</table>"
        )

    coverage = (
        f"<p class='meta'>Sources used: {esc(', '.join(report['sources_used']) or 'none')}. "
        f"Absent: {esc(', '.join(report['sources_absent']) or 'none')}.</p>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FPL squad, {esc(report['gameweek']['name'])}</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 2rem auto; max-width: 46rem; }}
.warn {{ background: #fde68a; padding: .75rem; border-radius: .375rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #e5e7eb; padding: .4rem .6rem; text-align: left;
          vertical-align: top; }}
.why {{ font-size: .85em; color: #4b5563; max-width: 26rem; }}
.meta {{ font-size: .85em; color: #6b7280; }}
details summary {{ cursor: pointer; }}
</style></head><body>
<h1>{esc(report['gameweek']['name'])}</h1>
<p>Deadline {esc(report['gameweek']['deadline'])} ·
   Projected XI total {esc(report['projected_total'])} pts ·
   Captain {esc(report['captain']['name'])}</p>
{banner}
{coverage}
<table><tr><th></th><th>Pos</th><th>Player</th><th>Price</th><th>xP</th>
<th>Why</th></tr>{rows}</table>
{transfers}
</body></html>"""
