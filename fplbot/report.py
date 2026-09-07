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


def transfer_diff(current_ids, target_ids, players, projections) -> list[dict]:
    by_id = {p.id: p for p in players}
    out_ids = [i for i in current_ids if i not in set(target_ids)]
    in_ids = [i for i in target_ids if i not in set(current_ids)]

    moves = []
    for out_id, in_id in zip(out_ids, in_ids):
        moves.append({
            "out": {"id": out_id, "name": by_id[out_id].web_name,
                    "projected_points": round(projections[out_id].total, 2)},
            "in": {"id": in_id, "name": by_id[in_id].web_name,
                   "projected_points": round(projections[in_id].total, 2)},
            "gain": round(projections[in_id].total - projections[out_id].total, 2),
            "cost_change": by_id[in_id].now_cost - by_id[out_id].now_cost,
        })
    return sorted(moves, key=lambda m: m["gain"], reverse=True)


def build_report(squad: Squad, players, projections, sentiment, stats,
                 gameweek: Gameweek, current_squad=None) -> dict:
    by_id = {p.id: p for p in players}

    reasons = []
    if stats.get("degraded"):
        reasons.append(
            "No comment-level Reddit access this run; sentiment came from "
            "post titles only and is materially weaker."
        )
    if stats.get("cache_age_seconds", 0) > 6 * 3600:
        reasons.append(
            f"FPL data served from a cache "
            f"{stats['cache_age_seconds'] / 3600:.1f}h old."
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "gameweek": {"id": gameweek.id, "name": gameweek.name,
                     "deadline": gameweek.deadline_time.isoformat()},
        "degraded": bool(reasons),
        "degraded_reason": " ".join(reasons),
        "stats": stats,
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
        "transfers": (
            transfer_diff(current_squad, squad.players, players, projections)
            if current_squad else []
        ),
    }


def render_html(report: dict) -> str:
    def esc(value) -> str:
        return html.escape(str(value))

    banner = (
        f'<p class="warn">Reduced coverage: {esc(report["degraded_reason"])}</p>'
        if report["degraded"] else ""
    )
    rows = "".join(
        f"<tr><td>{esc(p['position'])}</td><td>{esc(p['name'])}</td>"
        f"<td>£{esc(p['price_m'])}m</td><td>{esc(p['projected_points'])}</td></tr>"
        for p in report["squad"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FPL squad, {esc(report['gameweek']['name'])}</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 2rem auto; max-width: 46rem; }}
.warn {{ background: #fde68a; padding: .75rem; border-radius: .375rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #e5e7eb; padding: .4rem .6rem; text-align: left; }}
</style></head><body>
<h1>{esc(report['gameweek']['name'])}</h1>
<p>Deadline {esc(report['gameweek']['deadline'])} ·
   Projected XI total {esc(report['projected_total'])} pts ·
   Captain {esc(report['captain']['name'])}</p>
{banner}
<table><tr><th>Pos</th><th>Player</th><th>Price</th><th>xP</th></tr>{rows}</table>
</body></html>"""
