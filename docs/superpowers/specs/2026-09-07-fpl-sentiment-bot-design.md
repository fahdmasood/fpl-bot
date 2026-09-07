# FPL Sentiment Bot — Design

**Date:** 2026-09-07
**Status:** Approved for planning

## Purpose

Pick a Fantasy Premier League squad by combining the official FPL statistical
feed with sentiment mined from football news and community discussion, then
publish the result to a page the user reads before each deadline.

The bot advises. It never touches the user's FPL account, holds no credentials
for it, and makes no transfers.

## Success criteria

1. A run produces a legal 15-man squad — with a starting XI and a captain —
   satisfying the constraints *as read from the API*
   (`game_settings.squad_total_spend`, `squad_squadsize`, `squad_team_limit`,
   and `element_types[].squad_min_play`/`squad_max_play`), not hardcoded. At
   time of writing those are £100.0m, 15 players, max 3 per club, and an XI of
   1 GKP / 3-5 DEF / 2-5 MID / 1-3 FWD.
2. Every selection carries a reason a human can check: the projected points,
   and the sentiment signals that moved it.
3. When given an FPL team ID, the report also lists the transfers that would
   move the user's real squad toward the optimum.
4. It runs twice per gameweek without being asked, and the report lands
   somewhere the user will see it.
5. A failure in any sentiment source degrades the run — to titles-only, or to
   stats-only — rather than failing it, and the report says which.

## Non-goals

- Executing transfers, chips, or captain changes. The user was offered this and
  declined; the FPL write endpoints are unofficial and fragile.
- Live in-play tracking. FPL is only actionable at the deadline, so
  sub-deadline freshness buys nothing.
- Modelling chips (Wildcard, Bench Boost, Triple Captain) in v1.

## Data sources

All verified reachable on 2026-09-07.

### FPL API — unauthenticated, the quantitative backbone

| Endpoint | Gives us |
|---|---|
| `/api/bootstrap-static/` | 654 players; price, position, club, ownership, form, `status`, `chance_of_playing_next_round`, `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`, ICT. Also the 38 events with `deadline_time` and `is_next`. |
| `/api/fixtures/?event=N` | Fixtures with per-side difficulty ratings. |
| `/api/element-summary/{id}/` | Per-player match history and upcoming fixtures. |
| `/api/entry/{id}/event/{gw}/picks/` | A public squad's picks. No login required. |

The deadline schedule comes from `events[].deadline_time`; nothing is
hardcoded. At time of writing the next deadline is GW4,
`2026-09-12T12:30:00Z`.

### Sentiment sources

Comment-level volume is a requirement, not an upgrade. Post titles alone are a
thin and lagging signal; the useful content — "he's been carrying a knock all
week", "Pep hinted at rotation" — appears in comment threads, often hours
before it reaches a headline.

**Reddit API (required).** A free Reddit OAuth "script" app, created once at
`reddit.com/prefs/apps`, gives read access to r/FantasyPL listings *and* their
comment trees under the client-credentials grant. No user account is exposed;
the bot only reads public content. Credentials live in `REDDIT_CLIENT_ID` and
`REDDIT_CLIENT_SECRET`.

Comment volume is large, so the collector filters before anything reaches the
language model:

- Only threads that matter: the daily/matchday discussion threads, injury and
  press-conference threads, and posts above a score threshold.
- Only comments above a small score floor, discarding the long tail of
  one-word replies.
- Deduplicated by normalised text, since the same claim gets copy-pasted across
  threads and would otherwise be double-counted as independent evidence.
- Capped per run, newest first, so cost stays bounded on a busy matchday.

**RSS (supporting).** BBC Sport football and Guardian football feeds, for
injury reports and press-conference coverage from outside the community bubble.
Both verified working unauthenticated.

**Fallback.** If the Reddit credentials are missing or the API errors, the
collector falls back to `https://www.reddit.com/r/FantasyPL/hot.rss` and
`/new.rss` — verified working without auth, but titles only. A run in this
state is explicitly labelled as reduced-coverage in the report, because the
sentiment signal is meaningfully weaker without comments.

Note: `reddit.com/r/.../hot.json` returns 403 to non-browser clients as of
2026-09-07, and `old.reddit.com` 302-redirects. The `.rss` path is the working
unauthenticated route. Do not reintroduce the `.json` path.

X/Twitter is excluded: the API pricing does not justify the marginal signal.

## Prerequisites

One manual step before the first run: create a free Reddit script app at
`reddit.com/prefs/apps` and put the client ID and secret in `.env`.

Nothing else is required for the scheduled path, which scores sentiment inside
the Claude session. An `ANTHROPIC_API_KEY` is only needed to run the CLI
standalone. The FPL API needs no key, and no FPL credentials are held anywhere
in this system.

## Architecture

Deterministic core, LLM at the edges. Everything that decides the squad is
plain Python that runs offline against cached fixtures; the language model only
turns prose into numbers.

```
fetch.py  ──►  projection.py  ──►  optimize.py  ──►  report.py
   ▲               ▲
news.py ──► sentiment.py
```

### `fetch.py`
FPL API client. Caches every response to disk with a timestamp so runs are
reproducible and tests never hit the network. One retry with backoff; a stale
cache is preferred over a failed run.

### `news.py`
Collects raw text items — posts and comments — into a common shape:
`{source, url, published_at, title, body, score}`. Applies the thread, score,
dedupe, and volume-cap filters described above. Knows nothing about players.

### `sentiment.py`
Resolves player mentions and scores them. Two stages:

1. **Mention resolution** — deterministic string matching of items against the
   player list from `bootstrap-static`, handling web names, surnames, and a
   hand-maintained alias table for the ambiguous cases. Ambiguity is resolved
   by club context in the same text; unresolvable mentions are dropped, not
   guessed.
2. **Scoring** — resolved mentions batched to Claude Haiku 4.5, returning per
   mention `{player_id, sentiment: -1..1, category, confidence: 0..1}` where
   category is one of `injury`, `rotation`, `form`, `hype`.

Scoring sits behind a `Scorer` interface with two implementations, so the
package does not care who is running it:

- `ApiScorer` — calls the Anthropic API directly using `ANTHROPIC_API_KEY`.
  Used when the CLI runs standalone, and the path a future GitHub Actions
  deployment would take.
- `SessionScorer` — writes the batch to a file and reads scores back, letting
  the scheduled Claude routine do the scoring inside its own session with no
  API key involved. This is the default for the scheduled runs.

A third `NullScorer` returns no scores, which is how the stats-only degraded
run is implemented rather than as a special case threaded through the
pipeline.

An LLM earns its place here because football text is context-heavy: "Haaland
has a knock" contains no negative word but is the most decision-relevant
sentence in the corpus. Keyword scoring cannot do this.

Per player the mentions collapse into a confidence-weighted mean, with recency
decay over a 7-day half-life, plus mention volume as a separate field. Volume
is reported but never fed to the model — it measures popularity, not quality.

### `projection.py`
Expected points per player over the next N gameweeks (default 3, decayed; see
Resolved parameters):

- **Minutes probability** from recent minutes, `status`, and
  `chance_of_playing_next_round`.
- **Attacking returns** from per-90 `expected_goals` and `expected_assists`,
  scaled by the position's points values.
- **Clean sheet probability** from the club's `expected_goals_conceded` against
  fixture difficulty; applies to goalkeepers and defenders, and partially to
  midfielders.
- **Defensive contribution** from the `defensive_contribution_per_90` field.
  Defenders score for clearances/blocks/interceptions above a threshold;
  midfielders and forwards on a combined tackles-and-recoveries measure. This
  is a real and often-overlooked points source for defensive midfielders.
- **Bonus** estimated from ICT and historical BPS rate.

Sentiment enters as a **bounded modifier**, capped at ±15%, applied to minutes
probability and to form — never directly to the points total.

This cap is the most important guardrail in the design. Community sentiment is
loud, herd-driven, and frequently wrong; its genuine edge is early injury and
rotation news that the statistical feed has not yet absorbed. The cap lets it
express exactly that and nothing more. `injury` and `rotation` categories move
minutes probability; `form` and `hype` move form. Any player whose `status` is
not `a` (available) has the modifier floor removed — the API's own injury flag
outranks the internet's opinion.

### `optimize.py`
Integer linear program via PuLP, maximising total projected points subject to
budget, squad size, positional quota, and the 3-per-club limit. Then a second
smaller ILP picks the best legal XI from the 15 and the captain. An ILP finds
the true optimum; a greedy pick does not, and the difference is worth real
points.

### `report.py`
Renders the squad, the reasoning, the sentiment evidence, and the optional
transfer diff. Emits JSON as the source of truth, plus a rendered page.

### `cli.py`
`fplbot run [--team-id N] [--horizon N] [--output PATH]`. One command, no
hidden state.

## Scheduling and reporting

Two runs per gameweek, both derived from `deadline_time`:

- **T-48h** — early look while prices are still moving.
- **T-3h** — the call that matters, after Friday press conferences.

The schedule is driven by a Claude routine that invokes the CLI, reads the
JSON, and publishes an Artifact page that updates in place at one stable URL.
The Python core stays free of any reporting-destination knowledge, so moving to
a Discord or Slack webhook later is a change to the routine, not the package.

## Error handling

- FPL API unreachable → use cache, mark the report stale with the cache age.
- Reddit credentials missing or rejected → fall back to the RSS route and
  label the run reduced-coverage.
- A sentiment source fails → continue with the rest; the report names what was
  missing.
- All sentiment fails → stats-only run, clearly labelled. Still a valid squad.
- The LLM returns malformed JSON → discard that batch, do not fall back to
  guessed scores.
- The ILP is infeasible → surface the binding constraint rather than returning
  a partial squad.

The principle throughout: a degraded run that says so beats a confident wrong
answer.

## Testing

- Cached API snapshots as fixtures. The projection and optimizer are pure
  functions over them — deterministic, offline, fast.
- Optimizer tests assert the hard constraints hold on the output: budget,
  squad shape, club limit, XI legality.
- A property test that raising a player's sentiment never lowers their
  projection.
- Mention resolution tested against a hand-built set including the known
  ambiguous surnames.
- The sentiment scorer checked against a small hand-labelled set of real posts;
  this is a calibration check, not a pass/fail gate.
- One end-to-end test running the full pipeline offline from fixtures.

## Layout

```
fpl-bot/
  fplbot/
    fetch.py
    news.py
    sentiment.py
    projection.py
    optimize.py
    report.py
    cli.py
  tests/
    fixtures/
  docs/superpowers/specs/
```

## Resolved parameters

Settled by research against public FPL repos and Reddit's own API docs; see
`docs/research/2026-09-07-fpl-horizon-and-sentiment-volume.md` for sourcing.

**Horizon = 3 gameweeks, `decay_base = 0.84`** (weights 1.00 / 0.84 / 0.71).
`--horizon` stays configurable. The established public solvers default to 5-8,
but they optimise a *sequence of transfers with banked free transfers*; this
design makes a single-period squad pick, so their number would arrive without
the machinery that justifies it. No public backtest compares horizons head to
head — these are conventions, not measured optima, and ours should be revisited
against real results.

**The sentiment modifier applies to the GW+1 term only.** Availability
forecasting degrades sharply with horizon while attacking-return forecasting
does not, and sentiment here is an availability signal. Spreading ±15% evenly
across three decayed weeks would leave GW+1 holding ~39% of the objective, so
the guardrail would move the total by only ~6% — far weaker than intended.

**Comment collection: 600 per run** (hard ceiling 1,000), from at most 8
threads, score floor >= 3, age <= 36h, deduplicated on normalised text. At
Haiku 4.5 rates this is ~$0.20/run, ~$15/season across two runs per gameweek —
so cost is not the binding constraint; signal quality is. Instrument
`comments_fetched`, `comments_after_filter`, `mentions_resolved`, and
`unique_players_touched` per run, and move the cap to where
`unique_players_touched` plateaus.

Exclude per-match live and bonus threads. They generate the most comments and
the least decision-relevant text — the fastest way to burn the cap on noise.
Target the daily megathread, the current "How Did ____ Play?" thread, and
listing posts above the score floor.

## Reddit implementation constraints

- **Rate limiting is not a real constraint here.** Reddit allows 100 queries
  per minute per OAuth client, averaged over a 10-minute window; a run costs
  ~15-25 calls. Do not build elaborate throttling — read `x-ratelimit-remaining`
  off live responses (it is a *float*) and back off if it drops, rather than
  hardcoding a limit against a window Reddit describes as "currently" 10 minutes.
- **`/comments/{article}` is not a listing and has no `after`/`before` cursor.**
  "Newest first, capped" must be implemented with `sort=new`, `depth`, and
  selective `/api/morechildren` expansion — not a `limit`/`after` loop. Reddit
  documents no maximum `limit` on this endpoint; verify empirically.
- **User-Agent format is mandated**: `python:fpl-bot:v1.0.0 (by /u/<username>)`.
  Generic UAs are heavily throttled.
- **One client id.** Registering multiple accounts or apps for the same use
  case is prohibited under Reddit's Responsible Builder Policy and grounds for
  a permanent block.

## Testing addendum

Multi-gameweek projections must **freeze lag features at the deadline**. When
projecting GW+2 and beyond, the lagged inputs cannot use those weeks' own
results; each subsequent week reuses the first week's lag values. Getting this
wrong leaks future data into the backtest and produces a model that looks
excellent and performs badly. The end-to-end test asserts that a projection for
GW+3 is unchanged when GW+1 and GW+2 results are mutated in the fixture.
