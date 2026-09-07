# Scheduling

Two runs per gameweek, both derived from the API's `deadline_time`:

- **T-48h**, early look, while prices are still moving.
- **T-3h**, the run that matters, after Friday press conferences.

Deadlines shift week to week, so do not use a fixed weekly cron. Query the
next deadline and schedule against it:

```bash
python3 -c "from fplbot.config import Settings; from fplbot.fetch import FplClient; print(FplClient(Settings.from_env()).next_gameweek().deadline_time)"
```

Example output:

```
Gameweek 4 2026-09-12 12:30:00+00:00
```

## Claude routine (default)

A scheduled Claude routine runs the CLI with `--batch`, scores the written
batch inside its own session, re-runs to pick up the scores, and publishes the
report as an Artifact that updates in place at one URL. No API key needed.

This is the recommended path since it requires no external API credentials.

## Standalone alternative

Set `ANTHROPIC_API_KEY` and run the CLI directly from any scheduler. The
package holds no knowledge of where reports go, so pointing it at a Discord or
Slack webhook is a change to the caller, not to `fplbot`.

## Reddit data

Reddit sentiment requires approved Data API access under the Responsible
Builder Policy, requested through a support ticket, and a dedicated bot account
registered at developers.reddit.com/app-registration. Until approval arrives,
the bot runs on news feeds alone and reports Reddit as absent, which is the
expected default rather than an error.

## Graceful degradation and debugging

If no scorer is configured at all, the bot produces a statistics-only squad
(xP model only, no sentiment) and notes this in the report.

To check when a scheduled run looks wrong, examine these fields in the report:

- `degraded` (boolean) and `degraded_reason` (string): indicates the run fell
  back to degraded mode, typically due to API failures or missing data sources.
- `sources_used` (list) and `sources_absent` (list): shows which data sources
  were available and which were not.
- `reddit_status` (one of `not_configured`, `ok`, or `failed`): if `failed`,
  check the exception type included in the report.
- `stats.cache_age_seconds`: indicates how old the cached FPL data is. A stale
  cache means the FPL API was unreachable and the run used older data rather
  than failing outright.
