# fplbot

Picks a Fantasy Premier League squad by combining the official FPL statistics
API with football news sentiment.

Personal, non-commercial, and **advisory only**: it holds no FPL credentials and
never touches anyone's team.

## What it does

1. Reads the public FPL API for prices, expected goals and assists, expected
   goals conceded, availability flags, and fixture difficulty.
2. Reads public football news feeds (and, optionally, r/FantasyPL) for early
   injury, fitness and rotation news.
3. Scores that text for sentiment about **players**, and folds it into an
   expected-points model as a bounded modifier, capped at ±15% and applied
   only to the next gameweek.
4. Solves an integer linear program for the best legal 15-man squad within the
   £100.0m budget, plus a starting XI and captain.
5. Writes a private report before each deadline.

Sentiment scoring is not on by default. It requires either an
`ANTHROPIC_API_KEY` or a `--batch` run scored by a Claude session; with
neither set, the CLI skips scoring entirely and builds the squad from
statistics alone, marking the report as reduced coverage rather than
silently pretending sentiment was considered.

## Data use

**Read-only.** Nothing is posted, commented, voted on, moderated, or messaged.

If Reddit access is enabled:

- No comment author is ever stored. `NewsItem` has no `author` field, by design.
- No characteristics are inferred about any Reddit user. Sentiment is attributed
  to **footballers**, never to the person writing about them.
- Nothing is retained beyond the run, and nothing trains any model, text is
  scored at inference time and discarded.
- Roughly 15-25 API calls per run, twice per gameweek.

Reddit is optional and gated on approved Data API access. Without it the bot
runs on public news feeds and reports Reddit coverage as absent.

## Setup

    python3 -m pip install -e '.[dev]'
    cp .env.example .env

No credentials are required to run.

## Use

    fplbot --output report          # writes report.json and report.html
    fplbot --team-id 1234567        # also diff against your current squad
    fplbot --horizon 1              # next gameweek only
    fplbot --team-id 1234567 --free-transfers 2   # limit recommendations to what you can actually play

With `--team-id`, transfers are only recommended when they use a free
transfer or gain more than the 4 point hit cost of an extra one. Your bank
balance is read from the same public endpoint as your picks, so an upgrade
you can actually fund is priced as affordable. A move you cannot fund is
still shown, marked as not recommended, and never given a free transfer.
Everything else is shown too, so nothing is hidden.

An entry that has not played a gameweek yet has no picks endpoint. The run
reports that and produces a squad anyway.

## Tests

    python3 -m pytest

Every test runs offline against committed API snapshots. No test touches the
network.

## Design

- `docs/superpowers/specs/`, the design, and why each parameter is what it is
- `docs/research/`, sourced research behind the projection horizon and caps
