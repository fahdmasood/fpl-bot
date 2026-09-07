# Projection horizon and sentiment volume, research note

**Date:** 2026-09-07
**Feeds:** `docs/superpowers/specs/2026-09-07-fpl-sentiment-bot-design.md`, "Open items for planning"
**Method:** primary sources only. Every repo below was cloned and read locally, or the file was
fetched from the owner's own site. Claims are tagged:
**[CODE]** = read in source/config · **[PROSE]** = author's own claim in README/docs/paper ·
**[INFER]** = my reasoning, not anyone's claim · **[UNVERIFIED]** = could not confirm at source.

---

## Q1, Projection horizon

### Recommendation

> **`horizon = 3`, `decay_base = 0.84`** (weights 1.00 / 0.84 / 0.71; effective horizon ≈ 2.55 GW).
> **Apply the sentiment modifier to the GW+1 term only**, not to all three weeks.
> Keep `--horizon 1` and `--horizon 5` reachable from the CLI; do **not** default to 8.

Why not 1: fixture runs are real and cost nothing to price in, and every serious public tool
looks past the next week.
Why not 8: the 5–8 GW defaults in the public solvers exist because those tools optimise a
*sequence of transfers with banked free transfers*, which v1 of this design does not do. Copying
their horizon into a single-period squad pick imports the number without the machinery that
justifies it. **[INFER]**

Why sentiment should be GW+1-only: the one published head-to-head with per-horizon numbers shows
the **availability** part of an FPL forecast degrades sharply with horizon while the attacking-return
part does not (Table 4 below). Sentiment in this design is an availability signal. Spreading a
±15% minutes modifier evenly over three decayed weeks would leave GW+1 holding ~39% of the
objective, so the guardrail would move the total by only ~6%, much weaker than the spec intends. **[INFER]**

### What the public projects actually default to

All values below were read in the repo, at the path given.

| Project | Horizon default | Decay default | Where I read it |
|---|---|---|---|
| **sertalpbilal/FPL-Optimization-Tools** (HEAD, commit `bf7b2fa`, 2026-09-04) | **8** | **0.9** | `data/user_settings.json` and `data/comprehensive_settings.json`, keys `horizon` / `decay_base` **[CODE]** |
| same repo, in-code fallback when the key is absent | **3** | **0.84** | `dev/solver.py:291` `options.get("horizon", 3)`; `dev/solver.py:293` `options.get("decay_base", 0.84)` **[CODE]** |
| same repo, historical default, 2023-02 → 2025-03 | **5** | **0.84** | `git log -p data/regular_settings.json`; stable at 5/0.84 across ~40 commits from `d9b0ae7` (2023-02-21) to `fcdbe30` (2025-03-03) **[CODE]** |
| **TDRoss/JFPL-Optimization** (Julia port of the above, last commit 2024-10-06) | **5** | **0.84** | `data/regular_settings.json` **[CODE]** |
| **FPL Review** solver docs (the projection provider both of the above consume) | depth **6** | **0.85** | <https://docs.fplreview.com/the-model/solvers/settings/> **[PROSE]** |
| **joconnor-ml/forecasting-fantasy-football** (last commit 2023-08-10) | **5** | **none, flat mean** | `app/utils.py:19` `forecast_horizon: int = 5`; `scripts/compute_features.py:35` `max_horizon = 5`; `app/pages/4_🧠_Team_Optimiser.py` aggregates with `.agg({"score_pred": "mean", ...})` over the 5 horizon rows **[CODE]** |
| **177arc/fpl-advisor** (last commit 2021-07-24) | **8** | n/a | `advisor.ipynb`, `ctx.def_next_gws = get_next_gw_name(min(ctx.total_gws - ctx.next_gw + 1, 8), ctx)` **[CODE]** |
| **dbozbay/FPL-Optimization** (last commit 2024-09-18) | **5** | **no decay parameter exists** | `settings.json` is two keys: `team_id`, `horizon: 5`. `grep -ri "decay\|discount"` over the whole repo returns nothing **[CODE]** |
| **solpaul/fpl-prediction** (last commit 2022-08-02) | **6** (evaluation window, not an optimiser) | n/a | `valid_len = 6` appears in 10 notebook cells; `fpl_predictor/util.py:442 validation_gw_idx(df, season, gw, length)` **[CODE]**; README: "calculate the mean absolute error for the following 6 gameweeks" **[PROSE]** |
| **Torvaney/fpl-optimiser** (last commit 2019-04-16) | n/a, whole-season retrospective | n/a | `optimise.py:23` `get_optimal_squad(..., season='2016/17', optimise_on='total_points')`. This is a hindsight "what was the best possible squad" tool, **not** a forward planner. It is frequently cited as an FPL optimiser; it does not answer the horizon question. **[CODE]** |

Notes on the table:

- **The horizon default rose 5 → 8 and decay 0.84 → 0.9 in the Aug 2025 rewrite** of the
  sertalpbilal repo (the rewrite that introduced `data/user_settings.json`). I read the values
  either side of the change. I searched the commit messages between 2025-03 and 2025-09 and
  **found no commit message stating the reason**. Do not attribute a rationale to that change. **[CODE + gap]**
- The repo's own README now points installation at `github.com/solioanalytics/open-fpl-solver`
  and lists `chris.musson@hotmail.com` as maintainer contact, i.e. the widely-used "Alpha"
  toolchain has moved/rebranded. `README.md`, "Clone the Repository" and "Issues" sections. **[CODE]**
- ChrisMusson/FPL_Optimiser (last commit 2020-10-09) is a *lineup* optimiser (best XI from a
  fixed 15), so it is single-GW by construction and carries no horizon key. **[CODE]**
- spinalwiz/fpl-optimiser (last commit 2021-07-29): `grep -i horizon|decay|lookahead` over the
  whole repo returns nothing. No horizon concept. **[CODE]**

### How decay is actually applied

`dev/solver.py:810-813` **[CODE]**:

```python
if objective == "regular":
    objective_expr = sum_(gw_total[w] for w in gws)
else:
    objective_expr = sum_(gw_total[w] * pow(decay_base, w - next_gw) for w in gws)
```

and `dev/solver.py:292`, `objective = options.get("objective", "decay")`, so **decay is on by
default**; `"regular"` (flat) is the opt-out. The weight on gameweek *w* is `decay_base^(w - next_gw)`,
so the next GW always has weight 1.0 and later weeks are geometrically discounted. **[CODE]**

`decay_base` is documented in the repo as `data/README.md:20-21`: *"value assigned to decay rate of
expected points (discounts future GWs) … `"decay_base": 0.9` (10% discount per GW)"*. **[PROSE]**

**Effective horizon** (the sum of the decay weights) is the number worth comparing across tools,
because horizon and decay trade off against each other. **[INFER]**

| Config | GW+1 share of objective | Effective horizon (Σ weights) |
|---|---|---|
| h=8, d=0.90 (current sertalpbilal) | 17.6% | 5.70 |
| h=6, d=0.85 (FPL Review doc default) | 24.1% | 4.15 |
| h=5, d=0.84 (old sertalpbilal / JFPL) | 27.5% | 3.64 |
| **h=3, d=0.84 (recommended here)** | **39.3%** | **2.55** |
| h=5, d=1.0 (joconnor flat mean) | 20.0% | 5.00 |
| h=1 | 100% | 1.00 |

### The only per-horizon accuracy evidence I found

**OpenFPL**, Daniel Groos, arXiv:2508.09992 (submitted 2025-07-29), code at
<https://github.com/daniegr/OpenFPL>. Prospective 2024-25 evaluation of OpenFPL vs. the paid FPL
Review Massive Data Model vs. a "last 5 matches mean" baseline, at 1, 2 and 3 GW horizons.
Table 4 of the paper, RMSE (MAE), transcribed from the PDF **[CODE, read from the paper's own PDF]**:

| GW ahead | Method | Zeros | Blanks | Tickers | Haulers |
|---|---|---|---|---|---|
| 1 | FPL Review | **0.689** (0.237) | **1.189** (0.597) | 1.594 (1.227) | 5.172 (4.381) |
| 1 | OpenFPL | 0.818 (0.427) | 1.291 (0.749) | **1.517** (1.127) | **5.142** (4.317) |
| 2 | FPL Review | 0.826 (0.313) | 1.219 (0.631) | 1.605 (1.228) | 5.169 (4.447) |
| 2 | OpenFPL | 0.922 (0.475) | 1.309 (0.765) | 1.569 (1.192) | 5.051 (4.289) |
| 3 | FPL Review | 0.918 (0.372) | 1.260 (0.666) | 1.402 (1.099) | 5.197 (4.512) |
| 3 | OpenFPL | 0.966 (0.506) | 1.306 (0.775) | **1.369** (1.079) | 5.171 (4.467) |

(*Zeros* = did not play, 0 pts; *Blanks* = played, ≤2 pts; *Tickers* = 3–4 pts; *Haulers* = ≥5 pts.)

The paper's own conclusion, verbatim **[PROSE]**:

> "For all methods, shorter horizon improves forecasting for low-return categories, especially for
> Zeros where one-gameweek-ahead predictions lower RMSE by 15-25% relative to forecasts three
> gameweeks ahead. **No systematic horizon effect is observed for the high-return categories.**"

Read that as: *will he play* decays fast with horizon; *how many points if he plays* barely decays
at all. Which is the strongest single argument for the split recommendation above, keep a
multi-GW horizon for the attacking/clean-sheet terms, keep the availability/sentiment term short. **[INFER]**

The same paper's discussion also directly validates the sentiment premise of this design **[PROSE]**:

> "Although the FPL API encodes injuries and suspensions, it does not indicate whether a player is
> expected to start a match, come on as a substitute, or be rested ahead of more important fixtures.
> FPL Review and other commercial methods … integrate detailed expected-minutes projections derived
> from team news, fixture congestion, and betting odds. … future work could extend this line through
> **crowd-sourced minutes forecasts and web-scraping AI agents**."

That is the gap this bot's Reddit signal is aimed at, named by the author of the strongest open
FPL forecasting model. It is a hypothesis he flags as future work, **not a demonstrated result**.

### Transfer planning vs. single-GW greedy, why their horizons are big

This is the part that does *not* transfer to our v1, and is the main reason not to copy `horizon=8`.

- **sertalpbilal**: the model is multi-period. Decision variables are indexed `[player, week]`,
  `lineup[p, w]`, `captain[p, w]`, `transfer_in/out[p, w]`, and the objective sums
  `gw_total[w] = gw_xp[w] − hit_cost·penalized_transfers[w] + gw_ft_gain[w] − ft_penalty[w]
  + itb_value·in_the_bank[w] − cp_penalty[w]` over the horizon (`dev/solver.py`, `gw_xp` / `gw_total`
  construction ~line 796-808). The horizon is a *transfer-plan depth*, not a valuation window. **[CODE]**
- Two settings exist purely to stop the horizon's tail producing nonsense: `no_transfer_last_gws: 2`
  (`data/README.md:52`, "number of gameweeks at end where transfers are banned") and
  `ft_use_penalty: 0.2` (`data/README.md:86`, "penalty applied when a free transfer is used
  (prevents trivial scheduled transfers)"). **[CODE + PROSE]**
- FPL Review's docs give the same warning in prose: *"Leaving a buffer of a week or two, from the
  loaded projection horizon, can prevent unrealistic 'dead-end' moves in the final weeks."*
  <https://docs.fplreview.com/the-model/solvers/settings/> **[PROSE]**
- `ft_value_list: {"2": 2, "3": 1.6, "4": 1.3, "5": 1.1}` prices the option value of *banking* free
  transfers up to 5 (`data/user_settings.json`; explained at `data/README.md:22-23`). Banking FTs is
  what makes a long horizon pay: you plan a sequence, then execute only this week's move. **[CODE]**
- **joconnor-ml** shows the same plan-long-execute-one pattern in a much smaller form:
  `fpl_opt/transfers.py` has a `MultiHorizonTransferOptimiser` with 2-D `[week][player]` decision
  arrays and **no decay at all** (weeks summed flat), and the extraction helper is
  `get_transfer_dfs(transfer_in_decisions, transfer_out_decisions, player_df, week=0)`, it reads
  back **week 0 only**. **[CODE]**
- Note a real bug in that helper, if you ever lift the code:
  `transfers_out = [i for i in ... if transfer_in_decisions[week][i].value() == 1]`, it filters the
  out-list on the *in*-decisions. `fpl_opt/transfers.py`, `get_transfer_dfs`. **[CODE]**

**Implication for this design.** The spec's `optimize.py` is a single-period ILP plus an optional
transfer diff. Under that structure the horizon is a *valuation window*, and the right length is
"far enough to see a fixture swing, short enough that the availability term still means something".
3 with a 0.84 decay is where I land. If v2 ever adds multi-period transfer planning with banked
FTs, revisit and go to 5–8 with decay 0.84–0.9, the public consensus band. **[INFER]**

### Gaps I could not close

- **No published backtest comparing horizon values head-to-head.** I looked for one and did not
  find it. Nobody in these repos publishes "h=3 beat h=8 by N points/season". The 5/6/8 defaults
  are convention plus each author's judgement, not a measured optimum. Treat every horizon number
  in the table as a *convention*, not a result. **[gap]**
- No commit message, issue, or README in sertalpbilal/FPL-Optimization-Tools states *why* 8 and 0.9.
  **[gap]**
- solpaul's per-gameweek MAE printouts in `notebooks/02_...ipynb` and `05_...ipynb` are indexed by
  *gameweek of the season* (GW1…GW38 start points, each evaluated over the following 6), **not** by
  forecast horizon. They do **not** show horizon degradation and must not be cited as such. I
  initially misread them; flagging so nobody else does. **[CODE]**
- One methodological point from that repo worth stealing regardless, `fpl_predictor/util.py`
  comment above `create_lag_train` **[CODE]**: *"When making predictions for gw +2 and beyond we
  cannot use those weeks's lag features… Instead, each subsequent validation week should have the
  same lag values as the first."* Multi-GW projections must freeze features at the deadline, or the
  backtest leaks. Relevant to the spec's "Testing" section.

---

## Q2, Sentiment volume caps and filtering

### Recommendation

> **Per-run cap: 600 comments** (hard ceiling 1,000), drawn from **at most 8 threads**,
> **score floor ≥ 3**, **age ≤ 36h**, deduplicated on normalised text.
> Reddit budget per run: ~15-25 authenticated API calls, nowhere near the limit.
>
> These are *budget-derived and volume-derived*, not copied from a comparable project,
> **because no comparable public project exists** (see "Prior art" below). Treat them as a
> starting point to be replaced by a measured run, exactly as the spec already says.

Derivation, stated openly so the numbers can be argued with **[INFER]**:

| Input | Value | Basis |
|---|---|---|
| Cost per comment scored, Claude Haiku 4.5 (`claude-haiku-4-5`, $1.00/MTok in, $5.00/MTok out) | ~$0.0003 | ~120 input tok/comment amortised over a batched prompt + ~40 output tok for the `{player_id, sentiment, category, confidence}` record |
| 600 comments/run | **~$0.20/run** | |
| 2 runs/GW × 38 GW | **~$15/season**, ~$7.50 via the Batch API (50% discount) | |
| 1,000 comments/run | ~$0.33/run, ~$25/season | the ceiling, for a busy matchday |

Cost is not the binding constraint at these volumes; *signal quality* is. The cap should be set
by where marginal comments stop resolving to new player mentions, which is a measurable quantity
once the collector runs. Instrument it: log `comments_fetched`, `comments_after_filter`,
`mentions_resolved`, `unique_players_touched` per run, and move the cap to where
`unique_players_touched` plateaus.

### Reddit API limits, from Reddit's own pages

| Fact | Value | Source |
|---|---|---|
| OAuth client limit | **100 queries per minute per OAuth client id, averaged over a 10-minute window** (≈1,000 requests per rolling 600s, burstable) | <https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki>, verbatim: *"The limit is: 100 queries per minute (QPM) per OAuth client id. QPM limits will be an average over a time window (currently 10 minutes) to support bursting requests."* **[PROSE, Reddit-owned]** |
| Keyed on | `client_id` only (changed from client+user on 2023-07-01) | Same page, 2023 archived snapshot **[PROSE]** |
| Unauthenticated | **Blocked**, not throttled. Current page: *"Traffic not using OAuth or login credentials will be blocked, and the default rate limit will not apply."* The often-quoted 10 QPM figure is the pre-2026 policy. | Same page **[PROSE]** |
| Headers | `x-ratelimit-used` (int), `x-ratelimit-remaining` (**float**, parse as float), `x-ratelimit-reset` (seconds to end of period) | Same page; `prawcore/rate_limit.py` parses `remaining` via `int(float(...))` **[CODE]** |
| Window | 600s. Reddit hedges *"currently 10 minutes"*; `prawcore/const.py` has `WINDOW_SIZE = 600` | <https://raw.githubusercontent.com/praw-dev/prawcore/main/prawcore/const.py> **[CODE]** |
| `/r/{sub}/hot`, `/new`, `/top`, `limit` | default 25, **max 100**; supports `after` / `before` / `count` cursor pagination | <https://www.reddit.com/dev/api/> **[PROSE, Reddit-owned]** |
| `/r/{sub}/comments/{article}`, `limit` | *"(optional) an integer"*, **no maximum documented**. This endpoint is **not** a listing: there is **no `after`/`before`/`count`**. Use `depth`, `sort` (`confidence`/`top`/`new`/`controversial`/`old`/`random`/`qa`/`live`), and `/api/morechildren` to expand `more` stubs. | <https://www.reddit.com/dev/api/> **[PROSE]**, **[gap]** Reddit publishes no max here; verify empirically against your own client rather than assuming 100. |
| User-Agent | Mandated format `<platform>:<app ID>:<version string> (by /u/<reddit username>)`; e.g. `python:fpl-bot:v1.0.0 (by /u/yourname)`. Default UAs like `Python/urllib` are *"drastically limited"*. *"NEVER lie about your User-Agent."* | Reddit Data API Wiki **[PROSE]** |
| Multiple tokens/accounts to dodge limits | **Explicitly prohibited.** Responsible Builder Policy: *"This prohibits registering multiple accounts or submitting multiple requests for the same use case."* Data API Terms 3.2 reserves the right to *"permanently block your access"*. | <https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy>, <https://www.redditinc.com/policies/data-api-terms> **[PROSE]** |

Two things that change the design:

1. **The 100 QPM budget is enormous relative to this workload.** A run that reads one listing
   (1 call) plus 8 comment trees (1-3 calls each with `morechildren` expansion) is ~15-25 calls.
   You could run 40 of these per minute. **Rate limiting is not a real constraint for this bot**,
   so don't build elaborate throttling; just read `x-ratelimit-remaining` off live responses and
   back off if it drops, rather than hardcoding 100/min against a window Reddit says may change. **[INFER]**
2. **`/comments/{id}` has no cursor pagination.** The spec's "capped per run, newest first" needs
   to be implemented as `sort=new` + `depth` + selective `/api/morechildren`, not as a
   `limit`/`after` loop. **[INFER from the endpoint's documented parameters]**

### Measured locally on this machine, 2026-09-07

Verifying the spec's claims about the unauthenticated fallback path. All measured with
`curl -A "python:fpl-bot-research:v0.1 (by /u/fplbot)"`:

| Request | Result |
|---|---|
| `GET https://www.reddit.com/r/FantasyPL/hot.rss` | **200** ✅ spec's claim holds |
| `GET https://www.reddit.com/r/FantasyPL/hot.json?limit=5` | **403** ✅ spec's claim holds, keep the "do not reintroduce the `.json` path" note |
| Second `.rss` request, immediately after the first | **429**, with `x-ratelimit-used: 1`, `x-ratelimit-remaining: 0.0`, **`x-ratelimit-reset: 24`** |

**New finding the spec does not account for: the unauthenticated RSS route allows roughly
one request per ~24 seconds.** The `.rss` fallback works, but only if requests are spaced by the
`x-ratelimit-reset` value. A fallback that fires both feeds back-to-back will 429 on the second
and silently return an empty body (`content-length: 0`). Sleep on `x-ratelimit-reset` between
fallback fetches. **[measured]**

**Second new finding: the RSS route is *not* titles-only.** Fetching a specific submission's
comment feed,
`https://www.reddit.com/r/FantasyPL/comments/{id}/.rss?limit=500&sort=new`, returned
**83 `<entry>` elements with full comment bodies** in `<content type="html">`, unauthenticated.
Measured against `t3_1w9abnu` (the daily thread), fetched ~2.6h after it was posted. The `limit`
parameter *is* honoured here, `limit=10` returned exactly 10 entries, so 83 was the number
actually reachable, not a ceiling. **[measured]**

What the RSS route *does* lack is **`score`**, the entry schema is
`author, category, content, id, link, title, updated` (plus `published` on submission feeds).
So on the fallback path the spec's score-floor filter cannot be applied at all. That is the
honest characterisation of "reduced coverage": not *titles only*, but *comments without scores*,
one request per ~24s. Worth correcting in the spec. **[measured]**

### r/FantasyPL thread structure, observed live, not assumed

From the `hot.rss` fetch on 2026-09-07 (25 entries). Recurring, high-value thread types actually
present, with their real titles **[measured]**:

| Thread | Poster | Why it matters |
|---|---|---|
| `Rate My Team, Quick Questions & General Advice Daily Thread` | `/u/FPLModerator`, posted 22:30 UTC | The daily megathread. This is the volume. 83 comments within ~2.6h of posting; a matchday day's total will be far higher. |
| `How Did ____ Play? Gameweek N (26/27)` | mod | Post-match eye-test on minutes, roles, substitutions, **the single richest source for the minutes-probability signal** this design cares about |
| `GW3 Lineups \| Arsenal Vs. Chelsea \| 2026/27` | per-match | Confirmed XIs. Highest-precision rotation evidence, but only useful post-deadline, relevant for the *next* run, not the current one |
| `Bonus Points + Defcon (ARS v CHE) 🏆🛡️` | per-match | Live match thread; very high volume, very low signal-per-comment. **Recommend excluding.** |
| `Tonight's Predicted Price Changes` / `Price Changes` | daily | Price movement, not points. Out of scope for this design. |
| Ad-hoc injury/presser posts, e.g. `Arteta provides an update on Mosquera`, `Konsa (4.4) starts for Arsenal` | users | The high-value long tail. Catch these via a score threshold on the listing rather than by title matching. |

**Targeting recommendation** **[INFER, grounded in the observed titles]**: the daily megathread
+ the current `How Did ____ Play?` thread + any listing post above a score floor. Exclude the
per-match live/bonus threads, they generate the most comments and the least decision-relevant
text, so including them is the fastest way to burn the cap on noise.
