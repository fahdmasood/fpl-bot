# Known limitations

Found during review and recorded deliberately rather than fixed. Each is a
bounded accuracy issue, not a correctness break.

## Scoring model

**Goals conceded is charged as `E[X]/2` rather than `E[floor(X/2)]`.** The real
rule docks a point per two goals conceded, which is a floor, and the
expectation of the floor is smaller than half the expectation. Every
goalkeeper and defender is therefore charged roughly 0.22 points per match too
much. The slope is right (0.500 against a true 0.494), so ranking within a
position is essentially correct; the level is not. Fixing it moves defenders
in the top 50 from 11 to 12 and the best keeper up by 0.54 over a three
gameweek horizon.

**Cards and the sub-60-minute appearance point are not modelled.**
`SCORING` defines `yellow_card`, `red_card`, `own_goal`, `penalty_miss`,
`penalty_save` and `appearance_under_60`, and the projection reads none of
them. Cards are close to noise at this horizon. The appearance point matters
for fringe players who come off the bench.

## Mention resolution

**All-caps text defeats the common-word guard.** Player names that are also
ordinary English words (Cash, Rice, Mount, Hall, King and others in
`COMMON_WORD_NAMES`) are accepted only when the club is named nearby or the
name is capitalised mid-sentence. In an all-caps headline every letter is
upper, so the capitalisation signal carries no information and
"CHELSEA PAID CASH FOR THE DEAL" resolves to Matty Cash. Tabloid RSS titles do
use caps.

The converse also costs something: a lowercase occurrence is never treated as
a player, so a writer who does not capitalise a name loses that mention. That
is the deliberate side to err on, since a spurious mention becomes a player's
entire signal.

**Club matching is a curated alias list.** `CLUB_ALIASES` in `sentiment.py`
maps how people write club names onto how the API spells them, because the
API says "Man Utd", "Spurs", "Nott'm Forest" and "Hull City" and almost nobody
else does. Three letter codes are deliberately excluded: Sunderland is SUN,
and matching that would resolve any mention of the sun or the newspaper. The
list needs a new entry whenever a club is promoted.

**`COMMON_WORD_NAMES` needs maintenance.** It is a hand-curated list and goes
stale every transfer window.

Scope of the guard: 18 of 654 players are affected and only four are
meaningfully owned. Measured cost is roughly one dropped mention in 59, and a
lost signal is capped by the 15 percent sentiment bound at about 0.5 points on
a 3.5 point projection. The guard errs toward false negatives deliberately,
because a spurious mention becomes a player's entire signal.

## News collection

**A legitimately empty feed reads as an outage.** Failure is inferred from zero
entries, since `feedparser` does not raise on a dead host. A feed that is
healthy but genuinely has nothing new is indistinguishable.

**One failed feed of six marks the whole report degraded.** Reports will
therefore read degraded fairly often, which risks training the reader to
ignore the banner.

## Test coverage gaps

Confirmed by mutation testing, these three changes do not fail any test:

- removing shrinkage from the saves rate
- applying the saves term to outfield players (inert in practice, since
  outfielders record no saves)
- charging goals conceded to defenders only and not goalkeepers

`test_defenders_do_not_dominate_the_top_of_the_board` also has more slack than
intended. Its bound is 20 of the top 50 against an observed 11, and removing
only the goals conceded penalty still passes at 18. A bound of 15 would make it
a real guard.
