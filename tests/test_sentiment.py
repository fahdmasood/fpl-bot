from datetime import datetime, timedelta, timezone

from fplbot.models import Mention, MentionScore, NewsItem, Player
from fplbot.sentiment import NullScorer, aggregate, resolve_mentions

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def make_player(pid, name, team=1):
    return Player.from_api({
        "id": pid, "web_name": name, "element_type": 3, "team": team,
        "now_cost": 50, "status": "a", "chance_of_playing_next_round": None,
        "minutes": 900, "form": "3.0", "selected_by_percent": "5",
        "expected_goals": "1.0", "expected_assists": "1.0",
        "expected_goals_conceded": "10.0", "defensive_contribution_per_90": "1.0",
        "ict_index": "50", "bps": 200, "starts": 10,
    })


def item(body, hours_ago=1, src="bbc"):
    return NewsItem(source=src, url="u",
                    published_at=NOW - timedelta(hours=hours_ago),
                    title="t", body=body, score=10)


def test_resolves_a_plain_mention():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka looked sharp today")], players)
    assert len(mentions) == 1
    assert mentions[0].player_id == 1


def test_matching_is_case_insensitive_and_word_bounded():
    players = [make_player(1, "Saka")]
    assert len(resolve_mentions([item("SAKA is fit")], players)) == 1
    assert resolve_mentions([item("Sakamoto scored")], players) == []


def test_ambiguous_surname_without_club_context_is_dropped():
    """A wrongly attributed injury rumour is worse than a missing one."""
    players = [make_player(1, "Silva", team=1), make_player(2, "Silva", team=2)]
    assert resolve_mentions([item("Silva is injured")], players) == []


def test_ambiguous_surname_resolved_by_club_context():
    players = [make_player(1, "Silva", team=1), make_player(2, "Silva", team=2)]
    teams = {1: "Arsenal", 2: "Chelsea"}
    mentions = resolve_mentions([item("Chelsea's Silva is injured")], players, teams)
    assert [m.player_id for m in mentions] == [2]


def test_null_scorer_returns_nothing():
    assert NullScorer().score([1, 2, 3]) == {}


def test_scores_are_matched_by_index_not_position():
    """A discarded batch must not shift scores onto the wrong player."""
    players = [make_player(1, "Saka"), make_player(2, "Odegaard")]
    mentions = resolve_mentions(
        [item("Saka has a knock"), item("Odegaard has a knock")], players)
    assert [m.player_id for m in mentions] == [1, 2]
    # Only the second mention scored; the first was dropped as malformed.
    agg = aggregate(mentions, {1: MentionScore(2, -0.8, "injury", 0.9)}, NOW)
    assert 1 not in agg
    assert agg[2].availability_signal < 0


def test_aggregate_splits_availability_and_form_signals():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka has a knock")], players)
    scores = {0: MentionScore(1, sentiment=-0.8, category="injury", confidence=0.9)}
    agg = aggregate(mentions, scores, NOW)
    assert agg[1].availability_signal < 0
    assert agg[1].form_signal == 0


def test_aggregate_decays_old_mentions():
    players = [make_player(1, "Saka")]
    recent = resolve_mentions([item("Saka has a knock", hours_ago=1)], players)
    old = resolve_mentions([item("Saka has a knock", hours_ago=24 * 14)], players)
    score = {0: MentionScore(1, sentiment=-1.0, category="injury", confidence=1.0)}
    fresh = aggregate(recent, score, NOW)[1].availability_signal
    stale = aggregate(old, score, NOW)[1].availability_signal
    assert abs(stale) < abs(fresh)


def test_one_comment_yields_one_mention_per_player():
    """A player indexed under both a full name and a surname, such as
    "De Bruyne", must not match twice and count as two pieces of evidence."""
    players = [make_player(1, "De Bruyne")]
    mentions = resolve_mentions([item("De Bruyne was brilliant tonight")], players)
    assert len(mentions) == 1, [m.matched_text for m in mentions]


def test_a_swarm_of_weak_mentions_cannot_overturn_a_strong_one():
    """600 low confidence comments saying a player is fine must not flip a
    confident injury report from a trusted source. Volume measures how much a
    player is discussed, not how reliable the claim is."""
    players = [make_player(1, "Saka")]
    strong = item("Saka has been ruled out with a hamstring injury")
    weak = [item("Saka is fine, saw him training", src="reddit-comment")
            for _ in range(600)]
    mentions = resolve_mentions([strong] + weak, players)

    scores = {0: MentionScore(1, sentiment=-1.0, category="injury", confidence=1.0)}
    for i in range(1, len(mentions)):
        scores[i] = MentionScore(1, sentiment=0.6, category="injury", confidence=0.15)

    signal = aggregate(mentions, scores, NOW)[1].availability_signal
    assert signal < -0.5, f"a comment flood overturned a wire report: {signal}"


def test_moderate_trust_evidence_is_not_silently_inert():
    """Credible but lower confidence reporting must be able to move a signal.

    A floor set too high makes an entire tier of sources count for nothing at
    any volume, which is a false negative rather than caution: several
    Saturday reports that a player trained should eventually outweigh one
    Friday injury note.
    """
    players = [make_player(1, "Saka")]
    friday = item("Saka limped off and is a doubt", hours_ago=24)
    saturday = [item("Saka trained fully today", src="sky") for _ in range(5)]
    mentions = resolve_mentions([friday] + saturday, players)

    scores = {0: MentionScore(1, sentiment=-1.0, category="injury", confidence=1.0)}
    for i in range(1, len(mentions)):
        scores[i] = MentionScore(1, sentiment=0.7, category="injury", confidence=0.5)

    with_contradiction = aggregate(mentions, scores, NOW)[1].availability_signal
    alone = aggregate(mentions[:1], {0: scores[0]}, NOW)[1].availability_signal
    assert with_contradiction > alone, (
        f"credible contradicting reports had no effect: {alone} to {with_contradiction}"
    )


def test_volume_does_not_inflate_the_signal():
    """Ten people repeating a claim is not ten pieces of evidence. Volume
    measures popularity, not quality, and must never reach the model."""
    players = [make_player(1, "Saka")]
    one = resolve_mentions([item("Saka has a knock")], players)
    many = resolve_mentions([item("Saka has a knock")] * 10, players)
    score = MentionScore(1, sentiment=-0.8, category="injury", confidence=0.9)
    single = aggregate(one, {0: score}, NOW)[1].availability_signal
    repeated = aggregate(many, {i: score for i in range(10)}, NOW)[1].availability_signal
    assert abs(single - repeated) < 1e-9, (single, repeated)


def test_a_low_confidence_mention_moves_the_signal_less():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka has a knock")], players)
    confident = aggregate(
        mentions, {0: MentionScore(1, -1.0, "injury", 1.0)}, NOW)[1].availability_signal
    unsure = aggregate(
        mentions, {0: MentionScore(1, -1.0, "injury", 0.2)}, NOW)[1].availability_signal
    assert abs(unsure) < abs(confident)


def test_aggregate_signals_stay_in_range():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka has a knock")] * 1, players)
    scores = {0: MentionScore(1, sentiment=-5.0, category="injury", confidence=2.0)}
    agg = aggregate(mentions, scores, NOW)
    assert -1.0 <= agg[1].availability_signal <= 1.0


def test_volume_is_recorded_but_separate():
    players = [make_player(1, "Saka")]
    mentions = resolve_mentions([item("Saka good"), item("Saka great")], players)
    scores = {0: MentionScore(1, 0.5, "form", 0.8), 1: MentionScore(1, 0.5, "form", 0.8)}
    agg = aggregate(mentions, scores, NOW)
    assert agg[1].volume == 2


# Real API club names, which are what resolve_mentions is given in the
# pipeline. They matter: the API says "Man Utd", not "Manchester United".
CLUBS = {1: "Aston Villa", 2: "Man Utd", 3: "Arsenal", 4: "Everton", 5: "Chelsea"}


def test_an_ordinary_word_is_not_a_player_mention():
    """Live false positive: "Chelsea paid cash for the deal" resolved to
    Matty Cash of Aston Villa. Because RELEVANCE_FLOOR scales to the best
    evidence for a player, one spurious mention becomes his entire signal."""
    players = [make_player(1, "Cash", team=1)]
    got = resolve_mentions([item("Chelsea paid cash for the deal")], players, CLUBS)
    assert got == [], [m.matched_text for m in got]


def test_a_club_name_containing_a_player_name_is_not_a_mention():
    """"Manchester United will mount a challenge" resolved to Mason Mount."""
    players = [make_player(1, "Mount", team=2)]
    got = resolve_mentions(
        [item("Manchester United will mount a challenge")], players, CLUBS)
    assert got == [], [m.matched_text for m in got]


def test_a_sentence_of_plain_english_yields_no_mentions():
    """"Rice and beans is the king" resolved to both Rice and King."""
    players = [make_player(1, "Rice", team=3), make_player(2, "King", team=4)]
    got = resolve_mentions([item("Rice and beans is the king")], players, CLUBS)
    assert got == [], [m.matched_text for m in got]


def test_a_common_word_name_is_accepted_when_the_club_is_named():
    players = [make_player(1, "Cash", team=1)]
    got = resolve_mentions([item("Aston Villa's cash is fit again")], players, CLUBS)
    assert [m.player_id for m in got] == [1]


def test_a_common_word_name_is_accepted_when_capitalised_in_context():
    """Capitalised mid sentence, so the capital is doing real work rather
    than being the automatic capital every sentence opens with."""
    players = [make_player(1, "Cash", team=1)]
    got = resolve_mentions(
        [item("Emery said Cash was excellent at right back")], players, CLUBS)
    assert [m.player_id for m in got] == [1]


def test_a_name_that_is_not_an_ordinary_word_still_matches_in_lower_case():
    """The guard applies only to names that are also English words. Everyone
    else must keep resolving from lowercased comment text."""
    players = [make_player(1, "Saka", team=3)]
    assert len(resolve_mentions([item("saka is fit again")], players, CLUBS)) == 1


def test_a_bare_sentence_initial_common_word_name_is_dropped():
    """A deliberate false negative, and the one place this guard costs signal.

    "Cash was excellent at right back" with no club named and nothing before
    it is indistinguishable, by capitalisation alone, from "Rice and beans is
    the king": both are a capitalised common word opening a sentence. Since
    one spurious mention becomes a player's entire signal, the pair is
    resolved in favour of dropping both. A club name or any preceding clause
    brings the mention back.
    """
    players = [make_player(1, "Cash", team=1)]
    assert resolve_mentions([
        NewsItem(source="bbc", url="u", published_at=NOW, title="",
                 body="Cash was excellent at right back", score=10)
    ], players, CLUBS) == []
