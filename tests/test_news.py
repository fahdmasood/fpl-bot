from datetime import datetime, timedelta, timezone

from fplbot.config import Settings
from fplbot.models import NewsItem
from fplbot.news import FEEDS, SOURCE_TRUST, collect, filter_items, normalise

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def item(**kw):
    base = dict(source="bbc", url="u", published_at=NOW - timedelta(hours=1),
                title="t", body="Saka looked sharp", score=10)
    return NewsItem(**{**base, **kw})


def test_every_feed_has_a_trust_tier():
    for name, _url in FEEDS:
        assert name in SOURCE_TRUST, f"{name} has no trust tier"
    assert SOURCE_TRUST["bbc"] > SOURCE_TRUST["mirror"]


def test_excluded_feeds_are_not_present():
    """ESPN returns an empty document and Football365 404s (verified 2026-09-07)."""
    urls = " ".join(u for _n, u in FEEDS)
    assert "espn" not in urls
    assert "football365" not in urls


def test_drops_low_scoring_items():
    kept = filter_items([item(score=1), item(score=5)], Settings(cache_dir="/tmp/x"), NOW)
    assert len(kept) == 1


def test_drops_stale_items():
    old = item(published_at=NOW - timedelta(hours=72))
    assert filter_items([old], Settings(cache_dir="/tmp/x"), NOW) == []


def test_deduplicates_on_normalised_text():
    """The same claim is syndicated across outlets and would otherwise count
    as several independent pieces of evidence."""
    a = item(body="Saka is a doubt for Saturday!!")
    b = item(source="mirror", body="saka is a doubt for saturday")
    assert len(filter_items([a, b], Settings(cache_dir="/tmp/x"), NOW)) == 1


def test_respects_the_item_cap():
    settings = Settings(cache_dir="/tmp/x", comment_cap=5)
    many = [item(body=f"unique text {i}") for i in range(50)]
    assert len(filter_items(many, settings, NOW)) == 5


def test_keeps_newest_first_when_capping():
    settings = Settings(cache_dir="/tmp/x", comment_cap=1)
    older = item(body="older", published_at=NOW - timedelta(hours=10))
    newer = item(body="newer", published_at=NOW - timedelta(hours=1))
    assert filter_items([older, newer], settings, NOW)[0].body == "newer"


def test_normalise_strips_case_and_punctuation():
    assert normalise("Saka  is  OUT!!") == normalise("saka is out")


def test_news_item_has_no_author_field():
    """Reddit policy forbids inferring characteristics about users. The
    pipeline must not carry an author at all."""
    assert not hasattr(item(), "author")


def test_collect_without_reddit_reports_it_absent_not_failed(monkeypatch):
    import fplbot.news as news
    monkeypatch.setattr(news.RssCollector, "collect", lambda self: ([item()], set()))
    items, stats = collect(Settings(cache_dir="/tmp/x"), now=NOW)
    assert stats["reddit_available"] is False
    assert "reddit" in stats["sources_absent"]
    assert stats["reddit_status"] == "not_configured"
    assert items, "news-only run must still produce items"


def test_a_broken_reddit_is_distinguishable_from_an_absent_one(monkeypatch):
    """Both end up in sources_absent, but "never set up" and "set up and
    broken" need different responses from whoever reads the run."""
    import fplbot.news as news

    class Boom:
        def subreddit(self, name):
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(news.RssCollector, "collect", lambda self: ([item()], set()))
    items, stats = collect(Settings(cache_dir="/tmp/x"), reddit=Boom(), now=NOW)
    assert stats["reddit_available"] is False
    assert stats["reddit_status"].startswith("failed:")
    assert items, "a broken Reddit must not lose the news items"


def test_distinct_stories_sharing_an_opening_are_not_merged():
    """Syndicated copy shares lead sentences. Keying dedupe on a prefix would
    merge two different stories and throw away real evidence.

    The shared lead must exceed the old 200-character prefix, or this test
    passes against the buggy code too and guards nothing. The assertion below
    enforces that, so editing the text cannot silently disarm the test.
    """
    lead = (
        "The Premier League returns this weekend after the international break, "
        "with several managers facing selection headaches ahead of a congested "
        "run of fixtures that will test squad depth right across the division "
        "over the coming weeks. Here is the latest team news. "
    )
    assert len(normalise(lead)) > 200, "lead too short to exercise the old bug"

    a = item(body=lead + "Saka is expected to start against Chelsea.")
    b = item(source="sky", body=lead + "Haaland has been ruled out with a knock.")

    # Precondition: under the old truncating key these two collided.
    assert normalise(a.body)[:200] == normalise(b.body)[:200]

    kept = filter_items([a, b], Settings(cache_dir="/tmp/x"), NOW)
    assert len(kept) == 2, "two different stories were merged as duplicates"


class DeadFeed:
    """What feedparser actually returns for an unreachable or empty feed: it
    does not raise, it sets bozo and hands back zero entries."""

    bozo = 1
    entries: list = []
    bozo_exception = Exception("document declared as us-ascii, but parsed as utf-8")


def test_a_total_feed_outage_is_reported_as_an_absent_source(monkeypatch):
    """feedparser.parse never raises, so the try/except around it could not
    fire and a complete news blackout was reported as full coverage."""
    import fplbot.news as news

    monkeypatch.setattr(news.feedparser, "parse", lambda url: DeadFeed())
    items, stats = collect(Settings(cache_dir="/tmp/x"), now=NOW)

    assert items == []
    assert "news-rss" in stats["sources_absent"]
    assert "news-rss" not in stats["sources_used"]
    assert sorted(stats["feeds_failed"]) == sorted(n for n, _u in FEEDS)


def test_a_partial_feed_outage_keeps_the_source_and_names_the_failures(monkeypatch):
    """One dead feed is not a blackout. The run keeps its coverage and says
    which feeds went missing."""
    import time
    import fplbot.news as news

    class Live:
        bozo = 0
        entries = [{"published_parsed": time.gmtime(NOW.timestamp() - 3600),
                    "link": "u", "title": "t", "summary": "Saka trained fully"}]

    monkeypatch.setattr(
        news.feedparser, "parse",
        lambda url: DeadFeed() if "bbci" in url else Live())
    items, stats = collect(Settings(cache_dir="/tmp/x"), now=NOW)

    assert items, "a live feed must still deliver items"
    assert "news-rss" in stats["sources_used"]
    assert "news-rss" not in stats["sources_absent"]
    assert stats["feeds_failed"] == ["bbc"]
