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
    monkeypatch.setattr(news.RssCollector, "collect", lambda self: [item()])
    items, stats = collect(Settings(cache_dir="/tmp/x"), now=NOW)
    assert stats["reddit_available"] is False
    assert "reddit" in stats["sources_absent"]
    assert items, "news-only run must still produce items"
