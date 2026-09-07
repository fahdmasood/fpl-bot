"""Collect football text from news feeds, and optionally Reddit.

News is the primary source: the signal worth having is early availability
news, which reaches the press before it reaches the statistics. Reddit is
optional and requires approved Data API access.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import feedparser

from fplbot.config import Settings, user_agent
from fplbot.models import NewsItem

log = logging.getLogger(__name__)

SUBREDDIT = "FantasyPL"

# Verified returning items on 2026-09-07. ESPN's soccer feed returns an empty
# document and Football365's 404s; both are excluded deliberately.
FEEDS = [
    ("bbc", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
    ("guardian", "https://www.theguardian.com/football/rss"),
    ("sky", "https://www.skysports.com/rss/12040"),
    ("talksport", "https://talksport.com/football/feed/"),
    ("metro", "https://metro.co.uk/sport/football/feed/"),
    ("mirror", "https://www.mirror.co.uk/sport/football/?service=rss"),
]

# Multiplies a mention's confidence during aggregation (Task 7). The tabloids
# break real team news often enough to read and speculate often enough to
# discount.
SOURCE_TRUST = {
    "bbc": 1.0, "guardian": 1.0, "sky": 1.0,
    "talksport": 0.6, "metro": 0.6, "mirror": 0.6,
    "reddit-post": 0.7, "reddit-comment": 0.6,
}

# These threads carry the most comments and the least decision-relevant text.
EXCLUDED_TITLE_PATTERNS = re.compile(
    r"(match thread|live thread|bonus point|rate my team|who to captain\?)", re.I
)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower()).strip())


def filter_items(items: list[NewsItem], settings: Settings, now: datetime) -> list[NewsItem]:
    cutoff = now - timedelta(hours=settings.max_age_hours)
    seen: set[str] = set()
    kept: list[NewsItem] = []

    for it in sorted(items, key=lambda i: i.published_at, reverse=True):
        if it.score < settings.score_floor:
            continue
        if it.published_at < cutoff:
            continue
        # Key on the whole normalised body, not a prefix. Truncating to a
        # fixed prefix merges distinct stories that share a syndicated opening
        # sentence, and silently discarding real evidence is worse than
        # counting one claim twice.
        key = normalise(it.body)
        if key in seen:
            continue
        seen.add(key)
        kept.append(it)
        if len(kept) >= settings.comment_cap:
            break

    return kept


class RssCollector:
    def __init__(self, feeds=FEEDS, default_score: int = 5):
        self.feeds = feeds
        self.default_score = default_score

    def collect(self) -> tuple[list[NewsItem], set[str]]:
        """Return the items collected and the names of the feeds that failed.

        feedparser.parse does not raise on a dead feed. It sets `bozo` and
        hands back zero entries, so wrapping it in a try/except detects
        nothing: a total blackout used to be indistinguishable from a quiet
        news day. A feed that yields no entries is treated as failed, which
        is the only signal available and the honest reading of it.
        """
        items: list[NewsItem] = []
        failed: set[str] = set()
        for name, url in self.feeds:
            try:
                parsed = feedparser.parse(url)
            except Exception as exc:
                log.warning("feed %s failed: %s", name, exc)
                failed.add(name)
                continue
            entries = list(getattr(parsed, "entries", []) or [])
            if not entries:
                log.warning("feed %s returned no entries (bozo=%s)",
                            name, getattr(parsed, "bozo", 0))
                failed.add(name)
                continue
            if getattr(parsed, "bozo", 0):
                # Malformed but readable. Worth a line in the log, not a
                # reason to throw away entries we did parse.
                log.info("feed %s parsed with warnings", name)
            for entry in entries:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if not published:
                    continue
                items.append(
                    NewsItem(
                        source=name,
                        url=entry.get("link", ""),
                        published_at=datetime(*published[:6], tzinfo=timezone.utc),
                        title=entry.get("title", ""),
                        body=entry.get("summary", entry.get("title", "")),
                        # RSS carries no score; pass the floor so these are kept.
                        score=self.default_score,
                    )
                )
        return items, failed


class RedditCollector:
    """Optional. Requires approved Reddit Data API access."""

    def __init__(self, settings: Settings, reddit=None):
        self.settings = settings
        self.reddit = reddit or self._connect()

    def _connect(self):
        import praw

        return praw.Reddit(
            client_id=self.settings.reddit_client_id,
            client_secret=self.settings.reddit_client_secret,
            user_agent=user_agent(self.settings.reddit_username or "fplbot"),
        )

    def collect(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        subreddit = self.reddit.subreddit(SUBREDDIT)

        threads = [
            s for s in subreddit.hot(limit=25)
            if not EXCLUDED_TITLE_PATTERNS.search(s.title)
        ][: self.settings.max_threads]

        for submission in threads:
            items.append(
                NewsItem(
                    source="reddit-post",
                    url=f"https://reddit.com{submission.permalink}",
                    published_at=datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc),
                    title=submission.title,
                    body=submission.selftext or submission.title,
                    score=submission.score,
                )
            )
            submission.comment_sort = "new"
            # /comments/{id} has no cursor; depth and replace_more are the
            # only levers on volume.
            submission.comments.replace_more(limit=2)
            for comment in submission.comments.list():
                # No author is recorded, deliberately: Reddit policy forbids
                # inferring characteristics about users.
                items.append(
                    NewsItem(
                        source="reddit-comment",
                        url=f"https://reddit.com{comment.permalink}",
                        published_at=datetime.fromtimestamp(
                            comment.created_utc, tz=timezone.utc),
                        title=submission.title,
                        body=comment.body,
                        score=comment.score,
                    )
                )
                if len(items) >= self.settings.comment_ceiling:
                    return items
        return items


def collect(settings: Settings, reddit=None, now: datetime | None = None):
    now = now or datetime.now(tz=timezone.utc)
    raw: list[NewsItem] = []
    sources_used: list[str] = []
    sources_absent: list[str] = []

    collector = RssCollector()
    feeds_failed: list[str] = []
    try:
        news_items, failed = collector.collect()
        feeds_failed = sorted(failed)
        # One dead feed is thinner coverage, not a blackout. Only a total
        # outage makes the source absent; anything short of that is reported
        # through feeds_failed so the run says which ones went missing.
        if failed and len(failed) == len(collector.feeds):
            sources_absent.append("news-rss")
        else:
            raw += news_items
            sources_used.append("news-rss")
    except Exception as exc:
        log.warning("news feeds failed: %s", exc)
        feeds_failed = [name for name, _url in collector.feeds]
        sources_absent.append("news-rss")

    have_credentials = bool(settings.reddit_client_id and settings.reddit_client_secret)
    reddit_available = False
    # "Not configured" and "configured but broken" are different situations:
    # the first is the expected default, the second is worth alerting on.
    # Collapsing both into one absent-source string hides that from anyone
    # reading the run's stats afterwards.
    if have_credentials or reddit is not None:
        try:
            raw += RedditCollector(settings, reddit).collect()
            sources_used.append("reddit")
            reddit_available = True
            reddit_status = "ok"
        except Exception as exc:
            log.warning("Reddit collection failed: %s", exc)
            sources_absent.append("reddit")
            reddit_status = f"failed: {type(exc).__name__}"
    else:
        # Expected default: Reddit requires approved Data API access.
        log.info("no Reddit credentials; running on news feeds alone")
        sources_absent.append("reddit")
        reddit_status = "not_configured"

    kept = filter_items(raw, settings, now)
    stats = {
        "collection_attempted": True,
        "items_fetched": len(raw),
        "items_after_filter": len(kept),
        "sources_used": sources_used,
        "sources_absent": sources_absent,
        "feeds_failed": feeds_failed,
        "reddit_available": reddit_available,
        "reddit_status": reddit_status,
    }
    return kept, stats
