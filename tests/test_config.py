import pytest
from fplbot.config import Settings, SCORING, user_agent


def test_defaults_match_spec():
    s = Settings(cache_dir="/tmp/x")
    assert s.horizon == 3
    assert s.decay_base == 0.84
    assert s.sentiment_cap == 0.15
    assert s.comment_cap == 600
    assert s.score_floor == 3


def test_user_agent_matches_reddit_mandated_format():
    assert user_agent("someone") == "python:fpl-bot:v1.0.0 (by /u/someone)"


def test_scoring_table_covers_all_four_positions():
    for pos in ("GKP", "DEF", "MID", "FWD"):
        assert pos in SCORING["goal"]
        assert pos in SCORING["clean_sheet"]


def test_from_env_reads_reddit_credentials(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "abc")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "def")
    monkeypatch.setenv("REDDIT_USERNAME", "ghi")
    s = Settings.from_env()
    assert s.reddit_client_id == "abc"
    assert s.reddit_client_secret == "def"


def test_from_env_tolerates_missing_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    s = Settings.from_env()
    assert s.reddit_client_id is None
