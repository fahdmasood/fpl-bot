import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def bootstrap():
    return json.loads((FIXTURES / "bootstrap-static.json").read_text())


@pytest.fixture
def raw_fixtures():
    return json.loads((FIXTURES / "fixtures.json").read_text())


@pytest.fixture
def client(tmp_path, bootstrap, raw_fixtures):
    from fplbot.config import Settings
    from fplbot.fetch import FplClient

    class FakeSession:
        """Serves the snapshots; fails loudly if a test reaches the network."""

        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(url)
            if "bootstrap-static" in url:
                payload = bootstrap
            elif "fixtures" in url:
                payload = raw_fixtures
            else:
                raise AssertionError(f"unexpected network call: {url}")

            class R:
                status_code = 200

                @staticmethod
                def json():
                    return payload

                @staticmethod
                def raise_for_status():
                    return None

            return R()

    return FplClient(Settings(cache_dir=tmp_path), session=FakeSession())
