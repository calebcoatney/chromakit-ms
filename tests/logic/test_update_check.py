import io
import json
import ssl

import pytest

from logic import update_check as uc


def _opener(payload=None, exc=None, raw=None):
    def opener(req, timeout):
        assert req.full_url == uc.API_URL and timeout == 5.0
        if exc:
            raise exc
        return io.BytesIO(raw if raw is not None else json.dumps(payload).encode())
    return opener


@pytest.mark.parametrize("latest,current,newer", [
    ("0.2.10", "0.2.9", True), ("v0.3.0", "0.2.9", True), ("0.2.0", "0.2.0", False),
    ("0.2", "0.2.0", False), ("0.1.9", "0.2.0", False), ("0.3.0-rc1", "0.2.0", False),
    ("garbage", "0.2.0", False), ("0.3.0", "", False),
])
def test_is_newer(latest, current, newer):
    assert uc.is_newer(latest, current) is newer


def test_latest_release_reads_tag_and_page():
    got = uc.latest_release(opener=_opener({"tag_name": "v0.2.1", "html_url": "https://x/r/0.2.1",
                                            "draft": False, "prerelease": False}))
    assert got == ("0.2.1", "https://x/r/0.2.1")


@pytest.mark.parametrize("payload", [
    {"message": "API rate limit exceeded"},
    {"tag_name": "v0.3.0", "html_url": "https://x", "prerelease": True},
    {"tag_name": "v0.3.0-rc1", "html_url": "https://x"},
    {"tag_name": "v0.3.0"},
    [],
])
def test_unusable_answers_give_none(payload):
    assert uc.latest_release(opener=_opener(payload)) is None


@pytest.mark.parametrize("exc", [OSError("offline"), ssl.SSLError("NLR TLS"), TimeoutError()])
def test_network_failures_give_none(exc):
    assert uc.latest_release(opener=_opener(exc=exc)) is None


def test_malformed_json_gives_none():
    assert uc.latest_release(opener=_opener(raw=b"<html>")) is None
