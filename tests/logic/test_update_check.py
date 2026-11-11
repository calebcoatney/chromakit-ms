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


# ---- install type and update instructions -------------------------------------------------------
import json as _json
import sys as _sys

from logic import update_check as uc


def _reader(payload):
    text = payload if isinstance(payload, str) or payload is None else _json.dumps(payload)
    return lambda: text


def test_frozen_build_wins_over_any_distribution():
    kind, path = uc.detect_install(frozen=True, read_direct_url=_reader({"dir_info": {"editable": True}, "url": "file:///x"}))
    assert (kind, path) == ("frozen", None)


def test_git_install():
    payload = {"url": "https://github.com/calebcoatney/chromakit-ms", "vcs_info": {"vcs": "git", "commit_id": "abc"}}
    assert uc.detect_install(frozen=False, read_direct_url=_reader(payload)) == ("git", None)


def test_editable_install_keeps_posix_source_path():
    payload = {"url": "file:///home/me/my%20src/chromakit-qt", "dir_info": {"editable": True}}
    kind, path = uc.detect_install(frozen=False, read_direct_url=_reader(payload))
    assert kind == "editable" and path.endswith("my src/chromakit-qt")


def test_editable_install_windows_drive_path(monkeypatch):
    import urllib.request
    import ntpath
    monkeypatch.setattr(urllib.request, "url2pathname", lambda p: ntpath_unquote(p))
    payload = {"url": "file:///C:/Users/me/chromakit-qt", "dir_info": {"editable": True}}
    kind, path = uc.detect_install(frozen=False, read_direct_url=_reader(payload))
    assert kind == "editable" and path == "C:\\Users\\me\\chromakit-qt"


def ntpath_unquote(p):
    # what url2pathname does on Windows, for the one shape we need
    import urllib.parse
    p = urllib.parse.unquote(p)
    if p.startswith("/") and len(p) > 2 and p[2] == ":":
        p = p[1:]
    return p.replace("/", "\\")


@pytest.mark.parametrize("payload", [
    None,                                        # no distribution / no direct_url.json
    "not json",
    "[]",
    {"url": "https://example.com/x.zip", "archive_info": {}},
    {"dir_info": {"editable": False}, "url": "file:///x"},
    {"dir_info": {"editable": True}},            # editable with no url: nothing to show
    {"vcs_info": {"vcs": "hg"}},
])
def test_everything_else_is_unknown(payload):
    assert uc.detect_install(frozen=False, read_direct_url=_reader(payload)) == ("unknown", None)


def test_reader_that_raises_is_unknown():
    def boom():
        raise RuntimeError("broken metadata")
    assert uc.detect_install(frozen=False, read_direct_url=boom) == ("unknown", None)


def test_default_detection_never_raises():
    kind, _ = uc.detect_install()
    assert kind in {"frozen", "editable", "git", "unknown"}


def test_pip_command_text_is_exact():
    assert uc.pip_upgrade_command("/opt/py/bin/python") == (
        '"/opt/py/bin/python" -m pip install --upgrade '
        '"chromakit-ms[ms] @ git+https://github.com/calebcoatney/chromakit-ms"')
    assert "@" in uc.pip_upgrade_command("p") and "\uff20" not in uc.pip_upgrade_command("p")


def test_ms_is_a_real_extra():
    import pathlib, re
    setup_py = (pathlib.Path(__file__).resolve().parents[2] / "setup.py").read_text(encoding="utf-8")
    assert re.search(r'"ms"\s*:', setup_py)


def test_git_pull_command():
    assert uc.git_pull_command("/src/chromakit-qt") == 'git -C "/src/chromakit-qt" pull'
