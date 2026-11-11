"""Is there a newer ChromaKit release on GitHub? Never raises: any failure means "don't know"."""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from importlib import metadata

API_URL = "https://api.github.com/repos/calebcoatney/chromakit-ms/releases/latest"


def parse_version(text: str) -> tuple[int, ...] | None:
    """'v0.2.10' -> (0, 2, 10); pre-releases and anything else -> None."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", (text or "").strip())
    if not m:
        return None
    parts = [int(p) for p in m.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def is_newer(latest: str, current: str) -> bool:
    a, b = parse_version(latest), parse_version(current)
    return bool(a and b and a > b)


def latest_release(timeout: float = 5.0, opener=None) -> tuple[str, str] | None:
    """(version, release page URL) of the latest published, non-pre-release release, else None."""
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(API_URL, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "ChromaKit-update-check"})
    try:
        with opener(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
        return None
    tag, url = data.get("tag_name"), data.get("html_url")
    if not (isinstance(tag, str) and isinstance(url, str) and parse_version(tag)):
        return None
    return tag.strip().lstrip("v"), url


def _read_direct_url() -> str | None:
    """PEP 610 direct_url.json text of the installed chromakit-ms distribution, if any."""
    return metadata.distribution("chromakit-ms").read_text("direct_url.json")


def detect_install(frozen: bool | None = None, read_direct_url=None) -> tuple[str, str | None]:
    """How ChromaKit got here: ("frozen" | "editable" | "git" | "unknown", editable source dir or None).

    Never raises: unreadable or odd metadata is "unknown". `frozen` and `read_direct_url` are
    injectable for tests.
    """
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        return "frozen", None
    try:
        text = (read_direct_url or _read_direct_url)()
        data = json.loads(text) if text else None
        if not isinstance(data, dict):
            return "unknown", None
        dir_info, vcs_info = data.get("dir_info"), data.get("vcs_info")
        if isinstance(dir_info, dict) and dir_info.get("editable") is True:
            url = data.get("url")
            if isinstance(url, str) and url.startswith("file:"):
                # shortcut: a UNC editable URL (file://server/share/...) loses its host; handle netloc if anyone installs from a share
                # url2pathname, not slicing: turns file:///C:/x into C:\x on Windows
                return "editable", urllib.request.url2pathname(urllib.parse.urlparse(url).path)
            return "unknown", None
        if isinstance(vcs_info, dict) and vcs_info.get("vcs") == "git":
            return "git", None
    except Exception:
        pass
    return "unknown", None


def pip_upgrade_command(executable: str) -> str:
    return (f'"{executable}" -m pip install --upgrade '
            '"chromakit-ms[ms] @ git+https://github.com/calebcoatney/chromakit-ms"')


def git_pull_command(source_dir: str) -> str:
    return f'git -C "{source_dir}" pull'
