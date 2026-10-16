"""Is there a newer ChromaKit release on GitHub? Never raises: any failure means "don't know"."""
from __future__ import annotations

import json
import re
import urllib.request

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
