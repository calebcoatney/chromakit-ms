from pathlib import Path

import chromakit_version

ROOT = Path(__file__).resolve().parent.parent


def test_spec_names_the_entry_point_icon_and_excludes():
    spec = (ROOT / "packaging/windows/chromakit.spec").read_text(encoding="utf-8")
    for needle in ("main.py", "resources/file.ico", "'torch'", "ms_toolkit", "name='ChromaKit'"):
        assert needle in spec


def test_workflow_checks_the_tag_and_runs_the_self_test():
    wf = (ROOT / ".github/workflows/windows-build.yml").read_text(encoding="utf-8")
    for needle in ("windows-latest", "--self-test", "chromakit_version", "update_chromakit.bat",
                   "--unattended", "ChromaKit-", "RELEASE_NOTES.md"):
        assert needle in wf


def test_bat_files_use_crlf_and_quote_paths():
    for name in ("update_chromakit.bat", "ChromaKit.bat"):
        raw = (ROOT / "tools/windows" / name).read_bytes()
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
        assert b'"%PY%"' in raw


def test_version_is_valid_for_a_tag():
    assert chromakit_version.__version__.count(".") == 2
