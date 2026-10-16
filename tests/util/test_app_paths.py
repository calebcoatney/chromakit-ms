import os
import sys
from pathlib import Path

import util.app_paths as ap

REPO = Path(ap.__file__).resolve().parent.parent


def test_source_install_keeps_folders_beside_the_code(monkeypatch, tmp_path):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert Path(ap.writable_dir("cache")) == REPO / "cache"


def test_frozen_build_writes_under_user_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(ap, "user_data_dir", lambda: tmp_path / "ChromaKit")
    d = Path(ap.writable_dir("overrides"))
    assert d == tmp_path / "ChromaKit" / "overrides" and d.is_dir()


def test_user_data_dir_on_windows_uses_localappdata(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert ap.user_data_dir() == tmp_path / "ChromaKit"


def test_configure_frozen_runtime_pins_matplotlib_and_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "user_data_dir", lambda: tmp_path / "Chroma Kit")    # a space, like Kerry's
    monkeypatch.setenv("MPLCONFIGDIR", "placeholder")   # records the original so teardown restores it
    monkeypatch.delenv("MPLCONFIGDIR")
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    log = ap.configure_frozen_runtime()
    try:
        print("hello from a windowed exe")
        sys.stdout.flush()
        assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "Chroma Kit" / "mpl")
        assert log == tmp_path / "Chroma Kit" / "logs" / "chromakit.log"
        assert "hello from a windowed exe" in log.read_text(encoding="utf-8")
    finally:
        sys.stdout.close()


def test_configure_frozen_runtime_survives_an_unwritable_data_dir(monkeypatch, tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    monkeypatch.setattr(ap, "user_data_dir", lambda: blocker / "ChromaKit")   # mkdir under a file fails
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    out, err = sys.stdout, sys.stderr
    assert ap.configure_frozen_runtime() is None
    assert sys.stdout is out and sys.stderr is err
    assert "MPLCONFIGDIR" not in os.environ
