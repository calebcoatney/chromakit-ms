"""THROWAWAY: proves Velopack can install and update the frozen Qt app. Never merge this branch.

ChromaKit.exe --spike-version FILE        write the installed version to FILE
ChromaKit.exe --spike-download FEED FILE  download the update in FEED, do not apply it
ChromaKit.exe --spike-update FEED FILE    download it and apply it once this process exits
Normal launch: looks for updates in %USERPROFILE%\\Downloads\\ck-update-test (or CHROMAKIT_UPDATE_SOURCE).
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path


def startup():
    """First thing in main(): Velopack's install/update hooks, with auto-apply at startup OFF."""
    import velopack
    velopack.App().set_auto_apply_on_startup(False).run()


def headless(argv) -> bool:
    if len(argv) < 3 or not argv[1].startswith("--spike-"):
        return False
    import velopack
    cmd, out = argv[1], Path(argv[-1])
    code = 0
    try:
        mgr = velopack.UpdateManager(argv[2] if len(argv) > 3 else str(out.parent))
        if cmd == "--spike-version":
            out.write_text(mgr.get_current_version(), encoding="utf-8")
        else:
            info = mgr.check_for_updates()
            if info is None:
                raise RuntimeError("no update found")
            mgr.download_updates(info)
            out.write_text(f"downloaded {info.TargetFullRelease.Version}; "
                           f"deltas {len(info.DeltasToTarget)}", encoding="utf-8")
            if cmd == "--spike-update":
                mgr.wait_exit_then_apply_updates(info, silent=True, restart=False)
    except Exception:
        out.write_text(traceback.format_exc(), encoding="utf-8")
        code = 1
    sys.exit(code)


def same_results(installed: str, offered: str) -> bool:
    """Patch releases only: same major.minor."""
    return installed.split(".")[:2] == offered.split(".")[:2]


def attach(app, window):
    """Status-bar label, background check, the question, install after close."""
    from PySide6.QtCore import QObject, QSettings, Signal
    from PySide6.QtWidgets import QLabel, QMessageBox
    import velopack

    feed = os.environ.get("CHROMAKIT_UPDATE_SOURCE") or str(Path.home() / "Downloads" / "ck-update-test")
    label = QLabel("TEST BUILD")
    window.statusBar().addPermanentWidget(label)
    try:
        mgr = velopack.UpdateManager(feed)
        current = mgr.get_current_version()
    except Exception as e:
        label.setText(f"TEST BUILD: not installed by Velopack ({e})")
        return
    label.setText(f"TEST BUILD {current}")
    settings = QSettings("CalebCoatney", "ChromaKitSpike")
    pending = {}

    class Bridge(QObject):
        ready = Signal(object)

    bridge = Bridge(window)

    def check():
        try:
            info = mgr.check_for_updates()
            if info is not None:
                mgr.download_updates(info)
                bridge.ready.emit(info)
        except Exception:
            traceback.print_exc()

    def on_ready(info):
        new = info.TargetFullRelease.Version
        if same_results(current, new):
            pending["info"] = info
            label.setText(f"TEST BUILD {current}: {new} installs when you close")
            return
        label.setText(f"TEST BUILD {current}: {new} available")
        if settings.value("skipped", "") == new:
            return
        box = QMessageBox(window)
        box.setText(f"ChromaKit {new} is ready.")
        box.setInformativeText("Results can change: (test text, nothing really changed).")
        yes = box.addButton("Install when I close", QMessageBox.AcceptRole)
        box.addButton("Not now", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is yes:
            pending["info"] = info
            label.setText(f"TEST BUILD {current}: {new} installs when you close")
        else:
            settings.setValue("skipped", new)

    def on_quit():
        if "info" in pending:
            mgr.wait_exit_then_apply_updates(pending["info"], silent=True, restart=False)

    bridge.ready.connect(on_ready)
    app.aboutToQuit.connect(on_quit)
    if Path(feed).exists():
        threading.Thread(target=check, daemon=True).start()
