"""`ChromaKit.exe --self-test <dir>`: proves a frozen build has everything it needs.

Imports every first-party module and ms_toolkit, opens the main window, then writes a synthetic
SepSolve run, detects on the DBC TIC and exports, checking the MGF. Results go to
<dir>/self-test.txt because a windowed exe has no console.
"""
from __future__ import annotations

import importlib
import pkgutil
import os
import shutil
import traceback
from pathlib import Path

PACKAGES = ("logic", "ui", "util")
MIN_FIRST_PARTY_MODULES = 50      # a walk that finds fewer means the frozen build lost modules
SKIP = {"logic.deconvolution"}       # torch is not bundled; its import is guarded but slow to probe


def _import_everything(log):
    count = 0
    for pkg_name in PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
            if info.name in SKIP or ".tests" in info.name:
                continue
            importlib.import_module(info.name)
            count += 1
    if count < MIN_FIRST_PARTY_MODULES:
        raise RuntimeError(f"only {count} first-party modules found (expected at least "
                           f"{MIN_FIRST_PARTY_MODULES}); the build is missing modules")
    importlib.import_module("ms_toolkit.api")
    log.append(f"imports: {count} modules + ms_toolkit.api")


def _open_main_window(log):
    from PySide6.QtWidgets import QApplication
    from ui.app import ChromaKitApp
    app = QApplication.instance() or QApplication([])
    win = ChromaKitApp()
    win.show()
    app.processEvents()
    win.close()
    log.append("main window: ok")


def _check_export(out_dir: Path, log):
    from logic.gcxgc_export import export_gcxgc_results
    from logic.gcxgc_processor import GCxGCProcessor
    from logic.gcxgc_run import GCxGCRun
    from util import sepsolve_synth as synth
    shutil.rmtree(Path(out_dir) / "Synth.C", ignore_errors=True)
    scene = synth.Scene(peaks=[synth.Peak(t1_min=2.0, t2_s=1.0, height=10.0, base_mz=57),
                               synth.Peak(t1_min=4.0, t2_s=1.6, height=3.0, base_mz=105)],
                        duration_min=6.0)
    c = synth.build_run(out_dir, scene)
    run = GCxGCRun.from_c_folder(c)
    try:
        proc = GCxGCProcessor(run.frame("dbc_tic"), run)
        proc.apply_baseline(lam=1e5)
        peaks = proc.detect_peaks(min_prominence=50)
        paths = export_gcxgc_results(peaks, run, Path(c) / "results",
                                     settings={"detection_signal": "dbc_tic"},
                                     get_spectrum=proc.extract_spectrum, min_snr=5.0)
    finally:
        run.close()
    text = paths.mgf.read_text(encoding="utf-8")
    if not peaks or "MSLEVEL=2" not in text:
        raise RuntimeError(f"export check failed: {len(peaks)} peaks, MSLEVEL=2 in MGF: {'MSLEVEL=2' in text}")
    log.append(f"peaks: {len(peaks)}; mgf: {paths.mgf.name}")


def run_self_test(out_dir) -> int:
    os.environ.setdefault("CHROMAKIT_NO_UPDATE_CHECK", "1")
    os.environ.setdefault("CHROMAKIT_NO_MS_AUTOLOAD", "1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    try:
        from chromakit_version import __version__
        log.append(f"ChromaKit {__version__}")
        _import_everything(log)
        _open_main_window(log)
        _check_export(out_dir, log)
        status = 0
    except Exception:
        log.append(traceback.format_exc())
        status = 1
    (out_dir / "self-test.txt").write_text("\n".join(["OK" if status == 0 else "FAIL", *log]) + "\n",
                                           encoding="utf-8")
    return status
