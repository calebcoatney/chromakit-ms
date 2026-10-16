"""Write GCxGC results to <run>.C/results/: ChromSpace-layout CSV, full JSON record, MSP and MGF spectra, GNPS quant CSV."""
from __future__ import annotations

import csv
import datetime
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from logic.spectrum_noise import snr_gate

CHROMSPACE_COLUMNS = ["Peak #", "Source", "Compound", "Synonyms", "Library RI",
                      "1tR (min)", "2tR (min)", "Area", "Area %"]
#: ChromSpace labels FID-channel peaks "TIC [Front Signal]" in its Area-percent export.
SOURCE_LABELS = {"fid": "TIC [Front Signal]", "tic": "TIC", "dbc_tic": "DBC TIC"}
SCHEMA = "chromakit.gcxgc.v1"
VOLUME_UNITS = "detector counts x samples (sum of per-modulation trapezoid areas)"


@dataclass(frozen=True)
class GCxGCExportPaths:
    csv: Path
    json: Path
    msp: Path
    mgf: Path
    gnps_quant: Path


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return cleaned or "sample"


def area_percents(peaks) -> list[float]:
    total = sum(max(p.volume, 0.0) for p in peaks)
    return [100.0 * max(p.volume, 0.0) / total if total > 0 else 0.0 for p in peaks]


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _chromakit_version() -> str:
    try:
        from chromakit_version import __version__
        return __version__
    except Exception:
        return ""


def _chromakit_commit() -> str:
    """The commit this ChromaKit came from: the frozen build's _build_info, else the distribution's
    PEP 610 vcs_info, else ``git rev-parse HEAD`` in the package's own folder, else "". Never raises."""
    return _build_info_commit() or _direct_url_commit() or _git_commit()


def _build_info_commit() -> str:
    try:
        import _build_info
        commit = getattr(_build_info, "COMMIT", "")
        return commit if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{7,40}", commit) else ""
    except Exception:
        return ""


def _direct_url_commit() -> str:
    try:
        from importlib.metadata import distribution
        info = json.loads(distribution("chromakit-ms").read_text("direct_url.json") or "{}")
        commit = (info.get("vcs_info") or {}).get("commit_id") or ""
        return commit if isinstance(commit, str) else ""
    except Exception:
        return ""


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=3)
        commit = out.stdout.strip()
        return commit if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{7,40}", commit) else ""
    except Exception:
        return ""


def _intensity(x) -> str:
    """Fixed-point intensity text for MSP/MGF (``.6g`` switched to exponents above 1e6)."""
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def _write_csv(f, peaks, percents) -> None:
    w = csv.writer(f)
    w.writerow(CHROMSPACE_COLUMNS)
    for p, pct in zip(peaks, percents):
        w.writerow([p.peak_number, SOURCE_LABELS.get(p.detection_signal, p.detection_signal),
                    p.compound_name or "", "", "", f"{p.rt1:.6f}", f"{p.rt2 / 60.0:.8f}",
                    f"{p.volume:.6f}", f"{pct:.6f}"])


def _write_msp(f, peaks, spectra) -> None:
    for p in peaks:
        spec = spectra.get(p.peak_number)
        if not spec:
            continue
        title = f"Peak {p.peak_number}" + (f" - {p.compound_name}" if p.compound_name else "")
        f.write(f"Name: {title}\n")
        f.write(f"Comments: 1tR={p.rt1:.4f} min; 2tR={p.rt2:.3f} s; volume={p.volume:.6g}; "
                f"scan={p.spectrum_scan}; source={p.spectrum_source}\n")
        f.write(f"Num Peaks: {len(spec)}\n")
        for mz, inten in spec:
            f.write(f"{int(mz)} {_intensity(inten)}\n")
        f.write("\n")


def _base_peak_mz(spec) -> int:
    return int(max(spec, key=lambda t: t[1])[0])


def _write_mgf(f, peaks, spectra) -> None:
    """One entry per peak with a spectrum, in the field order of MZmine's GC (GNPS) export
    (AdapMgfExportTask): GC-GNPS needs MSLEVEL=2 even for EI. PEPMASS is the base peak, a stand-in
    (EI has no precursor). RTINSECONDS is the first dimension; 2tR stays in the CSV/JSON."""
    for p in peaks:
        spec = spectra.get(p.peak_number)
        if not spec:
            continue
        title = f"Peak {p.peak_number}" + (f" - {p.compound_name}" if p.compound_name else "")
        f.write("BEGIN IONS\n")
        f.write(f"TITLE={title}\n")
        f.write(f"FEATURE_ID={p.peak_number}\n")
        f.write(f"PEPMASS={_base_peak_mz(spec)}\n")
        f.write(f"RTINSECONDS={p.rt1 * 60.0:.2f}\n")
        f.write(f"SCANS={p.peak_number}\n")
        f.write("MSLEVEL=2\n")
        f.write("CHARGE=1+\n")
        f.write(f"Num peaks={len(spec)}\n")
        for mz, inten in spec:
            f.write(f"{int(mz)} {_intensity(inten)}\n")
        f.write("END IONS\n\n")


def _write_gnps_quant(f, peaks, spectra, sample_id) -> None:
    """GNPS feature table for one sample: the MGF's features, rt in minutes, detection-signal volume."""
    w = csv.writer(f)
    w.writerow(["row ID", "row m/z", "row retention time", f"{sample_id} Peak area"])
    for p in peaks:
        spec = spectra.get(p.peak_number)
        if spec:
            w.writerow([p.peak_number, _base_peak_mz(spec), f"{p.rt1:.6f}", f"{p.volume:.6f}"])


def _record(peaks, percents, run, settings, library, search_method, search_complete) -> dict:
    offset = None
    if settings.get("detection_signal", "fid") == "fid":       # TIC detection never maps FID->TOF
        off = run.offset()
        offset = {"a_scans": off.a, "b_scans_per_min": off.b, "ok": off.ok,
                  "n_windows": off.n_windows, "mode": off.mode}
    return {
        "schema": SCHEMA,
        "sample_id": run.sample_id,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "chromakit_version": _chromakit_version(),
        "chromakit_commit": _chromakit_commit(),
        "run": {
            "modulation_period_s": run.modulation_period_s,
            "phase_shift_s": run.phase_shift_s,
            "provenance": run.provenance,
            "fid_rate_hz": run.fid_hz,
            "tof_rate_hz": run.tof_hz,
            "offset": offset,
            "files": {k: os.path.basename(v) for k, v in run.paths.items() if v},
        },
        "settings": settings,
        "library": library,
        "search_method": search_method,
        "search_complete": search_complete,
        "volume_units": VOLUME_UNITS,
        "peaks": [dict(p.as_dict(), area_percent=pct) for p, pct in zip(peaks, percents)],
    }


def export_gcxgc_results(peaks, run, results_dir, *, settings: dict, get_spectrum,
                         library=None, search_method=None, search_complete=None,
                         min_snr: float = 0.0) -> GCxGCExportPaths:
    """Write the result files; get_spectrum(peak) -> [(mz, intensity)] | None.

    min_snr > 0 gates the MSP/MGF spectra (logic.spectrum_noise.snr_gate); the CSV and JSON are
    unaffected. search_complete: True after a full MS Search All, False when it was cancelled
    part-way (peaks it never reached have no ID), None when no search produced the IDs.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(run.sample_id or "sample")
    paths = GCxGCExportPaths(results_dir / f"{stem} - GCxGC peaks.csv",
                             results_dir / f"{stem} - GCxGC peaks.json",
                             results_dir / f"{stem} - GCxGC spectra.msp",
                             results_dir / f"{stem} - GCxGC spectra.mgf",
                             results_dir / f"{stem} - GNPS quant.csv")
    peaks = sorted(peaks, key=lambda p: p.peak_number)
    spectra = {}
    for p in peaks:
        s = snr_gate(get_spectrum(p), min_snr)
        if s:
            spectra[p.peak_number] = s
    percents = area_percents(peaks)
    _atomic_write(paths.csv, lambda f: _write_csv(f, peaks, percents))
    _atomic_write(paths.msp, lambda f: _write_msp(f, peaks, spectra))
    _atomic_write(paths.mgf, lambda f: _write_mgf(f, peaks, spectra))
    _atomic_write(paths.gnps_quant,
                  lambda f: _write_gnps_quant(f, peaks, spectra, run.sample_id or stem))
    record = _record(peaks, percents, run, settings, library, search_method, search_complete)
    _atomic_write(paths.json, lambda f: json.dump(record, f, indent=1, default=_json_default))
    return paths
