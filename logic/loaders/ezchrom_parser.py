"""Reader and .C converter for OpenLab EZChrom ``.dat`` raw files (Agilent 490 Micro GC).

An EZChrom ``.dat`` is an OLE compound document.  The streams this module reads:

``Detector Data/Detector <N> Trace``  (one per acquired channel)
    20-byte header, five little-endian int32: ``(5, n, n, 1, 0)``;
    ``n`` float64 LE samples (whole-number raw counts);
    40-byte trailer: f64 ``1.0``, i32 ``0``, i32 ``2``, f32 sample interval
    in seconds, f32 counts-to-``mV`` scale, 8 zero bytes, f64 ``1.0``.
    The two f32 fields are read as the decimals they encode (0.01 s, 1e-05).
    Sample ``i`` sits at ``t = i * interval`` from injection.
    Only the leading ``5`` (taken to be the sample type code for float64) has
    been seen; any other header is refused rather than guessed at.  (yadg's
    EZChrom reader expects int32 samples, so other EZChrom variants exist.)

``Detector Trace Handler``
    per trace: the trace number ``N`` as an MFC CString, the channel
    description, f32 sample interval, the y unit, f32 y scale.

``Chrom Header``
    f64 OLE-automation date at byte 16, in UTC; then CStrings: method path,
    sample name, software version, operator, ..., instrument name, ...,
    sequence file.

``Contents``
    the original result path on the acquisition PC.

Validated on 249 injections (498 traces) from a 490 Micro GC running OpenLab
EZChrom A.04.07: every vendor peak's LevelStart/LevelEnd equals the decoded
sample at BaselineStart/BaselineEnd, and the vendor's areas (counts x s) and
heights are reproduced from the decoded trace to within integer rounding.

Conversion writes one ``.C`` folder per injection.  Both channels stay in the
one folder and are selected by detector name (``Ch1_PPQ``, ``Ch2_5CB``),
mirroring how a multi-detector Agilent ``.D`` is one sample with several
``.ch`` channels.  The source ``.dat`` is *copied* (never moved) into
``data/`` and its sha256 is checked after the copy.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

SOURCE_FORMAT = "ezchrom_dat"
SIGNAL_TYPE = "gc"
DEFAULT_INSTRUMENT = "Agilent 490 Micro GC"
CONVERTER_VERSION = 1

TRACE_HEADER_SIZE = 20
TRACE_TRAILER_SIZE = 40
TRACE_TYPE_FLOAT64 = 5

_CSTRING_MARK = b"\xff\xfe\xff"
_OLE_EPOCH = _dt.datetime(1899, 12, 30, tzinfo=_dt.timezone.utc)
_TRACE_STREAM_RE = re.compile(r"^Detector (\d+) Trace$")
_ACAML_NS = "{urn:schemas-agilent-com:acaml15}"


class EZChromFormatError(ValueError):
    """The file is not an EZChrom .dat layout this reader has been validated on."""


# ── low-level decoding ────────────────────────────────────────────────────────

@dataclass
class TraceBlock:
    counts: np.ndarray        # raw detector counts, float64
    interval_s: float         # seconds between samples
    y_scale: float            # multiply counts by this to get y_unit
    header: tuple
    trailer: bytes


def decode_trace(raw: bytes) -> TraceBlock:
    """Decode one ``Detector Data/Detector <N> Trace`` stream."""
    if len(raw) < TRACE_HEADER_SIZE + TRACE_TRAILER_SIZE:
        raise EZChromFormatError(f"trace stream too short ({len(raw)} bytes)")
    header = struct.unpack_from("<5i", raw, 0)
    type_code, n, n_again = header[0], header[1], header[2]
    if type_code != TRACE_TYPE_FLOAT64:
        raise EZChromFormatError(
            f"unsupported trace sample type code {type_code} (only {TRACE_TYPE_FLOAT64} = float64 is validated)"
        )
    if n != n_again or n <= 0:
        raise EZChromFormatError(f"inconsistent trace point counts in header {header}")
    expected = TRACE_HEADER_SIZE + 8 * n + TRACE_TRAILER_SIZE
    if len(raw) != expected:
        raise EZChromFormatError(
            f"trace stream is {len(raw)} bytes; header says {n} points, which needs {expected}"
        )
    counts = np.frombuffer(raw, dtype="<f8", count=n, offset=TRACE_HEADER_SIZE).astype(np.float64)
    trailer = raw[TRACE_HEADER_SIZE + 8 * n:]
    interval_s, y_scale = (_f32_decimal(v) for v in struct.unpack_from("<2f", trailer, 16))
    if not (interval_s > 0 and np.isfinite(interval_s)):
        raise EZChromFormatError(f"implausible sample interval {interval_s!r} in trace trailer")
    if not (y_scale > 0 and np.isfinite(y_scale)):
        raise EZChromFormatError(f"implausible y scale {y_scale!r} in trace trailer")
    return TraceBlock(counts=counts, interval_s=float(interval_s), y_scale=float(y_scale),
                      header=header, trailer=trailer)


def _f32_decimal(value: float) -> float:
    """The decimal a float32 setting encodes: 0.01, not 0.009999999776482582."""
    return float(str(np.float32(value)))


def read_cstrings(raw: bytes) -> list[tuple[int, int, str]]:
    """Return ``(start, end, text)`` for each MFC Unicode CString in *raw*.

    A CString is ``FF FE FF`` then a length (one byte, or ``FF`` + uint16) in
    UTF-16 code units, then the UTF-16LE text.  ``end`` is the offset just past
    the text, where the next field starts.
    """
    out = []
    i = 0
    while True:
        j = raw.find(_CSTRING_MARK, i)
        if j < 0 or j + 4 > len(raw):
            break
        n = raw[j + 3]
        k = j + 4
        if n == 0xFF:
            if k + 2 > len(raw):
                break
            n = struct.unpack_from("<H", raw, k)[0]
            k += 2
        end = k + 2 * n
        if end > len(raw):
            break
        out.append((j, end, raw[k:end].decode("utf-16-le", errors="replace")))
        i = end
    return out


def ole_date_to_utc(days: float) -> _dt.datetime:
    return _OLE_EPOCH + _dt.timedelta(days=days)


def parse_detector_trace_handler(raw: bytes) -> dict[str, dict]:
    """Map trace number (``"0"``, ``"100"``) to its description, unit and scales."""
    strings = read_cstrings(raw)
    out: dict[str, dict] = {}
    for idx, (_, end, text) in enumerate(strings):
        if not text.isdigit() or text in out or idx + 2 >= len(strings):
            continue
        _, desc_end, desc = strings[idx + 1]
        _, unit_end, unit = strings[idx + 2]
        try:
            interval_s = _f32_decimal(struct.unpack_from("<f", raw, desc_end)[0])
            y_scale = _f32_decimal(struct.unpack_from("<f", raw, unit_end)[0])
        except struct.error:
            continue
        out[text] = {"description": desc, "y_unit": unit,
                     "interval_s": float(interval_s), "y_scale": float(y_scale)}
    return out


def channel_detector_id(description: str, trace_number: str) -> str:
    """Short, filename-safe detector name: ``Ch1_PPQ`` from "Channel 1, 10m PPQ ..."."""
    m = re.match(r"\s*Channel\s*(\d+)", description or "", re.IGNORECASE)
    if not m:
        return f"Det{trace_number}"
    col = channel_column(description)
    phase = col.split()[-1] if col else ""
    phase = re.sub(r"[^A-Za-z0-9]", "", phase)
    return f"Ch{m.group(1)}_{phase}" if phase else f"Ch{m.group(1)}"


def channel_column(description: str) -> str | None:
    """Column from a 490 channel description: "10m PPQ" from "Channel 1, 10m PPQ Heated ..."."""
    m = re.search(r"(\d+(?:\.\d+)?\s*m)\s+([A-Za-z0-9][A-Za-z0-9\-]*)", description or "")
    return f"{m.group(1).replace(' ', '')} {m.group(2)}" if m else None


# ── whole-file reading ────────────────────────────────────────────────────────

@dataclass
class EZChromChannel:
    detector: str             # short id, e.g. "Ch1_PPQ"
    trace_number: str         # "0", "100" (the N of "Detector N Trace")
    stream: str
    description: str
    column: str | None
    y_unit: str
    interval_s: float
    y_scale: float
    counts: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.counts.size)

    @property
    def sampling_rate_hz(self) -> float:
        return 1.0 / self.interval_s

    def time_min(self) -> np.ndarray:
        return np.arange(self.n_points, dtype=np.float64) * self.interval_s / 60.0

    def signal(self) -> np.ndarray:
        return self.counts * self.y_scale

    def info(self) -> dict:
        return {
            "detector": self.detector,
            "trace_number": self.trace_number,
            "stream": self.stream,
            "name": self.description,
            "column": self.column,
            "n_points": self.n_points,
            "sampling_rate_hz": round(self.sampling_rate_hz, 6),
            "sample_interval_s": self.interval_s,
            "run_time_min": round((self.n_points - 1) * self.interval_s / 60.0, 6),
            "y_units": self.y_unit,
            "y_scale_per_count": self.y_scale,
        }


@dataclass
class EZChromInjection:
    channels: list[EZChromChannel]
    injection_time_utc: _dt.datetime | None = None
    sample_name: str | None = None
    method_path: str | None = None
    instrument_name: str | None = None
    operator: str | None = None
    software_version: str | None = None
    sequence_file: str | None = None
    original_result_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def channel(self, detector: str | None = None) -> EZChromChannel:
        """Pick a channel by name; ``None`` means the first channel.

        Matches (case-insensitively) the short id, the trace number, the ACAML
        trace id (``"0_0"``) or the full description.  Never falls back to a
        different channel: an unknown name raises ``KeyError``.
        """
        if not self.channels:
            raise KeyError("no channels in this file")
        if detector is None:
            return self.channels[0]
        key = str(detector).strip().lower()
        for ch in self.channels:
            names = {ch.detector.lower(), ch.trace_number, ch.description.lower(),
                     f"detector {ch.trace_number} trace"}
            # ACAML TraceID is "<trace number>_<index>", e.g. "0_0", "100_1".
            if key in names or re.fullmatch(rf"{ch.trace_number}_\d+", key):
                return ch
        raise KeyError(
            f"detector {detector!r} not in this file; available: {[c.detector for c in self.channels]}"
        )


def read_ezchrom_streams(streams: Mapping[str, bytes], version: str | None = None) -> EZChromInjection:
    """Build an :class:`EZChromInjection` from a mapping of stream name → bytes.

    Stream names use ``/`` as separator (``"Detector Data/Detector 0 Trace"``).
    Kept separate from the OLE layer so it can be exercised with synthetic bytes.
    """
    warnings: list[str] = []
    dth = parse_detector_trace_handler(streams.get("Detector Trace Handler", b""))

    traces = []
    for name in streams:
        parts = name.split("/")
        if len(parts) == 2 and parts[0] == "Detector Data":
            m = _TRACE_STREAM_RE.match(parts[1])
            if m:
                traces.append((int(m.group(1)), m.group(1), name))
    if not traces:
        raise EZChromFormatError("no 'Detector Data/Detector N Trace' streams found")
    traces.sort()

    channels = []
    for _, number, stream in traces:
        block = decode_trace(streams[stream])
        meta = dth.get(number, {})
        if meta:
            for key, val in (("interval_s", block.interval_s), ("y_scale", block.y_scale)):
                if not np.isclose(meta[key], val, rtol=1e-6, atol=0):
                    raise EZChromFormatError(
                        f"trace {number}: {key} disagrees between trailer ({val}) and Detector Trace Handler ({meta[key]})"
                    )
        else:
            warnings.append(f"trace {number} has no Detector Trace Handler entry; unit assumed mV")
        description = meta.get("description", f"Detector {number}")
        channels.append(EZChromChannel(
            detector=channel_detector_id(description, number),
            trace_number=number,
            stream=stream,
            description=description,
            column=channel_column(description),
            y_unit=meta.get("y_unit") or "mV",
            interval_s=block.interval_s,
            y_scale=block.y_scale,
            counts=block.counts,
        ))
    ids = [c.detector for c in channels]
    if len(set(ids)) != len(ids):
        for c in channels:
            c.detector = f"{c.detector}_{c.trace_number}"

    inj = EZChromInjection(channels=channels, software_version=version, warnings=warnings)

    header = streams.get("Chrom Header")
    if header:
        if len(header) >= 24:
            days = struct.unpack_from("<d", header, 16)[0]
            if 32874 < days < 73051:   # 1990-01-01 .. 2100-01-01
                # Stored to the second; round away float noise (…:12.999999).
                t = ole_date_to_utc(days) + _dt.timedelta(microseconds=500_000)
                inj.injection_time_utc = t.replace(microsecond=0)
            else:
                warnings.append(f"Chrom Header date {days!r} out of range; injection time unknown")
        # CString order seen in A.04.07 files: method, sample name, "Version
        # A.04.07", operator, instrument name, blanks, sequence file.
        strings = [s for _, _, s in read_cstrings(header)]
        if strings:
            inj.method_path = strings[0] or None
        if len(strings) > 1:
            inj.sample_name = strings[1]
        for s in strings[2:]:
            if s.startswith("Version ") and inj.software_version is None:
                inj.software_version = s[len("Version "):]
            elif s.lower().endswith(".seq") and inj.sequence_file is None:
                inj.sequence_file = s
        if len(strings) > 3:
            inj.operator = strings[3] or None
        if len(strings) > 4:
            inj.instrument_name = strings[4] or None
    else:
        warnings.append("no Chrom Header stream; injection time and sample name unknown")

    contents = streams.get("Contents")
    if contents:
        cs = read_cstrings(contents)
        if cs:
            inj.original_result_path = cs[0][2]
    return inj


def read_ezchrom_dat(path: str) -> EZChromInjection:
    """Read an EZChrom ``.dat`` (OLE compound file).  Opens the file read-only."""
    import olefile

    if not olefile.isOleFile(path):
        raise EZChromFormatError(f"not an OLE compound file: {path}")
    wanted = {"Chrom Header", "Detector Trace Handler", "Contents"}
    streams: dict[str, bytes] = {}
    version = None
    with olefile.OleFileIO(path) as ole:
        for entry in ole.listdir(streams=True, storages=True):
            name = "/".join(entry)
            if len(entry) == 3 and entry[0] == "Signature" and entry[1] == "Product" \
                    and entry[2].startswith("Version "):
                version = entry[2][len("Version "):]
            if name in wanted or (len(entry) == 2 and entry[0] == "Detector Data"):
                if ole.get_type(entry) == olefile.STGTY_STREAM:
                    streams[name] = ole.openstream(entry).read()
    return read_ezchrom_streams(streams, version=version)


# ── ACAML (vendor results) cross-reference ───────────────────────────────────

def acaml_injection_summary(acam_path: str) -> dict:
    """Pull the injection-level facts from a per-injection ACAML result file."""
    root = ET.parse(acam_path).getroot()
    ns = _ACAML_NS
    content = root.find(f"{ns}Doc/{ns}Content")
    md = content.find(f"{ns}Injections/{ns}MeasData")
    signals = []
    for s in md.findall(f"{ns}Signal"):
        signals.append({"name": s.findtext(f"{ns}Name"), "trace_id": s.findtext(f"{ns}TraceID")})
    return {
        "data_file": md.findtext(f"{ns}BinaryData/{ns}DataItem/{ns}Name"),
        "created": md.findtext(f"{ns}Info/{ns}CreatedDate"),
        "sample_name": content.findtext(f"{ns}SampleParams/{ns}IdentParam/{ns}Name"),
        "instrument_name": content.findtext(f"{ns}Resources/{ns}Instrument/{ns}Name"),
        "signals": signals,
    }


def map_acaml_by_data_file(rslt_dir: str) -> dict[str, str]:
    """``{"long_001.dat": "<path>.rst.0.0001.acam_", ...}`` for a result-set folder."""
    out = {}
    for name in sorted(os.listdir(rslt_dir)):
        if not name.endswith(".acam_"):
            continue
        p = os.path.join(rslt_dir, name)
        try:
            data_file = acaml_injection_summary(p)["data_file"]
        except (ET.ParseError, AttributeError):
            continue
        if data_file:
            out[data_file] = p
    return out


def _cross_check_acaml(inj: EZChromInjection, summary: dict) -> list[str]:
    problems = []
    if summary.get("sample_name") is not None and inj.sample_name is not None \
            and summary["sample_name"] != inj.sample_name:
        problems.append(f"sample name: .dat {inj.sample_name!r} vs ACAML {summary['sample_name']!r}")
    if summary.get("created") and inj.injection_time_utc is not None:
        acaml_t = _dt.datetime.strptime(summary["created"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
        if abs((acaml_t - inj.injection_time_utc).total_seconds()) > 1:
            problems.append(f"injection time: .dat {inj.injection_time_utc.isoformat()} vs ACAML {summary['created']}")
    for sig in summary.get("signals", []):
        number = (sig.get("trace_id") or "").split("_")[0]
        match = [c for c in inj.channels if c.trace_number == number]
        if not match:
            problems.append(f"ACAML signal {sig} has no trace in the .dat")
        elif sig.get("name") and match[0].description != sig["name"]:
            problems.append(f"trace {number} name: .dat {match[0].description!r} vs ACAML {sig['name']!r}")
    return problems


# ── conversion to .C ──────────────────────────────────────────────────────────

_RSLT_RE = re.compile(r"\(GMT\s*([+-])(\d{2})-(\d{2})\)\s*(.*?)\.rslt$", re.IGNORECASE)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_rslt_name(rslt_name: str) -> tuple[str | None, _dt.timezone | None]:
    """``("8566-043", UTC-06:00)`` from "2025-05-27 17-27-27 (GMT -06-00)8566-043.rslt"."""
    m = _RSLT_RE.search(os.path.basename(rslt_name.rstrip("/\\")))
    if not m:
        return None, None
    sign = -1 if m.group(1) == "-" else 1
    offset = _dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3))) * sign
    return (m.group(4) or None), _dt.timezone(offset)


def convert_dat(dat_path: str, output_dir: str, *, acam_path: str | None = None,
                c_name: str | None = None, instrument: str = DEFAULT_INSTRUMENT,
                overwrite: bool = False) -> str:
    """Convert one EZChrom ``.dat`` into a ``.C`` folder under *output_dir*.

    The ``.dat`` is copied (not moved; the source is opened read-only) into
    ``<name>.C/data/``.  Returns the ``.C`` path.  An existing ``.C`` of the
    same name is left alone and its path returned, unless *overwrite*.
    """
    dat_path = os.path.abspath(dat_path)
    output_dir = os.path.abspath(output_dir)
    rslt_dir = os.path.dirname(dat_path)
    rslt_name = os.path.basename(rslt_dir)
    seq_id, tz = parse_rslt_name(rslt_name) if rslt_name.lower().endswith(".rslt") else (None, None)
    stem = os.path.splitext(os.path.basename(dat_path))[0]
    if c_name is None:
        c_name = f"{seq_id}_{stem}" if seq_id else stem
    c_path = os.path.join(output_dir, c_name + ".C")
    if os.path.commonpath([c_path, rslt_dir]) == rslt_dir:
        raise ValueError("output_dir must not be inside the source result-set folder")
    if os.path.exists(c_path):
        if not overwrite:
            return c_path
        shutil.rmtree(c_path)

    src_sha = _sha256(dat_path)
    inj = read_ezchrom_dat(dat_path)
    warnings = list(inj.warnings)

    acaml_summary = None
    if acam_path:
        acaml_summary = acaml_injection_summary(acam_path)
        warnings += _cross_check_acaml(inj, acaml_summary)

    t_utc = inj.injection_time_utc
    created = None
    if t_utc is not None:
        created = (t_utc.astimezone(tz) if tz is not None else t_utc).isoformat()
    first = inj.channels[0]
    manifest = {
        "signal_type": SIGNAL_TYPE,
        "source_format": SOURCE_FORMAT,
        "created": created or _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "sample_id": c_name,
        "sample_name": inj.sample_name,
        "instrument": instrument,
        "instrument_name": inj.instrument_name or (acaml_summary or {}).get("instrument_name"),
        "acquisition_software": f"OpenLab EZChrom {inj.software_version}" if inj.software_version else "OpenLab EZChrom",
        "injection_time_utc": t_utc.isoformat().replace("+00:00", "Z") if t_utc else None,
        "sequence_name": os.path.splitext(rslt_name)[0] if rslt_name.lower().endswith(".rslt") else None,
        "sequence_id": seq_id,
        "sequence_file": inj.sequence_file,
        "method": inj.method_path,
        "operator": inj.operator,
        "data_file": os.path.basename(dat_path),
        "x_units": "min",
        "y_units": first.y_unit,
        "sampling_rate_hz": round(first.sampling_rate_hz, 6),
        "default_detector": first.detector,
        "detectors": [c.detector for c in inj.channels],
        "channels": [c.info() for c in inj.channels],
        "source_path": dat_path,
        "source_sha256": src_sha,
        "source_size_bytes": os.path.getsize(dat_path),
        "original_result_path": inj.original_result_path,
        "acaml_path": os.path.abspath(acam_path) if acam_path else None,
        "acaml_sha256": _sha256(acam_path) if acam_path else None,
        "converter": "chromakit logic.loaders.ezchrom_parser",
        "converter_version": CONVERTER_VERSION,
        "converted_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "conversion_warnings": warnings,
    }

    os.makedirs(output_dir, exist_ok=True)
    tmp = c_path + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    try:
        os.makedirs(os.path.join(tmp, "data"))
        os.makedirs(os.path.join(tmp, "results"))
        dest = os.path.join(tmp, "data", os.path.basename(dat_path))
        shutil.copyfile(dat_path, dest)
        if _sha256(dest) != src_sha:
            raise IOError(f"copy of {dat_path} does not match its sha256")
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        os.rename(tmp, c_path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return c_path


def convert_rslt(rslt_dir: str, output_dir: str, *, overwrite: bool = False,
                 progress: Callable[[str], None] | None = None) -> list[str]:
    """Convert every ``.dat`` in an EZChrom result-set folder (``*.rslt``) to ``.C`` folders."""
    rslt_dir = os.path.abspath(rslt_dir)
    acams = map_acaml_by_data_file(rslt_dir)
    created = []
    for name in sorted(os.listdir(rslt_dir)):
        if not name.lower().endswith(".dat"):
            continue
        c = convert_dat(os.path.join(rslt_dir, name), output_dir,
                        acam_path=acams.get(name), overwrite=overwrite)
        created.append(c)
        if progress:
            progress(c)
    return created


def parse_ezchrom(path: str, output_dir: str, **kwargs) -> list[str]:
    """Convert a ``.dat`` file or a ``.rslt`` folder; returns the ``.C`` paths."""
    if os.path.isdir(path):
        return convert_rslt(path, output_dir, **kwargs)
    acam = None
    parent = os.path.dirname(os.path.abspath(path))
    if parent.lower().endswith(".rslt"):
        acam = map_acaml_by_data_file(parent).get(os.path.basename(path))
    return [convert_dat(path, output_dir, acam_path=acam, **kwargs)]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Convert OpenLab EZChrom .dat files / .rslt folders to ChromaKit .C folders.")
    ap.add_argument("sources", nargs="+", help=".dat files or .rslt result-set folders")
    ap.add_argument("-o", "--output-dir", required=True, help="where the .C folders are written")
    ap.add_argument("--overwrite", action="store_true", help="replace existing .C folders")
    args = ap.parse_args(argv)
    n = 0
    for src in args.sources:
        for c in parse_ezchrom(src, args.output_dir, overwrite=args.overwrite):
            print(c)
            n += 1
    print(f"{n} .C folder(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
