# -*- coding: utf-8 -*-
"""Pure (Qt-free) writer for one per-channel result block in an Excel sheet.

Shared by the JSON-to-Excel converter (``util/json_to_xlsx.py``) and the
GC1&2 paste-workbook export. Must not import ``ui/``, ``api/`` or ``util/``.
"""

import copy
import datetime
import re

from openpyxl.styles import Alignment, Font, PatternFill


DEFAULT_FORMAT_CONFIG = {
    'header_fields': [
        {'key': 'signal',     'label': 'Signal:',     'enabled': True,  'custom_label': '', 'show_label': False},
        {'key': 'sample_id',  'label': 'Sample ID:',  'enabled': True,  'custom_label': '', 'show_label': False},
        {'key': 'timestamp',  'label': 'Timestamp:',  'enabled': True,  'custom_label': '', 'show_label': False},
        {'key': 'method',     'label': 'Method:',     'enabled': False, 'custom_label': '', 'show_label': False},
        {'key': 'detector',   'label': 'Detector:',   'enabled': False, 'custom_label': '', 'show_label': False},
        {'key': 'notebook',   'label': 'Notebook:',   'enabled': False, 'custom_label': '', 'show_label': False},
    ],
    'timestamp_format': 'as_stored',
    'columns': [
        {'key': 'Compound ID',    'display': 'Compound ID',   'enabled': False, 'custom_name': ''},
        {'key': 'peak_number',    'display': 'Peak #',        'enabled': True,  'custom_name': ''},
        {'key': 'retention_time', 'display': 'Ret Time',      'enabled': True,  'custom_name': ''},
        {'key': 'integrator',     'display': 'Integrator',    'enabled': True,  'custom_name': ''},
        {'key': 'width',          'display': 'Width',         'enabled': True,  'custom_name': ''},
        {'key': 'area',           'display': 'Area',          'enabled': True,  'custom_name': ''},
        {'key': 'start_time',     'display': 'Start Time',    'enabled': True,  'custom_name': ''},
        {'key': 'end_time',       'display': 'End Time',      'enabled': True,  'custom_name': ''},
        {'key': 'Qual',           'display': 'Match Score',   'enabled': False, 'custom_name': ''},
        {'key': 'casno',          'display': 'CAS No',        'enabled': False, 'custom_name': ''},
        {'key': 'quality_issues', 'display': 'Quality Issues','enabled': False, 'custom_name': ''},
        {'key': 'mol_C',          'display': 'mol C',         'enabled': False, 'custom_name': ''},
        {'key': 'mol_C_percent',  'display': 'mol C %',       'enabled': False, 'custom_name': ''},
        {'key': 'num_carbons',    'display': '# Carbons',     'enabled': False, 'custom_name': ''},
        {'key': 'mol',            'display': 'mol',           'enabled': False, 'custom_name': ''},
        {'key': 'mass_mg',        'display': 'mass (mg)',     'enabled': False, 'custom_name': ''},
        {'key': 'mol_percent',    'display': 'mol %',         'enabled': False, 'custom_name': ''},
        {'key': 'wt_percent',     'display': 'wt %',          'enabled': False, 'custom_name': ''},
    ],
}

# Format for the GC1&2 "paste-ready" workbook: native Excel dates in the
# timestamp row, as the paste macro expects.
GC12_PASTE_FORMAT_CONFIG = copy.deepcopy(DEFAULT_FORMAT_CONFIG)
GC12_PASTE_FORMAT_CONFIG['timestamp_format'] = 'm/dd/yy h:mm'

# Timestamp parse patterns to try when reformatting
_TS_PARSE_FMTS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%d %b %y  %I:%M %p",
    "%d %b %y %I:%M %p",
    "%d %b %Y %I:%M %p",
    "%d-%b-%y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d %b %y  %H:%M",
    "%d %b %y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d",
]

# Pattern for auto-generated placeholder compound IDs written by the integrator
# when no MS assignment is available (e.g. "Unknown (5.234)").
_UNKNOWN_RT_PATTERN = re.compile(r"Unknown \(\d+\.\d+\)")

_HEADER_FONT = Font(bold=True, size=12)
_SUBHEADER_FONT = Font(bold=True, size=10)
_QUALITY_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")


def is_unidentified(peak: dict) -> bool:
    """Return True when the peak has no real compound assignment.

    A peak is unidentified when its compound_id (or "Compound ID") is None,
    empty, "unknown" (any case), or matches the "Unknown (RT)" placeholder.
    """
    cid = peak.get('compound_id') or peak.get('Compound ID')
    if not cid:
        return True
    cid_str = str(cid).strip()
    if not cid_str or cid_str.lower() == "unknown":
        return True
    if _UNKNOWN_RT_PATTERN.fullmatch(cid_str):
        return True
    return False


def is_unquantitated(peak: dict, all_peaks: list) -> bool:
    """Return True when this peak is missing quantitation results.

    Only meaningful when quantitation was attempted on the file at all.
    If no peak in *all_peaks* has wt_percent set, we treat the file as
    not having been quantitated and return False (don't filter).
    """
    any_quantitated = any(
        p.get('wt_percent') is not None
        for p in all_peaks
    )
    if not any_quantitated:
        return False
    return peak.get('wt_percent') is None


def parse_timestamp(ts_string: str):
    """Try to parse *ts_string* using known Agilent patterns and common ISO patterns.
    Returns a :class:`datetime.datetime` on success, or ``None``."""
    ts_string = ts_string.strip()
    if not ts_string:
        return None

    # Try standard ISO first
    try:
        return datetime.datetime.fromisoformat(ts_string)
    except (ValueError, AttributeError):
        pass

    for pattern in _TS_PARSE_FMTS:
        try:
            return datetime.datetime.strptime(ts_string, pattern)
        except ValueError:
            continue
    return None


def write_timestamp_cell(ws, row: int, col: int, ts, fmt: str, font):
    """Write a timestamp to *ws* at (*row*, *col*).

    * ``fmt == 'as_stored'``   → plain string, no number format.
    * ``fmt`` starts with '%'  → strftime text cell (backward-compat).
    * anything else            → parse (or take the given datetime), write as Excel native date
                                 with ``cell.number_format = fmt``.
    """
    cell = ws.cell(row=row, column=col)
    cell.font = font

    if isinstance(ts, datetime.datetime):
        dt, ts_string = ts, ts.isoformat(sep=' ')
    else:
        ts_string = ts
        dt = None

    if fmt == 'as_stored' or not fmt:
        cell.value = ts if dt is None else ts_string
        return cell

    if dt is None:
        dt = parse_timestamp(ts_string)

    if fmt.startswith('%'):
        # strftime string → text cell
        cell.value = dt.strftime(fmt) if dt else ts_string
    else:
        # Excel number-format string → native datetime cell
        if dt is not None:
            cell.value = dt
            cell.number_format = fmt
        else:
            cell.value = ts_string  # couldn't parse; write raw

    return cell


def get_peak_value(peak: dict, key: str):
    """Return the cell value for *key* from a peak dict, with type coercion."""
    if key == 'quality_issues':
        # Derived: combine is_saturated / is_convoluted / quality_issues list
        issues = []
        if peak.get("is_saturated"):
            issues.append("Saturated")
        if peak.get("is_convoluted"):
            issues.append("Convoluted")
        raw = peak.get("quality_issues")
        if raw:
            if isinstance(raw, list):
                issues.extend(raw)
            else:
                issues.append(str(raw))
        return ", ".join(issues) if issues else ""

    _float_keys = {'retention_time', 'width', 'area', 'start_time', 'end_time',
                   'Qual', 'mol_C', 'mol_C_percent', 'mol', 'mass_mg',
                   'mol_percent', 'wt_percent'}
    _int_keys = {'peak_number', 'num_carbons'}

    # 'Compound ID' in the config maps to 'compound_id' in serialized peak dicts
    if key == 'Compound ID':
        raw = peak.get('Compound ID') or peak.get('compound_id')
    else:
        raw = peak.get(key)
    if raw is None:
        return ""
    try:
        if key in _int_keys:
            return int(raw)
        if key in _float_keys:
            return round(float(raw), 4)
    except (ValueError, TypeError):
        pass
    return raw


def channel_width(cfg: dict) -> int:
    """Width in columns of one channel block (excluding the spacer column).

    Header rows that show labels occupy 2 columns; the peak table occupies one
    per enabled column. Use whichever is wider so neighbouring blocks never overlap.
    """
    n_cols = sum(1 for c in cfg['columns'] if c['enabled'])
    any_label = any(hf.get('show_label', True)
                    for hf in cfg['header_fields'] if hf['enabled'])
    return max(n_cols, 2 if any_label else 1)


def write_channel_block(ws, row: int, col: int, json_data: dict, cfg: dict, *,
                        skip_unidentified: bool = False,
                        skip_unquantitated: bool = False):
    """Write one channel's header fields, column headers and peak rows at (*row*, *col*).

    Returns ``(next_empty_row, skipped_count)``.
    """
    enabled_headers = [h for h in cfg['header_fields'] if h['enabled']]
    enabled_columns = [c for c in cfg['columns'] if c['enabled']]
    ts_fmt = cfg.get('timestamp_format', 'as_stored')
    n_cols = len(enabled_columns)
    skipped = 0

    # --- Header fields ---
    for hf in enabled_headers:
        if hf.get('key') == 'blank_line':
            row += 1
            continue
        label = hf['custom_label'].strip() or hf['label']
        raw_value = json_data.get(hf['key'], "")
        if hf.get('show_label', True):
            cell = ws.cell(row=row, column=col, value=label)
            cell.font = _HEADER_FONT
            val_col = col + 1
        else:
            val_col = col

        if hf['key'] == 'timestamp' and raw_value:
            write_timestamp_cell(ws, row, val_col, raw_value if isinstance(raw_value, datetime.datetime) else str(raw_value),
                                 ts_fmt, _HEADER_FONT)
        else:
            cell = ws.cell(row=row, column=val_col, value=raw_value)
            cell.font = _HEADER_FONT
        row += 1

    # --- Peak table column headers (no blank spacer row) ---
    for i, col_cfg in enumerate(enabled_columns):
        col_name = col_cfg['custom_name'].strip() or col_cfg['display']
        cell = ws.cell(row=row, column=col + i, value=col_name)
        cell.font = _SUBHEADER_FONT
        cell.alignment = Alignment(horizontal='center')
    row += 1

    # --- Peak rows ---
    all_peaks = json_data.get("peaks", [])
    for peak in all_peaks:
        if skip_unidentified and is_unidentified(peak):
            skipped += 1
            continue
        if skip_unquantitated and is_unquantitated(peak, all_peaks):
            skipped += 1
            continue
        has_quality_issue = bool(get_peak_value(peak, 'quality_issues'))
        for i, col_cfg in enumerate(enabled_columns):
            ws.cell(row=row, column=col + i, value=get_peak_value(peak, col_cfg['key']))
        if has_quality_issue:
            for i in range(n_cols):
                ws.cell(row=row, column=col + i).fill = _QUALITY_FILL
        row += 1

    return row, skipped
