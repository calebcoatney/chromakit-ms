"""Pure helpers for per-RT-window method parameters and baseline sections.

Dependency-free (no pydantic, no ui/api imports). All functions take plain dicts
shaped like ``ChromaMethod.to_processor_params()`` output. Override dicts are
dumped by alias, so the smoothing/baseline lambda key is ``'lambda'``.
"""

import copy
import logging

logger = logging.getLogger(__name__)

DETECTION_SECTIONS = ('smoothing', 'peaks', 'shoulders', 'negative_peaks')
_OUTSIDE_MODES = ('ignore', 'global')
_EDGE_TOL = 1e-9


def validate_windows(windows):
    """Raise ValueError naming ids for bad spans, overlaps or repeated ids."""
    seen = set()
    for w in windows:
        if w['id'] in seen:
            raise ValueError(f"Duplicate window id '{w['id']}'")
        seen.add(w['id'])
        if not w['start'] < w['end']:
            raise ValueError(
                f"Window '{w['id']}' has start {w['start']} >= end {w['end']}")
    ordered = sorted(windows, key=lambda w: (w['start'], w['end']))
    for a, b in zip(ordered, ordered[1:]):
        if b['start'] < a['end']:
            raise ValueError(
                f"Windows '{a['id']}' ({a['start']}-{a['end']}) and "
                f"'{b['id']}' ({b['start']}-{b['end']}) overlap")


def _empty_window(wid, start, end):
    return {'id': wid, 'label': wid, 'start': start, 'end': end,
            'smoothing': {}, 'peaks': {}, 'shoulders': {},
            'negative_peaks': {}, 'events': []}


def windows_from_range_filters(range_filters):
    """Build windows from legacy range filters, merging only overlapping ones.

    Malformed entries (len != 2) are ignored, as ``processor.process`` does.
    Returns ``(windows, messages)``, one message per merge.
    """
    spans = sorted(
        (float(rf[0]), float(rf[1])) for rf in (range_filters or [])
        if len(rf) == 2)
    merged, messages = [], []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            prev = merged[-1]
            messages.append(
                f"Range filters {prev[0]}-{prev[1]} and {start}-{end} overlap; "
                f"merged into {prev[0]}-{max(prev[1], end)}")
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])
    windows = [_empty_window(f"w{i}", s, e) for i, (s, e) in enumerate(merged, 1)]
    return windows, messages


def resolve_windows(params):
    """Return ``(sorted validated windows, outside_windows mode)`` for params."""
    if 'peak_windows' in params:
        windows = copy.deepcopy(list(params['peak_windows'] or []))
    else:
        windows, messages = windows_from_range_filters(
            (params.get('peaks') or {}).get('range_filters'))
        for msg in messages:
            logger.warning(msg)
    windows.sort(key=lambda w: w['start'])
    validate_windows(windows)
    outside = params.get('outside_windows', 'ignore')
    if outside not in _OUTSIDE_MODES:
        raise ValueError(
            f"outside_windows must be one of {_OUTSIDE_MODES}, got {outside!r}")
    return windows, outside


def window_at(windows, rt):
    """Window holding rt (inclusive both ends); the later window wins an edge."""
    best = None
    for w in windows:
        if w['start'] <= rt <= w['end'] and (best is None or w['start'] > best['start']):
            best = w
    return best


def apex_clamp(windows, rt, outside):
    """Allowed apex span ``(lo, hi)`` for a peak at rt; None means unbounded."""
    w = window_at(windows, rt)
    if w is None:
        return (None, None)
    if outside == 'ignore':
        return (w['start'], w['end'])
    lo = w['start'] if any(abs(o['end'] - w['start']) < _EDGE_TOL
                           for o in windows if o is not w) else None
    hi = w['end'] if any(abs(o['start'] - w['end']) < _EDGE_TOL
                         for o in windows if o is not w) else None
    return (lo, hi)


def merge_overrides(base, overrides):
    """Deep copy of base with every non-None override set (nested dicts merge)."""
    out = copy.deepcopy(base)
    for key, val in (overrides or {}).items():
        if val is None:
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            for k2, v2 in val.items():
                if v2 is not None:
                    out[key][k2] = copy.deepcopy(v2)
        else:
            out[key] = copy.deepcopy(val)
    return out


def window_params(params, window):
    """Deep copy of params with the window's detection-section overrides applied."""
    out = copy.deepcopy(params)
    for section in DETECTION_SECTIONS:
        out[section] = merge_overrides(out.get(section) or {}, window.get(section))
    return out


def overrides_any(window, sections=DETECTION_SECTIONS):
    """True if the window has any non-None override in the given sections."""
    for section in sections:
        for val in (window.get(section) or {}).values():
            if isinstance(val, dict):
                if any(v is not None for v in val.values()):
                    return True
            elif val is not None:
                return True
    return False


def resolve_break_points(windows, break_points):
    """Normalise break points, resolve anchors, reject duplicates, sort by time."""
    by_id = {w['id']: w for w in windows}
    out = []
    for bp in break_points or []:
        if isinstance(bp, dict):
            item = {'time': float(bp['time']),
                    'tolerance': bp.get('tolerance'),
                    'overrides': copy.deepcopy(bp.get('overrides') or {}),
                    'anchor': bp.get('anchor')}
        else:
            item = {'time': float(bp), 'tolerance': None,
                    'overrides': {}, 'anchor': None}
        anchor = item['anchor']
        if anchor:
            win = by_id.get(anchor['window_id'])
            if win is None:
                raise ValueError(
                    f"Break point at {item['time']} is anchored to missing "
                    f"window '{anchor['window_id']}'")
            item['time'] = float(win['start'] if anchor['edge'] == 'start' else win['end'])
        out.append(item)
    out.sort(key=lambda b: b['time'])
    for a, b in zip(out, out[1:]):
        if abs(b['time'] - a['time']) < _EDGE_TOL:
            raise ValueError(f"Duplicate break point at {b['time']}")
    return out


def section_overrides(seg_start_idx, bp_indices, break_points):
    """Overrides of the latest break whose sample index is <= seg_start_idx."""
    best = None
    for i, idx in enumerate(bp_indices):
        if idx <= seg_start_idx and (best is None or idx > bp_indices[best]):
            best = i
    return copy.deepcopy(break_points[best].get('overrides') or {}) if best is not None else {}
