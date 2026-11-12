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
    return {'id': wid, 'label': '', 'start': start, 'end': end,
            'smoothing': {}, 'peaks': {}, 'shoulders': {},
            'negative_peaks': {}, 'events': []}


def windows_from_range_filters(range_filters):
    """Build windows from legacy range filters, merging only overlapping ones.

    Malformed entries (len != 2) are ignored, as ``processor.process`` does.
    Inverted or zero-width filters are dropped (v1 methods tolerated them).
    Returns ``(windows, messages)``, one message per merge or dropped filter.
    """
    spans, messages = [], []
    for rf in (range_filters or []):
        if len(rf) != 2:
            continue
        start, end = float(rf[0]), float(rf[1])
        if not start < end:
            messages.append(
                f"Range filter {start}-{end} is inverted or zero-width; dropped")
            continue
        spans.append((start, end))
    spans.sort()
    merged = []
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


def dedupe_v1_break_points(break_points):
    """Drop repeated break times from a v1 list (they carry no overrides).

    Items are numbers or dicts with ``time``. Returns ``(list, messages)``,
    one message per dropped duplicate. Not for v2 input, which stays strict.
    """
    out, seen, messages = [], [], []
    for bp in break_points or []:
        t = float(bp['time'] if isinstance(bp, dict) else bp)
        if any(abs(t - s) < _EDGE_TOL for s in seen):
            messages.append(f"Duplicate break point at {t} dropped")
            continue
        seen.append(t)
        out.append(bp)
    return out, messages


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


def blend_points(a, b, n_window):
    """Samples of linear blend inside a window edge between two smoothings.

    Each enabled side contributes its savgol window (11 for other methods); the
    larger wins (0 if neither is enabled), capped at a quarter of the window.
    """
    k = 0
    for s in (a, b):
        if s.get('enabled'):
            k = max(k, s.get('savgol_window', 11) if s.get('method') == 'savgol' else 11)
    return min(k, n_window // 4)


def shared_edge_neighbour(windows, window, edge):
    """The window touching ``window`` at ``edge`` ('start' or 'end'), else None."""
    other = 'end' if edge == 'start' else 'start'
    for o in windows:
        if o is not window and abs(o[other] - window[edge]) < _EDGE_TOL:
            return o
    return None


def stitch_smoothed(x, base, pieces):
    """One smoothed trace from full-trace smoothings, blended at window edges.

    ``pieces`` is a list of ``(window, full_trace, k_left, k_right)``. Inside a
    window the result is its trace, outside every window it is ``base``. On the
    first/last k samples inside a free edge the value moves linearly from the
    neighbour (``base``) to the window's own trace:
    ``(1 - w) * neighbour + w * own`` with ``w = (j + 1) / (k + 1)``, j counting
    inward. At a shared edge only the later window blends (its ``k_left``, from
    the earlier window's trace); the earlier window's end stays pure and its
    ``k_right`` is ignored, so the trace is continuous. Samples are found by
    value, so descending axes work.
    """
    import numpy as np

    x = np.asarray(x)
    out = np.array(base, dtype=float, copy=True)
    wins = [w for w, _, _, _ in pieces]
    masks = [(x >= w['start']) & (x <= w['end']) for w in wins]
    for mask, (_, trace, _, _) in zip(masks, pieces):
        out[mask] = np.asarray(trace)[mask]
    for mask, (w, trace, k_left, k_right) in zip(masks, pieces):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(x[idx], kind='stable')]  # ascending value: start edge first
        own = np.asarray(trace)
        for edge, k, at_start in (('start', k_left, True), ('end', k_right, False)):
            neighbour = shared_edge_neighbour(wins, w, edge)
            if neighbour is not None and not at_start:
                continue  # the later window owns the blend at a shared edge
            k = min(k, idx.size // 2)
            if k <= 0:
                continue
            nb = base
            if neighbour is not None:
                nb = pieces[wins.index(neighbour)][1]
            nb = np.asarray(nb)
            sel = idx[:k] if at_start else idx[::-1][:k]  # j counts inward
            wt = (np.arange(k) + 1.0) / (k + 1.0)
            out[sel] = (1 - wt) * nb[sel] + wt * own[sel]
    return out
