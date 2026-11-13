"""Per-segment editor records <-> phase 1's ``peak_windows`` + ``baseline.break_points``.

A segment is a peak window. Its baseline lives in the break points (design doc
2026-11-12-piecewise-editor-design.md §4.1):

- start break: anchored to ``(id, 'start')``, or unanchored exactly at the start
  (a file the editor didn't write); shown as ``fresh_baseline``;
- partway rows: unanchored breaks strictly inside the window, stored merged over
  the segment's effective baseline, shown as the segment's own values only;
- end break: anchored to ``(id, 'end')``, carrying the whole-run overrides in force
  just before the segment ('resume') unless the file set others ('custom');
- whole-run breaks: every other break.

Pure dict code (no Qt, no pydantic), like ``logic/windows.py``. Params are shaped
like ``ChromaMethod.to_gui_params()``: windows in ``params['peak_windows']``, breaks
in ``params['baseline']['break_points']`` (may be None), override keys by alias
(``'lambda'``). Every function returns new params and never mutates its input.
"""

import copy

from logic.windows import _EDGE_TOL, merge_overrides, resolve_break_points, validate_windows

SEGMENT_SECTIONS = ('smoothing', 'peaks', 'shoulders', 'negative_peaks', 'rider')
_DEFAULT_TOLERANCE = 0.1  # BreakPoint.tolerance default in logic/method.py; unused by the processor


# ── reading ────────────────────────────────────────────────────────────────────

def _strip(overrides):
    """Copy of an override dict holding only set values (nested dicts too; empty ones dropped)."""
    out = {}
    for key, val in (overrides or {}).items():
        if isinstance(val, dict):
            val = {k: copy.deepcopy(v) for k, v in val.items() if v is not None}
            if val:
                out[key] = val
        elif val is not None:
            out[key] = copy.deepcopy(val)
    return out


def _same(a, b):
    return abs(float(a) - float(b)) < _EDGE_TOL


def _name(window):
    return window.get('label') or window['id']


def _breaks(params):
    return (params.get('baseline') or {}).get('break_points') or []


def _owners(windows, breaks):
    """Per break: ``(window_id, 'start'|'partway'|'end')`` if a segment owns it, else None."""
    ids = {w['id'] for w in windows}
    owners = []
    for bp in breaks:
        anchor = bp.get('anchor')
        if anchor:
            if anchor['window_id'] not in ids:
                raise ValueError(f"Break point at {bp['time']} is anchored to missing "
                                 f"window '{anchor['window_id']}'")
            owners.append((anchor['window_id'], anchor['edge']))
            continue
        t, owner = float(bp['time']), None
        for w in windows:
            if _same(t, w['start']):
                owner = (w['id'], 'start')
                break
            if w['start'] + _EDGE_TOL < t < w['end'] - _EDGE_TOL:
                owner = (w['id'], 'partway')
                break
        owners.append(owner)
    return owners


def _owned(owners, wid):
    """Indices of the breaks segment ``wid`` owns: ``{'start': i|None, 'partway': [i], 'end': i|None}``."""
    out = {'start': None, 'partway': [], 'end': None}
    for i, owner in enumerate(owners):
        if owner and owner[0] == wid:
            if owner[1] == 'partway':
                out['partway'].append(i)
            else:
                out[owner[1]] = i
    return out


def _resume(breaks, owners, t):
    best = None
    for i, bp in enumerate(breaks):
        if owners[i] is None and float(bp['time']) <= t + _EDGE_TOL and (
                best is None or float(bp['time']) > float(breaks[best]['time'])):
            best = i
    return _strip(breaks[best].get('overrides')) if best is not None else {}


def _parse(params, validate=True):
    """``(windows sorted by start, breaks, owners)``; the lists are params' own."""
    windows = params.get('peak_windows') or []
    breaks = _breaks(params)
    if validate:
        validate_windows(windows)
        resolve_break_points(windows, breaks)
    windows = sorted(windows, key=lambda w: w['start'])
    return windows, breaks, _owners(windows, breaks)


def _custom_end_ids(windows, breaks, owners):
    """Ids of segments whose end break differs from the resume value."""
    custom = set()
    for w in windows:
        end = _owned(owners, w['id'])['end']
        if end is not None and _strip(breaks[end].get('overrides')) != _resume(breaks, owners, w['start']):
            custom.add(w['id'])
    return custom


def resume_overrides(params, t):
    """Overrides of the latest whole-run break at or before t, else {}."""
    windows, breaks, owners = _parse(params)
    return _resume(breaks, owners, float(t))


def segments_view(params):
    """``{'segments': [record, ...] sorted by start, 'whole_run_breaks': [break dict, ...]}``."""
    windows, breaks, owners = _parse(params)
    segments = []
    for w in windows:
        own = _owned(owners, w['id'])
        fresh = own['start'] is not None
        baseline = _strip(breaks[own['start']].get('overrides')) if fresh else {}
        resume = _resume(breaks, owners, w['start'])
        effective = baseline if fresh else resume
        partway = []
        for i in sorted(own['partway'], key=lambda i: float(breaks[i]['time'])):
            stored = _strip(breaks[i].get('overrides'))
            partway.append({
                'time': float(breaks[i]['time']),
                'overrides': {k: v for k, v in stored.items()
                              if not (k in effective and effective[k] == v)},
            })
        if own['end'] is None:
            after = {'mode': 'none', 'overrides': {}}
        else:
            ov = _strip(breaks[own['end']].get('overrides'))
            after = {'mode': 'resume' if ov == resume else 'custom', 'overrides': ov}
        record = {'id': w['id'], 'label': w.get('label', ''),
                  'start': float(w['start']), 'end': float(w['end'])}
        record.update({s: _strip(w.get(s)) for s in SEGMENT_SECTIONS})
        record.update({'events': copy.deepcopy(w.get('events') or []),
                       'fresh_baseline': fresh, 'baseline': baseline,
                       'partway': partway, 'after': after})
        segments.append(record)
    whole = [copy.deepcopy(bp) for bp, owner in zip(breaks, owners) if owner is None]
    whole.sort(key=lambda bp: float(bp['time']))
    return {'segments': segments, 'whole_run_breaks': whole}


def new_segment(params, start, end):
    """A blank record over ``start``-``end``; id is the smallest unused ``w{n}``."""
    used = {w['id'] for w in params.get('peak_windows') or []}
    n = 1
    while f"w{n}" in used:
        n += 1
    record = {'id': f"w{n}", 'label': '', 'start': float(start), 'end': float(end)}
    record.update({s: {} for s in SEGMENT_SECTIONS})
    record.update({'events': [], 'fresh_baseline': False, 'baseline': {}, 'partway': [],
                   'after': {'mode': 'none', 'overrides': {}}})
    return record


# ── writing ────────────────────────────────────────────────────────────────────

def _set_breaks(out, breaks, original):
    """Store breaks sorted by time; an empty result keeps the input's None."""
    breaks.sort(key=lambda bp: float(bp['time']))
    out['baseline']['break_points'] = breaks if (breaks or original is not None) else None


def _sync(out, custom):
    """Regenerate every end break in ``out`` (in place); ``custom`` ids keep theirs.

    Not validated on entry: a fresh start just written may share its time with the
    neighbour's end break this drops. ``_finish`` validates the result.
    """
    windows, breaks, owners = _parse(out, validate=False)
    fresh_starts = {w['id']: w['start'] for w in windows
                    if _owned(owners, w['id'])['start'] is not None}
    drop, add = set(), []
    for w in windows:
        own = _owned(owners, w['id'])
        end = own['end']
        neighbour = next((o for o in windows if o is not w and o['id'] in fresh_starts
                          and _same(fresh_starts[o['id']], w['end'])), None)
        if end is not None and w['id'] in custom:
            if neighbour is not None:
                # shortcut: refuse instead of choosing between the custom end break and the
                # neighbour's fresh start; revisit if the editor should let one replace the other.
                raise ValueError(
                    f"Segment '{_name(neighbour)}' starts a fresh baseline at {w['end']:g}, where "
                    f"segment '{_name(w)}' has a custom baseline after it; reset that one first")
            continue
        if not (own['start'] is not None or own['partway']) or neighbour is not None:
            if end is not None:
                drop.add(end)
            continue
        resume = _resume(breaks, owners, w['start'])
        if end is None:
            add.append({'time': float(w['end']), 'tolerance': _DEFAULT_TOLERANCE,
                        'overrides': resume, 'anchor': {'window_id': w['id'], 'edge': 'end'}})
        elif _strip(breaks[end].get('overrides')) != resume:
            breaks[end] = dict(breaks[end], overrides=resume)
    original = out['baseline'].get('break_points')
    _set_breaks(out, [bp for i, bp in enumerate(breaks) if i not in drop] + add, original)


def _finish(out):
    """Validate the written params and refresh the range-filter mirror."""
    _parse(out)
    out['peak_windows'].sort(key=lambda w: w['start'])
    out['peaks']['range_filters'] = (
        [[w['start'], w['end']] for w in out['peak_windows']]
        if out.get('outside_windows', 'ignore') == 'ignore' else [])
    return out


def apply_segment(params, record):
    """New params with ``record`` written back; raises ValueError naming the time or segment."""
    out = copy.deepcopy(params)
    windows, breaks, owners = _parse(out)
    custom = _custom_end_ids(windows, breaks, owners)
    rid = record['id']
    name = record.get('label') or rid
    start, end = float(record['start']), float(record['end'])
    if not start < end:
        raise ValueError(f"Segment '{name}' has From {start:g} >= To {end:g}")
    for w in windows:
        if w['id'] != rid and start < w['end'] and w['start'] < end:
            raise ValueError(f"Segment '{name}' ({start:g}-{end:g}) overlaps segment "
                             f"'{_name(w)}' ({w['start']:g}-{w['end']:g})")
    for row in record.get('partway') or []:
        t = float(row['time'])
        if t <= start + _EDGE_TOL or t >= end - _EDGE_TOL:
            raise ValueError(f"Baseline change at {t:g} is not inside segment '{name}' "
                             f"({start:g}-{end:g})")

    window = {'id': rid, 'label': record.get('label', ''), 'start': start, 'end': end}
    window.update({s: _strip(record.get(s)) for s in SEGMENT_SECTIONS})
    window['events'] = copy.deepcopy(record.get('events') or [])
    out['peak_windows'] = [w for w in windows if w['id'] != rid] + [window]
    out['peak_windows'].sort(key=lambda w: w['start'])

    own = _owned(owners, rid)
    tolerance = {}  # a rewritten break keeps its tolerance (matched by role, partway by time)
    for role in ('start', 'end'):
        if own[role] is not None:
            tolerance[role] = breaks[own[role]].get('tolerance', _DEFAULT_TOLERANCE)
    old_partway = [breaks[i] for i in own['partway']]
    original = out['baseline'].get('break_points')
    kept = [bp for bp, owner in zip(breaks, owners) if not (owner and owner[0] == rid)]

    if record.get('fresh_baseline'):
        effective = _strip(record.get('baseline'))
        kept.append({'time': start, 'tolerance': tolerance.get('start', _DEFAULT_TOLERANCE),
                     'overrides': copy.deepcopy(effective),
                     'anchor': {'window_id': rid, 'edge': 'start'}})
    else:
        effective = _resume(kept, _owners(out['peak_windows'], kept), start)
    for row in record.get('partway') or []:
        t = float(row['time'])
        tol = next((bp.get('tolerance', _DEFAULT_TOLERANCE) for bp in old_partway
                    if _same(bp['time'], t)), _DEFAULT_TOLERANCE)
        kept.append({'time': t, 'tolerance': tol,
                     'overrides': merge_overrides(effective, _strip(row.get('overrides'))),
                     'anchor': None})
    after = record.get('after') or {'mode': 'none', 'overrides': {}}
    custom.discard(rid)
    if after['mode'] == 'custom':
        kept.append({'time': end, 'tolerance': tolerance.get('end', _DEFAULT_TOLERANCE),
                     'overrides': _strip(after.get('overrides')),
                     'anchor': {'window_id': rid, 'edge': 'end'}})
        custom.add(rid)
    _set_breaks(out, kept, original)
    _sync(out, custom)
    return _finish(out)


def remove_segment(params, segment_id):
    """Drop the window, its anchored breaks and partway rows; an unanchored start break stays."""
    out = copy.deepcopy(params)
    windows, breaks, owners = _parse(out)
    if not any(w['id'] == segment_id for w in windows):
        raise ValueError(f"No segment with id '{segment_id}'")
    custom = _custom_end_ids(windows, breaks, owners) - {segment_id}
    original = out['baseline'].get('break_points')
    kept = [bp for bp, owner in zip(breaks, owners)
            if not (owner and owner[0] == segment_id
                    and (owner[1] != 'start' or bp.get('anchor')))]
    out['peak_windows'] = [w for w in windows if w['id'] != segment_id]
    _set_breaks(out, kept, original)
    _sync(out, custom)
    return _finish(out)


def set_whole_run_breaks(params, breaks):
    """Replace the breaks outside every segment, then regenerate the resume end breaks."""
    out = copy.deepcopy(params)
    windows, stored, owners = _parse(out)
    custom = _custom_end_ids(windows, stored, owners)
    new = []
    for bp in breaks:
        bp = copy.deepcopy(bp)
        if bp.get('anchor'):
            raise ValueError(f"Whole-run baseline change at {bp['time']} is anchored to a segment")
        if bp.get('tolerance') is None:
            bp['tolerance'] = _DEFAULT_TOLERANCE
        bp.setdefault('overrides', {})
        bp['anchor'] = None
        new.append(bp)
    original = out['baseline'].get('break_points')
    _set_breaks(out, [bp for bp, owner in zip(stored, owners) if owner is not None] + new, original)
    _sync(out, custom)
    return _finish(out)


def sync_end_breaks(params):
    """New params with every non-custom end break regenerated."""
    out = copy.deepcopy(params)
    windows, breaks, owners = _parse(out)
    _sync(out, _custom_end_ids(windows, breaks, owners))
    return _finish(out)


# ── display helpers ────────────────────────────────────────────────────────────

def _fmt(val):
    if isinstance(val, bool):
        return 'on' if val else 'off'
    if isinstance(val, float):
        return f"{val:g}"
    if isinstance(val, dict):
        return ' '.join(f"{k} {_fmt(v)}" for k, v in val.items())
    return str(val)


def _fmt_lambda(val):
    mantissa, exponent = f"{float(val):e}".split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    return f"{mantissa}e{int(exponent)}"


def _switch(token, overrides):
    """'token on/off' when ``enabled`` is set, 'token' for any other change, else None."""
    if 'enabled' in overrides:
        return f"{token} {_fmt(overrides['enabled'])}"
    return token if overrides else None


def summarize(record):
    """Short 'what it changes' text, e.g. 'λ 1e3→1e4 · prom 0.02 · width 3'; '—' when nothing."""
    baseline = record.get('baseline') or {}
    rows = [row.get('overrides') or {} for row in record.get('partway') or []]
    lambdas = [o['lambda'] for o in [baseline] + rows if 'lambda' in o]
    tokens = []
    if lambdas:
        tokens.append('λ ' + '→'.join(_fmt_lambda(v) for v in lambdas))
    for overrides in [baseline] + rows:
        for key, val in overrides.items():
            token = f"{key} {_fmt(val)}"
            if key != 'lambda' and token not in tokens:
                tokens.append(token)
    peaks = record.get('peaks') or {}
    for key, token in (('min_prominence', 'prom'), ('min_width', 'width'), ('min_height', 'height')):
        if key in peaks:
            tokens.append(f"{token} {_fmt(peaks[key])}")
    tokens.append(_switch('smooth', record.get('smoothing') or {}))
    tokens.append(_switch('shoulders', record.get('shoulders') or {}))
    rider = record.get('rider') or {}
    if 'rider_integration' in rider:
        tokens.append(f"skim {rider['rider_integration']}")
    elif rider:
        tokens.append('skim')
    tokens.append(_switch('neg', record.get('negative_peaks') or {}))
    return ' · '.join(t for t in tokens if t) or '—'


def breaks_inside_peaks(peaks, break_times):
    """One message per break time strictly inside a peak's start_time-end_time bounds."""
    messages = []
    for t in break_times:
        for p in peaks:
            lo, hi = sorted((float(p.start_time), float(p.end_time)))
            if lo + _EDGE_TOL < t < hi - _EDGE_TOL:
                messages.append(f"Baseline change at {t:g} falls inside the peak at "
                                f"{lo:g}-{hi:g}; its area spans two baselines")
                break
    return messages
