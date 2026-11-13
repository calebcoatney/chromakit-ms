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
import logging

from logic.windows import _EDGE_TOL, merge_overrides, resolve_break_points, validate_windows, window_at

logger = logging.getLogger(__name__)

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


def resume_overrides(params, t):
    """Overrides of the latest whole-run break at or before t, else {}."""
    _, breaks, owners = _parse(params)
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


# ── validating ─────────────────────────────────────────────────────────────────

# Section names as the editor's group titles show them (ui/frames/segments.py GROUPS)
_SECTION_TITLES = {'baseline': 'Baseline', 'peaks': 'Peak detection', 'smoothing': 'Smoothing',
                   'shoulders': 'Shoulders', 'rider': 'Tail skim', 'negative_peaks': 'Negative peaks'}


def _allowed(model, key):
    """'1 to 10', '>= 3', '> 0 and <= 1' from a model field's bounds; '' when it has none."""
    field = next((f for name, f in model.model_fields.items() if key in (name, f.alias)), None)
    bounds = {}
    for meta in (field.metadata if field is not None else []):
        for op in ('ge', 'gt', 'le', 'lt'):
            if getattr(meta, op, None) is not None:
                bounds[op] = getattr(meta, op)
    if set(bounds) == {'ge', 'le'}:
        return f"{bounds['ge']:g} to {bounds['le']:g}"
    signs = {'ge': '>=', 'gt': '>', 'le': '<=', 'lt': '<'}
    return ' and '.join(f"{signs[op]} {bounds[op]:g}" for op in ('ge', 'gt', 'le', 'lt') if op in bounds)


def validate_record(record) -> None:
    """Raise ValueError naming the field and its allowed range if a value breaks the limits of
    its ``*Overrides`` model in logic/method.py (what ``ChromaMethod`` enforces on load)."""
    from logic.method import (  # lazy: logic.method is pydantic; this module stays free of it
        BaselineOverrides, NegativePeakOverrides, PeakOverrides, RiderOverrides,
        ShoulderOverrides, SmoothingOverrides,
    )
    models = {'smoothing': SmoothingOverrides, 'peaks': PeakOverrides,
              'shoulders': ShoulderOverrides, 'negative_peaks': NegativePeakOverrides,
              'rider': RiderOverrides}
    name = record.get('label') or record['id']
    checks = [(_SECTION_TITLES[s], models[s], record.get(s)) for s in SEGMENT_SECTIONS]
    checks.append((_SECTION_TITLES['baseline'], BaselineOverrides, record.get('baseline')))
    for row in record.get('partway') or []:
        checks.append((f"Baseline change at {float(row['time']):g}:", BaselineOverrides,
                       row.get('overrides')))
    after = record.get('after') or {}
    if after.get('mode') == 'custom':
        checks.append(('After the segment:', BaselineOverrides, after.get('overrides')))
    for where, model, values in checks:
        try:
            model.model_validate(values or {})
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            err = exc.errors()[0] if hasattr(exc, 'errors') else None
            if err is None:
                raise
            key = '.'.join(str(p) for p in err['loc'])
            allowed = _allowed(model, str(err['loc'][0])) if len(err['loc']) == 1 else ''
            if allowed:
                reason = f"{err.get('input')} is out of range (allowed {allowed})"
            else:
                reason = f"is not valid: {err['msg']}"
            raise ValueError(f"Segment '{name}': {where} {key} {reason}") from None


# ── writing ────────────────────────────────────────────────────────────────────
# Edits never rewrite a segment the user didn't edit (design §4.1): only the edited
# segment's own breaks change, plus the end break of a segment ending where the edited
# segment's own start break sat before the edit.

def _set_breaks(out, breaks, original):
    """Store breaks sorted by time; an empty result keeps the input's None."""
    breaks.sort(key=lambda bp: float(bp['time']))
    out['baseline']['break_points'] = breaks if (breaks or original is not None) else None


def _break_at(breaks, t):
    return any(_same(bp['time'], t) for bp in breaks)


def _end_break(wid, t, overrides, tolerance=_DEFAULT_TOLERANCE):
    return {'time': float(t), 'tolerance': tolerance, 'overrides': overrides,
            'anchor': {'window_id': wid, 'edge': 'end'}}


def _restore_neighbour_ends(out, times):
    """A segment ending at one of ``times`` regains a resume end break.

    Only when it has a fresh start or partway rows, has no end break, and no break
    sits at its end (a duplicate time is invalid). Not validated on entry.
    """
    windows, breaks, owners = _parse(out, validate=False)
    add = []
    for w in windows:
        if not any(_same(w['end'], t) for t in times):
            continue
        own = _owned(owners, w['id'])
        if own['end'] is not None or not (own['start'] is not None or own['partway']):
            continue
        if _break_at(breaks + add, w['end']):
            continue
        add.append(_end_break(w['id'], w['end'], _resume(breaks, owners, w['start'])))
    if add:
        _set_breaks(out, breaks + add, out['baseline'].get('break_points'))


def _finish(out):
    """Validate the written params and refresh the range-filter mirror."""
    _parse(out)
    out['peak_windows'].sort(key=lambda w: w['start'])
    out['peaks']['range_filters'] = (
        [[w['start'], w['end']] for w in out['peak_windows']]
        if out.get('outside_windows', 'ignore') == 'ignore' else [])
    return out


def apply_segment(params, record):
    """New params with ``record`` written back; raises ValueError naming the time or segment.

    The end break follows ``record['after']['mode']``: 'resume' writes the whole-run
    value in force before the segment, 'custom' the given overrides, 'none' nothing.
    'none' counts as 'resume' when the segment gains its first fresh start or partway
    row. A record with neither keeps only a custom end break (reset by setting the
    mode to 'resume' or 'none'). No end break is written where another break already
    sits. The segment ending at this one's old start regains a resume end break only
    if this segment's start break sat there and no break sits there now.
    """
    out = copy.deepcopy(params)
    windows, breaks, owners = _parse(out)
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
    # R21: a whole-run break (owned by no segment before the edit) that the new span would
    # cover becomes this segment's own, so deleting the segment would delete it too
    for bp, owner in zip(breaks, owners):
        t = float(bp['time'])
        if owner is None and (_same(t, start) or start + _EDGE_TOL < t < end - _EDGE_TOL):
            raise ValueError(f"Whole-run baseline change at {t:g} min is inside segment "
                             f"'{name}'; move or delete it first")

    old = next((w for w in windows if w['id'] == rid), None)
    window = {'id': rid, 'label': record.get('label', ''), 'start': start, 'end': end}
    window.update({s: _strip(record.get(s)) for s in SEGMENT_SECTIONS})
    window['events'] = copy.deepcopy(record.get('events') or [])
    out['peak_windows'] = [w for w in windows if w['id'] != rid] + [window]
    out['peak_windows'].sort(key=lambda w: w['start'])

    own = _owned(owners, rid)
    had_baseline = own['start'] is not None or bool(own['partway'])
    tolerance = {}  # a rewritten break keeps its tolerance (matched by role, partway by time)
    for role in ('start', 'end'):
        if own[role] is not None:
            tolerance[role] = breaks[own[role]].get('tolerance', _DEFAULT_TOLERANCE)
    old_partway = [breaks[i] for i in own['partway']]
    old_end = breaks[own['end']] if own['end'] is not None else None
    original = out['baseline'].get('break_points')
    kept = [bp for bp, owner in zip(breaks, owners) if not (owner and owner[0] == rid)]

    fresh = bool(record.get('fresh_baseline'))
    if fresh:
        for bp in [bp for bp in kept if bp.get('anchor') and bp['anchor']['edge'] == 'end'
                   and _same(bp['time'], start)]:
            logger.info("End break at %g of segment '%s' dropped: segment '%s' starts a "
                        "fresh baseline there", start, bp['anchor']['window_id'], name)
            kept.remove(bp)
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
    mode = after['mode']
    if mode not in ('resume', 'custom', 'none'):
        raise ValueError(f"Segment '{name}': unknown after-segment mode {mode!r}")
    if not (fresh or record.get('partway')):
        if mode != 'custom':  # a custom end break stays until the user resets it (R10)
            mode = 'none'
    elif mode == 'none' and not had_baseline:
        mode = 'resume'
    if mode != 'none':
        if _break_at(kept, end):
            logger.info("Segment '%s' gets no end break: a baseline change already sits at %g",
                        name, end)
        else:
            overrides = (_strip(after.get('overrides')) if mode == 'custom'
                         else _resume(kept, _owners(out['peak_windows'], kept), start))
            if (old_end is not None and _same(old_end['time'], end)
                    and _strip(old_end.get('overrides')) == overrides):
                kept.append(old_end)  # unchanged: keep the stored dict as it is
            else:
                kept.append(_end_break(rid, end, overrides,
                                       tolerance.get('end', _DEFAULT_TOLERANCE)))
    _set_breaks(out, kept, original)
    if old is not None and own['start'] is not None:  # only where its own start break sat
        _restore_neighbour_ends(out, [old['start']])
    return _finish(out)


def remove_segment(params, segment_id):
    """Drop the window, its anchored breaks and partway rows; an unanchored start break stays."""
    out = copy.deepcopy(params)
    windows, breaks, owners = _parse(out)
    gone = next((w for w in windows if w['id'] == segment_id), None)
    if gone is None:
        raise ValueError(f"No segment with id '{segment_id}'")
    original = out['baseline'].get('break_points')
    kept = [bp for bp, owner in zip(breaks, owners)
            if not (owner and owner[0] == segment_id
                    and (owner[1] != 'start' or bp.get('anchor')))]
    out['peak_windows'] = [w for w in windows if w['id'] != segment_id]
    _set_breaks(out, kept, original)
    start = _owned(owners, segment_id)['start']
    if start is not None and breaks[start].get('anchor'):  # only a start break actually removed
        _restore_neighbour_ends(out, [gone['start']])
    return _finish(out)


def set_whole_run_breaks(params, breaks):
    """Replace the breaks outside every segment.

    End breaks that were in 'resume' mode before the edit take the new resume value;
    no end break is added or removed.
    """
    out = copy.deepcopy(params)
    windows, stored, owners = _parse(out)
    follows = set()  # ids of segments whose end break was 'resume'
    for w in windows:
        end = _owned(owners, w['id'])['end']
        if end is not None and _strip(stored[end].get('overrides')) == _resume(stored, owners, w['start']):
            follows.add(w['id'])
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
    merged = [bp for bp, owner in zip(stored, owners) if owner is not None] + new
    merged_owners = _owners(windows, merged)
    for w in windows:
        end = _owned(merged_owners, w['id'])['end']
        if w['id'] in follows:
            resume = _resume(merged, merged_owners, w['start'])
            if _strip(merged[end].get('overrides')) != resume:
                merged[end] = dict(merged[end], overrides=resume)
    _set_breaks(out, merged, out['baseline'].get('break_points'))
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


def breaks_inside_peaks_timed(peaks, break_times):
    """``(break time, message)`` per break time strictly inside a peak's start_time-end_time bounds."""
    out = []
    for t in break_times:
        for p in peaks:
            lo, hi = sorted((float(p.start_time), float(p.end_time)))
            if lo + _EDGE_TOL < t < hi - _EDGE_TOL:
                out.append((t, f"Baseline change at {t:g} falls inside the peak at "
                               f"{lo:g}-{hi:g}; its area spans two baselines"))
                break
    return out


def breaks_inside_peaks(peaks, break_times):
    """One message per break time strictly inside a peak's start_time-end_time bounds."""
    return [msg for _, msg in breaks_inside_peaks_timed(peaks, break_times)]


def baseline_break_times(params):
    """Every time the baseline restarts: params' break points, anchors resolved, sorted."""
    return [b['time'] for b in resolve_break_points(params.get('peak_windows') or [], _breaks(params))]


def warnings_by_segment(windows, timed):
    """``{segment id: [message, ...]}`` for ``(time, message)`` pairs.

    A message goes to the window holding its time (``window_at``: inclusive, the later
    window wins a shared edge), else to Whole run, keyed ``''``.
    """
    out = {}
    for t, msg in timed:
        w = window_at(windows, t)
        out.setdefault(w['id'] if w else '', []).append(msg)
    return out
