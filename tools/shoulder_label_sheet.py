"""Build a click-to-label HTML sheet for marking shoulders and riders on real runs.

    python -m tools.shoulder_label_sheet RUN.D|RUN.C [...] --out DIR [--method M.chromethod] [--detector FID1A]

Writes DIR/label_sheet.html: one panel per peak cluster of each run, showing the
baseline-corrected signal only (no detector output, so the labels are not
biased by it). Click a panel to mark a shoulder or rider at that RT; click a
mark to remove it. "Save labels" downloads labels.json, which
``tools.shoulder_eval`` scores detectors against.

Keep DIR outside the repo (e.g. ~/chromakit-worktrees/_data/shoulders/):
the sheet embeds lab data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths


def resolve_d(path: str) -> str:
    """A GC2 run is a `<run>.C` folder wrapping `data/<run>.D`; return the .D."""
    p = Path(path)
    if p.suffix == '.C':
        ds = sorted((p / 'data').glob('*.D'))
        if len(ds) != 1:
            raise ValueError(f'{path}: expected one data/*.D inside, found {len(ds)}')
        return str(ds[0])
    return str(p)


def load_run(path: str, method_path: str | None = None, prominence: float | None = None,
             detector: str | None = None):
    """Load a .D (or .C) run and process it the way the app would.
    `detector` picks the channel (e.g. FID1A, FID2B, TCD3C); None auto-detects.
    Returns (x, corrected_y, params)."""
    from logic.data_handler import DataHandler
    from logic.processor import ChromatogramProcessor
    from logic.method import ChromaMethod
    from api.utils import convert_params_for_processor

    data = DataHandler().load_data_directory(resolve_d(path), detector=detector)
    x = np.asarray(data['chromatogram']['x'], dtype=float)
    y = np.asarray(data['chromatogram']['y'], dtype=float)
    method = ChromaMethod.from_file(method_path) if method_path else ChromaMethod(name="label-sheet", signal_type="gc")
    params = convert_params_for_processor(method.to_processor_params())
    params['shoulders'] = {'enabled': False}
    if prominence is not None:
        params['peaks']['min_prominence'] = prominence
    out = ChromatogramProcessor().process(x, y, params=params)
    return x, np.asarray(out['corrected_y'], dtype=float), params


def panels_for_run(x, y, min_snr=20.0, pad_fwhm=3.0, max_fwhm=40.0):
    """Peak clusters worth reviewing: peaks above `min_snr` x noise, windows
    of +-pad_fwhm FWHM around each, overlapping windows merged. A merged
    stretch longer than `max_fwhm` median FWHMs (a crowded run merges into a
    few huge panels) is cut at its lowest points into panels of about that width."""
    d2 = np.diff(y, 2)
    sd = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
    peaks, _ = find_peaks(y, prominence=min_snr * sd)
    if len(peaks) == 0:
        return [], sd
    w = peak_widths(y, peaks, rel_height=0.5)[0]
    spans = sorted((max(0, int(p - pad_fwhm * wi - 5)), min(len(y) - 1, int(p + pad_fwhm * 2 * wi + 5)))
                   for p, wi in zip(peaks, w))
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    width = max(int(max_fwhm * np.median(w)), 20)
    slack = max(int(5 * np.median(w)), 2)
    out = []
    for a, b in merged:
        while b - a > width + slack:
            lo, hi = a + width - slack, a + width + slack
            cut = lo + int(np.argmin(y[lo:hi + 1]))
            out.append([a, cut])
            a = cut
        out.append([a, b])
    return out, sd


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Shoulder labels</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#1f5fbf;--sh:#c2410c;--rd:#7c3aed;--card:#f6f6f8}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa3;--line:#7fb0ff;--sh:#fb923c;--rd:#c4b5fd;--card:#1e1e22}}
body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}
header{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}
.card{background:var(--card);border-radius:8px;padding:8px}
.card h3{margin:0 0 4px;font-size:12px;color:var(--mut);font-weight:500}
svg{width:100%;height:200px;cursor:crosshair;display:block}
.sig{fill:none;stroke:var(--line);stroke-width:1.2}
.mk-shoulder{stroke:var(--sh);stroke-width:2}.mk-rider{stroke:var(--rd);stroke-width:2}
.lbl{font-size:10px}
button,select{font:inherit;padding:4px 10px}
</style></head><body>
<header>
 <strong>Click where a second peak is</strong>
 <label>Mark as <select id="kind"><option value="shoulder">shoulder (merged, no valley)</option><option value="rider">rider (on a tail)</option></select></label>
 <label>Y zoom <select id="zoom"><option>1</option><option>3</option><option>10</option><option>30</option><option>100</option></select></label>
 <button id="save">Save labels</button> <span id="count"></span>
</header>
<p style="color:var(--mut)">Mark every shoulder or rider you would want split, at its own apex or hump. Leave panels with none unmarked: every panel counts as reviewed. Click a mark to remove it.</p>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const marks = {}; // key run|panel -> [{rt,kind}]
let zoom = 1;
function key(r,p){return r+'|'+p}
function draw(){
  const grid=document.getElementById('grid'); grid.innerHTML=''; let n=0;
  DATA.runs.forEach((run,ri)=>run.panels.forEach((pn,pi)=>{
    const card=document.createElement('div'); card.className='card';
    card.innerHTML=`<h3>${run.name} — panel ${pi+1} — ${pn.x[0].toFixed(3)}–${pn.x[pn.x.length-1].toFixed(3)} min</h3>`;
    const W=380,H=200, xs=pn.x, ys=pn.y;
    const x0=xs[0],x1=xs[xs.length-1], ymax=Math.max(...ys)/zoom, ymin=Math.min(0,Math.min(...ys));
    const X=v=>(v-x0)/(x1-x0)*W, Y=v=>H-6-(Math.min(v,ymax)-ymin)/(ymax-ymin)*(H-12);
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
    const pl=document.createElementNS(svg.namespaceURI,'polyline'); pl.setAttribute('class','sig');
    pl.setAttribute('points',xs.map((v,i)=>X(v).toFixed(1)+','+Y(ys[i]).toFixed(1)).join(' ')); svg.appendChild(pl);
    (marks[key(ri,pi)]||[]).forEach((m,mi)=>{ n++;
      const l=document.createElementNS(svg.namespaceURI,'line'); l.setAttribute('class','mk-'+m.kind);
      l.setAttribute('x1',X(m.rt));l.setAttribute('x2',X(m.rt));l.setAttribute('y1',0);l.setAttribute('y2',H);
      l.style.cursor='pointer'; l.addEventListener('click',e=>{e.stopPropagation();marks[key(ri,pi)].splice(mi,1);draw()}); svg.appendChild(l);
      const t=document.createElementNS(svg.namespaceURI,'text'); t.setAttribute('class','lbl'); t.setAttribute('x',X(m.rt)+3); t.setAttribute('y',12);
      t.setAttribute('fill','currentColor'); t.textContent=m.kind[0].toUpperCase()+' '+m.rt.toFixed(3); svg.appendChild(t);
    });
    svg.addEventListener('click',e=>{const r=svg.getBoundingClientRect(); const rt=x0+(e.clientX-r.left)/r.width*(x1-x0);
      (marks[key(ri,pi)]=marks[key(ri,pi)]||[]).push({rt, kind:document.getElementById('kind').value}); draw();});
    card.appendChild(svg); grid.appendChild(card);
  }));
  document.getElementById('count').textContent=n+' marks';
}
document.getElementById('zoom').onchange=e=>{zoom=+e.target.value;draw()};
document.getElementById('save').onclick=()=>{
  const out={runs:DATA.runs.map((run,ri)=>({name:run.name,path:run.path,method:run.method,detector:run.detector,
    panels:run.panels.map(p=>[p.x[0],p.x[p.x.length-1]]),
    labels:run.panels.flatMap((p,pi)=>(marks[key(ri,pi)]||[]).map(m=>({rt:+m.rt.toFixed(4),kind:m.kind})))}))};
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  a.download='labels.json'; a.click();
};
draw();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--method', default=None, help='.chromethod to process with (default: ChromaMethod defaults)')
    ap.add_argument('--min-snr', type=float, default=20.0, help='smallest peak (x noise) that gets a panel')
    ap.add_argument('--detector', default=None, help='channel, e.g. FID1A, FID2B, TCD3C (default: auto)')
    args = ap.parse_args(argv)
    runs = []
    for path in args.runs:
        x, y, _ = load_run(path, args.method, detector=args.detector)
        spans, _ = panels_for_run(x, y, args.min_snr)
        runs.append(dict(name=Path(path).name, path=str(path), method=args.method, detector=args.detector,
                         panels=[dict(x=[round(float(v), 5) for v in x[a:b + 1]],
                                      y=[round(float(v), 3) for v in y[a:b + 1]]) for a, b in spans]))
        print(f'{Path(path).name}: {len(spans)} panels')
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / 'label_sheet.html'
    out.write_text(HTML.replace('__DATA__', json.dumps(dict(runs=runs))))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
