/**
 * MSSpectrumViewer Component — interactive mass spectrum stick plot with search.
 */
import React, { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const MSSpectrumViewer = ({
  spectrum,
  searchResults,
  selectedPeakRT,
  hasMS,
  onExtractSpectrum,
  onSearchSpectrum,
  searching,
  disabled,
}) => {
  const [rtInput, setRtInput] = useState('');
  const [mzShift, setMzShift] = useState(0);

  React.useEffect(() => {
    if (selectedPeakRT != null) {
      setRtInput(selectedPeakRT.toFixed(3));
    }
  }, [selectedPeakRT]);

  const handleExtract = useCallback(() => {
    const rt = parseFloat(rtInput);
    if (!isNaN(rt) && rt > 0 && onExtractSpectrum) {
      onExtractSpectrum(rt);
    }
  }, [rtInput, onExtractSpectrum]);

  const normalizedData = useMemo(() => {
    if (!spectrum?.mz?.length) return null;
    const maxI = Math.max(...spectrum.intensities);
    const norm = maxI > 0
      ? spectrum.intensities.map(i => (i / maxI) * 100)
      : spectrum.intensities;
    const shiftedMz = mzShift !== 0
      ? spectrum.mz.map(m => m + mzShift)
      : spectrum.mz;
    return { mz: shiftedMz, intensities: norm };
  }, [spectrum, mzShift]);

  const traces = useMemo(() => {
    if (!normalizedData) return [];
    return [{
      x: normalizedData.mz,
      y: normalizedData.intensities,
      type: 'bar',
      width: 0.6,
      marker: { color: cssVar('--primary-color') || '#4f6bed' },
      hovertemplate: 'm/z: %{x}<br>Intensity: %{y:.1f}%<extra></extra>',
    }];
  }, [normalizedData]);

  const layout = useMemo(() => {
    const plotBg = cssVar('--plot-bg') || 'white';
    const paperBg = cssVar('--plot-paper') || 'white';
    const gridColor = cssVar('--plot-grid') || '#eef1f5';
    const textColor = cssVar('--plot-text') || '#4a5568';
    const minMz = normalizedData ? Math.max(1, Math.min(...normalizedData.mz) - 5) : 1;
    const maxMz = normalizedData ? Math.max(...normalizedData.mz) + 5 : 150;
    return {
      autosize: true,
      height: 220,
      margin: { l: 35, r: 10, t: 25, b: 35 },
      xaxis: {
        title: { text: 'm/z', font: { size: 10, color: textColor } },
        range: [minMz, maxMz],
        showgrid: true, gridcolor: gridColor,
        tickfont: { size: 9, color: textColor },
      },
      yaxis: {
        title: { text: 'Rel. %', font: { size: 10, color: textColor } },
        range: [0, 105],
        showgrid: false,
        showticklabels: false,
      },
      title: spectrum?.rt != null
        ? { text: `RT ${spectrum.rt.toFixed(3)} min${mzShift ? ` (shift: ${mzShift})` : ''}`, font: { size: 11, color: textColor } }
        : undefined,
      plot_bgcolor: plotBg,
      paper_bgcolor: paperBg,
      bargap: 0.1,
    };
  }, [normalizedData, spectrum, mzShift]);

  if (!hasMS) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <h2>Mass Spectrum</h2>
      </div>
      <div className="card-body">
        {/* Controls row */}
        <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
          <label className="form-label" style={{ marginBottom: 0, whiteSpace: 'nowrap' }}>RT:</label>
          <input type="text" className="form-control" style={{ width: '80px' }}
            value={rtInput} onChange={(e) => setRtInput(e.target.value)}
            placeholder="5.123" onKeyDown={(e) => e.key === 'Enter' && handleExtract()} />
          <button className="btn btn-sm btn-primary" onClick={handleExtract} disabled={disabled || !rtInput}>
            Extract
          </button>
          <div className="btn-divider" />
          <label className="form-label" style={{ marginBottom: 0, whiteSpace: 'nowrap' }}>m/z shift:</label>
          <input type="number" className="form-control" style={{ width: '55px' }}
            value={mzShift} onChange={(e) => setMzShift(parseInt(e.target.value) || 0)} />
          {spectrum && onSearchSpectrum && (
            <>
              <div className="btn-divider" />
              <button className="btn btn-sm btn-secondary" onClick={onSearchSpectrum} disabled={searching || disabled}>
                {searching ? 'Searching...' : 'Search Library'}
              </button>
            </>
          )}
        </div>

        {/* Spectrum plot */}
        {normalizedData ? (
          <Plot
            data={traces}
            layout={layout}
            config={{
              responsive: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ['lasso2d', 'select2d', 'pan2d'],
              displaylogo: false,
              toImageButtonOptions: {
                format: 'png', filename: 'mass_spectrum',
                height: 500, width: 800, scale: 2,
              },
            }}
            style={{ width: '100%' }}
          />
        ) : (
          <div className="text-center text-muted" style={{ padding: '1.5rem' }}>
            Enter a retention time or click a peak to view its mass spectrum
          </div>
        )}

        {/* Search results */}
        {searchResults?.length > 0 && (
          <div className="mt-2">
            <div style={{ fontWeight: 600, fontSize: '0.8rem', marginBottom: '0.375rem' }}>
              Library Search Results
            </div>
            <div style={{ maxHeight: '180px', overflowY: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Compound</th>
                    <th style={{ width: '60px' }}>Score</th>
                    <th style={{ width: '80px' }}>CAS#</th>
                    <th style={{ width: '50px' }}>MW</th>
                  </tr>
                </thead>
                <tbody>
                  {searchResults.map((r, i) => (
                    <tr key={i}>
                      <td>{r.compound_name || r.name}</td>
                      <td>{(r.match_score ?? r.score ?? 0).toFixed(0)}</td>
                      <td>{r.cas_number || r.cas || '\u2014'}</td>
                      <td>{r.molecular_weight?.toFixed(1) ?? '\u2014'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MSSpectrumViewer;
