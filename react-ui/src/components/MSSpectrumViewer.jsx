/**
 * MSSpectrumViewer Component
 *
 * Interactive mass spectrum viewer mirroring the desktop MSFrame:
 * - Stick plot of m/z vs intensity
 * - RT input with "Extract" button
 * - m/z shift control
 * - Peak click → auto-extract spectrum
 * - MS library search results display
 */
import React, { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';

const MSSpectrumViewer = ({
  spectrum,          // { rt, mz: number[], intensities: number[] } | null
  searchResults,     // MSSearchResult[] | null
  selectedPeakRT,    // auto-fill RT from peak selection
  hasMS,             // whether file has MS data
  onExtractSpectrum, // (rt: number) => void
  onSearchSpectrum,  // () => void
  searching,
  disabled,
}) => {
  const [rtInput, setRtInput] = useState('');
  const [mzShift, setMzShift] = useState(0);

  // Sync RT input when a peak is selected
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

  // Normalize intensities for display
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

  // Build Plotly traces — stick plot using thin bars
  const traces = useMemo(() => {
    if (!normalizedData) return [];
    return [{
      x: normalizedData.mz,
      y: normalizedData.intensities,
      type: 'bar',
      width: 0.6,
      marker: { color: '#3182ce' },
      hovertemplate: 'm/z: %{x}<br>Intensity: %{y:.1f}%<extra></extra>',
    }];
  }, [normalizedData]);

  const layout = useMemo(() => {
    const minMz = normalizedData ? Math.max(1, Math.min(...normalizedData.mz) - 5) : 1;
    const maxMz = normalizedData ? Math.max(...normalizedData.mz) + 5 : 150;
    return {
      autosize: true,
      height: 250,
      margin: { l: 40, r: 15, t: 30, b: 40 },
      xaxis: {
        title: 'm/z',
        range: [minMz, maxMz],
        showgrid: true, gridcolor: '#edf2f7',
      },
      yaxis: {
        title: 'Rel. Intensity (%)',
        range: [0, 105],
        showgrid: false,
        showticklabels: false,
      },
      title: spectrum?.rt != null
        ? { text: `Spectrum at RT ${spectrum.rt.toFixed(3)} min${mzShift ? ` (shift: ${mzShift})` : ''}`, font: { size: 12 } }
        : undefined,
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      bargap: 0.1,
    };
  }, [normalizedData, spectrum, mzShift]);

  if (!hasMS) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2>🔬 Mass Spectrum</h2>
      </div>
      <div className="card-body">

        {/* RT input + controls */}
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
          <label className="form-label" style={{ marginBottom: 0, whiteSpace: 'nowrap' }}>RT:</label>
          <input
            type="text"
            className="form-control"
            style={{ width: '90px' }}
            value={rtInput}
            onChange={(e) => setRtInput(e.target.value)}
            placeholder="e.g. 5.123"
            onKeyDown={(e) => e.key === 'Enter' && handleExtract()}
          />
          <button className="btn btn-sm btn-primary" onClick={handleExtract} disabled={disabled || !rtInput}>
            Extract
          </button>

          <span style={{ margin: '0 0.25rem', color: '#a0aec0' }}>|</span>

          <label className="form-label" style={{ marginBottom: 0, whiteSpace: 'nowrap' }}>m/z shift:</label>
          <input
            type="number"
            className="form-control"
            style={{ width: '65px' }}
            value={mzShift}
            onChange={(e) => setMzShift(parseInt(e.target.value) || 0)}
          />

          {spectrum && onSearchSpectrum && (
            <>
              <span style={{ margin: '0 0.25rem', color: '#a0aec0' }}>|</span>
              <button
                className="btn btn-sm btn-secondary"
                onClick={onSearchSpectrum}
                disabled={searching || disabled}
              >
                {searching ? '⏳ Searching...' : '🔍 Search Library'}
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
          <div className="text-center text-muted" style={{ padding: '2rem', fontSize: '0.9rem' }}>
            Enter a retention time or click a peak to view its mass spectrum
          </div>
        )}

        {/* Search results */}
        {searchResults && searchResults.length > 0 && (
          <div style={{ marginTop: '0.75rem' }}>
            <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.5rem' }}>
              Library Search Results
            </div>
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border-color)', background: '#f7fafc' }}>
                    <th style={thStyle}>Compound</th>
                    <th style={{ ...thStyle, width: '70px' }}>Score</th>
                    <th style={{ ...thStyle, width: '90px' }}>CAS#</th>
                    <th style={{ ...thStyle, width: '60px' }}>MW</th>
                  </tr>
                </thead>
                <tbody>
                  {searchResults.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={tdStyle}>{r.compound_name || r.name}</td>
                      <td style={tdStyle}>{(r.match_score ?? r.score ?? 0).toFixed(0)}</td>
                      <td style={tdStyle}>{r.cas_number || r.cas || '—'}</td>
                      <td style={tdStyle}>{r.molecular_weight?.toFixed(1) ?? '—'}</td>
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

const thStyle = { padding: '0.4rem 0.5rem', textAlign: 'left', fontWeight: 600 };
const tdStyle = { padding: '0.4rem 0.5rem' };

export default MSSpectrumViewer;
