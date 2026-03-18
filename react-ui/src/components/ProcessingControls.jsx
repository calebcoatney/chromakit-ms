/**
 * ProcessingControls Component
 *
 * Full-featured parameter panel mirroring the desktop ParametersFrame.
 * Covers: Smoothing (Whittaker/SavGol), Baseline (9 methods), Peak Detection,
 * Negative Peaks, Shoulder Detection, Range Filters, and Peak Grouping.
 */
import React, { useState, useEffect, useCallback } from 'react';

const LAM_METHODS = new Set(['asls', 'airpls', 'arpls', 'mixture_model', 'irsqr']);

const ProcessingControls = ({ onParametersChange, disabled }) => {
  const [params, setParams] = useState({
    smoothing: {
      enabled: false,
      method: 'whittaker',
      median_enabled: false,
      median_kernel: 5,
      lambda: 1e-1,
      diff_order: 1,
      savgol_window: 3,
      savgol_polyorder: 1,
    },
    baseline: {
      show_corrected: false,
      method: 'arpls',
      lambda: 1e4,
      asymmetry: 0.01,
      align_tic: false,
      break_points: [],
    },
    peaks: {
      enabled: false,
      mode: 'classical',
      min_prominence: 1e5,
      min_height: 0,
      min_width: 0,
      range_filters: [],
    },
    negative_peaks: {
      enabled: false,
      min_prominence: 1e5,
    },
    shoulders: {
      enabled: false,
      window_length: 41,
      polyorder: 3,
      sensitivity: 8,
      apex_distance: 10,
    },
    integration: {
      peak_groups: [],
    },
  });

  // Advanced sections visibility
  const [showShoulderAdvanced, setShowShoulderAdvanced] = useState(false);
  const [showNegativePeaks, setShowNegativePeaks] = useState(false);
  const [showRangeFilters, setShowRangeFilters] = useState(false);
  const [showPeakGrouping, setShowPeakGrouping] = useState(false);

  useEffect(() => {
    if (onParametersChange) onParametersChange(params);
  }, [params]);

  // ── Generic helpers ──

  const updateSection = useCallback((section, updates) => {
    setParams((prev) => ({
      ...prev,
      [section]: { ...prev[section], ...updates },
    }));
  }, []);

  const ensureOdd = (v) => (v % 2 === 0 ? v + 1 : v);

  // ── Computed ──

  const methodUsesLambda = LAM_METHODS.has(params.baseline.method);
  const lambdaExponent = Math.round(Math.log10(params.baseline.lambda));
  const isWhittaker = params.smoothing.method === 'whittaker';

  // ── Range filter / Peak group helpers ──

  const addRangeFilter = () => {
    const current = [...params.peaks.range_filters, [0, 0]];
    updateSection('peaks', { range_filters: current });
  };
  const removeRangeFilter = (idx) => {
    const current = params.peaks.range_filters.filter((_, i) => i !== idx);
    updateSection('peaks', { range_filters: current });
  };
  const updateRangeFilter = (idx, pos, value) => {
    const current = [...params.peaks.range_filters];
    current[idx] = [...current[idx]];
    current[idx][pos] = parseFloat(value) || 0;
    updateSection('peaks', { range_filters: current });
  };

  const addPeakGroup = () => {
    const current = [...params.integration.peak_groups, [0, 0]];
    updateSection('integration', { peak_groups: current });
  };
  const removePeakGroup = (idx) => {
    const current = params.integration.peak_groups.filter((_, i) => i !== idx);
    updateSection('integration', { peak_groups: current });
  };
  const updatePeakGroup = (idx, pos, value) => {
    const current = [...params.integration.peak_groups];
    current[idx] = [...current[idx]];
    current[idx][pos] = parseFloat(value) || 0;
    updateSection('integration', { peak_groups: current });
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>⚙️ Integration Parameters</h2>
      </div>
      <div className="card-body" style={{ maxHeight: 'calc(100vh - 200px)', overflowY: 'auto' }}>

        {/* ─── SIGNAL SMOOTHING ─── */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>Signal Smoothing</legend>

          <div className="form-check mb-2">
            <input type="checkbox" id="smoothing-enabled"
              checked={params.smoothing.enabled}
              onChange={(e) => updateSection('smoothing', { enabled: e.target.checked })}
            />
            <label htmlFor="smoothing-enabled" className="form-label" style={{ marginBottom: 0 }}>
              Enable Smoothing
            </label>
          </div>

          <div style={{ opacity: params.smoothing.enabled ? 1 : 0.4, pointerEvents: params.smoothing.enabled ? 'auto' : 'none' }}>
            {/* Method selector */}
            <div className="form-group">
              <label className="form-label">Method</label>
              <select className="form-control"
                value={params.smoothing.method}
                onChange={(e) => updateSection('smoothing', { method: e.target.value })}
              >
                <option value="whittaker">Whittaker</option>
                <option value="savgol">Savitzky-Golay</option>
              </select>
            </div>

            {/* Median pre-filter */}
            <div className="form-check mb-2">
              <input type="checkbox" id="median-enabled"
                checked={params.smoothing.median_enabled}
                onChange={(e) => updateSection('smoothing', { median_enabled: e.target.checked })}
              />
              <label htmlFor="median-enabled" className="form-label" style={{ marginBottom: 0 }}>
                Median Pre-Filter
              </label>
            </div>
            {params.smoothing.median_enabled && (
              <div className="form-group">
                <label className="form-label">Kernel Size: {params.smoothing.median_kernel}</label>
                <input type="range" className="form-control"
                  min="3" max="31" step="2"
                  value={params.smoothing.median_kernel}
                  onChange={(e) => updateSection('smoothing', { median_kernel: ensureOdd(parseInt(e.target.value)) })}
                />
              </div>
            )}

            {/* Whittaker params */}
            {isWhittaker && (
              <>
                <div className="form-group">
                  <label className="form-label">Lambda: {params.smoothing.lambda.toExponential(1)}</label>
                  <input type="range" className="form-control"
                    min="-3" max="3" step="0.5"
                    value={Math.log10(params.smoothing.lambda)}
                    onChange={(e) => updateSection('smoothing', { lambda: Math.pow(10, parseFloat(e.target.value)) })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Diff Order: {params.smoothing.diff_order}</label>
                  <select className="form-control"
                    value={params.smoothing.diff_order}
                    onChange={(e) => updateSection('smoothing', { diff_order: parseInt(e.target.value) })}
                  >
                    <option value={1}>1 (Slope)</option>
                    <option value={2}>2 (Curvature)</option>
                  </select>
                </div>
              </>
            )}

            {/* Savitzky-Golay params */}
            {!isWhittaker && (
              <>
                <div className="form-group">
                  <label className="form-label">Window: {params.smoothing.savgol_window}</label>
                  <input type="range" className="form-control"
                    min="3" max="51" step="2"
                    value={params.smoothing.savgol_window}
                    onChange={(e) => updateSection('smoothing', { savgol_window: ensureOdd(parseInt(e.target.value)) })}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Polynomial Order: {params.smoothing.savgol_polyorder}</label>
                  <input type="number" className="form-control"
                    min="1" max="5"
                    value={params.smoothing.savgol_polyorder}
                    onChange={(e) => updateSection('smoothing', { savgol_polyorder: parseInt(e.target.value) })}
                  />
                </div>
              </>
            )}
          </div>
        </fieldset>

        {/* ─── BASELINE CORRECTION ─── */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>Baseline Correction</legend>

          <div className="form-check mb-2">
            <input type="checkbox" id="baseline-show-corrected"
              checked={params.baseline.show_corrected}
              onChange={(e) => updateSection('baseline', { show_corrected: e.target.checked })}
            />
            <label htmlFor="baseline-show-corrected" className="form-label" style={{ marginBottom: 0 }}>
              Show Corrected Signal
            </label>
          </div>

          <div className="form-group">
            <label className="form-label">Algorithm</label>
            <select className="form-control"
              value={params.baseline.method}
              onChange={(e) => updateSection('baseline', { method: e.target.value })}
            >
              <option value="asls">ASLS – Asymmetric Least Squares</option>
              <option value="arpls">ARPLS – Asymmetrically Reweighted</option>
              <option value="airpls">AirPLS – Adaptive Iteratively Reweighted</option>
              <option value="imodpoly">IModPoly – Improved Modified Polynomial</option>
              <option value="modpoly">ModPoly – Modified Polynomial</option>
              <option value="snip">SNIP – Statistics-sensitive Non-linear</option>
              <option value="mixture_model">Mixture Model (spline)</option>
              <option value="irsqr">IRSQR – Reweighted Spline Quantile</option>
              <option value="fastchrom">FastChrom</option>
            </select>
          </div>

          {methodUsesLambda && (
            <div className="form-group">
              <label className="form-label">
                Lambda (λ): 10<sup>{lambdaExponent}</sup>
              </label>
              <input type="range" className="form-control"
                min="1" max="12" step="1"
                value={lambdaExponent}
                onChange={(e) => updateSection('baseline', { lambda: Math.pow(10, parseInt(e.target.value)) })}
              />
            </div>
          )}

          <div className="form-check mb-2">
            <input type="checkbox" id="align-tic"
              checked={params.baseline.align_tic}
              onChange={(e) => updateSection('baseline', { align_tic: e.target.checked })}
            />
            <label htmlFor="align-tic" className="form-label" style={{ marginBottom: 0 }}>
              Align TIC to FID
            </label>
          </div>
        </fieldset>

        {/* ─── PEAK DETECTION ─── */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>Peak Detection</legend>

          <div className="form-check mb-2">
            <input type="checkbox" id="peaks-enabled"
              checked={params.peaks.enabled}
              onChange={(e) => updateSection('peaks', { enabled: e.target.checked })}
            />
            <label htmlFor="peaks-enabled" className="form-label" style={{ marginBottom: 0 }}>
              Enable Peak Detection
            </label>
          </div>

          <div style={{ opacity: params.peaks.enabled ? 1 : 0.4, pointerEvents: params.peaks.enabled ? 'auto' : 'none' }}>
            <div className="form-group">
              <label className="form-label">Min Prominence</label>
              <input type="text" className="form-control"
                value={params.peaks.min_prominence}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!isNaN(v) && v >= 0) updateSection('peaks', { min_prominence: v });
                }}
                placeholder="e.g. 1e5"
              />
            </div>
          </div>
        </fieldset>

        {/* ─── NEGATIVE PEAKS ─── */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>
            <span style={{ cursor: 'pointer' }} onClick={() => setShowNegativePeaks(!showNegativePeaks)}>
              {showNegativePeaks ? '▼' : '▶'} Negative Peaks
            </span>
          </legend>
          {showNegativePeaks && (
            <>
              <div className="form-check mb-2">
                <input type="checkbox" id="neg-peaks-enabled"
                  checked={params.negative_peaks.enabled}
                  onChange={(e) => updateSection('negative_peaks', { enabled: e.target.checked })}
                />
                <label htmlFor="neg-peaks-enabled" className="form-label" style={{ marginBottom: 0 }}>
                  Enable Negative Peak Detection
                </label>
              </div>
              {params.negative_peaks.enabled && (
                <div className="form-group">
                  <label className="form-label">Min Prominence</label>
                  <input type="text" className="form-control"
                    value={params.negative_peaks.min_prominence}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v) && v >= 0) updateSection('negative_peaks', { min_prominence: v });
                    }}
                  />
                </div>
              )}
            </>
          )}
        </fieldset>

        {/* ─── SHOULDER DETECTION ─── */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>
            <span style={{ cursor: 'pointer' }} onClick={() => setShowShoulderAdvanced(!showShoulderAdvanced)}>
              {showShoulderAdvanced ? '▼' : '▶'} Shoulder Detection
            </span>
          </legend>
          {showShoulderAdvanced && (
            <>
              <div className="form-check mb-2">
                <input type="checkbox" id="shoulders-enabled"
                  checked={params.shoulders.enabled}
                  onChange={(e) => updateSection('shoulders', { enabled: e.target.checked })}
                />
                <label htmlFor="shoulders-enabled" className="form-label" style={{ marginBottom: 0 }}>
                  Enable Shoulder Detection
                </label>
              </div>

              <div style={{ opacity: params.shoulders.enabled ? 1 : 0.4, pointerEvents: params.shoulders.enabled ? 'auto' : 'none' }}>
                <div className="form-group">
                  <label className="form-label">Sensitivity: {params.shoulders.sensitivity}</label>
                  <input type="range" className="form-control"
                    min="1" max="10" step="1"
                    value={params.shoulders.sensitivity}
                    onChange={(e) => updateSection('shoulders', { sensitivity: parseInt(e.target.value) })}
                  />
                  <div className="text-muted" style={{ fontSize: '0.7rem' }}>Higher = more detections</div>
                </div>

                <div className="form-group">
                  <label className="form-label">Window Length: {params.shoulders.window_length}</label>
                  <input type="range" className="form-control"
                    min="5" max="101" step="2"
                    value={params.shoulders.window_length}
                    onChange={(e) => updateSection('shoulders', { window_length: ensureOdd(parseInt(e.target.value)) })}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Polynomial Order: {params.shoulders.polyorder}</label>
                  <input type="number" className="form-control"
                    min="1" max="5"
                    value={params.shoulders.polyorder}
                    onChange={(e) => updateSection('shoulders', { polyorder: parseInt(e.target.value) })}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Apex Distance: {params.shoulders.apex_distance}</label>
                  <input type="range" className="form-control"
                    min="1" max="50" step="1"
                    value={params.shoulders.apex_distance}
                    onChange={(e) => updateSection('shoulders', { apex_distance: parseInt(e.target.value) })}
                  />
                </div>
              </div>
            </>
          )}
        </fieldset>

        {/* ─── RANGE FILTERS ─── */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>
            <span style={{ cursor: 'pointer' }} onClick={() => setShowRangeFilters(!showRangeFilters)}>
              {showRangeFilters ? '▼' : '▶'} Range Filters
            </span>
          </legend>
          {showRangeFilters && (
            <>
              <div className="text-muted mb-2" style={{ fontSize: '0.75rem' }}>
                Only keep peaks within these time ranges (minutes).
              </div>
              {params.peaks.range_filters.map((rf, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', alignItems: 'center' }}>
                  <input type="number" className="form-control" style={{ width: '80px' }}
                    step="0.1" value={rf[0]} placeholder="Start"
                    onChange={(e) => updateRangeFilter(idx, 0, e.target.value)}
                  />
                  <span>–</span>
                  <input type="number" className="form-control" style={{ width: '80px' }}
                    step="0.1" value={rf[1]} placeholder="End"
                    onChange={(e) => updateRangeFilter(idx, 1, e.target.value)}
                  />
                  <button className="btn btn-sm btn-danger" onClick={() => removeRangeFilter(idx)}>✕</button>
                </div>
              ))}
              <button className="btn btn-sm btn-secondary" onClick={addRangeFilter}>+ Add Range</button>
            </>
          )}
        </fieldset>

        {/* ─── PEAK GROUPING ─── */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>
            <span style={{ cursor: 'pointer' }} onClick={() => setShowPeakGrouping(!showPeakGrouping)}>
              {showPeakGrouping ? '▼' : '▶'} Peak Grouping
            </span>
          </legend>
          {showPeakGrouping && (
            <>
              <div className="text-muted mb-2" style={{ fontSize: '0.75rem' }}>
                Group peaks within time windows into single integration regions.
              </div>
              {params.integration.peak_groups.map((pg, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', alignItems: 'center' }}>
                  <input type="number" className="form-control" style={{ width: '80px' }}
                    step="0.1" value={pg[0]} placeholder="Start"
                    onChange={(e) => updatePeakGroup(idx, 0, e.target.value)}
                  />
                  <span>–</span>
                  <input type="number" className="form-control" style={{ width: '80px' }}
                    step="0.1" value={pg[1]} placeholder="End"
                    onChange={(e) => updatePeakGroup(idx, 1, e.target.value)}
                  />
                  <button className="btn btn-sm btn-danger" onClick={() => removePeakGroup(idx)}>✕</button>
                </div>
              ))}
              <button className="btn btn-sm btn-secondary" onClick={addPeakGroup}>+ Add Group</button>
            </>
          )}
        </fieldset>

      </div>
    </div>
  );
};

const fieldsetStyle = {
  border: '1px solid var(--border-color)',
  borderRadius: '6px',
  padding: '1rem',
  marginBottom: '1rem',
};

const legendStyle = {
  fontWeight: 600,
  fontSize: '0.9rem',
  padding: '0 0.5rem',
  color: 'var(--text-color)',
};

export default ProcessingControls;
