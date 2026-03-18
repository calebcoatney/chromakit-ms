/**
 * ProcessingControls Component
 *
 * Full-featured parameter panel mirroring the desktop ParametersFrame.
 * Covers: Smoothing (Whittaker/SavGol), Baseline (9 methods + FastChrom + break points),
 * Peak Detection (classical + deconvolution), Negative Peaks, Shoulder Detection,
 * Range Filters, and Peak Grouping.
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
      fastchrom: { half_window: 0, smooth_half_window: 0 },
    },
    peaks: {
      enabled: false,
      mode: 'classical',
      min_prominence: 1e5,
      min_height: 0,
      min_width: 0,
      range_filters: [],
    },
    deconvolution: {
      splitting_method: 'geometric',
      windows: [],
      heatmap_threshold: 0.36,
      pre_fit_signal_threshold: 0.001,
      min_area_frac: 0.15,
      valley_threshold_frac: 0.48,
      mu_bound_factor: 0.68,
      fat_threshold_frac: 0.44,
      dedup_sigma_factor: 1.32,
      dedup_rt_tolerance: 0.005,
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

  // Collapsible sections
  const [showShoulderAdvanced, setShowShoulderAdvanced] = useState(false);
  const [showNegativePeaks, setShowNegativePeaks] = useState(false);
  const [showRangeFilters, setShowRangeFilters] = useState(false);
  const [showPeakGrouping, setShowPeakGrouping] = useState(false);
  const [showBreakPoints, setShowBreakPoints] = useState(false);
  const [showDeconvAdvanced, setShowDeconvAdvanced] = useState(false);

  useEffect(() => {
    if (onParametersChange) onParametersChange(params);
  }, [params]);

  const updateSection = useCallback((section, updates) => {
    setParams((prev) => ({
      ...prev,
      [section]: { ...prev[section], ...updates },
    }));
  }, []);

  const ensureOdd = (v) => (v % 2 === 0 ? v + 1 : v);

  // Computed
  const methodUsesLambda = LAM_METHODS.has(params.baseline.method);
  const isFastchrom = params.baseline.method === 'fastchrom';
  const lambdaExponent = Math.round(Math.log10(params.baseline.lambda));
  const isWhittaker = params.smoothing.method === 'whittaker';
  const isDeconv = params.peaks.mode === 'deconvolution';
  const isEMG = params.deconvolution.splitting_method === 'emg';

  // Range filter helpers
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

  // Peak group helpers
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

  // Break point helpers
  const addBreakPoint = () => {
    const bp = [...params.baseline.break_points, { time: 0, tolerance: 0.5 }];
    updateSection('baseline', { break_points: bp });
  };
  const removeBreakPoint = (idx) => {
    const bp = params.baseline.break_points.filter((_, i) => i !== idx);
    updateSection('baseline', { break_points: bp });
  };
  const updateBreakPoint = (idx, key, value) => {
    const bp = [...params.baseline.break_points];
    bp[idx] = { ...bp[idx], [key]: parseFloat(value) || 0 };
    updateSection('baseline', { break_points: bp });
  };

  // Deconvolution window helpers
  const addDeconvWindow = () => {
    const w = [...params.deconvolution.windows, [0, 0]];
    updateSection('deconvolution', { windows: w });
  };
  const removeDeconvWindow = (idx) => {
    const w = params.deconvolution.windows.filter((_, i) => i !== idx);
    updateSection('deconvolution', { windows: w });
  };
  const updateDeconvWindow = (idx, pos, value) => {
    const w = [...params.deconvolution.windows];
    w[idx] = [...w[idx]];
    w[idx][pos] = parseFloat(value) || 0;
    updateSection('deconvolution', { windows: w });
  };

  const resetDeconvDefaults = () => {
    updateSection('deconvolution', {
      heatmap_threshold: 0.36,
      pre_fit_signal_threshold: 0.001,
      min_area_frac: 0.15,
      valley_threshold_frac: 0.48,
      mu_bound_factor: 0.68,
      fat_threshold_frac: 0.44,
      dedup_sigma_factor: 1.32,
      dedup_rt_tolerance: 0.005,
    });
  };

  return (
    <div style={{ padding: '0.75rem' }}>

      {/* ─── SIGNAL SMOOTHING ─── */}
      <fieldset style={fieldsetStyle}>
        <legend style={legendStyle}>Signal Smoothing</legend>

        <div className="form-check mb-2">
          <input type="checkbox" id="smoothing-enabled"
            checked={params.smoothing.enabled}
            onChange={(e) => updateSection('smoothing', { enabled: e.target.checked })} />
          <label htmlFor="smoothing-enabled" className="form-label" style={{ marginBottom: 0 }}>Enable</label>
        </div>

        <div style={{ opacity: params.smoothing.enabled ? 1 : 0.4, pointerEvents: params.smoothing.enabled ? 'auto' : 'none' }}>
          <div className="form-group">
            <label className="form-label">Method</label>
            <select className="form-control" value={params.smoothing.method}
              onChange={(e) => updateSection('smoothing', { method: e.target.value })}>
              <option value="whittaker">Whittaker</option>
              <option value="savgol">Savitzky-Golay</option>
            </select>
          </div>

          <div className="form-check mb-2">
            <input type="checkbox" id="median-enabled" checked={params.smoothing.median_enabled}
              onChange={(e) => updateSection('smoothing', { median_enabled: e.target.checked })} />
            <label htmlFor="median-enabled" className="form-label" style={{ marginBottom: 0 }}>Median Pre-Filter</label>
          </div>
          {params.smoothing.median_enabled && (
            <div className="form-group">
              <label className="form-label">Kernel: {params.smoothing.median_kernel}</label>
              <input type="range" className="form-control" min="3" max="31" step="2"
                value={params.smoothing.median_kernel}
                onChange={(e) => updateSection('smoothing', { median_kernel: ensureOdd(parseInt(e.target.value)) })} />
            </div>
          )}

          {isWhittaker ? (
            <>
              <div className="form-group">
                <label className="form-label">Lambda: {params.smoothing.lambda.toExponential(1)}</label>
                <input type="range" className="form-control" min="-3" max="6" step="0.5"
                  value={Math.log10(params.smoothing.lambda)}
                  onChange={(e) => updateSection('smoothing', { lambda: Math.pow(10, parseFloat(e.target.value)) })} />
              </div>
              <div className="form-group">
                <label className="form-label">Diff Order</label>
                <select className="form-control" value={params.smoothing.diff_order}
                  onChange={(e) => updateSection('smoothing', { diff_order: parseInt(e.target.value) })}>
                  <option value={1}>d=1 (Slope)</option>
                  <option value={2}>d=2 (Curvature)</option>
                </select>
              </div>
            </>
          ) : (
            <>
              <div className="form-group">
                <label className="form-label">Window: {params.smoothing.savgol_window}</label>
                <input type="range" className="form-control" min="3" max="51" step="2"
                  value={params.smoothing.savgol_window}
                  onChange={(e) => updateSection('smoothing', { savgol_window: ensureOdd(parseInt(e.target.value)) })} />
              </div>
              <div className="form-group">
                <label className="form-label">Polynomial Order: {params.smoothing.savgol_polyorder}</label>
                <input type="range" className="form-control" min="1" max="5"
                  value={params.smoothing.savgol_polyorder}
                  onChange={(e) => updateSection('smoothing', { savgol_polyorder: parseInt(e.target.value) })} />
              </div>
            </>
          )}
        </div>
      </fieldset>

      {/* ─── BASELINE CORRECTION ─── */}
      <fieldset style={fieldsetStyle}>
        <legend style={legendStyle}>Baseline Correction</legend>

        <div className="form-check mb-2">
          <input type="checkbox" id="baseline-show-corrected" checked={params.baseline.show_corrected}
            onChange={(e) => updateSection('baseline', { show_corrected: e.target.checked })} />
          <label htmlFor="baseline-show-corrected" className="form-label" style={{ marginBottom: 0 }}>Show Corrected Signal</label>
        </div>

        <div className="form-group">
          <label className="form-label">Algorithm</label>
          <select className="form-control" value={params.baseline.method}
            onChange={(e) => updateSection('baseline', { method: e.target.value })}>
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
            <label className="form-label">Lambda (λ): 10<sup>{lambdaExponent}</sup></label>
            <input type="range" className="form-control" min="2" max="12" step="1"
              value={lambdaExponent}
              onChange={(e) => updateSection('baseline', { lambda: Math.pow(10, parseInt(e.target.value)) })} />
          </div>
        )}

        {isFastchrom && (
          <>
            <div className="form-group">
              <label className="form-label">Half Window (0=Auto): {params.baseline.fastchrom.half_window}</label>
              <input type="number" className="form-control" min="0" max="500"
                value={params.baseline.fastchrom.half_window}
                onChange={(e) => updateSection('baseline', {
                  fastchrom: { ...params.baseline.fastchrom, half_window: parseInt(e.target.value) || 0 }
                })} />
            </div>
            <div className="form-group">
              <label className="form-label">Smooth Half Window (0=Auto): {params.baseline.fastchrom.smooth_half_window}</label>
              <input type="number" className="form-control" min="0" max="500"
                value={params.baseline.fastchrom.smooth_half_window}
                onChange={(e) => updateSection('baseline', {
                  fastchrom: { ...params.baseline.fastchrom, smooth_half_window: parseInt(e.target.value) || 0 }
                })} />
            </div>
          </>
        )}

        <div className="form-check mb-2">
          <input type="checkbox" id="align-tic" checked={params.baseline.align_tic}
            onChange={(e) => updateSection('baseline', { align_tic: e.target.checked })} />
          <label htmlFor="align-tic" className="form-label" style={{ marginBottom: 0 }}>Align TIC to FID</label>
        </div>

        {/* Break points */}
        <div style={{ marginTop: '0.5rem' }}>
          <span style={{ cursor: 'pointer', fontSize: '0.85rem', fontWeight: 500 }}
            onClick={() => setShowBreakPoints(!showBreakPoints)}>
            {showBreakPoints ? '▼' : '▶'} Break Points
          </span>
          {showBreakPoints && (
            <div style={{ marginTop: '0.5rem' }}>
              <div className="text-muted mb-2" style={{ fontSize: '0.75rem' }}>
                Segment the baseline at specific time points.
              </div>
              {params.baseline.break_points.map((bp, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.4rem', alignItems: 'center' }}>
                  <input type="number" className="form-control" style={{ width: '70px' }}
                    step="0.1" value={bp.time} placeholder="Time"
                    onChange={(e) => updateBreakPoint(idx, 'time', e.target.value)} />
                  <span style={{ fontSize: '0.75rem' }}>±</span>
                  <input type="number" className="form-control" style={{ width: '60px' }}
                    step="0.1" value={bp.tolerance} placeholder="Tol"
                    onChange={(e) => updateBreakPoint(idx, 'tolerance', e.target.value)} />
                  <button className="btn btn-sm btn-danger" onClick={() => removeBreakPoint(idx)}>✕</button>
                </div>
              ))}
              <button className="btn btn-sm btn-secondary" onClick={addBreakPoint}>+ Add</button>
            </div>
          )}
        </div>
      </fieldset>

      {/* ─── PEAK DETECTION ─── */}
      <fieldset style={fieldsetStyle}>
        <legend style={legendStyle}>Peak Detection</legend>

        <div className="form-check mb-2">
          <input type="checkbox" id="peaks-enabled" checked={params.peaks.enabled}
            onChange={(e) => updateSection('peaks', { enabled: e.target.checked })} />
          <label htmlFor="peaks-enabled" className="form-label" style={{ marginBottom: 0 }}>Enable</label>
        </div>

        <div style={{ opacity: params.peaks.enabled ? 1 : 0.4, pointerEvents: params.peaks.enabled ? 'auto' : 'none' }}>
          <div className="form-group">
            <label className="form-label">Mode</label>
            <select className="form-control" value={params.peaks.mode}
              onChange={(e) => updateSection('peaks', { mode: e.target.value })}>
              <option value="classical">Classical</option>
              <option value="deconvolution">Deconvolution</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Min Prominence</label>
            <input type="text" className="form-control" value={params.peaks.min_prominence}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v) && v >= 0) updateSection('peaks', { min_prominence: v });
              }} placeholder="e.g. 1e5" />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            <div className="form-group">
              <label className="form-label">Min Height</label>
              <input type="text" className="form-control" value={params.peaks.min_height}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!isNaN(v) && v >= 0) updateSection('peaks', { min_height: v });
                }} placeholder="0" />
            </div>
            <div className="form-group">
              <label className="form-label">Min Width</label>
              <input type="text" className="form-control" value={params.peaks.min_width}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!isNaN(v) && v >= 0) updateSection('peaks', { min_width: v });
                }} placeholder="0" />
            </div>
          </div>

          {/* Deconvolution controls */}
          {isDeconv && (
            <div style={{ marginTop: '0.5rem', padding: '0.5rem', background: 'var(--hover-bg)', borderRadius: '6px' }}>
              <div className="form-group">
                <label className="form-label">Splitting Method</label>
                <select className="form-control" value={params.deconvolution.splitting_method}
                  onChange={(e) => updateSection('deconvolution', { splitting_method: e.target.value })}>
                  <option value="geometric">Geometric</option>
                  <option value="emg">EMG (Exponentially Modified Gaussian)</option>
                </select>
              </div>

              {/* Deconv windows */}
              <div className="text-muted mb-1" style={{ fontSize: '0.75rem' }}>Deconvolution Windows (optional)</div>
              {params.deconvolution.windows.map((w, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.4rem', alignItems: 'center' }}>
                  <input type="number" className="form-control" style={{ width: '70px' }}
                    step="0.1" value={w[0]} onChange={(e) => updateDeconvWindow(idx, 0, e.target.value)} />
                  <span>–</span>
                  <input type="number" className="form-control" style={{ width: '70px' }}
                    step="0.1" value={w[1]} onChange={(e) => updateDeconvWindow(idx, 1, e.target.value)} />
                  <button className="btn btn-sm btn-danger" onClick={() => removeDeconvWindow(idx)}>✕</button>
                </div>
              ))}
              <button className="btn btn-sm btn-secondary" onClick={addDeconvWindow} style={{ marginBottom: '0.5rem' }}>+ Add Window</button>

              {/* Advanced deconv */}
              <div>
                <span style={{ cursor: 'pointer', fontSize: '0.8rem', fontWeight: 500 }}
                  onClick={() => setShowDeconvAdvanced(!showDeconvAdvanced)}>
                  {showDeconvAdvanced ? '▼' : '▶'} Advanced
                </span>
                {showDeconvAdvanced && (
                  <div style={{ marginTop: '0.5rem' }}>
                    <div className="form-group">
                      <label className="form-label">Heatmap Threshold: {params.deconvolution.heatmap_threshold}</label>
                      <input type="range" className="form-control" min="0.10" max="0.50" step="0.02"
                        value={params.deconvolution.heatmap_threshold}
                        onChange={(e) => updateSection('deconvolution', { heatmap_threshold: parseFloat(e.target.value) })} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Pre-Fit Signal (%): {(params.deconvolution.pre_fit_signal_threshold * 100).toFixed(1)}</label>
                      <input type="range" className="form-control" min="0" max="5" step="0.1"
                        value={params.deconvolution.pre_fit_signal_threshold * 100}
                        onChange={(e) => updateSection('deconvolution', { pre_fit_signal_threshold: parseFloat(e.target.value) / 100 })} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Min Area Frac: {params.deconvolution.min_area_frac}</label>
                      <input type="range" className="form-control" min="0" max="0.30" step="0.01"
                        value={params.deconvolution.min_area_frac}
                        onChange={(e) => updateSection('deconvolution', { min_area_frac: parseFloat(e.target.value) })} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Valley Threshold: {params.deconvolution.valley_threshold_frac}</label>
                      <input type="range" className="form-control" min="0.20" max="0.80" step="0.05"
                        value={params.deconvolution.valley_threshold_frac}
                        onChange={(e) => updateSection('deconvolution', { valley_threshold_frac: parseFloat(e.target.value) })} />
                    </div>
                    {isEMG && (
                      <>
                        <div className="form-group">
                          <label className="form-label">μ Bound Factor: {params.deconvolution.mu_bound_factor}</label>
                          <input type="range" className="form-control" min="0.5" max="3.0" step="0.1"
                            value={params.deconvolution.mu_bound_factor}
                            onChange={(e) => updateSection('deconvolution', { mu_bound_factor: parseFloat(e.target.value) })} />
                        </div>
                        <div className="form-group">
                          <label className="form-label">Fat Threshold: {params.deconvolution.fat_threshold_frac}</label>
                          <input type="range" className="form-control" min="0.20" max="0.80" step="0.05"
                            value={params.deconvolution.fat_threshold_frac}
                            onChange={(e) => updateSection('deconvolution', { fat_threshold_frac: parseFloat(e.target.value) })} />
                        </div>
                        <div className="form-group">
                          <label className="form-label">Dedup σ Factor: {params.deconvolution.dedup_sigma_factor}</label>
                          <input type="range" className="form-control" min="0" max="2.0" step="0.1"
                            value={params.deconvolution.dedup_sigma_factor}
                            onChange={(e) => updateSection('deconvolution', { dedup_sigma_factor: parseFloat(e.target.value) })} />
                        </div>
                      </>
                    )}
                    {!isEMG && (
                      <div className="form-group">
                        <label className="form-label">Dedup RT Tolerance: {params.deconvolution.dedup_rt_tolerance}</label>
                        <input type="range" className="form-control" min="0" max="0.1" step="0.005"
                          value={params.deconvolution.dedup_rt_tolerance}
                          onChange={(e) => updateSection('deconvolution', { dedup_rt_tolerance: parseFloat(e.target.value) })} />
                      </div>
                    )}
                    <button className="btn btn-sm btn-secondary" onClick={resetDeconvDefaults}>Reset Defaults</button>
                  </div>
                )}
              </div>
            </div>
          )}
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
              <input type="checkbox" id="neg-peaks-enabled" checked={params.negative_peaks.enabled}
                onChange={(e) => updateSection('negative_peaks', { enabled: e.target.checked })} />
              <label htmlFor="neg-peaks-enabled" className="form-label" style={{ marginBottom: 0 }}>Enable</label>
            </div>
            {params.negative_peaks.enabled && (
              <div className="form-group">
                <label className="form-label">Min Prominence</label>
                <input type="text" className="form-control" value={params.negative_peaks.min_prominence}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v) && v >= 0) updateSection('negative_peaks', { min_prominence: v });
                  }} />
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
              <input type="checkbox" id="shoulders-enabled" checked={params.shoulders.enabled}
                disabled={isDeconv || !params.peaks.enabled}
                onChange={(e) => updateSection('shoulders', { enabled: e.target.checked })} />
              <label htmlFor="shoulders-enabled" className="form-label" style={{ marginBottom: 0 }}>
                Enable {isDeconv && <span className="text-muted">(classical only)</span>}
              </label>
            </div>

            <div style={{ opacity: params.shoulders.enabled && !isDeconv ? 1 : 0.4, pointerEvents: params.shoulders.enabled && !isDeconv ? 'auto' : 'none' }}>
              <div className="form-group">
                <label className="form-label">Sensitivity: {params.shoulders.sensitivity}</label>
                <input type="range" className="form-control" min="1" max="10" step="1"
                  value={params.shoulders.sensitivity}
                  onChange={(e) => updateSection('shoulders', { sensitivity: parseInt(e.target.value) })} />
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>Higher = more detections</div>
              </div>

              <div className="form-group">
                <label className="form-label">Apex Distance: {params.shoulders.apex_distance}</label>
                <input type="range" className="form-control" min="5" max="30" step="1"
                  value={params.shoulders.apex_distance}
                  onChange={(e) => updateSection('shoulders', { apex_distance: parseInt(e.target.value) })} />
              </div>

              <div className="form-group">
                <label className="form-label">Window Length: {params.shoulders.window_length}</label>
                <input type="range" className="form-control" min="5" max="101" step="2"
                  value={params.shoulders.window_length}
                  onChange={(e) => updateSection('shoulders', { window_length: ensureOdd(parseInt(e.target.value)) })} />
              </div>

              <div className="form-group">
                <label className="form-label">Polynomial Order: {params.shoulders.polyorder}</label>
                <input type="range" className="form-control" min="1" max="5"
                  value={params.shoulders.polyorder}
                  onChange={(e) => updateSection('shoulders', { polyorder: parseInt(e.target.value) })} />
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
              <div key={idx} style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.4rem', alignItems: 'center' }}>
                <input type="number" className="form-control" style={{ width: '70px' }}
                  step="0.1" value={rf[0]} placeholder="Start"
                  onChange={(e) => updateRangeFilter(idx, 0, e.target.value)} />
                <span>–</span>
                <input type="number" className="form-control" style={{ width: '70px' }}
                  step="0.1" value={rf[1]} placeholder="End"
                  onChange={(e) => updateRangeFilter(idx, 1, e.target.value)} />
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
              <div key={idx} style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.4rem', alignItems: 'center' }}>
                <input type="number" className="form-control" style={{ width: '70px' }}
                  step="0.1" value={pg[0]} placeholder="Start"
                  onChange={(e) => updatePeakGroup(idx, 0, e.target.value)} />
                <span>–</span>
                <input type="number" className="form-control" style={{ width: '70px' }}
                  step="0.1" value={pg[1]} placeholder="End"
                  onChange={(e) => updatePeakGroup(idx, 1, e.target.value)} />
                <button className="btn btn-sm btn-danger" onClick={() => removePeakGroup(idx)}>✕</button>
              </div>
            ))}
            <button className="btn btn-sm btn-secondary" onClick={addPeakGroup}>+ Add Group</button>
          </>
        )}
      </fieldset>

    </div>
  );
};

const fieldsetStyle = {
  border: '1px solid var(--border-color)',
  borderRadius: '6px',
  padding: '0.75rem',
  marginBottom: '0.75rem',
};

const legendStyle = {
  fontWeight: 600,
  fontSize: '0.85rem',
  padding: '0 0.5rem',
  color: 'var(--text-color)',
};

export default ProcessingControls;
