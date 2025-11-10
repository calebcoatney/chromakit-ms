/**
 * ProcessingControls Component
 * 
 * Provides UI controls for chromatogram processing parameters.
 * Updates happen in real-time as parameters change.
 */
import React, { useState, useEffect } from 'react';

const ProcessingControls = ({ onParametersChange, disabled }) => {
  const [params, setParams] = useState({
    smoothing: {
      enabled: false,
      median_filter: { kernel_size: 9 },
      savgol_filter: { window_length: 15, polyorder: 2 }
    },
    baseline: {
      show_corrected: false,
      method: 'arpls',
      lambda: 1e4,
      asymmetry: 0.01
    },
    peaks: {
      enabled: false,
      min_prominence: 1e5
    }
  });

  // Trigger real-time updates whenever parameters change
  useEffect(() => {
    if (onParametersChange) {
      onParametersChange(params);
    }
  }, [params]);

  // Smoothing handlers
  const handleSmoothingToggle = (e) => {
    setParams({
      ...params,
      smoothing: {
        ...params.smoothing,
        enabled: e.target.checked
      }
    });
  };

  const handleMedianKernelChange = (value) => {
    // Ensure odd values only
    const oddValue = value % 2 === 0 ? value + 1 : value;
    setParams({
      ...params,
      smoothing: {
        ...params.smoothing,
        median_filter: { kernel_size: oddValue }
      }
    });
  };

  const handleSavgolWindowChange = (value) => {
    // Ensure odd values only
    const oddValue = value % 2 === 0 ? value + 1 : value;
    setParams({
      ...params,
      smoothing: {
        ...params.smoothing,
        savgol_filter: { ...params.smoothing.savgol_filter, window_length: oddValue }
      }
    });
  };

  const handleSavgolPolyorderChange = (value) => {
    setParams({
      ...params,
      smoothing: {
        ...params.smoothing,
        savgol_filter: { ...params.smoothing.savgol_filter, polyorder: value }
      }
    });
  };

  // Baseline handlers
  const handleBaselineShowCorrected = (e) => {
    setParams({
      ...params,
      baseline: {
        ...params.baseline,
        show_corrected: e.target.checked
      }
    });
  };

  const handleBaselineMethodChange = (e) => {
    setParams({
      ...params,
      baseline: {
        ...params.baseline,
        method: e.target.value
      }
    });
  };

  const handleLambdaChange = (exponent) => {
    setParams({
      ...params,
      baseline: {
        ...params.baseline,
        lambda: Math.pow(10, exponent)
      }
    });
  };

  // Peak detection handlers
  const handlePeaksToggle = (e) => {
    setParams({
      ...params,
      peaks: {
        ...params.peaks,
        enabled: e.target.checked
      }
    });
  };

  const handleProminenceChange = (e) => {
    const value = e.target.value;
    // Parse scientific notation and regular numbers
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue >= 0) {
      setParams({
        ...params,
        peaks: {
          ...params.peaks,
          min_prominence: numValue
        }
      });
    }
  };

  // Helper to check if method uses lambda
  const methodUsesLambda = ['asls', 'airpls', 'arpls'].includes(params.baseline.method);
  const lambdaExponent = Math.round(Math.log10(params.baseline.lambda));

  return (
    <div className="card">
      <div className="card-header">
        <h2>⚙️ Integration Parameters</h2>
      </div>
      <div className="card-body" style={{ maxHeight: '600px', overflowY: 'auto' }}>
        
        {/* SIGNAL SMOOTHING */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>Signal Smoothing</legend>
          
          <div className="form-check mb-2">
            <input
              type="checkbox"
              id="smoothing-enabled"
              checked={params.smoothing.enabled}
              onChange={handleSmoothingToggle}
            />
            <label htmlFor="smoothing-enabled" className="form-label" style={{ marginBottom: 0 }}>
              Enable Smoothing
            </label>
          </div>

          <div style={{ opacity: params.smoothing.enabled ? 1 : 0.5, pointerEvents: params.smoothing.enabled ? 'auto' : 'none' }}>
            {/* Median Filter Kernel Size */}
            <div className="form-group">
              <label className="form-label">
                Median Filter Size: {params.smoothing.median_filter.kernel_size}
              </label>
              <input
                type="range"
                className="form-control"
                min="3"
                max="31"
                step="2"
                value={params.smoothing.median_filter.kernel_size}
                onChange={(e) => handleMedianKernelChange(parseInt(e.target.value))}
                disabled={!params.smoothing.enabled}
              />
            </div>

            {/* Savitzky-Golay Window */}
            <div className="form-group">
              <label className="form-label">
                Savitzky-Golay Window: {params.smoothing.savgol_filter.window_length}
              </label>
              <input
                type="range"
                className="form-control"
                min="5"
                max="51"
                step="2"
                value={params.smoothing.savgol_filter.window_length}
                onChange={(e) => handleSavgolWindowChange(parseInt(e.target.value))}
                disabled={!params.smoothing.enabled}
              />
            </div>

            {/* Savitzky-Golay Polynomial Order */}
            <div className="form-group">
              <label className="form-label">
                Polynomial Order: {params.smoothing.savgol_filter.polyorder}
              </label>
              <input
                type="number"
                className="form-control"
                min="1"
                max="5"
                value={params.smoothing.savgol_filter.polyorder}
                onChange={(e) => handleSavgolPolyorderChange(parseInt(e.target.value))}
                disabled={!params.smoothing.enabled}
              />
            </div>
          </div>
        </fieldset>

        {/* BASELINE CORRECTION */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>Baseline Correction</legend>

          <div className="form-check mb-2">
            <input
              type="checkbox"
              id="baseline-show-corrected"
              checked={params.baseline.show_corrected}
              onChange={handleBaselineShowCorrected}
            />
            <label htmlFor="baseline-show-corrected" className="form-label" style={{ marginBottom: 0 }}>
              Show Corrected Signal
            </label>
          </div>
          <div className="text-muted mb-2" style={{ fontSize: '0.75rem' }}>
            Unchecked: Show raw signal with baseline<br/>
            Checked: Show corrected signal (baseline at zero)
          </div>

          {/* Algorithm Selection */}
          <div className="form-group">
            <label className="form-label">Algorithm</label>
            <select
              className="form-control"
              value={params.baseline.method}
              onChange={handleBaselineMethodChange}
            >
              <option value="asls">asls - Asymmetric Least Squares</option>
              <option value="imodpoly">imodpoly - Improved Modified Polynomial</option>
              <option value="modpoly">modpoly - Modified Polynomial</option>
              <option value="snip">snip - Statistics-sensitive Non-linear</option>
              <option value="airpls">airpls - Adaptive Iteratively Reweighted</option>
              <option value="arpls">arpls - Asymmetrically Reweighted</option>
            </select>
          </div>

          {/* Lambda Parameter (only for certain methods) */}
          {methodUsesLambda && (
            <div className="form-group">
              <label className="form-label">
                Lambda (λ): 10^{lambdaExponent} = {params.baseline.lambda.toExponential(2)}
              </label>
              <input
                type="range"
                className="form-control"
                min="2"
                max="12"
                step="1"
                value={lambdaExponent}
                onChange={(e) => handleLambdaChange(parseInt(e.target.value))}
              />
            </div>
          )}
        </fieldset>

        {/* PEAK DETECTION */}
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>Peak Detection</legend>

          <div className="form-check mb-2">
            <input
              type="checkbox"
              id="peaks-enabled"
              checked={params.peaks.enabled}
              onChange={handlePeaksToggle}
            />
            <label htmlFor="peaks-enabled" className="form-label" style={{ marginBottom: 0 }}>
              Enable Peak Detection
            </label>
          </div>

          <div style={{ opacity: params.peaks.enabled ? 1 : 0.5, pointerEvents: params.peaks.enabled ? 'auto' : 'none' }}>
            {/* Min Prominence */}
            <div className="form-group">
              <label className="form-label">Min Prominence</label>
              <input
                type="text"
                className="form-control"
                value={params.peaks.min_prominence}
                onChange={handleProminenceChange}
                placeholder="Enter number or scientific notation (e.g. 1e5)"
                disabled={!params.peaks.enabled}
              />
              <div className="text-muted" style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>
                Enter a number or scientific notation (e.g. 1e5, 1e-3)
              </div>
            </div>
          </div>
        </fieldset>
      </div>
    </div>
  );
};

// Styling for fieldsets
const fieldsetStyle = {
  border: '1px solid var(--border-color)',
  borderRadius: '6px',
  padding: '1rem',
  marginBottom: '1rem'
};

const legendStyle = {
  fontWeight: 600,
  fontSize: '0.9rem',
  padding: '0 0.5rem',
  color: 'var(--text-color)'
};

export default ProcessingControls;
