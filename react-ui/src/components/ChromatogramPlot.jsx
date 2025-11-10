/**
 * ChromatogramPlot Component
 * 
 * Displays chromatogram data using Plotly with interactive controls.
 * Shows baseline, peaks, and shaded integration areas.
 * Preserves zoom state when data updates.
 */
import React, { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';

const ChromatogramPlot = ({ 
  data, 
  processedData, 
  integrationResults,
  showCorrectedSignal = false,
  showBaseline = true 
}) => {
  // Store zoom/pan state
  const [xAxisRange, setXAxisRange] = useState(null);
  const [yAxisRange, setYAxisRange] = useState(null);

  // Callback to capture layout updates (zoom/pan events)
  const handleRelayout = useCallback((event) => {
    console.log('Relayout event:', event); // Debug logging
    
    // Capture zoom/pan state for x-axis
    if (event['xaxis.range[0]'] !== undefined && event['xaxis.range[1]'] !== undefined) {
      const newRange = [event['xaxis.range[0]'], event['xaxis.range[1]']];
      setXAxisRange(newRange);
      console.log('Saved x range:', newRange);
    } else if (event['xaxis.autorange'] === true) {
      setXAxisRange(null);
      console.log('X autorange enabled');
    }
    
    // Capture zoom/pan state for y-axis
    if (event['yaxis.range[0]'] !== undefined && event['yaxis.range[1]'] !== undefined) {
      const newRange = [event['yaxis.range[0]'], event['yaxis.range[1]']];
      setYAxisRange(newRange);
      console.log('Saved y range:', newRange);
    } else if (event['yaxis.autorange'] === true) {
      setYAxisRange(null);
      console.log('Y autorange enabled');
    }
  }, []);

  // Prepare traces for plotting
  const getTraces = () => {
    const traces = [];

    if (!data && !processedData) {
      return traces;
    }

    const x = processedData?.x || data?.x || [];
    const baseline = processedData?.baseline_y || [];
    
    // Determine which y values to show based on showCorrectedSignal
    let y;
    if (showCorrectedSignal && processedData?.corrected_y) {
      // Show corrected signal (signal - baseline)
      y = processedData.corrected_y;
    } else if (processedData?.smoothed_y) {
      // Show smoothed signal (with baseline separately)
      y = processedData.smoothed_y;
    } else if (processedData?.original_y) {
      // Show original signal (no smoothing)
      y = processedData.original_y;
    } else {
      // Fallback to original data
      y = data?.y || [];
    }

    // Main chromatogram trace
    traces.push({
      x: x,
      y: y,
      type: 'scatter',
      mode: 'lines',
      name: showCorrectedSignal ? 'Corrected Signal' : 'Chromatogram',
      line: { color: '#667eea', width: 2 },
      hovertemplate: 'Time: %{x:.3f} min<br>Intensity: %{y:.2f}<extra></extra>'
    });

    // Baseline trace
    // If showing corrected signal, baseline is at zero (corrected - baseline = baseline - baseline = 0)
    // If showing uncorrected signal, show the actual baseline
    if (showBaseline && baseline.length > 0) {
      const baselineY = showCorrectedSignal 
        ? baseline.map(() => 0)  // Zero line for corrected signal
        : baseline;  // Actual baseline for uncorrected signal

      traces.push({
        x: x,
        y: baselineY,
        type: 'scatter',
        mode: 'lines',
        name: 'Baseline',
        line: { color: '#f56565', width: 1, dash: 'dash' },
        hovertemplate: 'Time: %{x:.3f} min<br>Baseline: %{y:.2f}<extra></extra>'
      });
    }

    // Peak markers (if peaks detected)
    if (processedData?.peaks_x && processedData.peaks_x.length > 0) {
      const peakX = processedData.peaks_x;
      let peakY = processedData.peaks_y;

      // If showing corrected signal, adjust peak heights
      if (showCorrectedSignal && processedData.corrected_peaks_y) {
        peakY = processedData.corrected_peaks_y;
      }

      traces.push({
        x: peakX,
        y: peakY,
        type: 'scatter',
        mode: 'markers',
        name: 'Peaks',
        marker: { 
          color: '#48bb78', 
          size: 10,
          symbol: 'diamond'
        },
        hovertemplate: 'RT: %{x:.3f} min<br>Intensity: %{y:.2f}<extra></extra>'
      });
    }

    // Integration areas (shaded regions between baseline and signal)
    if (integrationResults?.peaks && integrationResults.peaks.length > 0) {
      integrationResults.peaks.forEach((peak, index) => {
        if (peak.start_index !== undefined && peak.end_index !== undefined) {
          const startIdx = peak.start_index;
          const endIdx = peak.end_index;
          
          // Create x values for the shaded area
          const peakX = x.slice(startIdx, endIdx + 1);
          const peakY = y.slice(startIdx, endIdx + 1);
          
          // Baseline for shaded area
          const peakBaseline = showCorrectedSignal
            ? peakX.map(() => 0)  // Zero for corrected signal
            : baseline.slice(startIdx, endIdx + 1);  // Actual baseline for uncorrected

          // Create filled area between baseline and chromatogram
          traces.push({
            x: [...peakX, ...peakX.slice().reverse()],
            y: [...peakY, ...peakBaseline.slice().reverse()],
            fill: 'toself',
            fillcolor: `rgba(72, 187, 120, 0.3)`,
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            type: 'scatter'
          });
        }
      });
    }

    return traces;
  };

  const traces = getTraces();

  // Memoize layout to prevent unnecessary re-creation
  const layout = useMemo(() => {
    const baseLayout = {
      autosize: true,
      height: 500,
      margin: { l: 60, r: 40, t: 40, b: 60 },
      xaxis: {
        title: 'Time (min)',
        showgrid: true,
        gridcolor: '#e2e8f0',
        zeroline: false
      },
      yaxis: {
        title: 'Intensity',
        showgrid: true,
        gridcolor: '#e2e8f0',
        zeroline: true
      },
      hovermode: 'closest',
      showlegend: true,
      legend: {
        x: 1,
        xanchor: 'right',
        y: 1,
        bgcolor: 'rgba(255, 255, 255, 0.8)',
        bordercolor: '#e2e8f0',
        borderwidth: 1
      },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white'
    };

    // Apply saved zoom state
    if (xAxisRange) {
      baseLayout.xaxis.range = xAxisRange;
      baseLayout.xaxis.autorange = false;
    }
    if (yAxisRange) {
      baseLayout.yaxis.range = yAxisRange;
      baseLayout.yaxis.autorange = false;
    }

    return baseLayout;
  }, [xAxisRange, yAxisRange]);

  if (traces.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <h2>📈 Chromatogram</h2>
        </div>
        <div className="card-body">
          <div className="text-center text-muted" style={{ padding: '3rem' }}>
            Load a file to view chromatogram
          </div>
        </div>
      </div>
    );
  }

  // Count peaks
  const peakCount = processedData?.peaks_x?.length || 0;
  const integratedCount = integrationResults?.total_peaks || 0;

  return (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>📈 Chromatogram</h2>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          {peakCount > 0 && `${peakCount} peaks detected`}
          {integratedCount > 0 && ` | ${integratedCount} integrated`}
        </div>
      </div>
      <div className="card-body" style={{ padding: '0.5rem' }}>
        <Plot
          data={traces}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
            toImageButtonOptions: {
              format: 'png',
              filename: 'chromatogram',
              height: 800,
              width: 1200,
              scale: 2
            }
          }}
          onRelayout={handleRelayout}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default ChromatogramPlot;
