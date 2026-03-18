/**
 * ChromatogramPlot Component
 *
 * Enhanced interactive chromatogram viewer with:
 * - Peak click selection + highlight
 * - Distinctly colored integration shading (tab20 colormap)
 * - Optional TIC subplot (dual y-axis)
 * - Peak annotations (compound names, RT labels)
 * - Shoulder/negative peak visual indicators
 * - Zoom/pan state preservation
 */
import React, { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';

// tab20-style distinct colors for integration areas
const PEAK_COLORS = [
  '#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c',
  '#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5',
  '#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f',
  '#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5',
];

const ChromatogramPlot = ({
  data,
  processedData,
  integrationResults,
  ticData,
  showCorrectedSignal = false,
  showBaseline = true,
  onPeakClick,
  selectedPeakIndex,
}) => {
  const [xRange, setXRange] = useState(null);
  const [yRange, setYRange] = useState(null);
  const [showTIC, setShowTIC] = useState(false);

  const handleRelayout = useCallback((event) => {
    if (event['xaxis.range[0]'] !== undefined) {
      setXRange([event['xaxis.range[0]'], event['xaxis.range[1]']]);
    } else if (event['xaxis.autorange']) {
      setXRange(null);
    }
    if (event['yaxis.range[0]'] !== undefined) {
      setYRange([event['yaxis.range[0]'], event['yaxis.range[1]']]);
    } else if (event['yaxis.autorange']) {
      setYRange(null);
    }
  }, []);

  // Handle click on plot to select nearest peak
  const handleClick = useCallback((event) => {
    if (!onPeakClick || !integrationResults?.peaks?.length) return;
    const point = event.points?.[0];
    if (!point) return;
    const clickX = point.x;

    let closestIdx = -1;
    let minDist = Infinity;
    integrationResults.peaks.forEach((peak, i) => {
      const inBounds = peak.start_time <= clickX && clickX <= peak.end_time;
      const dist = Math.abs(peak.retention_time - clickX);
      if ((inBounds || dist < 0.05) && dist < minDist) {
        minDist = dist;
        closestIdx = i;
      }
    });
    if (closestIdx >= 0) onPeakClick(closestIdx);
  }, [onPeakClick, integrationResults]);

  const traces = useMemo(() => {
    const t = [];
    if (!data && !processedData) return t;

    const x = processedData?.x || data?.x || [];
    const baseline = processedData?.baseline_y || [];

    // Choose signal to display
    let y;
    if (showCorrectedSignal && processedData?.corrected_y) {
      y = processedData.corrected_y;
    } else if (processedData?.smoothed_y) {
      y = processedData.smoothed_y;
    } else if (processedData?.original_y) {
      y = processedData.original_y;
    } else {
      y = data?.y || [];
    }

    // ── Integration area shading (drawn first so signal is on top) ──
    if (integrationResults?.peaks?.length > 0) {
      integrationResults.peaks.forEach((peak, i) => {
        if (peak.start_index == null || peak.end_index == null) return;
        const si = peak.start_index;
        const ei = peak.end_index;
        const peakX = x.slice(si, ei + 1);
        const peakY = y.slice(si, ei + 1);
        const peakBL = showCorrectedSignal
          ? peakX.map(() => 0)
          : baseline.slice(si, ei + 1);

        const colorIdx = (i * 7) % PEAK_COLORS.length;
        const color = PEAK_COLORS[colorIdx];
        const isSelected = selectedPeakIndex === i;
        const isShoulder = peak.is_shoulder;
        const isNegative = peak.type === 'negative';

        let fillColor = color;
        if (isNegative) fillColor = '#00bcd4';

        t.push({
          x: [...peakX, ...[...peakX].reverse()],
          y: [...peakY, ...[...peakBL].reverse()],
          fill: 'toself',
          fillcolor: isSelected
            ? fillColor.replace(')', ', 0.6)').replace('rgb', 'rgba') || `${fillColor}99`
            : `${fillColor}4D`,
          line: {
            width: isSelected ? 2 : (isShoulder ? 1 : 0),
            color: isSelected ? '#000' : (isShoulder ? '#e91e63' : undefined),
            dash: isShoulder ? 'dot' : undefined,
          },
          showlegend: false,
          hoverinfo: 'skip',
          type: 'scatter',
        });
      });
    }

    // ── Main signal ──
    t.push({
      x, y,
      type: 'scatter', mode: 'lines',
      name: showCorrectedSignal ? 'Corrected Signal' : 'Chromatogram',
      line: { color: '#667eea', width: 1.5 },
      hovertemplate: 'RT: %{x:.3f} min<br>Intensity: %{y:.1f}<extra></extra>',
    });

    // ── Baseline ──
    if (showBaseline && baseline.length > 0) {
      t.push({
        x,
        y: showCorrectedSignal ? baseline.map(() => 0) : baseline,
        type: 'scatter', mode: 'lines',
        name: 'Baseline',
        line: { color: '#f56565', width: 1, dash: 'dash' },
        hovertemplate: 'RT: %{x:.3f} min<br>Baseline: %{y:.1f}<extra></extra>',
      });
    }

    // ── Peak markers ──
    if (processedData?.peaks_x?.length > 0) {
      const peakX = processedData.peaks_x;
      const peakY = (showCorrectedSignal && processedData.corrected_peaks_y)
        ? processedData.corrected_peaks_y
        : processedData.peaks_y;
      const meta = processedData.peak_metadata || [];

      // Separate shoulders from regular peaks
      const regularIdx = [];
      const shoulderIdx = [];
      meta.forEach((m, i) => {
        if (m?.is_shoulder) shoulderIdx.push(i);
        else regularIdx.push(i);
      });

      // Regular peaks
      if (regularIdx.length > 0 || meta.length === 0) {
        const indices = meta.length > 0 ? regularIdx : peakX.map((_, i) => i);
        t.push({
          x: indices.map(i => peakX[i]),
          y: indices.map(i => peakY[i]),
          type: 'scatter', mode: 'markers',
          name: 'Peaks',
          marker: { color: '#48bb78', size: 8, symbol: 'diamond' },
          hovertemplate: 'RT: %{x:.3f} min<br>Height: %{y:.1f}<extra></extra>',
        });
      }

      // Shoulder peaks
      if (shoulderIdx.length > 0) {
        t.push({
          x: shoulderIdx.map(i => peakX[i]),
          y: shoulderIdx.map(i => peakY[i]),
          type: 'scatter', mode: 'markers',
          name: 'Shoulders',
          marker: { color: '#e91e63', size: 7, symbol: 'triangle-up' },
          hovertemplate: 'Shoulder RT: %{x:.3f} min<br>Height: %{y:.1f}<extra></extra>',
        });
      }
    }

    // ── Peak annotations (compound names) ──
    // Handled via layout.annotations below

    // ── TIC subplot ──
    if (showTIC && ticData?.x?.length > 0) {
      t.push({
        x: ticData.x,
        y: ticData.y,
        type: 'scatter', mode: 'lines',
        name: 'TIC',
        yaxis: 'y2',
        line: { color: '#ed8936', width: 1 },
        hovertemplate: 'RT: %{x:.3f} min<br>TIC: %{y:.1f}<extra></extra>',
      });
    }

    return t;
  }, [data, processedData, integrationResults, showCorrectedSignal, showBaseline, showTIC, ticData, selectedPeakIndex]);

  // Build annotations for integrated peaks with compound names
  const annotations = useMemo(() => {
    if (!integrationResults?.peaks?.length) return [];
    return integrationResults.peaks
      .filter(p => p.match_name || p.compound_id)
      .map(p => ({
        x: p.retention_time,
        y: p.height || 0,
        text: p.match_name || p.compound_id || '',
        showarrow: true,
        arrowhead: 0,
        arrowcolor: '#718096',
        ax: 0, ay: -30,
        font: { size: 9, color: '#4a5568' },
        bgcolor: 'rgba(255,255,255,0.8)',
        borderpad: 2,
      }));
  }, [integrationResults]);

  const layout = useMemo(() => {
    const l = {
      autosize: true,
      height: showTIC && ticData?.x?.length ? 600 : 450,
      margin: { l: 60, r: showTIC ? 60 : 40, t: 30, b: 50 },
      xaxis: {
        title: 'Time (min)',
        showgrid: true, gridcolor: '#edf2f7', zeroline: false,
      },
      yaxis: {
        title: 'Intensity',
        showgrid: true, gridcolor: '#edf2f7', zeroline: true,
        domain: showTIC && ticData?.x?.length ? [0.3, 1] : [0, 1],
      },
      hovermode: 'closest',
      showlegend: true,
      legend: {
        x: 1, xanchor: 'right', y: 1,
        bgcolor: 'rgba(255,255,255,0.85)',
        bordercolor: '#e2e8f0', borderwidth: 1,
        font: { size: 11 },
      },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      annotations,
    };

    // TIC y-axis
    if (showTIC && ticData?.x?.length) {
      l.yaxis2 = {
        title: 'TIC Intensity',
        showgrid: true, gridcolor: '#feebc8', zeroline: false,
        domain: [0, 0.25],
        anchor: 'x',
      };
    }

    // Preserve zoom
    if (xRange) { l.xaxis.range = xRange; l.xaxis.autorange = false; }
    if (yRange) { l.yaxis.range = yRange; l.yaxis.autorange = false; }

    return l;
  }, [xRange, yRange, showTIC, ticData, annotations]);

  // Counts
  const peakCount = processedData?.peaks_x?.length || 0;
  const shoulderCount = processedData?.peak_metadata?.filter(m => m?.is_shoulder).length || 0;
  const integratedCount = integrationResults?.total_peaks || 0;

  if (traces.length === 0) {
    return (
      <div className="card">
        <div className="card-header"><h2>📈 Chromatogram</h2></div>
        <div className="card-body">
          <div className="text-center text-muted" style={{ padding: '3rem' }}>
            Load a file to view chromatogram
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h2>📈 Chromatogram</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <span>
            {peakCount > 0 && `${peakCount} peaks`}
            {shoulderCount > 0 && ` (${shoulderCount} shoulders)`}
            {integratedCount > 0 && ` · ${integratedCount} integrated`}
          </span>
          {ticData?.x?.length > 0 && (
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer', fontSize: '0.8rem' }}>
              <input type="checkbox" checked={showTIC} onChange={(e) => setShowTIC(e.target.checked)} />
              TIC
            </label>
          )}
        </div>
      </div>
      <div className="card-body" style={{ padding: '0.25rem' }}>
        <Plot
          data={traces}
          layout={layout}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
            toImageButtonOptions: {
              format: 'png', filename: 'chromatogram',
              height: 800, width: 1200, scale: 2,
            },
          }}
          onRelayout={handleRelayout}
          onClick={handleClick}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default ChromatogramPlot;
