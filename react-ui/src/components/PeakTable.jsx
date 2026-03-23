/**
 * PeakTable Component — integration results with CSV/JSON export and peak selection.
 */
import React, { useCallback } from 'react';

const PeakTable = ({ integrationResults, onIntegrate, onPeakClick, selectedPeakIndex, disabled }) => {
  if (!integrationResults) return null;

  const { peaks, total_peaks, integrated_areas } = integrationResults;
  const totalArea = integrated_areas?.reduce((sum, a) => sum + a, 0) || 0;

  const handleExportCSV = useCallback(() => {
    if (!peaks?.length) return;
    const headers = ['Peak #', 'RT (min)', 'Area', '% Area', 'Width (min)', 'Compound'];
    const rows = peaks.map((p, i) => {
      const pct = totalArea > 0 ? ((p.area / totalArea) * 100).toFixed(4) : '0';
      return [
        p.peak_number || i + 1,
        p.retention_time?.toFixed(4) ?? '',
        p.area?.toFixed(4) ?? '',
        pct,
        p.width?.toFixed(4) ?? '',
        p.compound_id || p.match_name || '',
      ].join(',');
    });
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'integration_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  }, [peaks, totalArea]);

  const handleExportJSON = useCallback(() => {
    if (!peaks?.length) return;
    const json = JSON.stringify({ total_peaks, peaks }, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'integration_results.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [peaks, total_peaks]);

  return (
    <div className="card">
      <div className="card-header">
        <h2>Integration Results</h2>
        <div className="card-header-meta">
          <span>{total_peaks} peaks &middot; Total: {totalArea.toFixed(1)}</span>
          <button className="btn btn-sm btn-secondary" onClick={handleExportCSV} title="Export CSV">CSV</button>
          <button className="btn btn-sm btn-secondary" onClick={handleExportJSON} title="Export JSON">JSON</button>
        </div>
      </div>
      <div style={{ overflowX: 'auto', maxHeight: '350px' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>RT (min)</th>
              <th>Area</th>
              <th>% Area</th>
              <th>Width</th>
              <th>Compound</th>
            </tr>
          </thead>
          <tbody>
            {peaks?.length > 0 ? peaks.map((peak, i) => {
              const pct = totalArea > 0 ? ((peak.area / totalArea) * 100).toFixed(2) : '0.00';
              const isSelected = selectedPeakIndex === i;
              return (
                <tr key={i}
                  className={`clickable${isSelected ? ' selected' : ''}`}
                  onClick={() => onPeakClick?.(i)}
                >
                  <td>{peak.peak_number || i + 1}</td>
                  <td>{peak.retention_time?.toFixed(3) ?? 'N/A'}</td>
                  <td>{peak.area?.toFixed(2) ?? 'N/A'}</td>
                  <td>{pct}%</td>
                  <td>{peak.width?.toFixed(4) ?? 'N/A'}</td>
                  <td>{peak.match_name || peak.compound_id || '\u2014'}</td>
                </tr>
              );
            }) : (
              <tr>
                <td colSpan="6" className="text-center text-muted" style={{ padding: '1.5rem' }}>
                  No peaks detected
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div style={{ padding: '0.5rem 0.75rem', borderTop: '1px solid var(--border-color)' }}>
        <button className="btn btn-success full-width" onClick={onIntegrate} disabled={disabled}>
          {disabled ? 'Integrating...' : 'Re-integrate'}
        </button>
      </div>
    </div>
  );
};

export default PeakTable;
