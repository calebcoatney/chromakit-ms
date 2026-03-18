/**
 * PeakTable Component
 *
 * Displays integration results with CSV export and peak selection.
 */
import React, { useCallback } from 'react';

const PeakTable = ({ integrationResults, onIntegrate, onPeakClick, selectedPeakIndex, disabled }) => {
  if (!integrationResults) return null;

  const { peaks, total_peaks, integrated_areas } = integrationResults;
  const totalArea = integrated_areas?.reduce((sum, a) => sum + a, 0) || 0;

  // CSV export
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

  // JSON export
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
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h2>📊 Peak Integration Results</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.85rem' }}>
          <span style={{ color: 'var(--text-secondary)' }}>
            {total_peaks} peaks · Total: {totalArea.toFixed(1)}
          </span>
          <button className="btn btn-sm btn-secondary" onClick={handleExportCSV} title="Export CSV">📄 CSV</button>
          <button className="btn btn-sm btn-secondary" onClick={handleExportJSON} title="Export JSON">📋 JSON</button>
        </div>
      </div>
      <div className="card-body" style={{ padding: 0 }}>
        <div style={{ overflowX: 'auto', maxHeight: '400px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ position: 'sticky', top: 0, background: '#f7fafc', borderBottom: '2px solid var(--border-color)' }}>
              <tr>
                <th style={thStyle}>#</th>
                <th style={thStyle}>RT (min)</th>
                <th style={thStyle}>Area</th>
                <th style={thStyle}>% Area</th>
                <th style={thStyle}>Width</th>
                <th style={thStyle}>Compound</th>
              </tr>
            </thead>
            <tbody>
              {peaks?.length > 0 ? peaks.map((peak, i) => {
                const pct = totalArea > 0 ? ((peak.area / totalArea) * 100).toFixed(2) : '0.00';
                const isSelected = selectedPeakIndex === i;
                return (
                  <tr key={i}
                    onClick={() => onPeakClick?.(i)}
                    style={{
                      borderBottom: '1px solid var(--border-color)',
                      backgroundColor: isSelected ? '#ebf8ff' : 'transparent',
                      cursor: onPeakClick ? 'pointer' : 'default',
                      transition: 'background-color 0.15s',
                    }}
                    onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = '#f7fafc'; }}
                    onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = 'transparent'; }}
                  >
                    <td style={tdStyle}>{peak.peak_number || i + 1}</td>
                    <td style={tdStyle}>{peak.retention_time?.toFixed(3) ?? 'N/A'}</td>
                    <td style={tdStyle}>{peak.area?.toFixed(2) ?? 'N/A'}</td>
                    <td style={tdStyle}>{pct}%</td>
                    <td style={tdStyle}>{peak.width?.toFixed(4) ?? 'N/A'}</td>
                    <td style={tdStyle}>{peak.match_name || peak.compound_id || '—'}</td>
                  </tr>
                );
              }) : (
                <tr>
                  <td colSpan="6" style={{ ...tdStyle, textAlign: 'center', padding: '2rem' }}>
                    <span className="text-muted">No peaks detected</span>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div style={{ padding: '0.75rem 1rem', borderTop: '1px solid var(--border-color)' }}>
          <button className="btn btn-success" onClick={onIntegrate} disabled={disabled} style={{ width: '100%' }}>
            {disabled ? '⏳ Integrating...' : '🔄 Re-integrate Peaks'}
          </button>
        </div>
      </div>
    </div>
  );
};

const thStyle = { padding: '0.6rem 0.75rem', textAlign: 'left', fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-color)' };
const tdStyle = { padding: '0.5rem 0.75rem', fontSize: '0.85rem', color: 'var(--text-color)' };

export default PeakTable;
