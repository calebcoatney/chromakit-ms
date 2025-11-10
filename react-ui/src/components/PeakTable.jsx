/**
 * PeakTable Component
 * 
 * Displays integration results in a table format.
 */
import React from 'react';

const PeakTable = ({ integrationResults, onIntegrate, disabled }) => {
  if (!integrationResults) {
    return null; // Don't show anything if no results
  }

  const { peaks, total_peaks, integrated_areas } = integrationResults;
  const totalArea = integrated_areas?.reduce((sum, area) => sum + area, 0) || 0;

  return (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>📊 Peak Integration Results</h2>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          {total_peaks} peaks | Total area: {totalArea.toFixed(2)}
        </div>
      </div>
      <div className="card-body" style={{ padding: 0 }}>
        <div style={{ overflowX: 'auto', maxHeight: '400px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ 
              position: 'sticky', 
              top: 0, 
              background: '#f7fafc',
              borderBottom: '2px solid var(--border-color)'
            }}>
              <tr>
                <th style={tableHeaderStyle}>Peak #</th>
                <th style={tableHeaderStyle}>RT (min)</th>
                <th style={tableHeaderStyle}>Area</th>
                <th style={tableHeaderStyle}>% Area</th>
                <th style={tableHeaderStyle}>Width (min)</th>
                <th style={tableHeaderStyle}>Compound</th>
              </tr>
            </thead>
            <tbody>
              {peaks && peaks.length > 0 ? (
                peaks.map((peak, index) => {
                  const percentArea = totalArea > 0 
                    ? ((peak.area / totalArea) * 100).toFixed(2)
                    : '0.00';
                  
                  return (
                    <tr 
                      key={index}
                      style={{
                        borderBottom: '1px solid var(--border-color)',
                        transition: 'background-color 0.2s'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f7fafc'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                    >
                      <td style={tableCellStyle}>{peak.peak_number || index + 1}</td>
                      <td style={tableCellStyle}>{peak.retention_time?.toFixed(3) || 'N/A'}</td>
                      <td style={tableCellStyle}>{peak.area?.toFixed(2) || 'N/A'}</td>
                      <td style={tableCellStyle}>{percentArea}%</td>
                      <td style={tableCellStyle}>{peak.width?.toFixed(4) || 'N/A'}</td>
                      <td style={tableCellStyle}>{peak.compound_id || 'Unknown'}</td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="6" style={{ ...tableCellStyle, textAlign: 'center', padding: '2rem' }}>
                    <div className="text-muted">No peaks detected</div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        
        {/* Integrate Button */}
        <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)' }}>
          <button
            className="btn btn-success"
            onClick={onIntegrate}
            disabled={disabled}
            style={{ width: '100%' }}
          >
            {disabled ? '⏳ Integrating...' : '🔄 Re-integrate Peaks'}
          </button>
        </div>
      </div>
    </div>
  );
};

const tableHeaderStyle = {
  padding: '0.75rem 1rem',
  textAlign: 'left',
  fontWeight: 600,
  fontSize: '0.875rem',
  color: 'var(--text-color)'
};

const tableCellStyle = {
  padding: '0.75rem 1rem',
  fontSize: '0.875rem',
  color: 'var(--text-color)'
};

export default PeakTable;
