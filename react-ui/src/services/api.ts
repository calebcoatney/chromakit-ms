/**
 * ChromaKit-MS API client.
 * Centralized API calls to the FastAPI backend.
 */
import axios from 'axios';
import type {
  BrowseResponse,
  LoadedFileData,
  ProcessingParams,
  ProcessedData,
  IntegrationResults,
  MassSpectrum,
  MSSearchResult,
  MSSearchOptions,
  DetectorInfo,
  RTTableSettings,
  QuantitationSettings,
  ScalingFactors,
  Peak,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ─── Health ──────────────────────────────────────────────────────────

export async function checkHealth(): Promise<{ status: string }> {
  const { data } = await api.get('/health');
  return data;
}

// ─── File Browsing (server-side fallback) ────────────────────────────

export async function browseDirectory(path: string = '.'): Promise<BrowseResponse> {
  const { data } = await api.get('/browse', { params: { path } });
  return data;
}

// ─── File Loading ────────────────────────────────────────────────────

export async function loadFile(filePath: string, detector?: string): Promise<LoadedFileData> {
  const { data } = await api.post('/load', { file_path: filePath, detector });
  return data;
}

export async function getDetectors(filePath: string): Promise<DetectorInfo[]> {
  const { data } = await api.get('/detectors', { params: { path: filePath } });
  return data;
}

// ─── Processing ──────────────────────────────────────────────────────

export async function processChromato(
  x: number[],
  y: number[],
  params: ProcessingParams,
  msRange?: [number, number]
): Promise<ProcessedData> {
  const { data } = await api.post('/process', { x, y, params, ms_range: msRange });
  return data;
}

// ─── Integration ─────────────────────────────────────────────────────

export async function integratePeaks(
  processedData: Record<string, any>,
  chemstationAreaFactor: number = 0.0784,
  rtTable?: Record<string, any>,
  peakGroups?: [number, number][]
): Promise<IntegrationResults> {
  const { data } = await api.post('/integrate', {
    processed_data: processedData,
    chemstation_area_factor: chemstationAreaFactor,
    rt_table: rtTable,
    peak_groups: peakGroups,
  });
  return data;
}

// ─── Spectrum Extraction ─────────────────────────────────────────────

export async function extractSpectrum(
  filePath: string,
  retentionTime: number,
  options?: Record<string, any>
): Promise<MassSpectrum> {
  const { data } = await api.post('/spectrum', {
    file_path: filePath,
    retention_time: retentionTime,
    options,
  });
  return data;
}

// ─── MS Search ───────────────────────────────────────────────────────

export async function searchSpectrum(
  spectrum: MassSpectrum,
  options?: Partial<MSSearchOptions>
): Promise<MSSearchResult[]> {
  const { data } = await api.post('/ms/search', { spectrum, options });
  return data;
}

export function batchSearchPeaks(
  filePath: string,
  peaks: Peak[],
  options?: Partial<MSSearchOptions>,
  onProgress?: (index: number, name: string) => void
): EventSource {
  const params = new URLSearchParams({
    file_path: filePath,
    peaks: JSON.stringify(peaks),
    options: JSON.stringify(options || {}),
  });
  const es = new EventSource(`/api/ms/batch-search?${params}`);
  es.addEventListener('progress', (e) => {
    const { index, compound_name } = JSON.parse(e.data);
    onProgress?.(index, compound_name);
  });
  return es;
}

// ─── RT Table ────────────────────────────────────────────────────────

export async function loadRTTable(filePath: string): Promise<RTTableSettings> {
  const { data } = await api.post('/rt-table/load', { file_path: filePath });
  return data;
}

export async function matchRTTable(
  peaks: Peak[],
  rtSettings: RTTableSettings
): Promise<Peak[]> {
  const { data } = await api.post('/rt-table/match', { peaks, settings: rtSettings });
  return data;
}

// ─── Quantitation ────────────────────────────────────────────────────

export async function quantitate(
  peaks: Peak[],
  settings: QuantitationSettings
): Promise<Peak[]> {
  const { data } = await api.post('/quantitate', { peaks, settings });
  return data;
}

// ─── Export ──────────────────────────────────────────────────────────

export async function exportResults(
  peaks: Peak[],
  filePath: string,
  format: 'json' | 'csv' | 'xlsx',
  options?: Record<string, any>
): Promise<Blob> {
  const { data } = await api.post(
    '/export',
    { peaks, file_path: filePath, format, options },
    { responseType: 'blob' }
  );
  return data;
}

// ─── TIC Alignment ──────────────────────────────────────────────────

export async function alignTIC(
  fidTime: number[],
  fidSignal: number[],
  ticTime: number[],
  ticSignal: number[]
): Promise<{ aligned_time: number[]; aligned_signal: number[]; lag_seconds: number }> {
  const { data } = await api.post('/align-tic', {
    fid_time: fidTime,
    fid_signal: fidSignal,
    tic_time: ticTime,
    tic_signal: ticSignal,
  });
  return data;
}

// ─── Manual Assignments ──────────────────────────────────────────────

export async function getAssignments(): Promise<Record<string, any>> {
  const { data } = await api.get('/assignments');
  return data;
}

export async function saveAssignment(
  retentionTime: number,
  compoundName: string,
  spectrum?: MassSpectrum
): Promise<void> {
  await api.post('/assignments', {
    retention_time: retentionTime,
    compound_name: compoundName,
    spectrum,
  });
}

// ─── MS Baseline Correction ─────────────────────────────────────────

export async function correctMSBaseline(
  filePath: string,
  baselineParams: Record<string, any>
): Promise<any> {
  const { data } = await api.post('/ms/baseline', {
    file_path: filePath,
    params: baselineParams,
  });
  return data;
}

export default api;
