/**
 * Core type definitions for ChromaKit-MS web app.
 * Mirrors logic/ data structures from the Python backend.
 */

// ─── Chromatogram Data ───────────────────────────────────────────────

export interface ChromatogramData {
  x: number[];
  y: number[];
}

export interface TICData {
  x: number[];
  y: number[];
}

export interface FileMetadata {
  filename?: string;
  detector?: string;
  method?: string;
  sample_name?: string;
  [key: string]: any;
}

export interface LoadedFileData {
  chromatogram: ChromatogramData;
  tic: TICData;
  has_ms: boolean;
  metadata: FileMetadata;
}

// ─── Processing Parameters (mirrors ParametersFrame.current_params) ─

export interface SmoothingParams {
  enabled: boolean;
  method: 'whittaker' | 'savgol';
  median_enabled: boolean;
  median_kernel: number;
  lambda: number;
  diff_order: number;
  savgol_window: number;
  savgol_polyorder: number;
}

export interface BreakPoint {
  time: number;
  tolerance: number;
}

export interface FastchromParams {
  half_window: number | null;
  smooth_half_window: number | null;
}

export interface BaselineParams {
  show_corrected: boolean;
  method: 'asls' | 'arpls' | 'airpls' | 'imodpoly' | 'modpoly' | 'snip' | 'mixture_model' | 'irsqr';
  lambda: number;
  asymmetry: number;
  align_tic: boolean;
  break_points: BreakPoint[];
  fastchrom: FastchromParams;
}

export interface PeakParams {
  enabled: boolean;
  mode: 'classical' | 'deconvolution';
  min_prominence: number;
  min_height: number;
  min_width: number;
  range_filters: [number, number][];
}

export interface DeconvolutionParams {
  splitting_method: 'geometric' | 'emg';
  windows: [number, number][];
  heatmap_threshold: number;
  pre_fit_signal_threshold: number;
  min_area_frac: number;
  valley_threshold_frac: number;
  mu_bound_factor: number;
  fat_threshold_frac: number;
  dedup_sigma_factor: number;
  dedup_rt_tolerance: number;
}

export interface NegativePeakParams {
  enabled: boolean;
  min_prominence: number;
}

export interface ShoulderParams {
  enabled: boolean;
  window_length: number;
  polyorder: number;
  sensitivity: number;
  apex_distance: number;
}

export interface IntegrationParams {
  peak_groups: [number, number][];
}

export interface ProcessingParams {
  smoothing: SmoothingParams;
  baseline: BaselineParams;
  peaks: PeakParams;
  deconvolution: DeconvolutionParams;
  negative_peaks: NegativePeakParams;
  shoulders: ShoulderParams;
  integration: IntegrationParams;
}

// ─── Processed Data ──────────────────────────────────────────────────

export interface PeakMetadataEntry {
  index: number;
  x: number;
  y: number;
  is_shoulder: boolean;
  type: string;
}

export interface ProcessedData {
  x: number[];
  original_y: number[];
  smoothed_y: number[];
  baseline_y: number[];
  corrected_y: number[];
  peaks_x: number[];
  peaks_y: number[];
  peak_metadata: PeakMetadataEntry[];
}

// ─── Integration / Peaks ─────────────────────────────────────────────

export interface Peak {
  compound_id: string;
  peak_number: number;
  retention_time: number;
  width: number;
  area: number;
  start_time: number;
  end_time: number;
  start_index?: number;
  end_index?: number;
  height?: number;
  is_shoulder?: boolean;
  type?: string;
  // MS match fields
  match_name?: string;
  match_score?: number;
  match_cas?: string;
  match_mw?: number;
  match_formula?: string;
  spectrum_mz?: number[];
  spectrum_intensities?: number[];
  // Quantitation fields
  mass_fraction?: number;
  mole_fraction?: number;
  carbon_number?: number;
}

export interface IntegrationResults {
  peaks: Peak[];
  retention_times: number[];
  integrated_areas: number[];
  total_peaks: number;
}

// ─── MS Data ─────────────────────────────────────────────────────────

export interface MassSpectrum {
  mz: number[];
  intensities: number[];
  retention_time?: number;
}

export interface MSSearchResult {
  compound_name: string;
  cas_number: string;
  match_score: number;
  molecular_weight?: number;
  formula?: string;
  spectrum?: MassSpectrum;
}

export interface MSSearchOptions {
  extraction_method: 'apex' | 'average' | 'peak_window';
  subtraction_enabled: boolean;
  subtraction_method: 'adjacent' | 'manual';
  search_method: 'identity' | 'similarity';
  min_match_score: number;
  max_results: number;
  quality_checks_enabled: boolean;
}

// ─── RT Table ────────────────────────────────────────────────────────

export interface RTTableEntry {
  compound_name: string;
  retention_time: number;
  tolerance?: number;
  formula?: string;
  molecular_weight?: number;
  cas_number?: string;
}

export interface RTTableSettings {
  matching_mode: 'window' | 'closest_apex';
  tolerance: number;
  weight_by_area: boolean;
  entries: RTTableEntry[];
}

// ─── Quantitation ────────────────────────────────────────────────────

export interface QuantitationSettings {
  enabled: boolean;
  internal_standard_name: string;
  internal_standard_mass: number;
  internal_standard_formula: string;
  internal_standard_mw: number;
  polyarc_enabled: boolean;
  sample_mass: number;
  carrier_gas_flow: number;
}

// ─── File System ─────────────────────────────────────────────────────

export interface FileEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  format?: 'agilent_d';
}

export interface BrowseResponse {
  current_path: string;
  parent_path: string | null;
  entries: FileEntry[];
}

// ─── Detector ────────────────────────────────────────────────────────

export interface DetectorInfo {
  name: string;
  description?: string;
  file_path?: string;
}

// ─── Export ──────────────────────────────────────────────────────────

export interface ExportSettings {
  format: 'json' | 'csv' | 'xlsx';
  triggers: {
    after_integration: boolean;
    after_ms_search: boolean;
    after_assignment: boolean;
    during_batch: boolean;
  };
  include_spectra: boolean;
  include_metadata: boolean;
}

// ─── Scaling ─────────────────────────────────────────────────────────

export interface ScalingFactors {
  signal_factor: number;
  area_factor: number;
}
