/**
 * Central Zustand store for ChromaKit-MS web app.
 * Replaces the scattered useState calls in App.jsx with a single,
 * structured store that mirrors the Qt signal/slot architecture.
 */
import { create } from 'zustand';
import type {
  LoadedFileData,
  ProcessingParams,
  ProcessedData,
  IntegrationResults,
  MassSpectrum,
  Peak,
  RTTableSettings,
  QuantitationSettings,
  ScalingFactors,
  DetectorInfo,
} from '../types';
import { DEFAULT_PROCESSING_PARAMS } from '../utils/defaults';

// ─── Store State ─────────────────────────────────────────────────────

interface ChromaKitState {
  // Connection
  apiStatus: 'connected' | 'disconnected' | 'checking';

  // File state
  selectedFilePath: string | null;
  fileData: LoadedFileData | null;
  availableDetectors: DetectorInfo[];
  currentDetector: string | null;

  // Processing
  params: ProcessingParams;
  processedData: ProcessedData | null;
  processing: boolean;

  // Integration
  integrationResults: IntegrationResults | null;
  integrating: boolean;

  // MS
  currentSpectrum: MassSpectrum | null;
  selectedPeakIndex: number | null;
  msSearching: boolean;

  // RT Table
  rtTableSettings: RTTableSettings | null;

  // Quantitation
  quantitationSettings: QuantitationSettings | null;

  // Scaling
  scalingFactors: ScalingFactors;

  // UI state
  loading: boolean;
  error: string | null;
  theme: 'light' | 'dark';

  // Actions
  setApiStatus: (status: 'connected' | 'disconnected' | 'checking') => void;
  setSelectedFile: (path: string | null) => void;
  setFileData: (data: LoadedFileData | null) => void;
  setAvailableDetectors: (detectors: DetectorInfo[]) => void;
  setCurrentDetector: (detector: string | null) => void;
  setParams: (params: ProcessingParams) => void;
  updateParams: (partial: Partial<ProcessingParams>) => void;
  setProcessedData: (data: ProcessedData | null) => void;
  setProcessing: (processing: boolean) => void;
  setIntegrationResults: (results: IntegrationResults | null) => void;
  setIntegrating: (integrating: boolean) => void;
  setCurrentSpectrum: (spectrum: MassSpectrum | null) => void;
  setSelectedPeakIndex: (index: number | null) => void;
  setMsSearching: (searching: boolean) => void;
  setRtTableSettings: (settings: RTTableSettings | null) => void;
  setQuantitationSettings: (settings: QuantitationSettings | null) => void;
  setScalingFactors: (factors: ScalingFactors) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  resetFileState: () => void;
}

// ─── Store ───────────────────────────────────────────────────────────

export const useChromaKitStore = create<ChromaKitState>((set) => ({
  // Initial state
  apiStatus: 'checking',
  selectedFilePath: null,
  fileData: null,
  availableDetectors: [],
  currentDetector: null,
  params: { ...DEFAULT_PROCESSING_PARAMS },
  processedData: null,
  processing: false,
  integrationResults: null,
  integrating: false,
  currentSpectrum: null,
  selectedPeakIndex: null,
  msSearching: false,
  rtTableSettings: null,
  quantitationSettings: null,
  scalingFactors: { signal_factor: 1.0, area_factor: 1.0 },
  loading: false,
  error: null,
  theme: 'light',

  // Actions
  setApiStatus: (status) => set({ apiStatus: status }),
  setSelectedFile: (path) => set({ selectedFilePath: path }),
  setFileData: (data) => set({ fileData: data }),
  setAvailableDetectors: (detectors) => set({ availableDetectors: detectors }),
  setCurrentDetector: (detector) => set({ currentDetector: detector }),
  setParams: (params) => set({ params }),
  updateParams: (partial) =>
    set((state) => ({
      params: { ...state.params, ...partial },
    })),
  setProcessedData: (data) => set({ processedData: data }),
  setProcessing: (processing) => set({ processing }),
  setIntegrationResults: (results) => set({ integrationResults: results }),
  setIntegrating: (integrating) => set({ integrating }),
  setCurrentSpectrum: (spectrum) => set({ currentSpectrum: spectrum }),
  setSelectedPeakIndex: (index) => set({ selectedPeakIndex: index }),
  setMsSearching: (searching) => set({ msSearching: searching }),
  setRtTableSettings: (settings) => set({ rtTableSettings: settings }),
  setQuantitationSettings: (settings) => set({ quantitationSettings: settings }),
  setScalingFactors: (factors) => set({ scalingFactors: factors }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setTheme: (theme) => set({ theme }),
  resetFileState: () =>
    set({
      fileData: null,
      processedData: null,
      integrationResults: null,
      currentSpectrum: null,
      selectedPeakIndex: null,
      availableDetectors: [],
      currentDetector: null,
      error: null,
    }),
}));
