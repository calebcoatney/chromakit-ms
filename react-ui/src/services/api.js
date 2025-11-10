/**
 * API service for ChromaKit-MS backend
 * 
 * This module provides all API calls to the FastAPI backend.
 * Uses axios for HTTP requests with proper error handling.
 */
import axios from 'axios';

// Use proxy in development, direct URL in production
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Browse a directory for .D files
 * @param {string} path - Directory path to browse
 * @returns {Promise} - Browse response with entries
 */
export const browseDirectory = async (path = '.') => {
  const response = await api.get('/browse', { params: { path } });
  return response.data;
};

/**
 * Load a .D file
 * @param {string} filePath - Path to .D file
 * @returns {Promise} - File data with chromatogram and TIC
 */
export const loadFile = async (filePath) => {
  const response = await api.post('/load', { file_path: filePath });
  return response.data;
};

/**
 * Process chromatogram data
 * @param {Object} data - Processing request
 * @param {Array} data.x - Time values
 * @param {Array} data.y - Intensity values
 * @param {Object} data.params - Processing parameters
 * @param {Array} data.ms_range - Optional MS time range
 * @returns {Promise} - Processed data
 */
export const processChromato = async (data) => {
  const response = await api.post('/process', data);
  return response.data;
};

/**
 * Integrate detected peaks
 * @param {Object} data - Integration request
 * @param {Object} data.processed_data - Processed chromatogram data
 * @param {Object} data.rt_table - Optional RT table
 * @param {number} data.chemstation_area_factor - Area scaling factor
 * @returns {Promise} - Integration results
 */
export const integratePeaks = async (data) => {
  const response = await api.post('/integrate', data);
  return response.data;
};

/**
 * Check API health
 * @returns {Promise} - Health status
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
