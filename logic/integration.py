import logging

import numpy as np
from scipy.integrate import simpson
from scipy.stats import skew
import pandas as pd

from logic.feature import Feature
from logic.tail_skim import classify_riders, demote, plan_skims

logger = logging.getLogger(__name__)


def _valley_index(values, offset, fallback):
    """Index of the minimum of ``values``, shifted by ``offset``, ignoring NaN.

    corrected_y and baseline_y are NaN where MS-gated baseline correction
    masked the signal. A plain argmin returns the first NaN, which drops the
    valley bound into the masked region and makes the area NaN. Returns
    ``fallback`` when ``values`` is empty or holds no finite sample.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return fallback
    return offset + int(np.nanargmin(values))

class ChromatographicPeak(Feature):
    """Chromatographic peak. Subclass of Feature with RT, MS, and quantitation fields.

    All original field names are preserved (retention_time, start_time, etc.).
    The base Feature.position property is overridden to return retention_time.
    """
    
    def __init__(self, compound_id, peak_number, retention_time, 
                 integrator, width, area, start_time, end_time,
                 start_index=None, end_index=None):
        """Initialize a ChromatographicPeak object."""
        # Initialize Feature base with generic fields
        super().__init__(
            feature_id=peak_number,
            position=retention_time,
            position_units="min",
            area=area,
            width=width,
            start=start_time,
            end=end_time,
            start_index=start_index or 0,
            end_index=end_index or 0,
        )
        # Chromatography-specific fields — names unchanged from original Peak
        self.compound_id = compound_id
        self.peak_number = peak_number
        self.retention_time = retention_time
        self.integrator = integrator
        self.start_time = start_time
        self.end_time = end_time
        
        # Add MS match properties
        self.compound_name = None
        self.match_score = None
        self.casno = None
        
        # Add quality assessment properties
        self.asymmetry = None
        self.spectral_coherence = None
        self.is_convoluted = False
        self.quality_issues = []
        
        # Add shoulder flag
        self.is_shoulder = False

        # Add negative peak flag
        self.is_negative = False

        # Add saturation properties
        self.is_saturated = False
        self.saturation_level = None

        # Add grouping properties
        self.is_grouped = False
        self.grouped_peak_count = None

        # MS spectral deconvolution results
        self.deconvolved_spectrum = None
        # When populated: {'mz': np.ndarray, 'intensities': np.ndarray}
        # Same format as SpectrumExtractor.extract_for_peak()

        self.deconvolution_component_count = None
        # int: components found in this peak's window; None = not yet run

        # Add quantitation properties (Polyarc + IS method)
        self.mol_C = None  # Moles of carbon
        self.mol_C_percent = None  # Mole percent of carbon
        self.num_carbons = None  # Number of carbons in molecule
        self.mol = None  # Moles of compound
        self.mass_mg = None  # Mass in milligrams
        self.mol_percent = None  # Mole percentage
        self.wt_percent = None  # Weight percentage
        self.raw_amount = None  # Pre-normalization amount (area / RF); RF-table strategy
        self.composition_percent = None  # normalized % in the method's rf_unit basis

        # Tail-skim fields (None = peak was not skimmed / not evaluated)
        self.skim_model = None  # str: tail model used
        self.skim_parent = None  # int: peak_number of the parent peak
        self.skim_source = None  # str: 'auto' or 'forced'
        self.area_before_skim = None  # float: area before skim correction
        self.skimmed_area = None  # float: area removed/added by the skim
        self.skim_children = None  # list[int]: peak_numbers skimmed off this peak
        self.skim_decision = None  # dict: decision, parent, hp_hc, hv_hp, hc_hv, tail_bend, failed
        self.skim_near_threshold = None  # bool: decision was close to a threshold
    
    @property
    def position(self) -> float:
        """Returns retention_time, satisfying the Feature.position interface."""
        return self.retention_time

    @staticmethod
    def column_headers() -> list:
        return ['Compound ID', 'Peak #', 'Ret Time', 'Integrator', 'Width', 'Area', 'Start Time', 'End Time']

    def as_row(self):
        """Return peak data as a row for display in a table."""
        return [
            self.compound_id, 
            self.peak_number, 
            round(self.retention_time, 3),
            self.integrator, 
            round(self.width, 3), 
            round(self.area, 1), 
            round(self.start_time, 3), 
            round(self.end_time, 3)
        ]
    
    def as_dict(self) -> dict:
        """Return all fields as a dict for export (JSON/CSV). Always-present fields first,
        then optional fields only when they hold a value."""
        d = {
            'compound_id': getattr(self, 'Compound_ID', None) or self.compound_id,
            'peak_number': self.peak_number,
            'retention_time': self.retention_time,
            'integrator': self.integrator,
            'width': self.width,
            'area': self.area,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'is_shoulder': self.is_shoulder,
            'is_negative': self.is_negative,
            'is_convoluted': self.is_convoluted,
            'is_saturated': self.is_saturated,
            'is_grouped': self.is_grouped,
            'quality_issues': self.quality_issues,
        }

        # MS identification fields (set post-search)
        qual = getattr(self, 'Qual', None)
        if qual is not None:
            d['Qual'] = float(qual)
        if self.casno:
            d['casno'] = self.casno
        if self.compound_name:
            d['compound_name'] = self.compound_name
        if self.match_score is not None:
            d['match_score'] = self.match_score

        # Quality detail fields (omit when None/empty)
        if self.asymmetry is not None:
            d['asymmetry'] = self.asymmetry
        if self.spectral_coherence is not None:
            d['spectral_coherence'] = self.spectral_coherence
        if self.saturation_level is not None:
            d['saturation_level'] = self.saturation_level
        if self.grouped_peak_count is not None:
            d['grouped_peak_count'] = self.grouped_peak_count

        # Quantitation fields (Polyarc + IS method)
        if self.mol_C is not None:
            d['mol_C'] = self.mol_C
        if self.mol_C_percent is not None:
            d['mol_C_percent'] = self.mol_C_percent
        if self.num_carbons is not None:
            d['num_carbons'] = self.num_carbons
        if self.mol is not None:
            d['mol'] = self.mol
        if self.mass_mg is not None:
            d['mass_mg'] = self.mass_mg
        if self.mol_percent is not None:
            d['mol_percent'] = self.mol_percent
        if self.wt_percent is not None:
            d['wt_percent'] = self.wt_percent
        if self.raw_amount is not None:
            d['raw_amount'] = self.raw_amount
        if self.composition_percent is not None:
            d['composition_percent'] = self.composition_percent

        # Tail-skim fields
        if self.skim_model is not None:
            d['skim_model'] = self.skim_model
        if self.skim_parent is not None:
            d['skim_parent'] = self.skim_parent
        if self.skim_source is not None:
            d['skim_source'] = self.skim_source
        if self.area_before_skim is not None:
            d['area_before_skim'] = self.area_before_skim
        if self.skimmed_area is not None:
            d['skimmed_area'] = self.skimmed_area
        if self.skim_children is not None:
            d['skim_children'] = self.skim_children
        if self.skim_decision is not None:
            d['skim_decision'] = self.skim_decision
        if self.skim_near_threshold is not None:
            d['skim_near_threshold'] = self.skim_near_threshold

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChromatographicPeak":
        """Reconstruct a ChromatographicPeak from its as_dict() serialization.

        Inverse of as_dict(). Tolerates missing optional fields (sets them to
        their __init__ defaults). When `deconvolved_spectrum` is present as
        dict-of-lists (e.g. after a JSON round-trip), arrays are restored via
        np.array() because downstream consumers expect np.ndarray.

        Used by API endpoints to lift peak dicts from request bodies back into
        Peak objects the logic/ layer can mutate.
        """
        peak = cls(
            compound_id=d['compound_id'],
            peak_number=d['peak_number'],
            retention_time=d['retention_time'],
            integrator=d['integrator'],
            width=d['width'],
            area=d['area'],
            start_time=d['start_time'],
            end_time=d['end_time'],
        )
        # Quality flags
        peak.is_shoulder = d.get('is_shoulder', False)
        peak.is_negative = d.get('is_negative', False)
        peak.is_convoluted = d.get('is_convoluted', False)
        peak.is_saturated = d.get('is_saturated', False)
        peak.is_grouped = d.get('is_grouped', False)
        peak.quality_issues = d.get('quality_issues', [])
        # MS identification fields (added by BatchSearchWorker at runtime)
        if 'Qual' in d:
            peak.Qual = d['Qual']
        if 'casno' in d:
            peak.casno = d['casno']
        if 'compound_name' in d:
            peak.compound_name = d['compound_name']
        if 'match_score' in d:
            peak.match_score = d['match_score']
        # Quality detail
        if 'asymmetry' in d:
            peak.asymmetry = d['asymmetry']
        if 'spectral_coherence' in d:
            peak.spectral_coherence = d['spectral_coherence']
        if 'saturation_level' in d:
            peak.saturation_level = d['saturation_level']
        if 'grouped_peak_count' in d:
            peak.grouped_peak_count = d['grouped_peak_count']
        # Quantitation
        if 'mol_C' in d:
            peak.mol_C = d['mol_C']
        if 'mol_C_percent' in d:
            peak.mol_C_percent = d['mol_C_percent']
        if 'num_carbons' in d:
            peak.num_carbons = d['num_carbons']
        if 'mol' in d:
            peak.mol = d['mol']
        if 'mass_mg' in d:
            peak.mass_mg = d['mass_mg']
        if 'mol_percent' in d:
            peak.mol_percent = d['mol_percent']
        if 'wt_percent' in d:
            peak.wt_percent = d['wt_percent']
        if 'raw_amount' in d:
            peak.raw_amount = d['raw_amount']
        if 'composition_percent' in d:
            peak.composition_percent = d['composition_percent']
        # Tail-skim fields
        if 'skim_model' in d:
            peak.skim_model = d['skim_model']
        if 'skim_parent' in d:
            peak.skim_parent = d['skim_parent']
        if 'skim_source' in d:
            peak.skim_source = d['skim_source']
        if 'area_before_skim' in d:
            peak.area_before_skim = d['area_before_skim']
        if 'skimmed_area' in d:
            peak.skimmed_area = d['skimmed_area']
        if 'skim_children' in d:
            peak.skim_children = d['skim_children']
        if 'skim_decision' in d:
            peak.skim_decision = d['skim_decision']
        if 'skim_near_threshold' in d:
            peak.skim_near_threshold = d['skim_near_threshold']
        # Deconvolved spectrum (not emitted by as_dict but may be in API payloads)
        if 'deconvolved_spectrum' in d and d['deconvolved_spectrum'] is not None:
            spec = d['deconvolved_spectrum']
            peak.deconvolved_spectrum = {
                'mz': np.asarray(spec['mz']),
                'intensities': np.asarray(spec['intensities']),
            }
        if 'deconvolution_component_count' in d:
            peak.deconvolution_component_count = d['deconvolution_component_count']
        return peak

    @property
    def apex_time(self):
        """Get the retention time at the peak apex."""
        return self.retention_time

Peak = ChromatographicPeak  # backward-compatibility alias — no call sites need to change

# Register ChromatographicPeak as the feature_class for gc/gcms profiles
from logic.signal_profiles import _update_chromatographic_profiles
_update_chromatographic_profiles(ChromatographicPeak)


def integrate_bands(x, y, bands, profile):
    """Integrate fixed x-windows, independent of peak detection.

    Args:
        x: 1-D array of x positions (native units; may be ascending or
           descending, e.g. inverted-x FTIR wavenumbers).
        y: 1-D array of the signal to integrate. Caller passes the
           baseline-corrected signal when baseline is enabled, else raw.
        bands: list of BandWindow (name, x_min, x_max) with x_min < x_max.
        profile: SignalProfile (used for position_units via x_label).

    Returns:
        list[SpectralFeature] — exactly one per band, in input order. A band
        with no samples in range yields a zero-area feature carrying a
        quality issue rather than being dropped.
    """
    from logic.feature import SpectralFeature

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    units = profile.x_label if profile is not None else ""

    features = []
    for i, band in enumerate(bands):
        x_min = min(band.x_min, band.x_max)
        x_max = max(band.x_min, band.x_max)
        mask = (x >= x_min) & (x <= x_max)
        idx = np.nonzero(mask)[0]

        if idx.size == 0:
            features.append(SpectralFeature(
                feature_id=i,
                position=(x_min + x_max) / 2.0,
                position_units=units,
                area=0.0, width=(x_max - x_min),
                start=x_min, end=x_max,
                start_index=-1, end_index=-1,
                band_assignment=band.name,
                absorbance=0.0,
                quality_issues=[f"no samples in [{x_min}, {x_max}]"],
            ))
            continue

        x_win = x[idx]
        y_win = y[idx]

        # Direction-safe integration: sort ascending by x before simpson.
        order = np.argsort(x_win)
        x_sorted = x_win[order]
        y_sorted = y_win[order]
        area = float(abs(simpson(y_sorted, x=x_sorted)))

        max_pos = int(np.argmax(y_win))
        absorbance = float(y_win[max_pos])
        position = float(x_win[max_pos])

        features.append(SpectralFeature(
            feature_id=i,
            position=position,
            position_units=units,
            area=area,
            width=(x_max - x_min),
            start=x_min, end=x_max,
            start_index=int(idx.min()), end_index=int(idx.max()),
            band_assignment=band.name,
            absorbance=absorbance,
        ))

    return features


class Integrator:
    """Provides functionality for integrating chromatographic peaks."""
    
    @staticmethod
    def identify_compound(retention_time, rt_table=None):
        """Identify a compound based on retention time.
        
        Args:
            retention_time: Retention time to match
            rt_table: Dictionary or DataFrame mapping retention times to compounds
            
        Returns:
            str: Identified compound or 'Unknown'
        """
        if rt_table is None:
            return f"Unknown ({retention_time:.3f})"
            
        # If rt_table is a DataFrame
        if isinstance(rt_table, pd.DataFrame):
            # Find the closest retention time within a tolerance
            # This is a simple implementation that can be refined
            tolerance = 0.1  # Tolerance in minutes
            closest_match = None
            min_diff = float('inf')
            
            for _, row in rt_table.iterrows():
                if 'retention_time' in row:
                    diff = abs(row['retention_time'] - retention_time)
                    if diff < min_diff and diff < tolerance:
                        min_diff = diff
                        closest_match = row.get('compound_name', f"Unknown ({retention_time:.3f})")
            
            return closest_match if closest_match else f"Unknown ({retention_time:.3f})"
            
        # If rt_table is a dictionary with retention time ranges
        elif isinstance(rt_table, dict):
            for compound, rt_range in rt_table.items():
                if isinstance(rt_range, tuple) and len(rt_range) == 2:
                    if rt_range[0] <= retention_time <= rt_range[1]:
                        return compound
                elif isinstance(rt_range, (int, float)):
                    # If exact match or within tolerance
                    if abs(rt_range - retention_time) <= 0.1:
                        return compound
            
            return f"Unknown ({retention_time:.3f})"
        
        # If rt_table format is not supported
        return f"Unknown ({retention_time:.3f})"
    
    @staticmethod
    def _calculate_derivatives(x, y, window_length=41, polyorder=3):
        """Calculate first and second derivatives using Savitzky-Golay filter.
        
        Args:
            x: Array of x values
            y: Array of y values
            window_length: Window length for Savitzky-Golay filter
            polyorder: Polynomial order for Savitzky-Golay filter
            
        Returns:
            tuple: (dy, d2y) first and second derivatives
        """
        from scipy.signal import savgol_filter
        
        # Ensure valid smoothing parameters
        data_length = len(y)
        if window_length >= data_length:
            window_length = min(data_length - 1, 41)
            if window_length % 2 == 0:
                window_length -= 1
        elif window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1
            
        # Smooth signal for derivative calculation
        y_smooth = savgol_filter(y, window_length, polyorder)
        
        # Calculate derivatives
        dy = np.gradient(y_smooth, x)
        d2y = np.gradient(dy, x)
        
        return dy, d2y
    
    @staticmethod
    def _find_second_derivative_bounds(x, y, apex_idx, dy=None, d2y=None, window_length=41, polyorder=3):
        """Find integration bounds using second derivative maxima.
        
        Args:
            x: Array of x values
            y: Array of y values
            apex_idx: Index of the peak apex
            dy: First derivative (optional, will calculate if not provided)
            d2y: Second derivative (optional, will calculate if not provided)
            window_length: Window length for Savitzky-Golay filter
            polyorder: Polynomial order for Savitzky-Golay filter
            
        Returns:
            tuple: (left_bound_idx, right_bound_idx)
        """
        from scipy.signal import find_peaks
        
        # Calculate derivatives if not provided
        if dy is None or d2y is None:
            dy, d2y = Integrator._calculate_derivatives(x, y, window_length, polyorder)
        
        # Find local maxima in second derivative (curvature change points)
        maxima_indices, _ = find_peaks(d2y)
        
        # Find left bound: largest maxima index < apex_idx
        left_bound = None
        left_candidates = maxima_indices[maxima_indices < apex_idx]
        if len(left_candidates) > 0:
            left_bound = int(np.max(left_candidates))
        
        # Find right bound: smallest maxima index > apex_idx
        right_bound = None
        right_candidates = maxima_indices[maxima_indices > apex_idx]
        if len(right_candidates) > 0:
            right_bound = int(np.min(right_candidates))
        
        # Fallbacks if bounds not found
        if left_bound is None:
            left_bound = max(0, apex_idx - 20)  # Default to 20 points left
        if right_bound is None:
            right_bound = min(len(y) - 1, apex_idx + 20)  # Default to 20 points right
        
        return left_bound, right_bound
    
    @staticmethod
    def _group_index_span(x, g_start, g_end):
        """Index span ``(start, end)`` of the peak-group window ``[g_start, g_end]`` on ascending ``x``."""
        start_idx = max(0, int(np.searchsorted(x, g_start, side='left')))
        end_idx = min(len(x) - 1, int(np.searchsorted(x, g_end, side='right')) - 1)
        return start_idx, end_idx

    @staticmethod
    def _apply_peak_grouping(peaks_list, peak_groups, x, y, baseline_y,
                              x_peaks, y_peaks, baseline_peaks,
                              ret_times, integrated_areas, integration_bounds, profile=None):
        """Merge peaks within user-specified group windows into composite peaks.
        
        Args:
            peaks_list: List of Peak objects from integration
            peak_groups: List of [start, end] time windows
            x: Full chromatogram time array
            y: Full chromatogram signal array (for finding apex)
            baseline_y: Full baseline array
            x_peaks, y_peaks, baseline_peaks: Per-peak data arrays
            ret_times, integrated_areas, integration_bounds: Per-peak summary lists
            profile: Optional SignalProfile for feature class selection
            
        Returns:
            Tuple of updated (peaks_list, x_peaks, y_peaks, baseline_peaks,
                              ret_times, integrated_areas, integration_bounds)
        """
        feature_cls = profile.feature_class if (profile is not None and profile.feature_class is not None) else Peak

        # Peak grouping keys on retention_time; it is only valid for
        # chromatographic peaks. Spectroscopy features have no RT — return the
        # inputs unchanged rather than raising AttributeError.
        if not all(hasattr(p, "retention_time") for p in peaks_list):
            return (peaks_list, x_peaks, y_peaks, baseline_peaks,
                    ret_times, integrated_areas, integration_bounds)

        # Track which peaks get consumed by a group
        consumed_indices = set()
        grouped_peaks = []  # (sort_key, Peak, x_peak, y_peak, baseline_peak)
        
        for group in peak_groups:
            if len(group) != 2:
                continue
            g_start, g_end = float(group[0]), float(group[1])
            
            # Find all peaks whose apex RT falls within this group window
            member_indices = []
            for i, peak in enumerate(peaks_list):
                if g_start <= peak.retention_time <= g_end and i not in consumed_indices:
                    member_indices.append(i)
            
            if len(member_indices) == 0:
                continue
            
            consumed_indices.update(member_indices)
            
            # Compute composite values
            total_area = sum(peaks_list[i].area for i in member_indices)
            
            # Apex = time of max signal within the group window
            window_mask = (x >= g_start) & (x <= g_end)
            if np.any(window_mask):
                window_y = y[window_mask]
                window_x = x[window_mask]
                apex_idx_in_window = np.argmax(window_y)
                apex_rt = float(window_x[apex_idx_in_window])
            else:
                # Fallback: use the peak with the largest area
                best_i = max(member_indices, key=lambda i: peaks_list[i].area)
                apex_rt = peaks_list[best_i].retention_time
            
            # Build start/end indices from the chromatogram arrays
            start_idx, end_idx = Integrator._group_index_span(x, g_start, g_end)
            
            # Create composite Peak
            width = g_end - g_start
            compound_id = "Group"
            if feature_cls is Peak or issubclass(feature_cls, ChromatographicPeak):
                composite = feature_cls(
                    compound_id=compound_id,
                    peak_number=0,
                    retention_time=apex_rt,
                    integrator='py',
                    width=width,
                    area=total_area,
                    start_time=g_start,
                    end_time=g_end,
                    start_index=start_idx,
                    end_index=end_idx
                )
            else:
                composite = feature_cls(
                    feature_id=0,
                    position=apex_rt,
                    position_units=profile.x_label if profile else "",
                    area=total_area,
                    width=width,
                    start=g_start,
                    end=g_end,
                    start_index=start_idx,
                    end_index=end_idx,
                )
            composite.is_grouped = True
            composite.grouped_peak_count = len(member_indices)
            
            # Build per-peak arrays for the group window
            gx = x[start_idx:end_idx + 1]
            gy = y[start_idx:end_idx + 1]
            gb = baseline_y[start_idx:end_idx + 1]
            
            grouped_peaks.append((g_start, composite, gx, gy, gb))
        
        # Build final lists: ungrouped peaks + grouped peaks, sorted by RT
        new_peaks = []
        new_x_peaks = []
        new_y_peaks = []
        new_baseline_peaks = []
        new_ret_times = []
        new_areas = []
        new_bounds = []
        
        # Add ungrouped peaks
        for i, peak in enumerate(peaks_list):
            if i not in consumed_indices:
                new_peaks.append((peak.retention_time, peak, x_peaks[i], y_peaks[i], baseline_peaks[i]))
        
        # Add grouped peaks
        for sort_key, composite, gx, gy, gb in grouped_peaks:
            new_peaks.append((sort_key, composite, gx, gy, gb))
        
        # Sort by retention time / start time
        new_peaks.sort(key=lambda t: t[0])
        
        # Unpack and re-number
        final_peaks = []
        for idx, (_, peak, xp, yp, bp) in enumerate(new_peaks):
            peak.peak_number = idx + 1
            final_peaks.append(peak)
            new_x_peaks.append(xp)
            new_y_peaks.append(yp)
            new_baseline_peaks.append(bp)
            new_ret_times.append(peak.retention_time)
            new_areas.append(peak.area)
            new_bounds.append((peak.start_time, peak.end_time))
        
        if len(consumed_indices) > 0:
            print(f"Peak grouping: Merged {len(consumed_indices)} peaks into {len(grouped_peaks)} groups")
        
        return final_peaks, new_x_peaks, new_y_peaks, new_baseline_peaks, new_ret_times, new_areas, new_bounds
    
    @staticmethod
    def _plan_riders(x, signal, apexes, peaks_x, peak_metadata, processed_data, group_spans):
        """Split-or-skim calls for every child apex and the skims they resolve to (logic.tail_skim).

        ``apexes`` are the sorted indices of kept and range-excluded apexes; an excluded apex has no
        metadata. Returns ``(calls, skims)``, both keyed by the child's apex index.
        """
        meta_at = {int(np.flatnonzero(x == v)[0]): (peak_metadata[i] if i < len(peak_metadata) else None) or {}
                   for i, v in enumerate(peaks_x)}
        negative = {k for k, m in meta_at.items() if m.get('is_negative')}
        calls = classify_riders(x, signal, apexes, [meta_at.get(int(k), {}) for k in apexes],
                                processed_data.get('rider'), processed_data.get('peak_windows') or [],
                                negative, group_spans)
        return plan_skims(x, signal, calls)

    @staticmethod
    def _skim_area(x, signal, skim, lo, hi, area_factor):
        """Area of a rider above its skim curve over ``[lo, hi]`` (inside its span), scaled like every
        peak. A rider with nothing left above the curve gets 0, with a logged warning."""
        a = skim.call.a
        area = simpson(signal[lo:hi + 1] - skim.curve.s[lo - a:hi - a + 1], x=x[lo:hi + 1])
        if area <= 0:
            logger.warning("tail skim: peak at %.3f min has no area above its parent's tail; area set to 0",
                           x[skim.call.child])
            return 0.0
        if area_factor:
            area *= area_factor
        return abs(area)

    @staticmethod
    def _net_parent_areas(peaks_list, peak_at, integrated_areas, skims, riders_of, x, signal, area_factor):
        """Take each reported parent's riders' areas off its area, keeping ``integrated_areas`` in step.

        A rider outside every window is not reported but is still skimmed and subtracted (spec,
        Interface with piecewise methods). Returns ``{parent apex: (area before skim, skimmed area)}``.
        """
        totals = {}
        for parent, riders in riders_of.items():
            peak = peak_at.get(parent)
            if peak is None:
                continue  # parent outside every window: nothing to subtract from
            skimmed = sum(peak_at[c].area if c in peak_at
                          else Integrator._skim_area(x, signal, skims[c], skims[c].call.a, skims[c].b,
                                                     area_factor)
                          for c in riders)
            totals[parent] = (peak.area, skimmed)
            peak.area = peak.area - skimmed
            integrated_areas[next(k for k, p in enumerate(peaks_list) if p is peak)] = peak.area
        return totals

    @staticmethod
    def _set_skim_fields(peaks_list, calls, skims, riders_of, peak_at, totals):
        """Tail-skim fields on the reported peaks, with their final peak numbers."""
        reported = {id(p) for p in peaks_list}

        def number(apex):
            peak = peak_at.get(apex)
            return peak.peak_number if peak is not None and id(peak) in reported else None

        for c, call in calls.items():
            peak = peak_at.get(c)
            if peak is None:
                continue
            ratios = {k: float(v) if np.isfinite(v) else None for k, v in call.ratios.items()}
            peak.skim_decision = {'decision': call.decision, 'parent': number(call.parent), **ratios,
                                  'failed': list(call.failed)}
            peak.skim_near_threshold = bool(call.near_threshold)
            if c in skims:
                peak.skim_model = skims[c].curve.model
                peak.skim_parent = number(call.parent)
                peak.skim_source = call.source
        for parent, (before, skimmed) in totals.items():
            peak = peak_at[parent]
            peak.area_before_skim = before
            peak.skimmed_area = skimmed
            peak.skim_children = [number(c) for c in riders_of[parent] if number(c) is not None]

    @staticmethod
    def _shade_arrays(peaks_list, y_peaks, baseline_peaks, corrected, skims, riders_of, peak_at, signal):
        """Top and floor of each peak's shaded area, in corrected space and aligned with ``x_peaks``.

        A rider's floor is its skim curve; a parent's top is the signal with its riders' skim curves
        put in under them. Every other peak is its integrated signal over a zero floor.
        """
        tops = [yp if corrected else yp - bp for yp, bp in zip(y_peaks, baseline_peaks)]
        floors = [np.zeros(len(yp)) for yp in y_peaks]
        slot = {id(p): k for k, p in enumerate(peaks_list)}

        def under(skim, lo, hi):
            """The skim curve over ``[lo, hi]`` clipped to its span, as (offset from lo, values)."""
            a, b = skim.call.a, skim.b
            s_lo, s_hi = max(a, lo), min(b, hi)
            if s_hi < s_lo:  # the peak's (clamped) bounds end before the rider starts
                return 0, skim.curve.s[:0]
            return s_lo - lo, skim.curve.s[s_lo - a:s_hi - a + 1]

        for c, skim in skims.items():
            k = slot.get(id(peak_at.get(c)))
            if k is not None:
                peak = peaks_list[k]
                off, s = under(skim, peak.start_index, peak.end_index)
                floors[k][off:off + len(s)] = s
        for parent, riders in riders_of.items():
            k = slot.get(id(peak_at.get(parent)))
            if k is None:
                continue
            peak = peaks_list[k]
            top = signal[peak.start_index:peak.end_index + 1].copy()
            for c in riders:
                off, s = under(skims[c], peak.start_index, peak.end_index)
                top[off:off + len(s)] = s
            tops[k] = top
        return tops, floors

    @staticmethod
    def integrate(processed_data, rt_table=None, area_factor=None, verbose=True, ms_data=None, quality_options=None, peak_groups=None, profile=None):
        """Integrate peaks in a chromatogram.
        
        Args:
            processed_data: Dictionary containing the processed chromatogram data
            rt_table: Dictionary or DataFrame mapping retention times to compounds
            area_factor: Optional area scaling multiplier applied to each peak
                area. None or 0.0 mean no scaling (x1); 0.0 additionally logs a
                warning (use 1.0 to leave areas unaltered).
            verbose: Whether to print integration results
            ms_data: Optional MS data object for peak quality assessment
            quality_options: Options for peak quality assessment
            peak_groups: Optional list of [start, end] time windows for peak grouping
            profile: Optional SignalProfile for feature class selection
            
        Returns:
            dict: Dictionary containing integration results
        """
        feature_cls = profile.feature_class if (profile is not None and profile.feature_class is not None) else Peak
        # Extract relevant values from the processed data
        x = processed_data['x']
        y = processed_data.get('smoothed_y', processed_data.get('original_y', processed_data.get('corrected_y')))
        baseline_y = processed_data['baseline_y']
        peaks_x = processed_data['peaks_x']
        peaks_y = processed_data['peaks_y']
        
        # Debug: Check if we have peaks to integrate
        if verbose:
            print(f"Integration starting: Found {len(peaks_x)} peaks to integrate")
            if len(peaks_x) == 0:
                print("ERROR: No peaks found in processed_data!")
                print(f"Available keys in processed_data: {list(processed_data.keys())}")
                return {
                    'peaks': [],
                    'x_peaks': [],
                    'y_peaks': [],
                    'baseline_peaks': [],
                    'retention_times': [],
                    'integrated_areas': [],
                    'integration_bounds': [],
                    'peaks_list': [],
                    'shade_top_peaks': [],
                    'shade_floor_peaks': [],
                }
        
        # Check if we have peak metadata with shoulder information
        has_shoulder_info = 'peak_metadata' in processed_data and len(processed_data.get('peak_metadata', [])) > 0
        peak_metadata = processed_data.get('peak_metadata', [])
        
        # Use the original signal for derivative calculation (shoulder detection).
        # Bound-walking threshold and valley-finding use integration_signal (smoothed +
        # baseline-corrected) to prevent raw noise spikes from triggering premature stops.
        bounds_detection_signal = processed_data.get('original_y', y)
        integration_signal = processed_data.get('corrected_y', y)  # Keep for area calculation
        
        # Calculate derivatives for bound detection using the bounds detection signal
        dy, d2y = Integrator._calculate_derivatives(x, bounds_detection_signal)
        
        # Integration criteria
        criteria = ['threshold', 'minimum']
        
        # Initialize lists to store the integration data
        peaks_list = []
        ret_times = []
        integrated_areas = []
        integration_bounds = []
        x_peaks = []
        y_peaks = []
        baseline_peaks = []
        
        # First, identify shoulders and their bounds
        shoulder_bounds = {}  # Index to (left_bound, right_bound)
        shoulder_indices = []  # Indices of shoulders
        
        if has_shoulder_info:
            for i, meta in enumerate(peak_metadata):
                if meta.get('is_shoulder', False):
                    idx = meta.get('index')
                    shoulder_indices.append(idx)
                    
                    # Get pre-calculated bounds if available
                    left_bound = meta.get('left_bound')
                    right_bound = meta.get('right_bound')
                    
                    # If bounds not in metadata, calculate them using second derivative
                    if left_bound is None or right_bound is None:
                        left_bound, right_bound = Integrator._find_second_derivative_bounds(
                            x, bounds_detection_signal, idx, dy, d2y
                        )
                    
                    shoulder_bounds[idx] = (left_bound, right_bound)
        
        # Warn once per integrate() call. area_factor=0.0 is a common mistake
        # ("disable"); None or 0.0 => no scaling (x1), never zeroing peaks.
        if area_factor == 0.0:
            logger.warning(
                "area_factor=0.0 does not disable scaling; treating as x1. "
                "Use 1.0 to leave areas unaltered."
            )

        # Neighbouring apexes bound each peak's valley search and overlap
        # clamp. Peaks dropped by a range filter still count as neighbours:
        # the valley between a kept peak and an excluded one is a real bound.
        # Ordered by array index, not value, so descending x-axes (FTIR) work.
        range_excluded = processed_data.get('range_excluded_peaks_x', [])
        neighbour_idx = np.unique([
            int(np.flatnonzero(x == v)[0])
            for v in np.concatenate([np.asarray(peaks_x), np.asarray(range_excluded)])
            if np.any(x == v)
        ]).astype(int)
        range_filters = processed_data.get('range_filters') or []

        # Tail skim, for chromatographic profiles only (spectral x-axes descend). Riders are worked
        # out before any bounds: a rider's span is set by its skim, and its parent's bounds ignore it.
        skim_signal = (integration_signal if integration_signal is processed_data.get('corrected_y')
                       else integration_signal - baseline_y)
        group_spans = [Integrator._group_index_span(x, float(g[0]), float(g[1]))
                       for g in (peak_groups or []) if len(g) == 2]
        calls, skims = {}, {}
        if issubclass(feature_cls, ChromatographicPeak):
            calls, skims = Integrator._plan_riders(x, skim_signal, neighbour_idx, peaks_x, peak_metadata,
                                                   processed_data, group_spans)
        riders_of = {}  # parent apex index -> apex indices of its skimmed riders, in RT order
        for c in sorted(skims):
            riders_of.setdefault(skims[c].call.parent, []).append(c)

        def clamp_to_window(i, apex_x, left_bound, right_bound):
            # Clamp bounds to the span allowed for this apex. The processor puts
            # a per-peak 'clamp' in the metadata (a None side is unbounded).
            # Processed dicts without one (older API clients) fall back to the
            # range filters holding the apex; these may overlap, so use their
            # union. By value, so descending x-axes work.
            clamp = peak_metadata[i].get('clamp') if i < len(peak_metadata) else None
            if clamp is None:
                windows = [(s, e) for s, e in range_filters if s <= apex_x <= e]
                clamp = ((min(s for s, _ in windows), max(e for _, e in windows))
                         if windows else (None, None))
            lo, hi = clamp
            if lo is not None or hi is not None:
                lo = -np.inf if lo is None else lo
                hi = np.inf if hi is None else hi
                in_window = np.flatnonzero((x >= lo) & (x <= hi))
                left_bound = max(left_bound, int(in_window.min()))
                right_bound = min(right_bound, int(in_window.max()))
            return left_bound, right_bound

        def walk_bounds(i, apex_x, peak_idx, is_shoulder, is_negative, riders):
            """Bounds of kept peak ``i``: the bound walk, the overlap clamp and the window clamp.

            ``riders`` are the apex indices of this peak's skimmed riders. They are not neighbours for
            the valley search and the overlap clamp, they do not limit the right bound as shoulders,
            and the right bound reaches at least the last one's end (spec section 2).
            """
            nbrs = np.setdiff1d(neighbour_idx, riders) if riders else neighbour_idx
            pos = np.searchsorted(nbrs, peak_idx)
            prev_apex_x = x[nbrs[pos - 1]] if pos > 0 else None
            nxt = np.searchsorted(nbrs, peak_idx, side='right')
            next_apex_x = x[nbrs[nxt]] if nxt < len(nbrs) else None
            # For the narrow-bounds warnings; a shoulder keeps its detected bounds
            left_stop_reason = right_stop_reason = "shoulder bounds"
            threshold = apex_vertical_distance = np.nan

            # Get bounds based on peak type
            if is_shoulder and peak_idx in shoulder_bounds:
                # If it's a shoulder, use the second derivative bounds
                left_bound, right_bound = shoulder_bounds[peak_idx]
            else:
                # For regular peaks, use standard approach but with shoulder awareness
                # Initialize left and right bounds based on peak apex
                left_bound = peak_idx
                right_bound = peak_idx
                
                # Define the range for min_left and min_right calculations
                if prev_apex_x is not None:  # If not the first peak
                    # Find the index of the previous peak's x value in the x array
                    start_idx = np.where(x == prev_apex_x)[0][0] if np.any(x == prev_apex_x) else None
                    if start_idx is not None:
                        # The neighbour may be a range-excluded apex inside the
                        # MS-gated NaN region, so the valley search skips NaN.
                        if is_negative:
                            # For negative peaks, find the point closest to the baseline
                            # between the previous peak and this one (baseline crossing)
                            segment = bounds_detection_signal[start_idx:peak_idx]
                            baseline_segment = baseline_y[start_idx:peak_idx]
                            min_left = _valley_index(np.abs(segment - baseline_segment), start_idx, peak_idx)
                        else:
                            # Use the corrected signal so the valley is found in baseline-corrected
                            # space. On a rising raw baseline argmin would always land at start_idx.
                            min_left = _valley_index(integration_signal[start_idx:peak_idx], start_idx, peak_idx)
                    else:
                        min_left = 0
                else:
                    # First peak: slice may include NaN from the MS-gated masked region.
                    # An all-NaN or empty slice falls back to 0, as before.
                    min_left = _valley_index(integration_signal[0:peak_idx], 0, 0)

                if next_apex_x is not None:  # If not the last peak
                    # Find the index of the next peak's x value in the x array
                    end_idx = np.where(x == next_apex_x)[0][0] if np.any(x == next_apex_x) else None
                    if end_idx is not None:
                        if is_negative:
                            # For negative peaks, find the point closest to the baseline
                            segment = bounds_detection_signal[peak_idx:end_idx]
                            baseline_segment = baseline_y[peak_idx:end_idx]
                            min_right = _valley_index(np.abs(segment - baseline_segment), peak_idx, peak_idx)
                        else:
                            min_right = _valley_index(integration_signal[peak_idx:end_idx], peak_idx, peak_idx)
                    else:
                        min_right = len(bounds_detection_signal) - 1
                else:
                    min_right = len(bounds_detection_signal) - 1
                
                # Calculate vertical distance between apex and baseline (peak height).
                # Use integration_signal for consistency with diff in bound-walking loops.
                apex_vertical_distance = integration_signal[peak_idx]

                # For negative peaks, distance is negative — use absolute value
                if is_negative or apex_vertical_distance < 0:
                    apex_vertical_distance = abs(apex_vertical_distance)
                
                # The threshold is 0.25% of the apex height
                threshold = apex_vertical_distance * 0.0025
                
                dx = 25  # Step size for slope calculation
                
                # Check for shoulder to the left that would limit this peak's left bound
                # (skip shoulder interactions for negative peaks)
                left_limit_by_shoulder = None
                if not is_negative:
                    for s_idx, s_bounds in shoulder_bounds.items():
                        # If shoulder apex is to the left of this peak and its right bound is after this peak's min_left
                        if s_idx < peak_idx and s_bounds[1] > min_left:
                            left_limit_by_shoulder = s_bounds[1]

                # Calculate the left bound using the specified criteria.
                # Use integration_signal (smoothed + corrected) for diff so that raw noise
                # spikes below baseline don't prematurely fire the threshold criterion.
                previous_slope = None
                left_stop_reason = "no_criteria_met"
                for j in range(peak_idx, 2, -1):
                    diff = integration_signal[j]
                    
                    # If we've hit a shoulder's right bound, stop
                    if left_limit_by_shoulder is not None and j <= left_limit_by_shoulder:
                        left_bound = max(j, left_limit_by_shoulder)
                        left_stop_reason = "shoulder_limit"
                        break
                    
                    # Central difference approximation
                    if j >= dx and j < len(bounds_detection_signal) - dx:
                        slope = (bounds_detection_signal[j] - bounds_detection_signal[j - dx]) / (x[j] - x[j - dx])
                    else:
                        # Edge case fallback to first-order difference
                        slope = (bounds_detection_signal[j] - bounds_detection_signal[j - 1]) / (x[j] - x[j - 1])
                    
                    # Check based on provided criteria
                    if 'threshold' in criteria:
                        # For negative peaks, compare abs(diff) since diff is negative
                        check_val = abs(diff) if is_negative else diff
                        if check_val <= threshold:
                            left_bound = j
                            left_stop_reason = f"threshold (check_val={check_val:.6f} <= {threshold:.6f})"
                            break

                    if 'minimum' in criteria and j == min_left:
                        left_bound = j
                        left_stop_reason = "minimum"
                        break

                    if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                        left_bound = j
                        left_stop_reason = f"slope_change (slope={slope:.6f}, prev_slope={previous_slope:.6f})"
                        break

                    previous_slope = slope

                # Check for shoulder to the right that would limit this peak's right bound
                # (skip shoulder interactions for negative peaks)
                right_limit_by_shoulder = None
                if not is_negative:
                    for s_idx, s_bounds in shoulder_bounds.items():
                        # If shoulder apex is to the right of this peak and its left bound is before this peak's min_right
                        if s_idx > peak_idx and s_bounds[0] < min_right and s_idx not in riders:
                            right_limit_by_shoulder = s_bounds[0]
                
                # Calculate the right bound similarly
                previous_slope = None
                right_stop_reason = "no_criteria_met"
                for j in range(peak_idx, len(bounds_detection_signal) - 3):
                    diff = integration_signal[j]
                    
                    # If we've hit a shoulder's left bound, stop
                    if right_limit_by_shoulder is not None and j >= right_limit_by_shoulder:
                        right_bound = min(j, right_limit_by_shoulder)
                        right_stop_reason = "shoulder_limit"
                        break
                    
                    # Central difference approximation
                    if j >= dx and j < len(bounds_detection_signal) - dx:
                        slope = (bounds_detection_signal[j + dx] - bounds_detection_signal[j]) / (x[j + dx] - x[j])
                    else:
                        # Edge case fallback to first-order difference
                        slope = (bounds_detection_signal[j + 1] - bounds_detection_signal[j]) / (x[j + 1] - x[j])
                    
                    # Check based on provided criteria
                    if 'threshold' in criteria:
                        # For negative peaks, compare abs(diff) since diff is negative
                        check_val = abs(diff) if is_negative else diff
                        if check_val <= threshold:
                            right_bound = j
                            right_stop_reason = f"threshold (check_val={check_val:.6f} <= {threshold:.6f})"
                            break
                    
                    if 'minimum' in criteria and j == min_right:
                        right_bound = j
                        right_stop_reason = "minimum"
                        break
                    
                    if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                        right_bound = j
                        right_stop_reason = f"slope_change (slope={slope:.6f}, prev_slope={previous_slope:.6f})"
                        break
                    
                    previous_slope = slope
            
            # Clamp bounds to prevent overlap with adjacent peaks
            if prev_apex_x is not None:
                prev_peak_idx = np.where(x == prev_apex_x)[0][0] if np.any(x == prev_apex_x) else None
                if prev_peak_idx is not None and left_bound <= prev_peak_idx:
                    # Don't extend past the previous peak's apex; use midpoint as boundary
                    left_bound = (prev_peak_idx + peak_idx) // 2
            if next_apex_x is not None:
                next_peak_idx = np.where(x == next_apex_x)[0][0] if np.any(x == next_apex_x) else None
                if next_peak_idx is not None and right_bound >= next_peak_idx:
                    # Don't extend past the next peak's apex; use midpoint as boundary
                    right_bound = (peak_idx + next_peak_idx) // 2

            if riders:
                right_bound = max(right_bound, max(skims[c].b for c in riders))
            left_bound, right_bound = clamp_to_window(i, apex_x, left_bound, right_bound)
            return left_bound, right_bound, (left_stop_reason, right_stop_reason, threshold,
                                             apex_vertical_distance)

        # Iterate through each peak for integration
        peak_at = {}  # apex index -> its peak, for the skim fields and shading
        for i, (apex_x, detected_apex_y) in enumerate(zip(peaks_x, peaks_y)):
            # Find the index of the apex in x and y arrays
            peak_idx = np.where(x == apex_x)[0][0]
            baseline_at_apex = baseline_y[peak_idx]
            
            # IMPORTANT: Use the bounds detection signal value at apex for threshold calculation
            # This ensures consistency between apex height calculation and integration bounds
            apex_y = bounds_detection_signal[peak_idx]
            
            # Define peak number early for use in verbose output
            peak_number = i + 1
            
            # Check if this peak is a shoulder
            is_shoulder = peak_idx in shoulder_indices

            # Check if this peak is a negative peak
            is_negative = False
            if has_shoulder_info and i < len(peak_metadata):
                is_negative = peak_metadata[i].get('is_negative', False)

            skim = skims.get(int(peak_idx))
            if skim is not None:
                # A rider: its span is the skim's (spec section 3); no bound walk, no overlap clamp
                left_bound, right_bound = clamp_to_window(i, apex_x, skim.call.a, skim.b)
                stop_info = ("tail skim", "tail skim", np.nan, np.nan)
            else:
                riders = riders_of.get(int(peak_idx), [])
                left_bound, right_bound, stop_info = walk_bounds(i, apex_x, peak_idx, is_shoulder,
                                                                 is_negative, riders)
                if riders and any(s <= right_bound and e >= left_bound for s, e in group_spans):
                    # A peak-group window overlaps the parent: it and its riders fall back to split
                    for c in riders_of.pop(int(peak_idx)):
                        calls[c] = demote(calls[c], x[c], "a peak-group window overlaps its parent")
                        del skims[c]
                    left_bound, right_bound, stop_info = walk_bounds(i, apex_x, peak_idx, is_shoulder,
                                                                     is_negative, [])
            left_stop_reason, right_stop_reason, threshold, apex_vertical_distance = stop_info

            # Check for problematic case: bounds too close to apex (indicating premature stop)
            if left_bound == peak_idx and right_bound == peak_idx:
                print(f"WARNING: Peak {peak_number} at RT={apex_x:.3f} has zero-width integration!")
                print(f"  Left stop reason: {left_stop_reason}")
                print(f"  Right stop reason: {right_stop_reason}")
                print(f"  Apex signal: {bounds_detection_signal[peak_idx]:.3f}, Baseline: {baseline_y[peak_idx]:.3f}")
                print(f"  Threshold: {threshold:.6f}, Apex height: {apex_vertical_distance:.3f}")
            elif abs(left_bound - peak_idx) <= 2 or abs(right_bound - peak_idx) <= 2:
                print(f"WARNING: Peak {peak_number} at RT={apex_x:.3f} has very narrow integration bounds")
                print(f"  Left bound: {left_bound} (apex: {peak_idx}) - {left_stop_reason}")
                print(f"  Right bound: {right_bound} (apex: {peak_idx}) - {right_stop_reason}")
                print(f"  Integration width: {abs(right_bound - left_bound)} points")
                print(f"  Threshold: {threshold:.6f}, Apex height: {apex_vertical_distance:.3f}")
            
            # Append retention time at the peak (apex)
            ret_times.append(apex_x)
            
            # Extract data for integration using the corrected signal for area calculation
            x_peak = x[left_bound:right_bound + 1]
            y_peak = integration_signal[left_bound:right_bound + 1]  # Use corrected signal for area
            baseline_peak = baseline_y[left_bound:right_bound + 1]
            
            x_peaks.append(x_peak)
            y_peaks.append(y_peak)
            baseline_peaks.append(baseline_peak)
            
            # Check if integration_signal is already baseline-corrected.
            # We use `is` (identity) rather than np.array_equal because
            # processed_data['corrected_y'] may contain NaN in MS-gated masked
            # regions, and np.array_equal returns False for arrays with any
            # NaN (NaN != NaN per IEEE 754). False here would incorrectly
            # trigger the "subtract baseline" branch on already-corrected data,
            # silently doubling the baseline subtraction. Identity check is
            # both safe and semantically correct: when 'corrected_y' is in
            # processed_data, integration_signal is set to that same object
            # by `processed_data.get('corrected_y', y)` above.
            if integration_signal is processed_data.get('corrected_y'):
                # Signal is already baseline-corrected, use directly
                y_peak_corrected = y_peak
            else:
                # Signal is not baseline-corrected, subtract baseline
                y_peak_corrected = y_peak - baseline_peak
            
            # Calculate the integrated area using Simpson's rule
            area = simpson(y_peak_corrected, x=x_peak)
            
            # Apply area scaling. None or 0.0 => no scaling (x1). 0.0 is a common
            # mistake ("disable"); see the once-per-call warning above.
            if area_factor:
                area *= area_factor

            # For negative peaks, report area as positive
            if is_negative:
                area = abs(area)

            # For decreasing x-axes (e.g. FTIR wavenumbers), area may be negative
            # even for non-negative peaks due to integration direction
            area = abs(area)

            if skim is not None:
                # A rider's area is what lies above its skim curve (spec section 5)
                area = Integrator._skim_area(x, skim_signal, skim, left_bound, right_bound, area_factor)

            # Store integrated area and bounds
            integrated_areas.append(area)
            integration_bounds.append((x[left_bound], x[right_bound]))
            
            # Create a Peak object
            retention_time = apex_x
            compound_id = Integrator.identify_compound(retention_time, rt_table)
            peak_number = i + 1
            integrator = 'py'
            start_time = x[left_bound]
            end_time = x[right_bound]
            width = end_time - start_time
            
            if feature_cls is Peak or issubclass(feature_cls, ChromatographicPeak):
                peak = feature_cls(compound_id, peak_number, retention_time,
                            integrator, width, area, start_time, end_time,
                            start_index=left_bound, end_index=right_bound)
            else:
                # SpectralFeature and other Feature subclasses use the generic signature
                peak = feature_cls(
                    feature_id=peak_number,
                    position=apex_x,
                    position_units=profile.x_label if profile else "",
                    area=area,
                    width=width,
                    start=start_time,
                    end=end_time,
                    start_index=left_bound,
                    end_index=right_bound,
                )
            
            # Add shoulder flag if applicable
            if has_shoulder_info:
                peak.is_shoulder = is_shoulder

            # Add negative peak flag
            if is_negative:
                peak.is_negative = True

            # Record which RT window the peak belongs to (not exported yet)
            peak.window_id = None
            if has_shoulder_info and i < len(peak_metadata):
                peak.window_id = peak_metadata[i].get('window_id')
            
            # Assess peak quality if MS data is available and quality checks are enabled
            if ms_data is not None and quality_options and quality_options.get('quality_checks_enabled', False):
                quality = assess_peak_quality(peak, x, integration_signal, ms_data, quality_options)
                peak.asymmetry = quality['asymmetry']
                peak.spectral_coherence = quality['spectral_coherence']
                peak.is_convoluted = quality['is_convoluted']
                
                # Add specific quality issues
                peak.quality_issues = []
                if quality['asymmetry'] is not None and quality_options.get('skew_check', False) and abs(quality['asymmetry']) > quality_options.get('skew_threshold', 0.5):
                    issue = "Asymmetric" if quality['asymmetry'] > 0 else "Fronting"
                    peak.quality_issues.append(f"{issue} (skew={quality['asymmetry']:.2f})")
                
                if quality['spectral_coherence'] is not None and quality_options.get('coherence_check', False) and quality['spectral_coherence'] < quality_options.get('coherence_threshold', 0.7):
                    peak.quality_issues.append(f"Low coherence ({quality['spectral_coherence']:.2f})")

            if skim is not None and area == 0:
                peak.quality_issues.append("Skim: no area above parent tail")
            peak_at[int(peak_idx)] = peak
            peaks_list.append(peak)

        # Parents are integrated whole, then their riders' areas are taken off
        skim_totals = Integrator._net_parent_areas(peaks_list, peak_at, integrated_areas, skims, riders_of,
                                                   x, skim_signal, area_factor)
        
        # Per-window events (registry is empty in phase 1; handlers must keep
        # the peak count so the parallel lists below stay aligned)
        from logic.window_events import apply_window_events
        peaks_list = apply_window_events(peaks_list, processed_data)

        # Apply peak grouping if specified
        if peak_groups and len(peak_groups) > 0 and len(peaks_list) > 0:
            peaks_list, x_peaks, y_peaks, baseline_peaks, ret_times, integrated_areas, integration_bounds = \
                Integrator._apply_peak_grouping(
                    peaks_list, peak_groups, x, 
                    integration_signal,
                    baseline_y, x_peaks, y_peaks, baseline_peaks, 
                    ret_times, integrated_areas, integration_bounds,
                    profile=profile
                )

        # Skim fields carry the final peak numbers, so they are set after grouping
        Integrator._set_skim_fields(peaks_list, calls, skims, riders_of, peak_at, skim_totals)
        shade_top_peaks, shade_floor_peaks = Integrator._shade_arrays(
            peaks_list, y_peaks, baseline_peaks, integration_signal is processed_data.get('corrected_y'),
            skims, riders_of, peak_at, skim_signal)

        if verbose:
            # Use the feature class's own headers/rows so the summary works for
            # any Feature subtype (ChromatographicPeak, SpectralFeature, ...).
            headers = (
                feature_cls.column_headers()
                if hasattr(feature_cls, "column_headers")
                else ['Compound ID', 'Peak #', 'Ret Time',
                      'Integrator', 'Width', 'Area', 'Start Time', 'End Time']
            )
            try:
                from tabulate import tabulate
                print(tabulate([p.as_row() for p in peaks_list], headers), end='\n\n')
            except ImportError:
                print("Tabulate package not found. Install with 'pip install tabulate' for better formatting.")
                # Basic format — polymorphic via each peak's as_row().
                print("Integration Results:")
                print("  " + " | ".join(str(h) for h in headers))
                for peak in peaks_list:
                    print("  " + " | ".join(str(v) for v in peak.as_row()))
        
        return {
            'peaks': peaks_list,
            'x_peaks': x_peaks,
            'y_peaks': y_peaks,
            'baseline_peaks': baseline_peaks,
            'retention_times': ret_times,
            'integrated_areas': integrated_areas,
            'integration_bounds': integration_bounds,
            'peaks_list': peaks_list,
            'shade_top_peaks': shade_top_peaks,
            'shade_floor_peaks': shade_floor_peaks,
        }

def assess_peak_quality(peak, x, y, ms_data=None, options=None):
    """Assess peak quality to detect possible convolution.
    
    Args:
        peak: Peak object with start_time, end_time, and retention_time
        x: Array of retention times for the chromatogram
        y: Array of intensities for the chromatogram
        ms_data: MS data object with data and xlabels arrays
        options: Dictionary with quality check options
        
    Returns:
        dict: Quality metrics
    """
    # Initialize default options if none provided
    if options is None:
        options = {
            'quality_checks_enabled': False,
            'skew_check': False,
            'coherence_check': False,
            'skew_threshold': 0.5,
            'coherence_threshold': 0.7,
            'high_corr_threshold': 0.5
        }
    
    quality = {
        'asymmetry': None,
        'spectral_coherence': None,
        'is_convoluted': False
    }
    
    # If quality checks are disabled, return early
    if not options.get('quality_checks_enabled', False):
        return quality
    
    # Find indices corresponding to peak bounds
    start_idx = np.searchsorted(x, peak.start_time)
    end_idx = np.searchsorted(x, peak.end_time)
    
    # Ensure we have enough points for analysis
    if end_idx - start_idx < 5:
        print(f"Not enough points for quality assessment of peak {peak.peak_number} (need at least 5)")
        return quality
    
    # Extract peak region
    peak_x = x[start_idx:end_idx+1]
    peak_y = y[start_idx:end_idx+1]
    
    # 1. TEST FOR PEAK SHAPE ASYMMETRY (SKEWNESS) if enabled
    if options.get('skew_check', False):
        try:
            # Normalize intensities to [0,1] for numerical stability
            peak_y_norm = (peak_y - np.min(peak_y)) / (np.max(peak_y) - np.min(peak_y)) if np.max(peak_y) > np.min(peak_y) else peak_y
            
            # Calculate skewness using scipy
            asymmetry = skew(peak_y_norm)
            quality['asymmetry'] = asymmetry
            
            # Get the threshold from options
            skew_threshold = options.get('skew_threshold', 0.5)
            
            # Flag high asymmetry
            if abs(asymmetry) > skew_threshold:
                quality['is_convoluted'] = True
                print(f"Peak {peak.peak_number}: High asymmetry detected ({asymmetry:.2f})")
        except Exception as e:
            print(f"Error calculating peak asymmetry: {str(e)}")
    
    # 2. TEST FOR ION CHROMATOGRAM CORRELATION (SPECTRAL COHERENCE) if enabled
    if options.get('coherence_check', False) and ms_data is not None:
        try:
            # Find corresponding indices in MS data
            ms_start_idx = np.searchsorted(ms_data.xlabels, peak.start_time)
            ms_end_idx = np.searchsorted(ms_data.xlabels, peak.end_time)
            apex_idx = np.searchsorted(ms_data.xlabels, peak.retention_time)
            
            # Ensure we have enough MS scans
            if ms_end_idx - ms_start_idx < 3:
                print(f"Not enough MS scans for spectral coherence test of peak {peak.peak_number}")
                return quality
            
            # Get spectrum at apex to identify top ions
            apex_spectrum = ms_data.data[apex_idx, :].astype(float)
            
            # Find top N ions (by intensity)
            N = min(10, len(apex_spectrum))  # Use top 10 ions
            top_mz_indices = np.argsort(apex_spectrum)[-N:]
            
            # Extract XICs for top ions
            xics = ms_data.data[ms_start_idx:ms_end_idx+1, top_mz_indices]
            
            # Calculate pairwise correlations
            correlations = []
            for i in range(N):
                for j in range(i+1, N):
                    # Skip ions with no variation
                    if np.std(xics[:, i]) > 0 and np.std(xics[:, j]) > 0:
                        r = np.corrcoef(xics[:, i], xics[:, j])[0, 1]
                        correlations.append(r)
            
            if correlations:
                # Calculate mean correlation
                mean_corr = np.mean(correlations)
                quality['spectral_coherence'] = mean_corr
                
                # Calculate percentage of highly correlated pairs (r > 0.9)
                pct_high_corr = sum(1 for r in correlations if r > 0.9) / len(correlations)
                
                # Get thresholds from options
                coherence_threshold = options.get('coherence_threshold', 0.7)
                high_corr_threshold = options.get('high_corr_threshold', 0.5)
                
                # Flag low coherence
                if mean_corr < coherence_threshold or pct_high_corr < high_corr_threshold:
                    quality['is_convoluted'] = True
                    print(f"Peak {peak.peak_number}: Low spectral coherence detected ({mean_corr:.2f}, {pct_high_corr:.2f} high corr)")
        except Exception as e:
            print(f"Error calculating spectral coherence: {str(e)}")
    
    return quality