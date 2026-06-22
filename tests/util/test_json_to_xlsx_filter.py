"""Tests for peak-filter helpers in util/json_to_xlsx."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from util.json_to_xlsx import _is_unidentified, _is_unquantitated


class TestIsUnidentified:
    def test_none_compound_id_is_unidentified(self):
        assert _is_unidentified({"compound_id": None}) is True

    def test_missing_compound_id_is_unidentified(self):
        assert _is_unidentified({"peak_number": 1}) is True

    def test_empty_string_is_unidentified(self):
        assert _is_unidentified({"compound_id": ""}) is True

    def test_whitespace_only_is_unidentified(self):
        assert _is_unidentified({"compound_id": "   "}) is True

    def test_literal_unknown_is_unidentified(self):
        assert _is_unidentified({"compound_id": "Unknown"}) is True
        assert _is_unidentified({"compound_id": "unknown"}) is True
        assert _is_unidentified({"compound_id": "UNKNOWN"}) is True

    def test_auto_unknown_with_rt_is_unidentified(self):
        """Pattern 'Unknown (X.XXX)' written by integration as placeholder."""
        assert _is_unidentified({"compound_id": "Unknown (5.234)"}) is True
        assert _is_unidentified({"compound_id": "Unknown (12.001)"}) is True

    def test_real_compound_name_is_not_unidentified(self):
        assert _is_unidentified({"compound_id": "Propanoic acid"}) is False
        assert _is_unidentified({"compound_id": "Nonane"}) is False

    def test_compound_id_capitalized_key_also_read(self):
        """JSON sometimes uses 'Compound ID' (notebook-style)."""
        assert _is_unidentified({"Compound ID": "Decane"}) is False
        assert _is_unidentified({"Compound ID": None}) is True


class TestIsUnquantitated:
    def test_no_quantitation_run_returns_false(self):
        """If no peak in the file has wt_percent, filter is a no-op."""
        peaks = [
            {"compound_id": "Decane", "wt_percent": None},
            {"compound_id": "Benzene", "wt_percent": None},
        ]
        assert _is_unquantitated(peaks[0], peaks) is False
        assert _is_unquantitated(peaks[1], peaks) is False

    def test_quantitation_run_unquantitated_peak_returns_true(self):
        """When other peaks have wt_percent, a None means 'was skipped'."""
        peaks = [
            {"compound_id": "Decane", "wt_percent": 50.0},
            {"compound_id": "Mysterium", "wt_percent": None},
        ]
        assert _is_unquantitated(peaks[1], peaks) is True

    def test_quantitation_run_quantitated_peak_returns_false(self):
        peaks = [
            {"compound_id": "Decane", "wt_percent": 50.0},
            {"compound_id": "Benzene", "wt_percent": 30.0},
        ]
        assert _is_unquantitated(peaks[0], peaks) is False
        assert _is_unquantitated(peaks[1], peaks) is False
