import os

import pytest

pytest.importorskip("pytestqt")

from ui.dialogs.batch_convert_dialog import _find_sepsolve_stems


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"x")


def _names(paths):
    return sorted(os.path.basename(p) for p in paths)


def test_stems_are_exact(tmp_path):
    _touch(tmp_path, "Sample 1.HDR", "Sample 1.rsd", "Sample 1.lsc", "Sample 1_0.5_secs.dbc.lsc",
           "Sample 1.DAT", "Sample 1.DAX", "Sample 10.HDR", "Sample 10.rsd", "Sample 10_0.5_secs.dbc.lsc")
    g = _find_sepsolve_stems(str(tmp_path))
    assert sorted(g) == ["Sample 1", "Sample 10"]
    assert _names(g["Sample 1"]) == ["Sample 1.HDR", "Sample 1.lsc", "Sample 1.rsd", "Sample 1_0.5_secs.dbc.lsc"]
    assert _names(g["Sample 10"]) == ["Sample 10.HDR", "Sample 10.rsd", "Sample 10_0.5_secs.dbc.lsc"]
    assert g["Sample 1"][0].endswith("Sample 1.HDR")


def test_dbc_goes_to_the_longest_matching_stem(tmp_path):
    _touch(tmp_path, "A.HDR", "A.rsd", "A_0.5_secs.dbc.lsc", "A_B.HDR", "A_B.rsd", "A_B_0.5_secs.dbc.lsc")
    g = _find_sepsolve_stems(str(tmp_path))
    assert _names(g["A"]) == ["A.HDR", "A.rsd", "A_0.5_secs.dbc.lsc"]
    assert _names(g["A_B"]) == ["A_B.HDR", "A_B.rsd", "A_B_0.5_secs.dbc.lsc"]


def test_group_needs_hdr_and_a_spectrum_file(tmp_path):
    _touch(tmp_path, "X.rsd", "X.lsc", "Y.HDR", "Y.rsd", "Z.HDR", "Z.rsd", "Z.lsc")
    assert sorted(_find_sepsolve_stems(str(tmp_path))) == ["Z"]
