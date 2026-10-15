"""What the MS tab says when ms-toolkit cannot be imported.

Kerry installed ms-toolkit from a clone and still read "MSToolkit is not installed": the
real ImportError was swallowed. The tab must say which of the two cases it is, and why.
"""
import pytest

pytest.importorskip("pytestqt")

from ui.frames.ms import toolkit_unavailable_text


def test_missing_package_names_the_interpreter():
    err = ModuleNotFoundError("No module named 'ms_toolkit'", name="ms_toolkit")
    headline, hint = toolkit_unavailable_text(err, "/envs/ck/bin/python")
    assert "not installed" in headline and "/envs/ck/bin/python" in headline
    assert hint == '"/envs/ck/bin/python" -m pip install ms-toolkit-nrel'


def test_broken_dependency_is_not_reported_as_missing():
    err = ModuleNotFoundError("No module named 'gensim'", name="gensim")
    headline, hint = toolkit_unavailable_text(err, "/envs/ck/bin/python")
    assert "not installed" not in headline
    assert "No module named 'gensim'" in headline
    assert "gensim" in hint


def test_other_import_errors_keep_their_text():
    err = ImportError("cannot import name 'triu' from 'scipy.linalg'")
    headline, _ = toolkit_unavailable_text(err, "/py")
    assert "cannot import name 'triu'" in headline
