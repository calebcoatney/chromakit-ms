import numpy as np
import pytest
from util import dxlsx_export as dx


class StubDataFile:
    def __init__(self, name, xlabels, data, metadata=None):
        self.name = name
        self.xlabels = np.asarray(xlabels)
        self.data = np.asarray(data)
        self.metadata = metadata or {}


class StubDataDir:
    def __init__(self, name, datafiles, metadata=None):
        self.name = name
        self.datafiles = datafiles
        self.metadata = metadata or {}

    def get_file(self, fname):
        for f in self.datafiles:
            if f.name == fname:
                return f
        raise KeyError(fname)


def make_dir():
    fid = StubDataFile("FID1A.ch", [0.0, 0.5, 1.0], [10.0, 20.0, 30.0],
                       metadata={"notebook": "SampleA"})
    ms = StubDataFile("data.ms", [0.5, 1.0], [[1, 2, 3], [10, 20, 30]])
    return StubDataDir("dirname", [fid, ms], metadata={"notebook": "DirNB"})


def test_list_signals_returns_ch_and_ms():
    d = make_dir()
    assert dx.list_signals(d) == ["FID1A.ch", "data.ms"]


def test_read_signal_ch():
    d = make_dir()
    x, y = dx.read_signal(d, "FID1A.ch")
    assert np.allclose(x, [0.0, 0.5, 1.0])
    assert np.allclose(y, [10.0, 20.0, 30.0])


def test_read_signal_ms_is_tic():
    d = make_dir()
    x, y = dx.read_signal(d, "data.ms")
    assert np.allclose(x, [0.5, 1.0])
    assert np.allclose(y, [6.0, 60.0])


def test_read_notebook_prefers_detector_file():
    d = make_dir()
    assert dx.read_notebook(d, "/some/path.D") == "SampleA"


def test_read_notebook_falls_back_to_dir_then_basename():
    fid = StubDataFile("FID1A.ch", [0.0], [1.0], metadata={})
    d = StubDataDir("dname", [fid], metadata={})
    assert dx.read_notebook(d, "/x/basename.D") == "dname"


def test_read_notebook_uses_basename_when_no_name():
    fid = StubDataFile("FID1A.ch", [0.0], [1.0], metadata={})
    d = StubDataDir(None, [fid], metadata={})  # name=None
    assert dx.read_notebook(d, "/data/MySample.D") == "MySample"


def test_read_notebook_basename_handles_trailing_slash():
    fid = StubDataFile("FID1A.ch", [0.0], [1.0], metadata={})
    d = StubDataDir(None, [fid], metadata={})  # name=None
    assert dx.read_notebook(d, "/data/MySample.D/") == "MySample"
