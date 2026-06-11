"""Test that /api/process doesn't crash due to missing numpy import.

Regression test for a latent bug: np is used at api/main.py:175 inside
/api/process, but `import numpy as np` was only inside /api/run.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import importlib
import api.main as api_main


def test_numpy_imported_at_module_level():
    """api.main must have np bound at module scope."""
    importlib.reload(api_main)
    assert hasattr(api_main, 'np'), \
        "api.main must `import numpy as np` at the top of the module"
