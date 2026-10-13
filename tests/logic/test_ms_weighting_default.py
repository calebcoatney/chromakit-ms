"""One default: every reader takes the weighting default from DEFAULT_WEIGHTING."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_no_other_module_hardcodes_a_nist_gc_default():
    """One default, defined once: every reader takes it from DEFAULT_WEIGHTING."""
    pattern = re.compile(r"""['"]weighting['"]\s*(?:,|:)\s*['"]NIST_GC['"]""")
    offenders = []
    for folder in ("ui", "logic", "api"):
        for path in (REPO / folder).rglob("*.py"):
            if path.name == "test_api.py":
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(REPO)}:{n}")
    assert offenders == []
