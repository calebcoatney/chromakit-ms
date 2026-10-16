"""Every third-party import in the shipped code is declared in setup.py (pandas once wasn't)."""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIPPED = ["logic", "ui", "util", "resources", "main.py"]
FIRST_PARTY = {"logic", "ui", "util", "resources", "main", "web", "api", "tests", "tools",
               "chromakit_version", "_build_info"}
# Not in setup.py's install_requires, on purpose: torch, tabulate and ms_toolkit are imported only inside
# guarded try/except blocks with a fallback. torch and ms_toolkit are declared as extras ([deconv] and
# [ms]); tabulate is not declared anywhere. fastapi and uvicorn belong to the web API and requests appears
# only in tools/ and api/ example scripts, which are not shipped; none of the three is declared.
OPTIONAL = {"torch", "tabulate", "ms_toolkit", "fastapi", "uvicorn", "requests"}
# import name -> distribution name where they differ
DIST = {"PySide6": "PySide6", "rainbow": "rainbow-api", "sklearn": "scikit-learn"}


def _imports():
    found = set()
    for root in SHIPPED:
        p = ROOT / root
        for f in ([p] if p.suffix == ".py" else p.rglob("*.py")):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    found |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m not in FIRST_PARTY}


def _declared():
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    block = re.search(r"install_requires=\[(.*?)\]", text, re.S).group(1)
    return {re.split(r"[<>=!~\[ ]", s.strip().strip('"\''))[0].lower()
            for s in block.split(",") if s.strip().strip('"\'')}


def test_every_shipped_import_is_declared():
    declared = _declared()
    missing = sorted(m for m in _imports() - OPTIONAL if DIST.get(m, m).lower() not in declared)
    assert missing == []


def test_no_declared_dependency_is_unused():
    used = {DIST.get(m, m).lower() for m in _imports()} | {"uvicorn", "fastapi"}
    unused = sorted(d for d in _declared() if d not in used)
    assert unused == []


def test_ms_extra_requires_the_fixed_ms_toolkit():
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"ms": ["ms-toolkit-nrel @ git+https://github.com/calebcoatney/ms-toolkit@v0.1.3"]' in text
