# PyInstaller spec for the ChromaKit Windows app (folder mode). Run from the repo root:
#   pyinstaller packaging/windows/chromakit.spec --noconfirm --distpath packaging/windows/dist --workpath packaging/windows/build
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent.parent
ICON = "resources/file.ico"   # relative to the repo root; bundled at resources/ (= _internal/resources/) where ui/app.py ICON_PATH finds it

datas = [(str(ROOT / "LICENSE"), "."), (str(ROOT / "NOTICE"), "."),
         (str(ROOT / "licenses"), "licenses"), (str(ROOT / "data"), "data"),
         (str(ROOT / ICON), "resources")]
for folder in ("ui",):
    for path in (ROOT / folder).rglob("*"):
        if path.suffix in (".qss", ".png", ".ico", ".svg"):
            datas.append((str(path), str(path.parent.relative_to(ROOT))))
datas += collect_data_files("ms_toolkit")
datas += collect_data_files("rainbow")
datas += copy_metadata("ms-toolkit-nrel")

# The frozen app redistributes these packages, so their dist-info (which carries most licence texts)
# travels with it. The PySide6/shiboken6 wheels carry only LicenseRef-Qt-Commercial.txt, so the
# LGPL-3.0 and GPL-3.0 texts they are used under ship from licenses/ (bundled above).
# On CI (GITHUB_ACTIONS set) a missing name fails the build, except requests, which is only pulled in
# transitively; a local build reports the miss and still freezes (check the log for "no metadata for").
from importlib.metadata import PackageNotFoundError

for _name in ("LGPL-3.0.txt", "GPL-3.0.txt"):
    if not (ROOT / "licenses" / _name).is_file():
        raise SystemExit(f"chromakit.spec: licenses/{_name} is missing; PySide6 is redistributed under it")

_ON_CI = bool(os.environ.get("GITHUB_ACTIONS"))
for _dist in ("PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6", "numpy", "scipy", "pandas",
              "matplotlib", "pybaselines", "rainbow-api", "scikit-learn", "gensim", "joblib", "orjson",
              "openpyxl", "olefile", "pydantic", "tqdm", "requests"):
    try:
        datas += copy_metadata(_dist)
    except PackageNotFoundError as _e:
        if _ON_CI and _dist != "requests":
            raise SystemExit(f"chromakit.spec: no metadata for {_dist} ({_e}); its licence text would not ship")
        print(f"chromakit.spec: no metadata for {_dist} ({_e}); its licence text is not bundled")

hiddenimports = (collect_submodules("logic") + collect_submodules("ui") + collect_submodules("util")
                 + collect_submodules("ms_toolkit") + collect_submodules("rainbow")
                 + ["chromakit_version"])
if (ROOT / "_build_info.py").exists():
    hiddenimports.append("_build_info")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['torch', 'torchvision', 'tkinter', 'customtkinter', 'fastapi', 'uvicorn',
              'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='ChromaKit',
          icon=str(ROOT / ICON), console=False)
coll = COLLECT(exe, a.binaries, a.datas, name='ChromaKit')
