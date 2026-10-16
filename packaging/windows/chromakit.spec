# PyInstaller spec for the ChromaKit Windows app (folder mode). Run from the repo root:
#   pyinstaller packaging/windows/chromakit.spec --noconfirm --distpath packaging/windows/dist --workpath packaging/windows/build
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
