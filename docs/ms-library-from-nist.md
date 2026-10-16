# Building ChromaKit's MS library from your NIST install

ChromaKit searches spectra with ms-toolkit. The Setup MS Library dialog asks for three files built
from a mass-spectral library: the library as JSON (ChromaKit builds the search vectors from it each
time it loads), a preselector (`.pkl`) and a Word2Vec model (`.model`). ChromaKit does not ship NIST
data. Build these files from your own licensed copy.

## Install

```
git clone https://github.com/calebcoatney/chromakit-ms
cd chromakit-ms
pip install ".[ms]"
```

The `[ms]` extra installs `ms-toolkit-nrel` 0.1.3 from its GitHub tag (PyPI still has 0.1.2). The two helper scripts
used below live in the repository's `tools/` folder, which pip does not install, so run them from the
clone. If you only want the application, this installs it without a clone:

```
pip install "chromakit-ms[ms] @ git+https://github.com/calebcoatney/chromakit-ms"
```

Both forms need git. Tested: on macOS (Python 3.13) pip resolved both install commands, and the build
steps below ran against ms-toolkit 0.1.2 on a small synthetic library. Not tested: Windows, Linux, an
install into a fresh environment, and a full-size NIST library.

## 1. Export NIST to text with lib2nist

lib2nist ships with NIST MS Search. Open lib2nist, add your library (for example `mainlib`), choose
a text output format (the NIST MS Search text / MSP format) and export to a file such as
`nist_mainlib.msp`. Menu labels differ between lib2nist versions. An export of the full EI library is
large (hundreds of MB), which is expected.

## 2. Convert to the layout ms-toolkit reads

```
python tools/nist_text_for_mstoolkit.py nist_mainlib.msp nist_mainlib.txt
```

lib2nist can put several `m/z intensity;` pairs on one line and writes `CAS#: ...; NIST#: ...`.
ms-toolkit wants one `m/z intensity` pair per line and a `CASNO:` field. The script rewrites each
record that way, keeps the name, formula, molecular weight, CAS number and peak list, and prints how
many spectra it wrote. Compare that count with the size of your library.

## ms-toolkit 0.1.2 and the GitHub code

ms-toolkit 0.1.2 (PyPI) and the GitHub code before 0.1.3 fail here: they import `orjson` without
listing it, and with numpy 2 and orjson 3.10 or newer the text-library parser stops with
`Type is not JSON serializable: numpy.float64`. 0.1.3 fixes both, and the `[ms]` extra installs it.
If you installed ms-toolkit some other way, upgrade it with
`pip install git+https://github.com/calebcoatney/ms-toolkit@v0.1.3`.

## 3. Build the library files with ms-toolkit

Save this as `build_library.py` next to `nist_mainlib.txt` and run `python build_library.py`:

```python
from ms_toolkit.api import MSToolkit

tk = MSToolkit()
tk.load_library(file_path="nist_mainlib.txt", json_path="NIST.json")  # parses, writes NIST.json, vectorizes
tk.train_preselector(save_path="NIST_preselector.pkl")
tk.train_w2v(save_path="NIST_w2v.model")                              # the slowest step on a full library
```

If `NIST.json` already exists, `load_library` reads it and skips the text file. Delete it to rebuild
from a new export. The search vectors are held in memory at about 8 KB per spectrum, so a full EI
library needs a few GB of RAM.

## 4. Load it in ChromaKit

Mass Spectrometry tab → Setup MS Library. Three file dialogs open in turn: pick `NIST.json`, then
`NIST_preselector.pkl`, then `NIST_w2v.model`. Pick the `.json` file for the library, not the `.txt`:
ChromaKit's loader reads only the JSON. Cancelling any of the three dialogs cancels the setup. When
loading finishes the status line reads "MS Library loaded" and Search Library becomes available.
ChromaKit remembers the three paths and loads them at the next start.
