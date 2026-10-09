# ChromaKit-MS

ChromaKit-MS is a Python application for inspecting and processing GC-MS (Gas Chromatography–Mass Spectrometry) data. It provides a PySide6 desktop GUI, a FastAPI REST API backend, and an experimental React/Vite web frontend.

## Features

* **Chromatogram processing** – smoothing, baseline correction, and peak detection via `pybaselines` and `scipy`.
* **Peak integration** – integrate detected peaks; export results to JSON, CSV, and Excel.
* **Mass spectrum search** – search against spectral libraries using `ms-toolkit-nrel` (single peak or batch).
* **Peak deconvolution** – 1D U-Net neural network for resolving overlapping peaks (requires `torch`).
* **GCxGC-TOFMS/FID** – SepSolve/Markes runs: 2D peaks on the FID or DBC TIC, spectra and library IDs, CSV/JSON/MSP/MGF export (see [docs/gcxgc.md](docs/gcxgc.md)).
* **Batch automation** – process directories of data files without manual interaction.
* **Quantitation** – Polyarc + internal standard method for compound quantitation.
* **Interactive plots** – inspect chromatograms, TIC traces, heatmaps, and mass spectra with Matplotlib.
* **REST API** – FastAPI backend for programmatic access and web frontends.

## Requirements

* Python 3.10+
* [PySide6](https://pypi.org/project/PySide6/) 6.9+
* [rainbow-api](https://github.com/reciprocal-space/rainbow-api) (Agilent `.D` file parsing)
* NumPy, SciPy, Matplotlib, pybaselines, openpyxl

**Optional:**
* `ms-toolkit-nrel` – MS library searching (`pip install -e ".[ms]"`)
* `torch` – neural network peak deconvolution (`pip install -e ".[deconv]"`)

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

To include MS library search support:

```bash
pip install -e ".[ms]"
```

This installs the `chromakit-ms` console entry point.

To install straight from GitHub instead of a clone (the MS library search extra is included):

```bash
pip install "chromakit-ms[ms] @ git+https://github.com/calebcoatney/chromakit-ms"
```

For MS library search with your own NIST copy, see [docs/ms-library-from-nist.md](docs/ms-library-from-nist.md).
The install commands were checked on macOS only; Windows and Linux are untested.

### REST API Backend

```bash
pip install -r api/requirements.txt
```

## GCxGC-TOFMS/FID

Supported: SepSolve/Markes runs (`.HDR`, `.rsd`, `.lsc`, `.dbc.lsc`) converted with Tools → Batch Convert to .C…, with 2D peak detection on the FID or DBC TIC, spectra and library IDs, and CSV/JSON/MSP/MGF export. The how-to is in [docs/gcxgc.md](docs/gcxgc.md).

Not yet: region/stencil drawing and group-type (PIONA) wt%, deconvolution, retention indices, multi-sample alignment, and batch processing of GCxGC runs.

GCxGC mode has been used on macOS only; Windows is untested.

## Usage

### Desktop GUI

```bash
chromakit-ms
```

Use the file tree to select an Agilent `.D` directory. The application displays the chromatogram and TIC, and provides controls for processing parameters, peak integration, MS search, quantitation, and batch automation.

### REST API

```bash
cd api
python main.py
```

Available at `http://127.0.0.1:8000` with interactive docs at `/docs`. See [api/README.md](api/README.md) for endpoint details.

### React Web Frontend

```bash
cd react-ui
npm install
npm run dev
```

## Project Structure

```
logic/          Data loading, processing, integration, and export (shared by GUI and API)
ui/             PySide6 desktop interface
api/            FastAPI REST API backend
react-ui/       Vite/React web frontend (experimental)
deconvolution/  Peak deconvolution models and training utilities
util/           Shared utilities
resources/      Icons and static assets
main.py         Desktop GUI entry point
setup.py        Package metadata and entry points
```

## License

ChromaKit-MS is released under the [Apache License 2.0](LICENSE). Third-party components included or linked:

* `ms-toolkit-nrel` – Apache License 2.0
* `rainbow-api` – GNU Lesser General Public License v3.0
* `pybaselines` – BSD 3-Clause License

See the [NOTICE](NOTICE) file for full attribution details.
