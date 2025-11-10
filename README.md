# ChromaKit-MS

ChromaKit-MS is a Python application for inspecting and processing GC-MS (Gas Chromatography–Mass Spectrometry) data.  The project provides both a PySide6-based desktop GUI and a FastAPI REST API backend that allows users to load Agilent `.D` directories, apply chromatogram processing steps and perform MS library searches.

## Features

* **Chromatogram processing** – smoothing, baseline correction and peak detection via `pybaselines` and `scipy` utilities.
* **Peak integration** – integrate detected peaks and export results.
* **Mass spectrum search** – optional search against spectral libraries using `ms-toolkit-nrel`.
* **Batch automation** – process a list of data directories without manual interaction.
* **Interactive plots** – inspect chromatograms, TIC traces and extracted mass spectra with Matplotlib.
* **REST API** – FastAPI backend for web-based frontends and programmatic access.

## Requirements

* Python 3.7+
* [PySide6](https://pypi.org/project/PySide6/)
* [rainbow-api](https://github.com/reciprocal-space/rainbow-api) (for reading Agilent data)
* NumPy, SciPy, Matplotlib and pybaselines

Optional features such as MS library searching require `ms-toolkit-nrel`.

## Installation

### Desktop GUI Application

Clone the repository and install in editable mode:

```bash
pip install -e .
```

This installs the `chromakit-ms` console entry point which launches the GUI.

### REST API Backend

Install API dependencies:

```bash
pip install -r api/requirements.txt
```

## Usage

### Desktop GUI

```bash
chromakit-ms
```

Use the file tree on the left of the window to select an Agilent `.D` directory.  The application will display the chromatogram and TIC, allowing you to adjust processing parameters, integrate peaks and export results.  Refer to the on-screen controls and dialogs for automation and MS search options.

### REST API

Start the API server:

```bash
cd api
python main.py
```

The API will be available at `http://127.0.0.1:8000` with interactive documentation at `/docs`.

See the [API README](api/README.md) for detailed endpoint documentation and usage examples.

## Project Structure

```
logic/       Data loading, processing and integration (shared by GUI and API)
ui/          PySide6 desktop interface implementation
api/         FastAPI REST API backend
resources/   Icons and other static resources
main.py      Desktop GUI application entry point
setup.py     Package metadata and entry point
```

## License
ChromaKit-MS is released under the terms of the
[Apache License 2.0](LICENSE). The distribution also includes or links
to third-party components under their own licenses:

* `ms-toolkit-nrel` – Apache License 2.0
* `rainbow-api` – GNU Lesser General Public License v3.0
* `pybaselines` – BSD 3-Clause License

See the [NOTICE](NOTICE) file for attribution and additional details.
