"""Dialog for configuring which parameter sections are visible."""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox, QLabel
)
from PySide6.QtCore import QSettings


# Ordered list of (key, display label) for each parameter section
PARAMETER_SECTIONS = [
    ('smoothing', 'Signal Smoothing'),
    ('baseline', 'Baseline Correction'),
    ('baseline_advanced', 'Advanced Baseline (MS, TIC Align, Break Points)'),
    ('peaks', 'Peak Detection'),
    ('negative_peaks', 'Negative Peak Detection'),
    ('shoulders', 'Shoulder Detection'),
    ('range_filters', 'Peak Range Filters'),
    ('peak_grouping', 'Peak Grouping'),
]

SETTINGS_PREFIX = "parameter_visibility/"


class ParameterVisibilityDialog(QDialog):
    """Dialog to select which parameter sections are shown in the sidebar."""

    def __init__(self, current_visibility: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Visible Parameters")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        info = QLabel("Choose which parameter sections to display:")
        info.setStyleSheet("margin-bottom: 8px;")
        layout.addWidget(info)

        self.checkboxes = {}
        for key, label in PARAMETER_SECTIONS:
            cb = QCheckBox(label)
            cb.setChecked(current_visibility.get(key, True))
            layout.addWidget(cb)
            self.checkboxes[key] = cb

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_visibility(self) -> dict:
        """Return dict of {section_key: bool} from current checkbox state."""
        return {key: cb.isChecked() for key, cb in self.checkboxes.items()}


def load_visibility_settings() -> dict:
    """Load section visibility from QSettings."""
    settings = QSettings("CalebCoatney", "ChromaKit")
    vis = {}
    for key, _ in PARAMETER_SECTIONS:
        val = settings.value(SETTINGS_PREFIX + key, True, type=bool)
        vis[key] = val
    return vis


def save_visibility_settings(visibility: dict):
    """Save section visibility to QSettings."""
    settings = QSettings("CalebCoatney", "ChromaKit")
    for key, val in visibility.items():
        settings.setValue(SETTINGS_PREFIX + key, val)
