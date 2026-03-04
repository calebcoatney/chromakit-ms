"""
Scaling Factors Dialog for configuring signal and area multipliers.

These factors match ChromaKit output to different versions of Agilent analysis software.
Presets can be saved/loaded for different instrument configurations.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QInputDialog,
    QMessageBox
)
from PySide6.QtCore import Qt, QSettings, Signal
import json


class ScalingFactorsDialog(QDialog):
    """Dialog for configuring signal and area scaling factors with preset support."""

    factors_changed = Signal(float, float)  # signal_factor, area_factor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scaling Factors")
        self.resize(400, 280)

        self.settings = QSettings("CalebCoatney", "ChromaKit")
        self.setup_ui()
        self.load_current_settings()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Preset selector ---
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self.preset_combo)

        self.save_preset_btn = QPushButton("Save As...")
        self.save_preset_btn.clicked.connect(self._save_preset)
        preset_layout.addWidget(self.save_preset_btn)

        self.delete_preset_btn = QPushButton("Delete")
        self.delete_preset_btn.clicked.connect(self._delete_preset)
        preset_layout.addWidget(self.delete_preset_btn)

        layout.addWidget(preset_group)

        # --- Factor inputs ---
        factors_group = QGroupBox("Scaling Factors")
        form = QFormLayout(factors_group)

        self.signal_factor_spin = QDoubleSpinBox()
        self.signal_factor_spin.setRange(0.0, 1e9)
        self.signal_factor_spin.setDecimals(6)
        self.signal_factor_spin.setValue(1.0)
        self.signal_factor_spin.setToolTip(
            "Multiplier applied to raw detector signal upon data loading"
        )
        form.addRow("Signal Factor:", self.signal_factor_spin)

        self.area_factor_spin = QDoubleSpinBox()
        self.area_factor_spin.setRange(0.0, 1e9)
        self.area_factor_spin.setDecimals(6)
        self.area_factor_spin.setValue(1.0)
        self.area_factor_spin.setToolTip(
            "Multiplier applied to integrated peak areas (ChemStation correction)"
        )
        form.addRow("Area Factor:", self.area_factor_spin)

        layout.addWidget(factors_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()

        self.restore_button = QPushButton("Restore Defaults")
        self.restore_button.clicked.connect(self._restore_defaults)
        button_layout.addWidget(self.restore_button)

        button_layout.addStretch()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    # ---- Settings persistence ----

    def load_current_settings(self):
        """Load saved factors and populate preset list."""
        signal_factor = self.settings.value("scaling/signal_factor", 1.0, type=float)
        area_factor = self.settings.value("scaling/area_factor", 1.0, type=float)
        self.signal_factor_spin.setValue(signal_factor)
        self.area_factor_spin.setValue(area_factor)
        self._refresh_presets()

    def _save_settings(self):
        """Persist current factor values to QSettings."""
        self.settings.setValue("scaling/signal_factor", self.signal_factor_spin.value())
        self.settings.setValue("scaling/area_factor", self.area_factor_spin.value())

    def _restore_defaults(self):
        """Reset factors to unity (no scaling)."""
        self.signal_factor_spin.setValue(1.0)
        self.area_factor_spin.setValue(1.0)
        self.preset_combo.setCurrentIndex(0)

    # ---- Preset management ----

    def _get_presets(self) -> dict:
        """Load presets dict from QSettings."""
        raw = self.settings.value("scaling/presets", "{}")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    def _set_presets(self, presets: dict):
        self.settings.setValue("scaling/presets", json.dumps(presets))

    def _refresh_presets(self):
        """Rebuild the preset combo box."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("(Custom)")
        presets = self._get_presets()
        for name in sorted(presets.keys()):
            self.preset_combo.addItem(name)
        self.preset_combo.blockSignals(False)

        # Try to match current values to a preset
        self._sync_combo_to_values()

    def _sync_combo_to_values(self):
        """Select the preset matching the current spinbox values, or (Custom)."""
        sig = self.signal_factor_spin.value()
        area = self.area_factor_spin.value()
        presets = self._get_presets()
        for name, vals in presets.items():
            if abs(vals["signal_factor"] - sig) < 1e-9 and abs(vals["area_factor"] - area) < 1e-9:
                idx = self.preset_combo.findText(name)
                if idx >= 0:
                    self.preset_combo.blockSignals(True)
                    self.preset_combo.setCurrentIndex(idx)
                    self.preset_combo.blockSignals(False)
                    return
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, index):
        if index <= 0:
            return
        name = self.preset_combo.currentText()
        presets = self._get_presets()
        if name in presets:
            self.signal_factor_spin.setValue(presets[name]["signal_factor"])
            self.area_factor_spin.setValue(presets[name]["area_factor"])

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = self._get_presets()
        if name in presets:
            reply = QMessageBox.question(
                self, "Overwrite Preset",
                f'Preset "{name}" already exists. Overwrite?',
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        presets[name] = {
            "signal_factor": self.signal_factor_spin.value(),
            "area_factor": self.area_factor_spin.value(),
        }
        self._set_presets(presets)
        self._refresh_presets()

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        if self.preset_combo.currentIndex() <= 0:
            return
        presets = self._get_presets()
        if name in presets:
            reply = QMessageBox.question(
                self, "Delete Preset",
                f'Delete preset "{name}"?',
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            del presets[name]
            self._set_presets(presets)
            self._refresh_presets()
            self.preset_combo.setCurrentIndex(0)

    # ---- Dialog accept ----

    def accept(self):
        self._save_settings()
        self.factors_changed.emit(
            self.signal_factor_spin.value(),
            self.area_factor_spin.value(),
        )
        super().accept()
