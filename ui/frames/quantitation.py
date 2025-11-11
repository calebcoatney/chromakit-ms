"""
Quantitation frame for Polyarc + Internal Standard method.

Provides UI for entering internal standard and sample preparation information,
and displays calculated quantitation results.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QCheckBox, QComboBox, QPushButton, QFormLayout
)
from PySide6.QtCore import Signal, Qt
from logic.quantitation import QuantitationCalculator


class QuantitationFrame(QWidget):
    """Frame for quantitation settings and calculations."""
    
    # Signal emitted when quantitation settings change
    quantitation_changed = Signal()
    # Signal emitted when user requests re-quantitation
    requantitate_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculator = QuantitationCalculator()
        self.ms_toolkit = None  # Will be set by main app
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Enable checkbox
        self.enable_checkbox = QCheckBox("Enable Quantitation")
        self.enable_checkbox.setChecked(False)
        layout.addWidget(self.enable_checkbox)

        # Overwrite results checkbox
        self.overwrite_checkbox = QCheckBox("Overwrite existing results (JSON/CSV)")
        self.overwrite_checkbox.setChecked(False)
        layout.addWidget(self.overwrite_checkbox)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("Polyarc + Internal Standard")
        self.method_combo.setEnabled(False)  # Only one method for now
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        # Internal Standard group
        is_group = QGroupBox("Internal Standard Information")
        is_layout = QFormLayout()
        
        # Compound name with search button
        compound_layout = QHBoxLayout()
        self.compound_edit = QLineEdit()
        self.compound_edit.setPlaceholderText("e.g., Nonane")
        compound_layout.addWidget(self.compound_edit)
        self.search_library_btn = QPushButton("Search MS Library")
        self.search_library_btn.setMaximumWidth(150)
        compound_layout.addWidget(self.search_library_btn)
        is_layout.addRow("Compound Name:", compound_layout)
        
        # Formula (autofilled)
        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("e.g., C9H20")
        is_layout.addRow("Formula:", self.formula_edit)
        
        # Molecular weight (autofilled)
        self.mw_edit = QLineEdit()
        self.mw_edit.setPlaceholderText("e.g., 128.259")
        is_layout.addRow("MW (g/mol):", self.mw_edit)
        
        # Density (user input)
        self.density_edit = QLineEdit()
        self.density_edit.setPlaceholderText("e.g., 0.718")
        is_layout.addRow("Density (g/mL):", self.density_edit)
        
        # Volume (user input)
        self.volume_is_edit = QLineEdit()
        self.volume_is_edit.setPlaceholderText("e.g., 2")
        is_layout.addRow("Volume Added (µL):", self.volume_is_edit)
        
        # Calculated mol C (read-only)
        self.mol_c_is_label = QLabel("—")
        self.mol_c_is_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        is_layout.addRow("mol C of IS:", self.mol_c_is_label)
        
        is_group.setLayout(is_layout)
        layout.addWidget(is_group)
        
        # Sample Preparation group
        sample_group = QGroupBox("Sample Preparation")
        sample_layout = QFormLayout()
        
        # Sample volume
        self.volume_sample_edit = QLineEdit()
        self.volume_sample_edit.setPlaceholderText("e.g., 4")
        sample_layout.addRow("Sample Volume (µL):", self.volume_sample_edit)
        
        # Sample density (optional)
        self.density_sample_edit = QLineEdit()
        self.density_sample_edit.setPlaceholderText("Optional")
        sample_layout.addRow("Sample Density (g/mL):", self.density_sample_edit)
        
        # Calculated sample mass (read-only)
        self.mass_sample_label = QLabel("—")
        self.mass_sample_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        sample_layout.addRow("Sample Mass (mg):", self.mass_sample_label)
        
        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)
        
        # Results group (shown after quantitation)
        results_group = QGroupBox("Quantitation Results")
        results_layout = QFormLayout()
        
        # Response factor
        self.rf_label = QLabel("—")
        self.rf_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        results_layout.addRow("Response Factor:", self.rf_label)
        
        # Carbon balance
        self.c_balance_label = QLabel("—")
        self.c_balance_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        results_layout.addRow("Carbon Balance (%):", self.c_balance_label)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Re-Quantitate button
        self.requantitate_btn = QPushButton("Re-Quantitate")
        self.requantitate_btn.setEnabled(False)
        self.requantitate_btn.setToolTip("Re-run quantitation with current settings without running MS search")
        layout.addWidget(self.requantitate_btn)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        # Initially disable all inputs
        self._set_inputs_enabled(False)
        
    def _connect_signals(self):
        """Connect signals to slots."""
        self.enable_checkbox.toggled.connect(self._on_enable_toggled)
        self.search_library_btn.clicked.connect(self._on_search_library)
        self.requantitate_btn.clicked.connect(self._on_requantitate)
        
        # Connect input changes to calculation updates
        self.compound_edit.textChanged.connect(self._on_inputs_changed)
        self.formula_edit.textChanged.connect(self._update_mol_c_is)
        self.mw_edit.textChanged.connect(self._update_mol_c_is)
        self.density_edit.textChanged.connect(self._update_mol_c_is)
        self.volume_is_edit.textChanged.connect(self._update_mol_c_is)
        
        self.volume_sample_edit.textChanged.connect(self._update_sample_mass)
        self.density_sample_edit.textChanged.connect(self._update_sample_mass)
        
    def _on_enable_toggled(self, checked):
        """Handle enable/disable checkbox."""
        self._set_inputs_enabled(checked)
        self.requantitate_btn.setEnabled(checked)
        self.quantitation_changed.emit()
        
    def _set_inputs_enabled(self, enabled):
        """Enable or disable all input fields."""
        self.compound_edit.setEnabled(enabled)
        self.search_library_btn.setEnabled(enabled)
        self.formula_edit.setEnabled(enabled)
        self.mw_edit.setEnabled(enabled)
        self.density_edit.setEnabled(enabled)
        self.volume_is_edit.setEnabled(enabled)
        self.volume_sample_edit.setEnabled(enabled)
        self.density_sample_edit.setEnabled(enabled)
    
    def _on_requantitate(self):
        """Handle re-quantitate button click."""
        self.requantitate_requested.emit()
        
    def _on_search_library(self):
        """Search MS library for compound and autofill formula/MW."""
        compound_name = self.compound_edit.text().strip()
        if not compound_name:
            return
        
        # Try to get compound from MS library
        if self.ms_toolkit and hasattr(self.ms_toolkit, 'library'):
            if compound_name in self.ms_toolkit.library:
                compound = self.ms_toolkit.library[compound_name]
                
                # Autofill formula
                if hasattr(compound, 'formula'):
                    self.formula_edit.setText(compound.formula)
                    
                # Autofill MW
                if hasattr(compound, 'mw'):
                    self.mw_edit.setText(str(compound.mw))
                    
                print(f"Autofilled {compound_name}: formula={getattr(compound, 'formula', 'N/A')}, MW={getattr(compound, 'mw', 'N/A')}")
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, 
                    "Compound Not Found",
                    f"Compound '{compound_name}' not found in MS library.\n\n"
                    "Make sure the library is loaded and the compound name is correct."
                )
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Library Not Available",
                "MS library is not loaded. Please load a library first."
            )
        
    def _on_inputs_changed(self):
        """Handle any input change."""
        self.quantitation_changed.emit()
        
    def _update_mol_c_is(self):
        """Update calculated mol C of internal standard."""
        try:
            volume_uL = float(self.volume_is_edit.text())
            density = float(self.density_edit.text())
            mw = float(self.mw_edit.text())
            formula = self.formula_edit.text().strip()
            
            mol_c = self.calculator.calculate_mol_C_internal_standard(
                volume_uL, density, mw, formula
            )
            
            if mol_c is not None:
                self.mol_c_is_label.setText(f"{mol_c:.6e} mol")
            else:
                self.mol_c_is_label.setText("Invalid formula")
                
        except (ValueError, AttributeError):
            self.mol_c_is_label.setText("—")
            
        self._on_inputs_changed()
        
    def _update_sample_mass(self):
        """Update calculated sample mass."""
        try:
            volume_uL = float(self.volume_sample_edit.text())
            
            if self.density_sample_edit.text().strip():
                density = float(self.density_sample_edit.text())
                mass_mg = self.calculator.calculate_sample_mass(volume_uL, density)
                
                if mass_mg is not None:
                    self.mass_sample_label.setText(f"{mass_mg:.4f} mg")
                else:
                    self.mass_sample_label.setText("—")
            else:
                self.mass_sample_label.setText("—")
                
        except (ValueError, AttributeError):
            self.mass_sample_label.setText("—")
            
        self._on_inputs_changed()
        
    def is_enabled(self):
        """Check if quantitation is enabled."""
        return self.enable_checkbox.isChecked()
        
    def get_settings(self):
        """Get current quantitation settings."""
        if not self.is_enabled():
            return None

        try:
            settings = {
                'enabled': True,
                'method': self.method_combo.currentText(),
                'internal_standard': {
                    'compound': self.compound_edit.text().strip(),
                    'formula': self.formula_edit.text().strip(),
                    'MW': float(self.mw_edit.text()) if self.mw_edit.text() else None,
                    'density': float(self.density_edit.text()) if self.density_edit.text() else None,
                    'volume_uL': float(self.volume_is_edit.text()) if self.volume_is_edit.text() else None,
                },
                'sample': {
                    'volume_uL': float(self.volume_sample_edit.text()) if self.volume_sample_edit.text() else None,
                    'density': float(self.density_sample_edit.text()) if self.density_sample_edit.text().strip() else None,
                },
                'overwrite_results': self.overwrite_checkbox.isChecked()
            }

            # Calculate mol C
            if all([settings['internal_standard']['volume_uL'],
                   settings['internal_standard']['density'],
                   settings['internal_standard']['MW'],
                   settings['internal_standard']['formula']]):
                mol_c = self.calculator.calculate_mol_C_internal_standard(
                    settings['internal_standard']['volume_uL'],
                    settings['internal_standard']['density'],
                    settings['internal_standard']['MW'],
                    settings['internal_standard']['formula']
                )
                settings['internal_standard']['mol_C'] = mol_c
            else:
                settings['internal_standard']['mol_C'] = None

            # Calculate sample mass
            if settings['sample']['volume_uL'] and settings['sample']['density']:
                mass_mg = self.calculator.calculate_sample_mass(
                    settings['sample']['volume_uL'],
                    settings['sample']['density']
                )
                settings['sample']['mass_mg'] = mass_mg
            else:
                settings['sample']['mass_mg'] = None

            return settings

        except (ValueError, TypeError):
            return None
            
    def set_response_factor(self, rf):
        """Set and display the calculated response factor."""
        if rf is not None:
            self.rf_label.setText(f"{rf:.2e}")
        else:
            self.rf_label.setText("—")
            
    def set_carbon_balance(self, c_balance):
        """Set and display the carbon balance."""
        if c_balance is not None:
            self.c_balance_label.setText(f"{c_balance:.1f}%")
        else:
            self.c_balance_label.setText("—")
            
    def autofill_from_library(self, compound_name, formula, mw):
        """Autofill formula and MW from MS library."""
        if formula:
            self.formula_edit.setText(formula)
        if mw:
            self.mw_edit.setText(str(mw))
