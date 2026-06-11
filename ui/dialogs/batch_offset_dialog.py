"""Modal dialog: apply an MS time offset to multiple sibling .D files.

Opened by ChromaKitApp when the user clicks 'Apply to folder...' in
the spectral deconvolution inspector. The dialog presents a checkbox
list of all sibling .D paths in the current file's parent folder,
with existing-offset markers on rows that would be overwritten.
On Accept, the caller queries selected_paths() and persists via
logic.sidecar_offsets.save_offsets_batch.
"""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from logic.sidecar_offsets import OffsetEntry


class BatchOffsetDialog(QDialog):
    """Modal dialog for applying an MS time offset to multiple .D files.

    The current file's row is always pre-checked AND disabled (cannot be
    unchecked). Sibling rows are pre-checked by default. Rows whose path
    has an existing sidecar entry show a warning marker with the current
    saved offset value.
    """

    def __init__(
        self,
        offset_min: float,
        siblings: list,            # absolute .D paths (sorted)
        current_path: str,         # absolute path of the .D currently loaded
        existing_offsets: dict,    # {abs_path: OffsetEntry} for paths with entries
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Apply MS Time Offset to Folder")
        self.setModal(True)
        self.setMinimumSize(520, 420)

        self._offset_min = float(offset_min)
        self._siblings = list(siblings)
        self._current_path = str(current_path)
        self._existing_offsets = dict(existing_offsets or {})

        # Each row: (abs_path, checkbox_widget)
        self._row_widgets: list = []

        self._build_ui()
        self._update_count_label()
        self._update_apply_button_state()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Offset value readout
        offset_str = f"{self._offset_min:+.4f} min ({self._offset_min * 60:+.2f} s)"
        offset_label = QLabel(f"Offset to apply: <b>{offset_str}</b>")
        layout.addWidget(offset_label)

        # Folder path
        parent_dir = os.path.dirname(self._current_path)
        layout.addWidget(QLabel("Apply to the following .D files in:"))
        folder_label = QLabel(f"    <code>{parent_dir}</code>")
        folder_label.setTextFormat(Qt.RichText)
        layout.addWidget(folder_label)

        # Scrollable list of sibling rows
        self._list_widget = QListWidget()
        for sib_path in self._siblings:
            self._add_row(sib_path)
        layout.addWidget(self._list_widget, 1)

        # Live count label
        self._count_label = QLabel()
        layout.addWidget(self._count_label)

        # Selection helper buttons
        helper_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        select_none_btn = QPushButton("Select none")
        select_non_conflict_btn = QPushButton("Select non-conflicts")
        select_all_btn.clicked.connect(self._on_select_all)
        select_none_btn.clicked.connect(self._on_select_none)
        select_non_conflict_btn.clicked.connect(self._on_select_non_conflicts)
        helper_row.addWidget(select_all_btn)
        helper_row.addWidget(select_none_btn)
        helper_row.addWidget(select_non_conflict_btn)
        helper_row.addStretch(1)
        layout.addLayout(helper_row)

        # Cancel / Apply buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        self._apply_btn = self._button_box.button(QDialogButtonBox.Apply)
        self._apply_btn.setText("Apply")
        self._apply_btn.setDefault(True)
        self._button_box.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        self._apply_btn.clicked.connect(self.accept)
        layout.addWidget(self._button_box)

    def _add_row(self, abs_path: str) -> None:
        """Add one row (checkbox + label) for one sibling .D path."""
        is_current = (abs_path == self._current_path)
        existing = self._existing_offsets.get(abs_path)

        # Build label text
        basename = os.path.basename(abs_path)
        text_parts = [basename]
        if is_current:
            text_parts.append("    (current)")
        if existing is not None:
            text_parts.append(
                f"    ⚠ already {existing.offset_min:+.4f} min"
            )
        label_text = " ".join(text_parts)

        checkbox = QCheckBox(label_text)
        checkbox.setChecked(True)  # all-checked default per spec
        checkbox.setToolTip(abs_path)
        if existing is not None and not is_current:
            # Conflict styling (current row stays plain even if it has existing)
            checkbox.setStyleSheet("color: #b27200;")  # warning amber
        if is_current:
            checkbox.setEnabled(False)  # disabled-checked
        checkbox.stateChanged.connect(self._on_row_toggled)

        item = QListWidgetItem(self._list_widget)
        item.setSizeHint(checkbox.sizeHint())
        self._list_widget.addItem(item)
        self._list_widget.setItemWidget(item, checkbox)

        self._row_widgets.append((abs_path, checkbox))

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_row_toggled(self, _state: int) -> None:
        self._update_count_label()
        self._update_apply_button_state()

    def _on_select_all(self) -> None:
        for _path, cb in self._row_widgets:
            if cb.isEnabled():
                cb.setChecked(True)
        self._update_count_label()
        self._update_apply_button_state()

    def _on_select_none(self) -> None:
        for _path, cb in self._row_widgets:
            if cb.isEnabled():
                cb.setChecked(False)
        self._update_count_label()
        self._update_apply_button_state()

    def _on_select_non_conflicts(self) -> None:
        for path, cb in self._row_widgets:
            if not cb.isEnabled():
                continue  # current row stays as-is (disabled-checked)
            has_existing = path in self._existing_offsets
            cb.setChecked(not has_existing)
        self._update_count_label()
        self._update_apply_button_state()

    def _update_count_label(self) -> None:
        total = len(self._row_widgets)
        checked = sum(1 for _p, cb in self._row_widgets if cb.isChecked())
        self._count_label.setText(f"☑ {checked} of {total} selected")

    def _update_apply_button_state(self) -> None:
        # Enabled only when at least one non-current row is checked.
        non_current_checked = any(
            cb.isChecked() and path != self._current_path
            for path, cb in self._row_widgets
        )
        self._apply_btn.setEnabled(non_current_checked)
        if non_current_checked:
            self._apply_btn.setToolTip("")
        else:
            self._apply_btn.setToolTip(
                "Use 'Apply' on the inspector for current-file-only."
            )

    # ── Public API ─────────────────────────────────────────────────────

    def selected_paths(self) -> list:
        """Return list of checked .D paths. Always includes current_path."""
        return [path for path, cb in self._row_widgets if cb.isChecked()]
