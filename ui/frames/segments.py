"""Segments table and segment editor for the Parameters pane (design 2026-11-12 §4.2).

``SegmentsPanel`` lists the Whole run row and one row per segment. ``SegmentEditor``
edits one segment record as shaped by ``logic.segments.segments_view``; the host
applies the records it emits with ``apply_segment``. Neither widget touches params.
"""
import copy
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget,
)

from logic.segments import summarize
from logic.windows import merge_overrides
from ui.widgets.override_form import _ERROR_STYLE, SECTION_FIELDS, OverrideField, OverrideGroup

WHOLE_RUN = 'Whole run (defaults)'
HINT = 'Grey = same as the whole run. Type a value to change it for this segment; ↺ puts it back.'
DECONV_NOTE = 'deconvolution mode: segments filter only'
SAVE_BLOCKED = 'Fix the fields marked in red before saving'
PARTWAY_TAG = 'same as segment'
GROUPS = (('baseline', 'Baseline'), ('peaks', 'Peak detection'), ('smoothing', 'Smoothing'),
          ('shoulders', 'Shoulders'), ('rider', 'Tail skim'), ('negative_peaks', 'Negative peaks'))
DETECTION_GROUPS = ('peaks', 'shoulders', 'rider', 'negative_peaks')
PARTWAY_KEYS = ('enabled', 'method', 'lambda', 'baseline_offset')
_OUTSIDE = (('ignore', 'ignore'), ('use whole-run settings', 'global'))
_MAX_VISIBLE_ROWS = 6
_GREY_STYLE = 'color: #666666; font-size: 10px;'
_WARN_STYLE = 'color: #b36b00; font-size: 10px;'
_RED_STYLE = 'color: #c0392b;'


def _split(values, keys):
    """``(shown, carried)``: the overrides a form has fields for (flat 'a.b' keys) and the rest.

    The rest (keys a file may hold that the form doesn't show) is carried through unchanged.
    """
    shown, carried = {}, {}
    for key, val in (values or {}).items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                (shown if f"{key}.{k2}" in keys else carried).setdefault(key, {})[k2] = copy.deepcopy(v2)
        else:
            (shown if key in keys else carried)[key] = copy.deepcopy(val)
    return shown, carried


class _TimeEdit(QLineEdit):
    """A required float typed as text; bad text turns red and keeps the last good value."""
    committed = Signal()

    def __init__(self, value: float, parent=None):
        super().__init__(parent)
        self._value = float(value)
        self._error = False
        self.set_value(value)
        self.editingFinished.connect(self._on_finished)

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = float(value)
        self.setText(f"{self._value:.12g}")
        self._set_error(None)

    def has_error(self) -> bool:
        return self._error

    def _set_error(self, msg):
        self._error = bool(msg)
        self.setStyleSheet(_ERROR_STYLE if msg else '')
        self.setToolTip(msg or '')

    def _on_finished(self):
        try:
            value = float(self.text().strip())
            if not math.isfinite(value):
                raise ValueError(self.text())
        except ValueError:
            self._set_error('Enter a time (number)')
            return
        self._set_error(None)
        if value != self._value:
            self._value = value
            self.committed.emit()


# ── Segments table ─────────────────────────────────────────────────────────────

class SegmentsPanel(QGroupBox):
    """Whole run row plus one row per segment; clicking a row asks the host to open it."""
    open_requested = Signal(str)        # segment id; '' = Whole run
    add_requested = Signal()
    delete_requested = Signal(str)
    outside_changed = Signal(str)       # 'ignore' | 'global'

    def __init__(self, parent=None):
        super().__init__('Segments', parent)
        self._editing = False
        self._names = {}     # id -> shown name, for re-marking warnings
        self._warnings = {}  # id -> tooltip text

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(QLabel('Outside segments:'))
        self.outside_combo = QComboBox()
        for text, value in _OUTSIDE:
            self.outside_combo.addItem(text, value)
        self.outside_combo.currentIndexChanged.connect(
            lambda _i: self.outside_changed.emit(self.outside_combo.currentData()))
        top.addWidget(self.outside_combo)
        layout.addLayout(top)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Name', 'From', 'To', 'What it changes'])
        header = self.table.horizontalHeader()
        for col in (0, 1, 2):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.cellClicked.connect(lambda row, _col: self.open_requested.emit(self._row_id(row)))
        self.table.cellActivated.connect(lambda row, _col: self.open_requested.emit(self._row_id(row)))
        self.table.itemSelectionChanged.connect(self._update_delete)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.add_button = QPushButton('+ Segment')
        self.add_button.clicked.connect(self.add_requested.emit)
        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(self._on_delete)
        hint = QLabel('or shift-drag on the plot')
        hint.setStyleSheet(_GREY_STYLE)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        buttons.addWidget(hint)
        layout.addLayout(buttons)
        self._update_delete()

    def _row_id(self, row):
        return self.table.item(row, 0).data(Qt.UserRole)

    def _selected_id(self):
        rows = self.table.selectionModel().selectedRows()
        return self._row_id(rows[0].row()) if rows else None

    def _update_delete(self):
        self.delete_button.setEnabled(not self._editing and bool(self._selected_id()))

    def _on_delete(self):
        sid = self._selected_id()
        if sid:
            self.delete_requested.emit(sid)

    def set_view(self, view: dict, whole_run_summary: str, outside: str) -> None:
        idx = self.outside_combo.findData(outside)
        if idx < 0:
            raise ValueError(f"unknown outside-segments mode {outside!r}")
        self.outside_combo.blockSignals(True)
        self.outside_combo.setCurrentIndex(idx)
        self.outside_combo.blockSignals(False)

        selected = self._selected_id()
        rows = [('', WHOLE_RUN, 'start', 'end', whole_run_summary)]
        rows += [(r['id'], r.get('label') or r['id'], f"{r['start']:g}", f"{r['end']:g}", summarize(r))
                 for r in view['segments']]
        self._names = {sid: name for sid, name, *_ in rows}  # '' is Whole run
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        italic = QFont(self.table.font())
        italic.setItalic(True)
        for i, (sid, *texts) in enumerate(rows):
            for col, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if not sid:
                    item.setFont(italic)
                self.table.setItem(i, col, item)
            self.table.item(i, 0).setData(Qt.UserRole, sid)
        self.table.clearSelection()
        for i, (sid, *_rest) in enumerate(rows):
            if selected is not None and sid == selected:
                self.table.selectRow(i)
        self.table.blockSignals(False)
        for sid in self._names:
            self._mark(sid)
        visible = min(len(rows), _MAX_VISIBLE_ROWS)
        self.table.setFixedHeight(self.table.horizontalHeader().sizeHint().height()
                                  + visible * self.table.verticalHeader().defaultSectionSize()
                                  + 2 * self.table.frameWidth())
        self._update_delete()

    def set_editing(self, editing: bool) -> None:
        """While a segment is open the table, + Segment, Delete and the outside combo are off.

        The outside combo too: the host's Cancel restores the method as it was when the
        segment opened, which would silently undo an outside change made meanwhile.
        """
        self._editing = editing
        self.table.setEnabled(not editing)
        self.add_button.setEnabled(not editing)
        self.outside_combo.setEnabled(not editing)
        self._update_delete()

    def set_row_warning(self, segment_id: str, text: str | None) -> None:
        if text:
            self._warnings[segment_id] = text
        else:
            self._warnings.pop(segment_id, None)
        self._mark(segment_id)

    def _mark(self, segment_id):
        for i in range(self.table.rowCount()):
            if self._row_id(i) != segment_id:
                continue
            text = self._warnings.get(segment_id)
            name = self._names[segment_id]
            self.table.item(i, 0).setText(f"⚠ {name}" if text else name)
            for col in range(self.table.columnCount()):
                self.table.item(i, col).setToolTip(text or '')


# ── Segment editor ─────────────────────────────────────────────────────────────

class _PartwayRow(QFrame):
    """One baseline change inside the segment: its time and its own overrides."""
    changed = Signal()
    delete_requested = Signal()

    def __init__(self, time: float, overrides: dict, inherited: dict, parent=None):
        super().__init__(parent)
        specs = {s.key: s for s in SECTION_FIELDS['baseline']}
        shown, self._carried = _split(overrides, PARTWAY_KEYS)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        self.time_edit = _TimeEdit(time)
        self.time_edit.committed.connect(self.changed.emit)
        self.delete_button = QToolButton()
        self.delete_button.setText('✕')
        self.delete_button.setAutoRaise(True)
        self.delete_button.setToolTip('Remove this baseline change')
        self.delete_button.clicked.connect(self.delete_requested.emit)
        top = QHBoxLayout()
        top.addWidget(self.time_edit, 1)
        top.addWidget(self.delete_button)
        form.addRow(QLabel('From'), top)
        self.fields = {}
        for key in PARTWAY_KEYS:
            f = OverrideField(specs[key], PARTWAY_TAG)
            self.fields[key] = f
            form.addRow(QLabel(specs[key].label), f)
        self.set_inherited(inherited)
        for key, val in shown.items():
            self.fields[key].set_value(val)
        for f in self.fields.values():
            f.changed.connect(self.changed.emit)

    def set_inherited(self, inherited: dict) -> None:
        for key, f in self.fields.items():
            f.set_inherited(inherited.get(key))

    def has_errors(self) -> bool:
        return self.time_edit.has_error() or any(f.has_error() for f in self.fields.values())

    def record(self) -> dict:
        own = {k: f.value() for k, f in self.fields.items() if f.value() is not None}
        return {'time': self.time_edit.value(), 'overrides': merge_overrides(self._carried, own)}


class _PartwayTable(QFrame):
    """'Change the baseline partway through this segment': rows plus + Change at…."""
    changed = Signal()
    add_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('partwayTable')
        self.setStyleSheet('QFrame#partwayTable { border: 1px dashed #d4a72c; border-radius: 3px; }')
        self.rows = []
        layout = QVBoxLayout(self)
        title = QLabel('Change the baseline partway through this segment')
        title.setStyleSheet('color: #8a5a00; font-weight: bold;')
        layout.addWidget(title)
        self._rows_layout = QVBoxLayout()
        layout.addLayout(self._rows_layout)
        buttons = QHBoxLayout()
        self.add_button = QPushButton('+ Change at…')
        self.add_button.clicked.connect(self.add_clicked.emit)
        hint = QLabel('or right-click the plot inside the segment')
        hint.setStyleSheet(_GREY_STYLE)
        buttons.addWidget(self.add_button)
        buttons.addWidget(hint)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def add_row(self, time, overrides, inherited) -> None:
        row = _PartwayRow(time, overrides, inherited)
        row.changed.connect(self.changed.emit)
        row.delete_requested.connect(lambda: self._delete(row))
        self.rows.append(row)
        self._rows_layout.addWidget(row)
        return row

    def _delete(self, row):
        self.rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self.changed.emit()

    def set_inherited(self, inherited: dict) -> None:
        for row in self.rows:
            row.set_inherited(inherited)


class SegmentEditor(QFrame):
    """Edits one segment record; emits records, never touches params.

    ``inherited`` holds the whole-run values per section (``{'baseline': {...}, 'peaks':
    {...}, ...}``); ``resume`` the whole-run baseline overrides in force before the
    segment (``logic.segments.resume_overrides``). While any field holds bad text,
    ``record()`` keeps returning the last valid record and nothing is previewed or saved.
    """
    preview_changed = Signal(dict)      # current record
    save_requested = Signal(dict)
    cancel_requested = Signal()
    delete_requested = Signal(str)      # segment id; never for a new (unsaved) segment

    def __init__(self, record: dict, inherited: dict, resume: dict, deconvolution: bool,
                 is_new: bool, parent=None):
        super().__init__(parent)
        self._base = copy.deepcopy(record)  # id, events and any key the editor doesn't show
        self._inherited = copy.deepcopy(inherited)
        self._resume = copy.deepcopy(resume)
        self._is_new = is_new
        self._after = copy.deepcopy(record.get('after') or {'mode': 'none', 'overrides': {}})
        self._committed_label = record.get('label', '')
        # apply_segment writes 'none' as 'resume' when the segment gains its first fresh
        # start or partway row, so 'none' is kept only if the stored segment had either
        self._had_baseline = not is_new and bool(record.get('fresh_baseline') or record.get('partway'))
        self._had_errors = False  # error state at the last check; clearing it previews
        self.setObjectName('segmentEditor')
        self.setStyleSheet('QFrame#segmentEditor { border: 2px solid #1f6feb; border-radius: 4px; }')

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setStyleSheet('font-weight: bold;')
        self.save_button = QPushButton('Save')
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._on_save)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.delete_button = QPushButton('Delete segment')
        self.delete_button.setHidden(is_new)
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self._base['id']))
        header.addWidget(self.title_label, 1)
        header.addWidget(self.delete_button)
        header.addWidget(self.save_button)
        header.addWidget(self.cancel_button)
        layout.addLayout(header)

        span = QHBoxLayout()
        self.label_edit = QLineEdit(record.get('label', ''))
        self.label_edit.setPlaceholderText(record['id'])
        self.label_edit.textChanged.connect(self._update_title)
        self.label_edit.editingFinished.connect(self._on_label_finished)
        self.from_edit = _TimeEdit(record['start'])
        self.to_edit = _TimeEdit(record['end'])
        for edit in (self.from_edit, self.to_edit):
            edit.setMaximumWidth(90)
            edit.committed.connect(self._on_span_committed)
        span.addWidget(QLabel('Name'))
        span.addWidget(self.label_edit, 1)
        span.addWidget(self.from_edit)
        span.addWidget(QLabel('–'))
        span.addWidget(self.to_edit)
        layout.addLayout(span)

        self.hint_label = QLabel(HINT)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(_GREY_STYLE)
        layout.addWidget(self.hint_label)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(_RED_STYLE)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(_WARN_STYLE)
        self.warning_label.hide()
        layout.addWidget(self.warning_label)
        self.deconv_note = QLabel(DECONV_NOTE)
        self.deconv_note.setStyleSheet(_WARN_STYLE)
        self.deconv_note.setVisible(deconvolution)
        layout.addWidget(self.deconv_note)

        self.groups = {}
        self._carried = {}  # per section: overrides with no field here, written back as loaded
        for section, title in GROUPS:
            keys = {s.key for s in SECTION_FIELDS[section]}
            shown, self._carried[section] = _split(record.get(section), keys)
            group = OverrideGroup(title, section, inherited.get(section) or {}, shown)
            self.groups[section] = group
            layout.addWidget(group)
        self._build_baseline_extras(record)
        for section in DETECTION_GROUPS:
            self.groups[section].setEnabled(not deconvolution)
        layout.addStretch(1)

        self._update_title()
        self._refresh_baseline()
        for section, group in self.groups.items():
            group.changed.connect(self._on_baseline_changed if section == 'baseline' else self._preview)
        self._watch_edits(self)
        self._last_valid = self._build()

    def _watch_edits(self, widget):
        """Check errors after every text edit (connected after the fields' own handlers).

        Bad text can be cleared without changing any value (retyping the old From, or
        blanking an inheriting field), so no field emits; this catches that case.
        """
        for edit in widget.findChildren(QLineEdit):
            edit.editingFinished.connect(self._on_any_edit_finished)

    def _build_baseline_extras(self, record):
        group = self.groups['baseline']
        form = group.body_layout()
        self.fresh_check = QCheckBox()
        self.fresh_check.setChecked(bool(record.get('fresh_baseline')))
        form.insertRow(0, self.fresh_check)
        self.partway = _PartwayTable()
        for row in record.get('partway') or []:
            self.partway.add_row(row['time'], row.get('overrides'), self._segment_baseline())
        self.partway.changed.connect(self._on_partway_changed)
        self.partway.add_clicked.connect(
            lambda: self.add_partway((self.from_edit.value() + self.to_edit.value()) / 2))
        form.addRow(self.partway)
        after = QHBoxLayout()
        self.after_label = QLabel()
        self.after_label.setWordWrap(True)
        self.after_label.setStyleSheet(_GREY_STYLE)
        self.after_reset = QToolButton()
        self.after_reset.setText('↺')
        self.after_reset.setAutoRaise(True)
        self.after_reset.setToolTip('Let the whole-run baseline take over again after this segment')
        self.after_reset.clicked.connect(self._on_after_reset)
        after.addWidget(self.after_label, 1)
        after.addWidget(self.after_reset)
        form.addRow(after)
        self.fresh_check.toggled.connect(self._on_fresh_toggled)
        group.set_expanded(bool(record.get('fresh_baseline') or record.get('partway')
                                or group.values()))

    # ── state → widgets ──

    def _section(self, section):
        return merge_overrides(self._carried[section], self.groups[section].values())

    def _segment_baseline(self):
        """The baseline partway rows inherit: own values when fresh, else resume, over the whole run."""
        own = self._section('baseline') if self.fresh_check.isChecked() else self._resume
        return merge_overrides(self._inherited.get('baseline') or {}, own)

    def _update_title(self):
        text = self.label_edit.text()
        if text:
            self.title_label.setText(f"Segment: {text}")
        else:
            self.title_label.setText('New segment' if self._is_new else f"Segment: {self._base['id']}")

    def _refresh_baseline(self):
        """Fresh tick text and field state, partway inheritance, after-line, change count."""
        fresh = self.fresh_check.isChecked()
        group = self.groups['baseline']
        self.fresh_check.setText(f"Start a fresh baseline at {self.from_edit.value():g}")
        for f in group.fields.values():
            f.setEnabled(fresh)
        self.partway.set_inherited(self._segment_baseline())
        end = self.to_edit.value()
        mode = self._after['mode']
        carries_on = (mode == 'none' and self._had_baseline
                      and (fresh or bool(self.partway.rows)))  # no end break: R11
        if mode == 'custom':
            summary = summarize({'baseline': self._after.get('overrides') or {}})
            self.after_label.setText(f"After {end:g}: custom ({summary})")
        elif carries_on:
            self.after_label.setText(f"After {end:g} this segment's baseline carries on")
        else:
            self.after_label.setText(f"After {end:g} the whole-run baseline takes over again.")
        self.after_reset.setHidden(not (mode == 'custom' or carries_on))
        n = (len(group.values()) if fresh else 0) + len(self.partway.rows)
        group.count_label.setText('same as whole run' if n == 0
                                  else ('1 change' if n == 1 else f"{n} changes"))

    # ── user edits ──

    def _preview(self):
        """Emit the record after a committed change; held while any field has bad text."""
        self._had_errors = self.has_errors()
        if self._had_errors:
            return
        if self.error_label.text() == SAVE_BLOCKED:
            self.show_error(None)
        self.preview_changed.emit(self.record())

    def _on_any_edit_finished(self):
        if self._had_errors and not self.has_errors():
            self._preview()  # edits held while the error stood
        else:
            self._had_errors = self.has_errors()

    def _on_label_finished(self):
        if self.label_edit.text() != self._committed_label:
            self._committed_label = self.label_edit.text()
            self._preview()

    def _on_span_committed(self):
        self._refresh_baseline()
        self._preview()

    def _on_fresh_toggled(self, _checked):
        self._refresh_baseline()
        self._preview()

    def _on_baseline_changed(self):
        self._refresh_baseline()  # also re-sets the count OverrideGroup just wrote
        self._preview()

    def _on_partway_changed(self):
        self._refresh_baseline()
        self._preview()

    def _on_after_reset(self):
        self._after = {'mode': 'resume', 'overrides': copy.deepcopy(self._resume)}
        self._refresh_baseline()
        self._preview()

    def _on_save(self):
        if self.has_errors():
            self.show_error(SAVE_BLOCKED)
            return
        self.show_error(None)
        self.save_requested.emit(self.record())

    # ── public API ──

    def has_errors(self) -> bool:
        groups = [g for s, g in self.groups.items()
                  if s != 'baseline' or self.fresh_check.isChecked()]  # unticked: values unused
        return (self.from_edit.has_error() or self.to_edit.has_error()
                or any(g.has_errors() for g in groups)
                or any(row.has_errors() for row in self.partway.rows))

    def _build(self):
        out = copy.deepcopy(self._base)
        fresh = self.fresh_check.isChecked()
        out.update({
            'label': self.label_edit.text(),
            'start': self.from_edit.value(),
            'end': self.to_edit.value(),
            'fresh_baseline': fresh,
            'baseline': self._section('baseline') if fresh else {},
            'partway': sorted((row.record() for row in self.partway.rows), key=lambda r: r['time']),
            'after': copy.deepcopy(self._after),
        })
        for section, _title in GROUPS:
            if section != 'baseline':
                out[section] = self._section(section)
        return out

    def record(self) -> dict:
        if not self.has_errors():
            self._last_valid = self._build()
        return copy.deepcopy(self._last_valid)

    def set_span(self, start: float, end: float) -> None:
        """Move From/To (e.g. an edge dragged on the plot); one preview if anything moved."""
        if ((float(start), float(end)) == (self.from_edit.value(), self.to_edit.value())
                and not (self.from_edit.has_error() or self.to_edit.has_error())):
            return
        self.from_edit.set_value(start)
        self.to_edit.set_value(end)
        self._on_span_committed()

    def set_resume(self, resume: dict) -> None:
        """New whole-run value in force before the segment (From moved across a whole-run
        break); partway rows and the after-line follow it. Emits nothing."""
        if resume == self._resume:
            return
        self._resume = copy.deepcopy(resume)
        self._refresh_baseline()

    def add_partway(self, time: float) -> None:
        self._watch_edits(self.partway.add_row(float(time), {}, self._segment_baseline()))
        self.groups['baseline'].set_expanded(True)
        self._on_partway_changed()

    def show_error(self, msg: str | None) -> None:
        self.error_label.setText(msg or '')
        self.error_label.setVisible(bool(msg))

    def set_warnings(self, lines: list[str]) -> None:
        self.warning_label.setText('\n'.join(f"⚠ {line}" for line in lines))
        self.warning_label.setVisible(bool(lines))
