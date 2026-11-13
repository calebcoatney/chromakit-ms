"""Inherit-or-override form widgets for per-segment settings.

An OverrideField edits one optional setting: blank/first combo item means
"inherit the whole-run value" (shown greyed), anything else is an override.
OverrideGroup stacks the fields of one *Overrides model.
"""
import math
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QToolButton, QVBoxLayout, QWidget,
)

INHERIT_TAG = 'same as whole run'
OVERRIDE_TAG = 'changed ↺'
_ERROR_STYLE = 'border: 1px solid #c0392b;'
_GREY = QColor('#888888')
_TAG_STYLE = 'color: #888888; font-style: italic;'
_CHANGED_STYLE = 'QToolButton { color: #1f6feb; border: none; }'
_OPEN, _CLOSED = '▾', '▸'


@dataclass(frozen=True)
class FieldSpec:
    key: str            # override key as dumped by alias ('lambda', not 'lambda_'); 'a.b' nests
    label: str          # matches the main group's label text so help icons map
    kind: str           # 'float' | 'int' | 'bool' | 'choice'
    choices: tuple = ()  # ((text, value), ...) for 'choice'


_RIDER_MODES = (("Automatic", "auto"), ("Always skim", "skim"), ("Always split", "split"))
_RIDER_MODELS = (("Exponential", "exponential"), ("Tangent", "tangent"))
_BASELINE_METHODS = (
    ("asls - Asymmetric Least Squares", "asls"),
    ("imodpoly - Improved Modified Polynomial", "imodpoly"),
    ("modpoly - Modified Polynomial", "modpoly"),
    ("snip - Statistics-sensitive Non-linear", "snip"),
    ("airpls - Adaptive Iteratively Reweighted", "airpls"),
    ("arpls - Asymmetrically Reweighted", "arpls"),
    ("mixture_model - Mixture Model", "mixture_model"),
    ("irsqr - Spline Quantile Regression", "irsqr"),
    ("fastchrom - FastChrom's Baseline", "fastchrom"),
)
_SMOOTH_METHODS = (("Whittaker", "whittaker"), ("Savitzky-Golay", "savgol"))

SECTION_FIELDS = {
    'baseline': (
        FieldSpec('enabled', 'Enable Baseline Correction', 'bool'),
        FieldSpec('method', 'Algorithm:', 'choice', _BASELINE_METHODS),
        FieldSpec('lambda', 'Lambda (λ):', 'float'),
        FieldSpec('baseline_offset', 'Baseline Offset:', 'float'),
        FieldSpec('fastchrom.half_window', 'Half Window:', 'int'),
        FieldSpec('fastchrom.smooth_half_window', 'Smooth Half Window:', 'int'),
    ),
    'smoothing': (
        FieldSpec('enabled', 'Enable Smoothing', 'bool'),
        FieldSpec('method', 'Method:', 'choice', _SMOOTH_METHODS),
        FieldSpec('median_enabled', 'Median Pre-filter', 'bool'),
        FieldSpec('median_kernel', 'Median kernel:', 'int'),
        FieldSpec('lambda', 'Smoothing λ:', 'float'),
        FieldSpec('diff_order', 'Penalty order:', 'int'),
        FieldSpec('savgol_window', 'Window length:', 'int'),
        FieldSpec('savgol_polyorder', 'Polynomial order:', 'int'),
    ),
    'peaks': (
        FieldSpec('min_prominence', 'Min Prominence:', 'float'),
        FieldSpec('min_width', 'Min Width (samples):', 'float'),
        FieldSpec('min_height', 'Min Height:', 'float'),
    ),
    'shoulders': (
        FieldSpec('enabled', 'Enable Shoulder Detection', 'bool'),
        FieldSpec('sensitivity', 'Sensitivity:', 'int'),
        FieldSpec('window_length', 'Window length:', 'int'),
        FieldSpec('polyorder', 'Polynomial order:', 'int'),
        FieldSpec('apex_distance', 'Apex distance:', 'int'),
    ),
    'negative_peaks': (
        FieldSpec('enabled', 'Enable Negative Peak Detection', 'bool'),
        FieldSpec('min_prominence', 'Min Prominence:', 'float'),
    ),
    'rider': (
        FieldSpec('rider_integration', 'Mode:', 'choice', _RIDER_MODES),
        FieldSpec('skim_model', 'Skim model:', 'choice', _RIDER_MODELS),
        FieldSpec('skim_height_ratio', 'Height ratio:', 'float'),
        FieldSpec('skim_max_valley_frac', 'Max valley fraction:', 'float'),
        FieldSpec('skim_valley_ratio', 'Valley ratio:', 'float'),
        FieldSpec('skim_tail_bend', 'Tail bend:', 'float'),
    ),
}


class OverrideField(QWidget):
    """One optional setting: inherit (None) or override."""
    changed = Signal()

    def __init__(self, spec: FieldSpec, inherited_tag: str = INHERIT_TAG, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._tag = inherited_tag
        self._inherited = None
        self._value = None
        self._error = False
        self._last_text = ''  # text as last committed; unchanged text is never re-parsed
        self.edit = None
        self.combo = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if spec.kind in ('float', 'int'):
            self.edit = QLineEdit()
            self.edit.editingFinished.connect(self._on_edit_finished)
            self._style_placeholder()
            layout.addWidget(self.edit, 1)
        elif spec.kind in ('bool', 'choice'):
            self.combo = QComboBox()
            self.combo.addItem(self._tag, None)
            self.combo.setItemData(0, QBrush(_GREY), Qt.ForegroundRole)
            grey_italic = QFont(self.combo.font())
            grey_italic.setItalic(True)
            self.combo.setItemData(0, grey_italic, Qt.FontRole)
            if spec.kind == 'bool':
                self.combo.addItem('on', True)
                self.combo.addItem('off', False)
            else:
                for text, value in spec.choices:
                    self.combo.addItem(text, value)
            self.combo.currentIndexChanged.connect(self._on_combo_changed)
            layout.addWidget(self.combo, 1)
        else:
            raise ValueError(f"unknown field kind {spec.kind!r}")

        self.tag_label = QLabel(self._tag)  # shown while inheriting
        self.tag_label.setStyleSheet(_TAG_STYLE)
        layout.addWidget(self.tag_label)

        self.reset_button = QToolButton()  # shown while overridden; click reverts
        self.reset_button.setText(OVERRIDE_TAG)
        self.reset_button.setToolTip("Click to go back to the inherited value")
        self.reset_button.setAutoRaise(True)
        self.reset_button.setStyleSheet(_CHANGED_STYLE)
        self.reset_button.setHidden(True)
        self.reset_button.clicked.connect(lambda: self.set_value(None))
        layout.addWidget(self.reset_button)

    def _style_placeholder(self):
        pal = self.edit.palette()
        pal.setColor(QPalette.PlaceholderText, _GREY)
        self.edit.setPalette(pal)
        font = self.edit.font()
        font.setItalic(True)  # italic only while showing the inherited placeholder
        self.edit.setFont(font)

    def _sync_italic(self):
        font = self.edit.font()
        font.setItalic(not self.edit.text())
        self.edit.setFont(font)

    def _fmt(self, value):
        if value is None:
            return ''
        if self.spec.kind == 'float':
            return f"{value:.12g}"  # enough digits that a typed value round-trips
        return str(value)

    def _inherited_text(self):
        v = self._inherited
        if v is None:
            return ''
        if self.spec.kind == 'bool':
            return 'on' if v else 'off'
        if self.spec.kind == 'choice':
            return dict((val, text) for text, val in self.spec.choices).get(v, str(v))
        return self._fmt(v)

    def set_inherited(self, value) -> None:
        self._inherited = value
        if self.edit is not None:
            self.edit.setPlaceholderText(f"{value:g}" if self.spec.kind == 'float' and value is not None else self._fmt(value))
        else:
            self.combo.setItemText(0, f"{self._tag} ({self._inherited_text()})" if value is not None else self._tag)

    def _parse(self, text):
        text = text.strip()
        if not text:
            return None
        if self.spec.kind == 'int':
            return int(text)
        v = float(text)
        if not math.isfinite(v):
            raise ValueError(text)
        return v

    def _commit(self, value):
        """Store value and refresh the reset button; emit only on a real change."""
        overridden = value is not None or bool(self.edit and self.edit.text())
        self.reset_button.setHidden(not overridden)
        self.tag_label.setHidden(overridden)
        if value != self._value:
            self._value = value
            self.changed.emit()

    def _on_edit_finished(self):
        self._sync_italic()
        if self.edit.text() == self._last_text:
            return
        self._last_text = self.edit.text()
        try:
            value = self._parse(self.edit.text())
        except ValueError:
            self.set_error('Enter a number' + (' (whole)' if self.spec.kind == 'int' else ' or scientific notation (e.g. 1e3)'))
            self._commit(None)
            return
        self.set_error(None)
        self._commit(value)

    def _on_combo_changed(self, _index):
        self._commit(self.combo.currentData())

    def set_value(self, value) -> None:
        if self.edit is not None:
            self.edit.setText(self._fmt(value))
            self._last_text = self.edit.text()
            self.set_error(None)
            self._sync_italic()
        else:
            idx = self.combo.findData(value)
            if value is not None and idx < 0:
                raise ValueError(f"{value!r} is not a valid {self.spec.key}")
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(max(idx, 0))
            self.combo.blockSignals(False)
        self._commit(value)

    def value(self):
        return self._value

    def set_error(self, msg) -> None:
        w = self.edit if self.edit is not None else self.combo
        self._error = bool(msg)
        w.setStyleSheet(_ERROR_STYLE if msg else '')
        w.setToolTip(msg or '')

    def has_error(self) -> bool:
        return self._error


def _flatten(d, prefix=''):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


class OverrideGroup(QGroupBox):
    """Collapsible group of OverrideFields for one overrides section."""
    changed = Signal()

    def __init__(self, title: str, section: str, inherited: dict, values: dict, parent=None):
        super().__init__(parent)
        self._base_title = title
        self.section = section

        self.toggle_button = QToolButton()
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setStyleSheet('font-weight: bold;')
        self.toggle_button.clicked.connect(lambda: self.set_expanded(not self._expanded))
        self.count_label = QLabel()
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header = QHBoxLayout()
        header.addWidget(self.toggle_button)
        header.addStretch(1)
        header.addWidget(self.count_label)

        self._body = QWidget()
        form = QFormLayout(self._body)
        self.fields = {}
        for spec in SECTION_FIELDS[section]:
            f = OverrideField(spec)
            self.fields[spec.key] = f
            form.addRow(QLabel(spec.label), f)
        outer = QVBoxLayout(self)
        outer.addLayout(header)
        outer.addWidget(self._body)

        self.set_inherited(inherited)
        self._load(values)
        for f in self.fields.values():
            f.changed.connect(self._on_field_changed)
        self._update_count()
        self.set_expanded(bool(self.values()))

    def _load(self, values):
        for key, v in _flatten(values).items():
            if key not in self.fields:
                raise KeyError(f"{self.section} has no override field {key!r}")
            self.fields[key].set_value(v)

    def set_values(self, values: dict) -> None:
        """Replace all overrides (reload); re-evaluates the collapsed state."""
        for f in self.fields.values():
            f.blockSignals(True)
        for f in self.fields.values():
            f.set_value(None)
        self._load(values)
        for f in self.fields.values():
            f.blockSignals(False)
        self._update_count()
        self.set_expanded(bool(self.values()))

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)  # hides only; never disables
        self.toggle_button.setText(f"{_OPEN if expanded else _CLOSED} {self._base_title}")

    def _update_count(self):
        n = sum(1 for f in self.fields.values() if f.value() is not None)
        self.count_label.setText(INHERIT_TAG if n == 0 else ('1 change' if n == 1 else f"{n} changes"))

    def _on_field_changed(self):
        self._update_count()
        self.changed.emit()

    def has_errors(self) -> bool:
        return any(f.has_error() for f in self.fields.values())

    def values(self) -> dict:
        out = {}
        for key, f in self.fields.items():
            v = f.value()
            if v is None:
                continue
            head, _, tail = key.partition('.')
            if tail:
                out.setdefault(head, {})[tail] = v
            else:
                out[key] = v
        return out

    def set_inherited(self, inherited: dict) -> None:
        flat = _flatten(inherited)
        for key, f in self.fields.items():
            f.set_inherited(flat.get(key))
