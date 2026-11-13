"""Inherit-or-override form widgets for per-segment settings.

An OverrideField edits one optional setting: blank/first combo item means
"inherit the whole-run value" (shown greyed), anything else is an override.
OverrideGroup stacks the fields of one *Overrides model.
"""
import math
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QToolButton, QVBoxLayout, QWidget,
)

INHERIT_TAG = 'same as whole run'
OVERRIDE_TAG = 'changed ↺'
_ERROR_STYLE = 'border: 1px solid #c0392b;'
_GREY = QColor('#888888')


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

        self.reset_button = QToolButton()
        self.reset_button.setText('↺')
        self.reset_button.setToolTip(f"{OVERRIDE_TAG}: click to go back to the inherited value")
        self.reset_button.setAutoRaise(True)
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
            return f"{value:g}"
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
            self.edit.setPlaceholderText(self._fmt(value))
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
        self.reset_button.setHidden(value is None and not (self.edit and self.edit.text()))
        if value != self._value:
            self._value = value
            self.changed.emit()

    def _on_edit_finished(self):
        self._sync_italic()
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
        w.setStyleSheet(_ERROR_STYLE if msg else '')
        w.setToolTip(msg or '')


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
        self.setCheckable(True)  # the check box is the expand/collapse toggle
        body = QWidget()
        form = QFormLayout(body)
        self.fields = {}
        for spec in SECTION_FIELDS[section]:
            f = OverrideField(spec)
            self.fields[spec.key] = f
            form.addRow(QLabel(spec.label), f)
        outer = QVBoxLayout(self)
        outer.addWidget(body)
        self._body = body
        self.toggled.connect(body.setVisible)

        self.set_inherited(inherited)
        for key, v in _flatten(values).items():
            if key not in self.fields:
                raise KeyError(f"{section} has no override field {key!r}")
            self.fields[key].set_value(v)
        for f in self.fields.values():
            f.changed.connect(self._on_field_changed)
        self._update_title()
        # shortcut: collapsed whenever nothing is set at construction, re-collapse is manual
        self.setChecked(bool(self.values()))

    def _update_title(self):
        n = sum(1 for f in self.fields.values() if f.value() is not None)
        suffix = INHERIT_TAG if n == 0 else ('1 change' if n == 1 else f"{n} changes")
        self.setTitle(f"{self._base_title} — {suffix}")

    def _on_field_changed(self):
        self._update_title()
        self.changed.emit()

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
