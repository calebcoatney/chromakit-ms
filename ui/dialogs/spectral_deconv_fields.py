"""One definition per spectral-deconvolution setting, shared by MS Options and the Inspector.

Both dialogs build these widgets with ``build_setting_boxes`` so a setting's
label, range, step, decimals, tooltip and special-value text cannot drift apart
between the two surfaces. Keys are the field names of ``WindowGroupingParams``
and ``CorrelationDeconvParams``. No QSettings here: each dialog loads and saves.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox, QWidget,
)


@dataclass(frozen=True)
class FieldSpec:
    label: str                     # form label, without the colon
    tooltip: str                   # plain text; widgets get it wrapped by tooltip_html
    kind: str = "float"            # "float" (QDoubleSpinBox), "int" (QSpinBox), "mz_list" (QLineEdit)
    minimum: float = 0.0
    maximum: float = 0.0
    step: float = 1.0
    decimals: int = 0
    special_value_text: str = ""   # shown instead of the minimum (0 = Auto / Unlimited)
    placeholder: str = ""


FIELDS: dict[str, FieldSpec] = {
    # ── Peak Window Grouping (WindowGroupingParams) ──
    "gap_tolerance": FieldSpec(
        "Gap tolerance (min)",
        "FID peaks closer than this (end of one to start of the next) share a window. "
        "Auto = 0.5 × the median FID peak width. Raise it if peaks that nearly touch land "
        "in separate windows; lower it if unrelated peaks are pooled.",
        minimum=0.0, maximum=5.0, step=0.05, decimals=3, special_value_text="Auto"),
    "padding_fraction": FieldSpec(
        "Padding fraction",
        "Each window extends this fraction of its FID peaks' total width past the first and "
        "last peak. Raise it if a compound's ion peaks are cut off at the window edge (an ion "
        "peak touching the edge is rejected); lower it if windows reach into neighbouring "
        "compounds.",
        minimum=0.0, maximum=2.0, step=0.1, decimals=2),
    "rt_match_tolerance": FieldSpec(
        "RT match tolerance (min)",
        "How far (min) a spectrum's RT (its lead ion's apex) may sit from an FID peak and "
        "still be matched to it. Raise it if spectra go unmatched; lower it if a spectrum is "
        "matched to the wrong neighbouring peak.",
        minimum=0.001, maximum=1.0, step=0.005, decimals=3),
    "max_window_peaks": FieldSpec(
        "Max peaks per window",
        "Split a window at its largest RT gap when it holds more than this many FID peaks "
        "(Unlimited = never). Raise it if a crowded region is split between peaks that belong "
        "together; lower it if a crowded region becomes one very wide window.",
        kind="int", minimum=0, maximum=100, special_value_text="Unlimited"),
    # ── Ion Peak Detection (CorrelationDeconvParams) ──
    "noise_percentile": FieldSpec(
        "Noise percentile",
        "Points below this quantile of an ion's intensities over the whole run count as noise "
        "(0.85 = 85th percentile). Raise it if noisy ions create junk peaks; lower it if weak "
        "real ions vanish.",
        minimum=0.0, maximum=1.0, step=0.05, decimals=3),
    "search_range": FieldSpec(
        "Search range (min)",
        "A dip splits two ion peaks only if it is the lowest point within ± this many minutes. "
        "Set near one peak width. Raise it if one compound splits into several peaks; lower it "
        "if close isomers merge.",
        minimum=0.001, maximum=1.0, step=0.005, decimals=3),
    "min_peak_height": FieldSpec(
        "Min peak height (counts)",
        "Ignore ion peaks lower than this (counts). Raise it if noise peaks get in; lower it "
        "if trace compounds are missed.",
        minimum=0.0, maximum=1e9, step=100.0, decimals=0),
    "min_top_edge_ratio": FieldSpec(
        "Min peak top/edge ratio",
        "A peak's apex must be at least this many times higher than both of its edges. Raise "
        "it if bumps on a high baseline count as peaks; lower it if real shouldered peaks are "
        "dropped.",
        minimum=1.0, maximum=100.0, step=0.1, decimals=2),
    "min_points": FieldSpec(
        "Min points per peak",
        "Ignore ion peaks with fewer scans than this. Raise it if spikes appear; lower it if "
        "very fast peaks are dropped.",
        kind="int", minimum=1, maximum=100),
    # ── Ion Grouping (CorrelationDeconvParams) ──
    "rt_tolerance": FieldSpec(
        "RT tolerance (min)",
        "An ion peak can join a group only if its apex is within this many minutes of the "
        "group's strongest ion. Raise it if one compound's ions land in separate groups; lower "
        "it if neighbouring compounds merge.",
        minimum=0.001, maximum=1.0, step=0.005, decimals=3),
    "min_r": FieldSpec(
        "Min shape correlation (r)",
        "An ion peak joins a group if its shape correlates with the group's strongest ion at "
        "least this well (Pearson r). An ion near a stronger group that misses this is "
        "discarded; it does not start its own group. Raise it if spectra pick up co-eluting "
        "ions; lower it if ions of noisy compounds are rejected.",
        minimum=0.0, maximum=1.0, step=0.05, decimals=2),
    "min_ions": FieldSpec(
        "Min ions per spectrum",
        "Drop groups with fewer ions than this; their FID peaks use the fallback spectrum. "
        "Raise it if noise forms spectra; lower it if small molecules with few ions disappear.",
        kind="int", minimum=1, maximum=100),
    "excluded_mz": FieldSpec(
        "Excluded m/z",
        "m/z values (±0.5), e.g. column-bleed ions, that can't lead a group or set its RT "
        "(unless every ion in the group is excluded). They stay in the spectrum. Separate "
        "values with commas or spaces.",
        kind="mz_list", placeholder="e.g. 73, 147, 221"),
}

# (box title, setting keys) in display order
SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Peak Window Grouping",
     ("gap_tolerance", "padding_fraction", "rt_match_tolerance", "max_window_peaks")),
    ("Ion Peak Detection",
     ("noise_percentile", "search_range", "min_peak_height", "min_top_edge_ratio", "min_points")),
    ("Ion Grouping",
     ("rt_tolerance", "min_r", "min_ions", "excluded_mz")),
)

INVALID_MZ_TITLE = "Excluded m/z not valid"


def invalid_mz_message(error) -> str:
    """What both dialogs say when the Excluded m/z field doesn't parse."""
    return ("Excluded m/z must be numbers separated by commas or spaces, "
            f"e.g. 73, 147, 207 ({error}).")


def tooltip_html(text: str) -> str:
    """Rich-text tooltip: Qt wraps it, where a plain-text tooltip is one long line."""
    return f"<p>{html.escape(text, quote=False)}</p>"


def make_field_widget(spec: FieldSpec) -> QWidget:
    if spec.kind == "mz_list":
        w = QLineEdit()
        w.setPlaceholderText(spec.placeholder)
    elif spec.kind == "int":
        w = QSpinBox()
        w.setRange(int(spec.minimum), int(spec.maximum))
        w.setSingleStep(int(spec.step))
    else:
        w = QDoubleSpinBox()
        w.setDecimals(spec.decimals)
        w.setRange(spec.minimum, spec.maximum)
        w.setSingleStep(spec.step)
    if spec.special_value_text:
        w.setSpecialValueText(spec.special_value_text)
    w.setToolTip(tooltip_html(spec.tooltip))
    return w


def build_setting_boxes() -> tuple[list[QGroupBox], dict[str, QWidget]]:
    """One QGroupBox (form layout) per section, and every setting's widget by key."""
    boxes: list[QGroupBox] = []
    widgets: dict[str, QWidget] = {}
    for title, keys in SECTIONS:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for key in keys:
            spec = FIELDS[key]
            widgets[key] = make_field_widget(spec)
            form.addRow(spec.label + ":", widgets[key])
        boxes.append(box)
    return boxes, widgets
