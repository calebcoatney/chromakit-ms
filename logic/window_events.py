"""Typed per-window events: registry and dispatch.

Shipped empty in phase 1; later features (tail skim, user-placed split)
register a pydantic model in EVENT_TYPES and a handler in EVENT_HANDLERS.
"""
from typing import Callable

from pydantic import BaseModel

EVENT_TYPES: dict[str, type[BaseModel]] = {}
# handler(event, window, window_peaks, processed_data) -> new window peaks
EVENT_HANDLERS: dict[str, Callable[[BaseModel, dict, list, dict], list]] = {}


def validate_event(event: dict) -> dict:
    """Validate one event dict against its registered model."""
    t = event.get("type")
    if t is None:
        raise ValueError("Window event is missing its 'type' field.")
    model = EVENT_TYPES.get(t)
    if model is None:
        raise ValueError(
            f"Event type '{t}' is not supported by this ChromaKit version; "
            "the method was probably saved by a newer version."
        )
    return model(**event).model_dump()


def apply_window_events(peaks_list: list, processed_data: dict) -> list:
    """Run each window's events, in order, on the peaks carrying its id.

    shortcut: handlers must return the same peaks (same count and order);
    Integrator keeps parallel lists indexed by peak, so a handler that adds
    or drops peaks raises here. Upgrade when the first event needs to split
    or remove peaks (rebuild the parallel lists in Integrator.integrate).
    """
    windows = processed_data.get("peak_windows") or []
    for window in windows:
        for raw in window.get("events", []):
            t = raw.get("type")
            handler = EVENT_HANDLERS.get(t)
            if handler is None:
                raise ValueError(f"No handler registered for window event type '{t}'.")
            event = EVENT_TYPES[t](**raw)
            members = [i for i, p in enumerate(peaks_list)
                       if getattr(p, "window_id", None) == window["id"]]
            subset = [peaks_list[i] for i in members]
            result = handler(event, window, subset, processed_data)
            if len(result) != len(subset):
                raise ValueError(
                    f"Handler for window event '{t}' changed the peak count "
                    f"({len(subset)} -> {len(result)}); not supported yet."
                )
            peaks_list = list(peaks_list)
            for i, p in zip(members, result):
                peaks_list[i] = p
    return peaks_list
