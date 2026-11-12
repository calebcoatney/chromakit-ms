import copy
import datetime

from openpyxl import Workbook

from logic.paste_blocks import (
    DEFAULT_FORMAT_CONFIG,
    GC12_PASTE_FORMAT_CONFIG,
    channel_width,
    write_channel_block,
    write_timestamp_cell,
)


def _data(peaks):
    return {"signal": "Signal: X.D\\FID1A.ch", "sample_id": "X",
            "timestamp": "2026-05-21 14:36:00", "peaks": peaks}


_PEAK = {"peak_number": 1, "retention_time": 8.241, "integrator": "py", "width": 0.04,
         "area": 907185.0, "start_time": 8.111, "end_time": 8.4}


def test_block_layout_matches_paste_format():
    wb = Workbook()
    ws = wb.active
    nxt, skipped = write_channel_block(ws, 1, 1, _data([_PEAK]),
                                       copy.deepcopy(GC12_PASTE_FORMAT_CONFIG))
    assert [ws.cell(r, 1).value for r in (1, 2)] == ["Signal: X.D\\FID1A.ch", "X"]
    assert isinstance(ws.cell(3, 1).value, datetime.datetime)
    assert ws.cell(3, 1).number_format == "m/dd/yy h:mm"
    assert [ws.cell(4, c).value for c in range(1, 8)] == [
        "Peak #", "Ret Time", "Integrator", "Width", "Area", "Start Time", "End Time"]
    assert ws.cell(5, 2).value == 8.241 and ws.cell(5, 3).value == "py"
    assert (nxt, skipped) == (6, 0)


def test_channel_width_is_seven_for_paste_config():
    assert channel_width(GC12_PASTE_FORMAT_CONFIG) == 7


def test_paste_config_is_independent_copy():
    assert DEFAULT_FORMAT_CONFIG["timestamp_format"] == "as_stored"
    assert GC12_PASTE_FORMAT_CONFIG["header_fields"] is not DEFAULT_FORMAT_CONFIG["header_fields"]


def test_skip_flags_count_skipped_peaks():
    ws = Workbook().active
    unknown = dict(_PEAK, compound_id="Unknown (5.234)")
    nxt, skipped = write_channel_block(ws, 1, 1, _data([unknown]),
                                       copy.deepcopy(GC12_PASTE_FORMAT_CONFIG),
                                       skip_unidentified=True)
    assert skipped == 1
    assert nxt == 5  # three header rows + column headers, no peak row
    assert ws.cell(5, 1).value is None


def test_write_timestamp_cell_accepts_datetime():
    ws = Workbook().active
    dt = datetime.datetime(2026, 5, 21, 14, 36)
    cell = write_timestamp_cell(ws, 1, 1, dt, "m/dd/yy h:mm", None)
    assert cell.value == dt and cell.number_format == "m/dd/yy h:mm"
