from logic.automation_worker import AutomationWorker


def test_format_timing_table_gcms():
    timings = [
        {'file': 'Sample_001.C', 'load': 0.31, 'process': 0.18, 'ms_search': 1.20, 'save': 0.02},
        {'file': 'Sample_002.C', 'load': 0.29, 'process': 0.17, 'ms_search': 1.18, 'save': 0.02},
    ]
    table = AutomationWorker._format_timing_table(timings)
    assert 'Batch Timing Report' in table
    assert 'Sample_001.C' in table
    assert 'Sample_002.C' in table
    assert 'Average' in table


def test_format_timing_table_mixed():
    """Non-GC-MS files show — in MS Search column."""
    timings = [
        {'file': 'Sample_001.C', 'load': 0.31, 'process': 0.18, 'ms_search': 1.20, 'save': 0.02},
        {'file': 'ReactIR_001.C', 'load': 0.20, 'process': 0.10, 'ms_search': None, 'save': 0.02},
    ]
    table = AutomationWorker._format_timing_table(timings)
    assert '—' in table
    assert 'Average' in table


def test_format_timing_table_empty():
    table = AutomationWorker._format_timing_table([])
    assert 'no data' in table
