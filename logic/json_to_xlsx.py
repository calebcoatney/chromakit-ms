# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:28:40 2024

@author: ccoatney
"""

import os
import json
import datetime
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# Timestamp patterns used by Agilent instruments and common ISO formats
_TS_PARSE_FMTS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%d %b %y  %I:%M %p",
    "%d %b %y %I:%M %p",
    "%d %b %Y %I:%M %p",
    # Agilent ChemStation writes "30-Jul-26, 16:31:10" (comma between date
    # and time). The comma-bearing variants must come with the plain ones.
    "%d-%b-%y, %H:%M:%S",
    "%d-%b-%Y, %H:%M:%S",
    "%d-%b-%y, %H:%M",
    "%d-%b-%Y, %H:%M",
    "%d-%b-%y %H:%M:%S",
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%y %H:%M",
    "%d-%b-%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%d %b %y  %H:%M",
    "%d %b %y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d",
]


def _parse_timestamp(ts_string: str):
    """Try to parse *ts_string* into a datetime object."""
    if not ts_string:
        return None
    ts_string = ts_string.strip()
    
    # Try ISO first
    try:
        return datetime.datetime.fromisoformat(ts_string)
    except (ValueError, AttributeError):
        pass

    for pattern in _TS_PARSE_FMTS:
        try:
            return datetime.datetime.strptime(ts_string, pattern)
        except ValueError:
            continue
    return None


def _build_rf_lookup(json_data):
    """Return {compound_name: response_factor} from a result JSON.

    Returns an empty dict when the method did not use an RF table (no
    ``quantitation`` block, a different ``strategy``, or an empty table), which
    is the signal to omit the rf / mol% / normalized columns for that run.
    """
    quant = json_data.get("quantitation") or {}
    if not isinstance(quant, dict):
        return {}
    table = quant.get("rf_table")
    lookup = {}
    if isinstance(table, dict):
        items = table.items()
    elif isinstance(table, (list, tuple)):
        items = []
        for entry in table:
            if isinstance(entry, dict):
                items.append((entry.get("compound"),
                              entry.get("response_factor")))
    else:
        return {}

    for compound, rf in items:
        if not compound:
            continue
        try:
            rf = float(rf)
        except (TypeError, ValueError):
            continue
        if rf == 0.0:
            continue  # would produce #DIV/0! in the mol% formula
        lookup[str(compound)] = rf
    return lookup


def process_json_to_excel(directory, output_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "Chromatograms"

    starting_row = 1  # Start row for writing each file

    # Collect all (root, json_files) pairs, then sort chronologically
    dir_file_pairs = []
    for root, _, files in os.walk(directory):
        json_files = [f for f in files if f.endswith('.json')]
        if json_files:
            dir_file_pairs.append((root, json_files))

    def _dir_sort_key(pair):
        root, jfiles = pair
        try:
            with open(os.path.join(root, jfiles[0]), 'r', encoding='utf-8') as fh:
                ts = json.load(fh).get('timestamp', '')
            dt = _parse_timestamp(str(ts)) if ts else None
            if dt is not None:
                return dt
        except Exception:
            pass
        return datetime.datetime.max

    dir_file_pairs.sort(key=_dir_sort_key)

    for root, json_files in dir_file_pairs:
        for file in json_files:
            json_path = os.path.join(root, file)
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                # Write header data
                headers = [
                    ("Sample ID:", json_data.get("sample_id", "")),
                    ("Timestamp:", json_data.get("timestamp", "")),
                    ("Method:", json_data.get("method", "")),
                    ("Detector:", json_data.get("detector", "")),
                ]

                current_row = starting_row  # Start at the current row
                col_offset = 1  # Always start at column 1

                for header, value in headers:
                    cell = ws.cell(row=current_row, column=col_offset, value=header)
                    cell.font = Font(bold=True)
                    
                    val_cell = ws.cell(row=current_row, column=col_offset + 1, value=value)
                    if header == "Timestamp:" and value:
                        dt = _parse_timestamp(str(value))
                        if dt:
                            val_cell.value = dt
                            val_cell.number_format = "m/d/yy h:mm" # Default Excel date/time format
                    
                    current_row += 1

                # Optional signal and notebook fields
                ws.cell(row=current_row, column=col_offset, value=json_data.get("signal", ""))
                current_row += 1
                ws.cell(row=current_row, column=col_offset, value=json_data.get("notebook", ""))
                current_row += 2

                # RF table drives the optional quantitation columns. When the
                # method was not an rf_table method there is nothing to look up,
                # so those columns are omitted entirely for this run.
                rf_lookup = _build_rf_lookup(json_data)
                peaks = json_data.get("peaks", []) or []
                has_quant = bool(rf_lookup)

                # Column letters (col_offset is 1, but stay generic)
                area_col = col_offset + 5
                rf_col = col_offset + 8
                mol_col = col_offset + 9
                norm_col = col_offset + 10
                area_letter = get_column_letter(area_col)
                rf_letter = get_column_letter(rf_col)
                mol_letter = get_column_letter(mol_col)
                norm_letter = get_column_letter(norm_col)

                # Write peaks table header
                peaks_headers = ["Compound ID", "Peak #", "Ret Time", "Integrator",
                                 "Width", "Area", "Start Time", "End Time"]
                if has_quant:
                    peaks_headers += ["rf", "mol%", "normalized mol%"]
                for col, header in enumerate(peaks_headers, start=col_offset):
                    cell = ws.cell(row=current_row, column=col, value=header)
                    cell.font = Font(bold=True)

                current_row += 1

                # Write peaks data
                first_peak_row = current_row
                for peak in peaks:
                    ws.cell(row=current_row, column=col_offset,
                            value=peak.get("compound_id") or "")
                    ws.cell(row=current_row, column=col_offset + 1,
                            value=int(peak.get("peak_number", 0)))
                    ws.cell(row=current_row, column=col_offset + 2,
                            value=float(peak.get("retention_time", 0.0)))
                    ws.cell(row=current_row, column=col_offset + 3, value=peak.get("integrator", ""))
                    ws.cell(row=current_row, column=col_offset + 4, value=float(peak.get("width", 0.0)))
                    ws.cell(row=current_row, column=col_offset + 5, value=float(peak.get("area", 0.0)))
                    ws.cell(row=current_row, column=col_offset + 6, value=float(peak.get("start_time", 0.0)))
                    ws.cell(row=current_row, column=col_offset + 7, value=float(peak.get("end_time", 0.0)))
                    if has_quant:
                        rf = rf_lookup.get(str(peak.get("compound_id") or ""))
                        if rf is not None:
                            ws.cell(row=current_row, column=rf_col, value=rf)
                        # mol% / normalized formulas are only meaningful when an
                        # RF exists; otherwise the cells stay blank so no
                        # #DIV/0! ever reaches the sheet.
                    current_row += 1

                last_peak_row = current_row - 1

                if has_quant and peaks:
                    mol_range = f"{mol_letter}{first_peak_row}:{mol_letter}{last_peak_row}"
                    norm_range = f"{norm_letter}{first_peak_row}:{norm_letter}{last_peak_row}"
                    for row in range(first_peak_row, last_peak_row + 1):
                        if ws.cell(row=row, column=rf_col).value is None:
                            continue
                        ws.cell(row=row, column=mol_col,
                                value=f"={area_letter}{row}/{rf_letter}{row}")
                        ws.cell(row=row, column=norm_col,
                                value=f"={mol_letter}{row}/SUM({mol_range})")

                    # Totals row
                    total_mol = ws.cell(row=current_row, column=mol_col,
                                        value=f"=SUM({mol_range})")
                    total_mol.font = Font(bold=True)
                    total_norm = ws.cell(row=current_row, column=norm_col,
                                         value=f"=SUM({norm_range})")
                    total_norm.font = Font(bold=True)
                    current_row += 1

                # After writing this file, update starting_row for the next file
                starting_row = current_row + 2  # Add a gap between files

            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    wb.save(output_file)
    print(f"Data successfully written to {output_file}")


def _main():
    """Interactive entry point: prompt for a folder and an output workbook."""
    import tkinter as tk
    from tkinter import filedialog

    # Initialize Tkinter root window (it won't show on the screen)
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Ask the user to select the folder containing the JSON files
    directory_to_search = filedialog.askdirectory(title="Select Folder to Search for .json Files")
    if not directory_to_search:
        print("No directory selected. Exiting.")
        return

    # Ask the user to select the output Excel file location
    output_excel_file = filedialog.asksaveasfilename(
        title="Save Excel File", defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if not output_excel_file:
        print("No output file selected. Exiting.")
        return

    # Process the selected directory and save the results
    process_json_to_excel(directory_to_search, output_excel_file)


# Importing this module must never open a GUI — the dialogs only run as a script.
if __name__ == "__main__":
    _main()
