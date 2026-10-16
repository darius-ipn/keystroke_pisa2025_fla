#!/usr/bin/env python3
"""
Create text_mega_lookup.tsv with ALL timepoint variants for every essay.

For each JSON keystroke file, reconstructs the essay text at 6 timepoints:
  full, 05min, 10min, 15min, 20min, 25min

Output:
  BEA Paper/text_mega_lookup.tsv
    Columns: filename, task, chat_used, source_timepoint, text

Newlines within essays are replaced with \\n literals so the TSV stays
single-line-per-record.
"""

import json
import re
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_BASE_DIR = Path("Originaldaten/allUser_corr")
FOLDERS = [
    "FLA25MS_advertisement",
    "FLA25MS_chat__advertisement",
    "FLA25MS_chat__teacher",
    "FLA25MS_teacher",
]

OUTPUT_FILE = Path("BEA Paper/text_mega_lookup.tsv")

# All timepoints to reconstruct for every file
TIMEPOINTS = ["05min", "10min", "15min", "20min", "25min", "full"]

# Map source_timepoint labels to cutoff in milliseconds (None = no cutoff)
TIMEPOINT_TO_CUTOFF_MS = {
    "full": None,
    "05min": 5 * 60 * 1000,
    "10min": 10 * 60 * 1000,
    "15min": 15 * 60 * 1000,
    "20min": 20 * 60 * 1000,
    "25min": 25 * 60 * 1000,
}


# ============================================================
# FUNCTIONS (consistent with the extraction scripts)
# ============================================================

def load_json_events(filepath: Path) -> list:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return [data]
        except json.JSONDecodeError:
            content = content.strip()
            if not content.startswith('['):
                content = '[' + content + ']'
            content = re.sub(r',\s*([}\]])', r'\1', content)
            return json.loads(content)


def truncate_events(events: list, cutoff_ms: int) -> list:
    """Return only events with timestamp_rel <= cutoff_ms.
    Events without timestamps are kept (metadata)."""
    truncated = []
    for e in events:
        ts = e.get('timestamp_rel')
        if ts is None:
            truncated.append(e)
        elif ts <= cutoff_ms:
            truncated.append(e)
    return truncated


def reconstruct_text(events: list, target_field: str = 'text') -> str:
    relevant_events = [
        e for e in events
        if (e.get('event') in ['KeyDown', 'TextPasted', 'TextCut']
            or e.get('event', '').startswith('cr'))
        and (e.get('target') == target_field or e.get('startElem') == target_field)
        and not e.get('IGNORE')
    ]

    text = ''
    for event in relevant_events:
        event_type = event.get('event', '')
        cursor_start = event.get('cursorStart', 0)
        cursor_end = event.get('cursorEnd', cursor_start)

        cursor_start = max(0, min(cursor_start, len(text)))
        cursor_end = max(0, min(cursor_end, len(text)))
        if cursor_start > cursor_end:
            cursor_start, cursor_end = cursor_end, cursor_start

        pre = text[:cursor_start]
        post = text[cursor_end:]

        if event_type == 'TextCut':
            insert = ''
        elif event_type == 'KeyDown':
            key = event.get('key', '')
            if key == 'Backspace':
                if cursor_start == cursor_end and cursor_start > 0:
                    pre = text[:cursor_start - 1]
                insert = ''
            elif key == 'Delete':
                if cursor_start == cursor_end and cursor_end < len(text):
                    post = text[cursor_end + 1:]
                insert = ''
            elif key == 'Enter':
                insert = '\n'
            elif key == 'Tab':
                insert = '\t'
            elif len(key) == 1:
                insert = key
            else:
                continue
        else:
            insert = event.get('text', '')

        text = pre + insert + post

    return text


def parse_folder_info(folder_name: str) -> tuple:
    chat_used = 'chat__' in folder_name or '_chat_' in folder_name
    parts = folder_name.replace('FLA25MS_', '').replace('chat__', '').split('_')
    task = parts[-1] if parts else 'unknown'
    return task, chat_used


def escape_for_tsv(text: str) -> str:
    """Escape text so it fits in a single TSV line."""
    return (
        text.replace('\\', '\\\\')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
            .replace('\t', '\\t')
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Mega Text Lookup – All timepoints for every essay")
    print("=" * 60)

    # --- Step 1: Load ALL JSON files and their folder metadata ---
    print("\nStep 1: Loading all keystroke JSON files...")
    file_records = []  # list of (filename, task, chat_used, events)

    for folder in FOLDERS:
        folder_path = DEFAULT_BASE_DIR / folder
        if not folder_path.exists():
            print(f"  Warning: {folder_path} not found, skipping")
            continue

        task, chat_used = parse_folder_info(folder)
        json_files = sorted(folder_path.glob('*.json'))
        loaded = 0
        for json_file in json_files:
            try:
                events = load_json_events(json_file)
                file_records.append((json_file.name, task, chat_used, events))
                loaded += 1
            except Exception as e:
                print(f"    Error loading {json_file.name}: {e}")
        print(f"  {folder}: loaded {loaded} files")

    print(f"  Total files loaded: {len(file_records)}")

    if not file_records:
        print("ERROR: No JSON files found. Exiting.")
        return

    # --- Step 2: Reconstruct text at every timepoint for every file ---
    print(f"\nStep 2: Reconstructing texts at {len(TIMEPOINTS)} timepoints each...")
    print(f"  Expected output rows: {len(file_records)} files × {len(TIMEPOINTS)} timepoints = {len(file_records) * len(TIMEPOINTS)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    empty = 0
    empty_by_tp = {tp: 0 for tp in TIMEPOINTS}

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write("filename\ttask\tchat_used\tsource_timepoint\ttext\n")

        for fname, task, chat_used, events in file_records:
            for timepoint in TIMEPOINTS:
                cutoff_ms = TIMEPOINT_TO_CUTOFF_MS[timepoint]

                if cutoff_ms is not None:
                    events_truncated = truncate_events(events, cutoff_ms)
                else:
                    events_truncated = events

                text = reconstruct_text(events_truncated, 'text')
                text_escaped = escape_for_tsv(text)

                out.write(f"{fname}\t{task}\t{chat_used}\t{timepoint}\t{text_escaped}\n")
                total += 1

                if len(text.strip()) == 0:
                    empty += 1
                    empty_by_tp[timepoint] += 1

    print(f"\nDone: {total} texts reconstructed ({empty} empty rows total)")
    print(f"Saved to: {OUTPUT_FILE}")

    print(f"\nEmpty texts by timepoint (out of {len(file_records)} files):")
    for tp in TIMEPOINTS:
        print(f"  {tp}: {empty_by_tp[tp]} empty")


if __name__ == '__main__':
    main()