#!/usr/bin/env python3
"""
Create text_hybrid_lookup.tsv for the hybrid training sets.

Reads the hybrid CSVs (which contain a '_source_timepoint' column indicating
whether each row is from 'full', '05min', '10min', etc.) and reconstructs the
essay text at the corresponding time cutoff.

Output:
  BEA Paper/text_hybrid_lookup.tsv
    Columns: filename, task, chat_used, source_timepoint, text

The text is reconstructed by replaying keystroke events up to the relevant
time cutoff (or all events for 'full'). Newlines within essays are replaced
with \\n literals so the TSV stays single-line-per-record.
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

HYBRID_DIR = Path("BEA Paper/splits/hybrid")
HYBRID_FILES = ["wo_chat_train.csv", "withchat_train.csv"]
OUTPUT_FILE = Path("BEA Paper/text_hybrid_lookup.tsv")

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
    print("Hybrid Text Lookup – Reconstruct texts at time cutoffs")
    print("=" * 60)

    # --- Step 1: Load hybrid CSVs to know which (filename, timepoint) pairs we need ---
    print("\nStep 1: Loading hybrid training CSVs...")
    hybrid_rows = []
    for hf in HYBRID_FILES:
        path = HYBRID_DIR / hf
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        if '_source_timepoint' not in df.columns:
            print(f"  WARNING: {path} has no '_source_timepoint' column, skipping.")
            continue
        print(f"  {hf}: {len(df)} rows")
        hybrid_rows.append(df[['filename', 'task', 'chat_used', '_source_timepoint']])

    if not hybrid_rows:
        print("ERROR: No hybrid data found. Exiting.")
        return

    df_hybrid = pd.concat(hybrid_rows, ignore_index=True)
    df_hybrid = df_hybrid.rename(columns={'_source_timepoint': 'source_timepoint'})

    # Deduplicate: same filename can appear twice (full + intermediate)
    # but each (filename, source_timepoint) pair should be unique
    df_hybrid = df_hybrid.drop_duplicates(subset=['filename', 'source_timepoint'])
    print(f"\n  Unique (filename, timepoint) pairs to reconstruct: {len(df_hybrid)}")

    # Summarize what we need
    tp_counts = df_hybrid['source_timepoint'].value_counts()
    for tp, count in sorted(tp_counts.items()):
        print(f"    {tp}: {count}")

    # --- Step 2: Build a filename -> events lookup from the JSON files ---
    print("\nStep 2: Loading keystroke JSON files...")
    needed_filenames = set(df_hybrid['filename'].unique())
    events_lookup = {}

    for folder in FOLDERS:
        folder_path = DEFAULT_BASE_DIR / folder
        if not folder_path.exists():
            print(f"  Warning: {folder_path} not found, skipping")
            continue

        json_files = sorted(folder_path.glob('*.json'))
        loaded = 0
        for json_file in json_files:
            if json_file.name in needed_filenames:
                try:
                    events_lookup[json_file.name] = load_json_events(json_file)
                    loaded += 1
                except Exception as e:
                    print(f"    Error loading {json_file.name}: {e}")
        print(f"  {folder}: loaded {loaded} files")

    print(f"  Total JSON files loaded: {len(events_lookup)}")
    missing = needed_filenames - set(events_lookup.keys())
    if missing:
        print(f"  WARNING: {len(missing)} filenames from hybrid CSVs not found in JSON folders")

    # --- Step 3: Reconstruct text for each (filename, timepoint) pair ---
    print("\nStep 3: Reconstructing texts...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    empty = 0
    not_found = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write("filename\ttask\tchat_used\tsource_timepoint\ttext\n")

        for _, row in df_hybrid.iterrows():
            fname = row['filename']
            task = row['task']
            chat_used = row['chat_used']
            timepoint = row['source_timepoint']

            if fname not in events_lookup:
                not_found += 1
                continue

            events = events_lookup[fname]

            # Apply time cutoff if not full
            cutoff_ms = TIMEPOINT_TO_CUTOFF_MS.get(timepoint)
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

    print(f"\nDone: {total} texts reconstructed ({empty} empty, {not_found} not found)")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()