#!/usr/bin/env python3
"""
Extract keystroke measures from JSON keystroke log files.
Filters outliers, then outputs two CSVs: raw statistics and z-score normalized values.
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats

# Configuration - can be overridden via command line
DEFAULT_BASE_DIR = Path("Originaldaten/allUser_corr")
FOLDERS = [
    "FLA25MS_advertisement",
    "FLA25MS_chat__advertisement",
    "FLA25MS_chat__teacher",
    "FLA25MS_teacher"
]

# Thresholds
BREAK_THRESHOLD_MS = 2000  # Pause threshold for breaks (ms)
MAX_IDLE_TIME_MS = 60000   # Max time to attribute to a target area (ms)

# Outlier filtering
MIN_CHARS = 10      # Remove essays shorter than this (empty/trivial)
MAX_CHARS = 10000   # Remove essays longer than this (extreme copy-pasting)

# Columns to exclude from z-score normalization
EXCLUDE_COLS = ['filename', 'task', 'chat_used', 'holistic_score']


def load_json_events(filepath: Path) -> list:
    """Load events from a JSON file."""
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


def get_events_with_timestamps(events: list) -> list:
    """Filter events that have timestamps and sort them."""
    timestamped = [e for e in events if 'timestamp_rel' in e]
    return sorted(timestamped, key=lambda x: x['timestamp_rel'])


def extract_measures(events: list) -> dict:
    """Extract all keystroke measures from a list of events."""

    timestamped_events = get_events_with_timestamps(events)

    if not timestamped_events:
        return None

    measures = {}

    # === TEMPORAL MEASURES ===
    first_ts = timestamped_events[0]['timestamp_rel']
    last_ts = timestamped_events[-1]['timestamp_rel']
    measures['total_writing_time'] = (last_ts - first_ts) / 1000  # seconds

    # Initial pause: time from 0 to first keystroke
    keydown_events = [e for e in timestamped_events if e.get('event') == 'KeyDown']
    if keydown_events:
        measures['initial_pause'] = keydown_events[0]['timestamp_rel']  # ms
    else:
        measures['initial_pause'] = first_ts

    # === PAUSES & BREAKS ===
    keydown_timestamps = [e['timestamp_rel'] for e in keydown_events]

    breaks = []
    if len(keydown_timestamps) > 1:
        for i in range(1, len(keydown_timestamps)):
            interval = keydown_timestamps[i] - keydown_timestamps[i-1]
            if interval >= BREAK_THRESHOLD_MS:
                breaks.append(interval)

    measures['break_count'] = len(breaks)
    measures['break_total_time'] = sum(breaks) / 1000 if breaks else 0  # seconds
    measures['break_mean_duration'] = np.mean(breaks) if breaks else 0  # ms
    measures['break_ratio'] = (measures['break_total_time'] / measures['total_writing_time']
                               if measures['total_writing_time'] > 0 else 0)

    # === BURSTS ===
    deletion_keys = {'Backspace', 'Delete'}

    bursts = []
    current_burst_chars = 0
    current_burst_start = None

    for i, event in enumerate(keydown_events):
        if event.get('IGNORE'):
            continue

        key = event.get('key', '')
        ts = event['timestamp_rel']

        is_break = False
        if i > 0:
            prev_ts = keydown_events[i-1]['timestamp_rel']
            if ts - prev_ts >= BREAK_THRESHOLD_MS:
                is_break = True

        is_revision = key in deletion_keys

        if is_break or is_revision:
            if current_burst_chars > 0 and current_burst_start is not None:
                burst_duration = keydown_events[i-1]['timestamp_rel'] - current_burst_start
                bursts.append({'chars': current_burst_chars, 'duration': burst_duration})
            current_burst_chars = 0
            current_burst_start = ts if not is_revision else None
        else:
            if current_burst_start is None:
                current_burst_start = ts
            if len(key) == 1:
                current_burst_chars += 1

    if current_burst_chars > 0 and current_burst_start is not None:
        burst_duration = keydown_timestamps[-1] - current_burst_start if keydown_timestamps else 0
        bursts.append({'chars': current_burst_chars, 'duration': burst_duration})

    measures['burst_count'] = len(bursts)
    measures['burst_mean_length_char'] = np.mean([b['chars'] for b in bursts]) if bursts else 0
    measures['burst_mean_duration'] = np.mean([b['duration'] for b in bursts]) if bursts else 0  # ms

    # === DELETIONS ===
    deletion_events = [e for e in keydown_events if e.get('key') in deletion_keys and not e.get('IGNORE')]
    measures['deletion_count'] = len(deletion_events)
    measures['deletion_ratio'] = (len(deletion_events) / len(keydown_events)
                                   if keydown_events else 0)

    chars_deleted = 0
    for e in events:
        if e.get('event') in ['TextCut'] and not e.get('IGNORE'):
            chars_deleted += len(e.get('text', ''))
        elif e.get('event') == 'KeyDown' and e.get('key') in deletion_keys and not e.get('IGNORE'):
            cursor_start = e.get('cursorStart', 0)
            cursor_end = e.get('cursorEnd', 0)
            if cursor_start != cursor_end:
                chars_deleted += abs(cursor_end - cursor_start)
            else:
                chars_deleted += 1
    measures['deletion_char_count'] = chars_deleted

    # === PRODUCTION & EFFICIENCY ===
    measures['total_keystrokes'] = len(keydown_events)

    final_text = reconstruct_text(events, 'text')
    final_chat_prompt = reconstruct_text(events, 'chat_prompt')

    measures['final_text_length_char'] = len(final_text)
    measures['final_text_length_word'] = len(final_text.split()) if final_text.strip() else 0

    active_time_seconds = measures['total_writing_time'] - measures['break_total_time']
    measures['chars_per_minute'] = ((measures['final_text_length_char'] / active_time_seconds * 60)
                                     if active_time_seconds > 0 else 0)

    total_chars_typed = sum(1 for e in keydown_events if len(e.get('key', '')) == 1 and not e.get('IGNORE'))
    measures['process_product_ratio'] = (total_chars_typed / measures['final_text_length_char']
                                          if measures['final_text_length_char'] > 0 else 0)

    # === NAVIGATION ===
    arrow_keys = {'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown'}
    navigation_keystrokes = len([e for e in keydown_events if e.get('key') in arrow_keys])
    mouse_clicks = len([e for e in timestamped_events if e.get('event') in ['MouseClick', 'MouseDown']])
    measures['navigation_count'] = navigation_keystrokes + mouse_clicks

    paste_events = [e for e in events if e.get('event') == 'TextPasted']
    measures['copy_paste_count'] = len(paste_events)

    cursor_positions = []
    for e in keydown_events:
        if 'cursorStart' in e and e.get('target') == 'text':
            cursor_positions.append(e['cursorStart'])

    if len(cursor_positions) > 1:
        forward_moves = sum(1 for i in range(1, len(cursor_positions))
                          if cursor_positions[i] >= cursor_positions[i-1])
        measures['linearity_index'] = forward_moves / (len(cursor_positions) - 1)
    else:
        measures['linearity_index'] = 1.0

    # === AREA/TARGET TIME TRACKING ===
    target_times = calculate_target_times(timestamped_events)
    measures['time_in_text_ms'] = target_times.get('text', 0)
    measures['time_in_task_text_ms'] = target_times.get('task_text', 0)
    measures['time_in_chat_ms'] = target_times.get('chat', 0)
    measures['time_in_chat_prompt_ms'] = target_times.get('chat_prompt', 0)
    measures['time_in_none_ms'] = target_times.get('none', 0) + target_times.get(None, 0)

    measures['target_switches'] = count_target_switches(timestamped_events)

    return measures


def reconstruct_text(events: list, target_field: str = 'text') -> str:
    """
    Reconstruct the final text from events.
    Based on the algorithm from EventDoku.md
    """
    relevant_events = [
        e for e in events
        if (e.get('event') in ['KeyDown', 'TextPasted', 'TextCut'] or
            e.get('event', '').startswith('cr'))
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
                    pre = text[:cursor_start-1]
                insert = ''
            elif key == 'Delete':
                if cursor_start == cursor_end and cursor_end < len(text):
                    post = text[cursor_end+1:]
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


def calculate_target_times(events: list) -> dict:
    """Calculate time spent in each target area."""
    target_times = defaultdict(int)

    for i, event in enumerate(events[:-1]):
        current_target = event.get('target', None)
        if current_target is None:
            current_target = 'none'

        next_ts = events[i + 1]['timestamp_rel']
        current_ts = event['timestamp_rel']
        duration = next_ts - current_ts

        if 0 < duration < MAX_IDLE_TIME_MS:
            target_times[current_target] += duration

    return dict(target_times)


def count_target_switches(events: list) -> int:
    """Count the number of times the user switched between target areas."""
    switches = 0
    prev_target = None

    for event in events:
        current_target = event.get('target')
        if current_target and prev_target and current_target != prev_target:
            switches += 1
        if current_target:
            prev_target = current_target

    return switches


def parse_folder_info(folder_name: str) -> tuple:
    """Extract task and chat_used from folder name."""
    chat_used = 'chat__' in folder_name or '_chat_' in folder_name
    parts = folder_name.replace('FLA25MS_', '').replace('chat__', '').split('_')
    task = parts[-1] if parts else 'unknown'
    return task, chat_used


def process_all_files(base_dir: Path, folders: list) -> pd.DataFrame:
    """Process all JSON files in the specified folders."""
    all_records = []

    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue

        task, chat_used = parse_folder_info(folder)

        json_files = list(folder_path.glob('*.json'))
        print(f"Processing {len(json_files)} files in {folder}...")

        for json_file in json_files:
            try:
                events = load_json_events(json_file)
                measures = extract_measures(events)

                if measures:
                    measures['filename'] = json_file.name
                    measures['task'] = task
                    measures['chat_used'] = chat_used
                    all_records.append(measures)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

    return pd.DataFrame(all_records)


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers based on text length thresholds."""
    initial_count = len(df)

    too_short = (df['final_text_length_char'] < MIN_CHARS).sum()
    too_long = (df['final_text_length_char'] > MAX_CHARS).sum()

    print(f"\nOutlier filtering:")
    print(f"  Text length range: {df['final_text_length_char'].min():.0f} – {df['final_text_length_char'].max():.0f}")
    print(f"  Removing {too_short} submissions < {MIN_CHARS} chars (empty/trivial)")
    print(f"  Removing {too_long} submissions > {MAX_CHARS} chars (excessive copy-pasting)")

    df_filtered = df[
        (df['final_text_length_char'] >= MIN_CHARS) &
        (df['final_text_length_char'] <= MAX_CHARS)
    ].copy()

    removed = initial_count - len(df_filtered)
    print(f"  Kept {len(df_filtered)} of {initial_count} rows ({removed} removed, {100*removed/initial_count:.1f}%)")

    return df_filtered


def normalize_zscore(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """Z-score normalize numeric columns, excluding specified columns."""
    df_norm = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_normalize = [c for c in numeric_cols if c not in exclude_cols]

    for col in cols_to_normalize:
        if df[col].std() > 0:
            df_norm[col] = stats.zscore(df[col], nan_policy='omit')
        else:
            df_norm[col] = 0

    return df_norm


def main():
    print("Keystroke Measures Extraction Script")
    print("=" * 60)

    # Get base directory from command line or use default
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = DEFAULT_BASE_DIR

    print(f"Base directory: {base_dir.absolute()}")

    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        print("Usage: python extract_keystroke_measures.py [base_directory]")
        sys.exit(1)

    # 1. Extract measures from all files
    df = process_all_files(base_dir, FOLDERS)

    if df.empty:
        print("No data extracted. Check if the folder paths are correct.")
        return

    # Reorder columns: metadata first, then measures
    meta_cols = ['filename', 'task', 'chat_used']
    measure_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + measure_cols]

    print(f"\nExtracted {len(df)} records total")

    # 2. Filter outliers
    df_filtered = filter_outliers(df)

    # 3. Save filtered raw statistics
    output_raw = 'pisa_keylog_statistics.csv'
    df_filtered.to_csv(output_raw, index=False)
    print(f"\nSaved filtered raw statistics to: {output_raw}")

    # 4. Normalize and save
    df_normalized = normalize_zscore(df_filtered, EXCLUDE_COLS)
    output_norm = 'pisa_keylog_statistics_normalised.csv'
    df_normalized.to_csv(output_norm, index=False)
    print(f"Saved normalized statistics to: {output_norm}")

    # Summary for paper
    initial_count = len(df)
    final_count = len(df_filtered)
    too_short = (df['final_text_length_char'] < MIN_CHARS).sum()
    too_long = (df['final_text_length_char'] > MAX_CHARS).sum()

    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"""
Data Cleaning:
- Initial sample: N = {initial_count}
- Removed {too_short} submissions with < {MIN_CHARS} characters (empty/trivial)
- Removed {too_long} submissions with > {MAX_CHARS} characters (excessive copy-pasting)
- Final sample: N = {final_count} ({100*final_count/initial_count:.1f}% retained)

Measures extracted: {len(measure_cols)}
  {', '.join(measure_cols)}
Tasks: {df_filtered['task'].value_counts().to_dict()}
Chat used: {df_filtered['chat_used'].value_counts().to_dict()}
""")


if __name__ == '__main__':
    main()