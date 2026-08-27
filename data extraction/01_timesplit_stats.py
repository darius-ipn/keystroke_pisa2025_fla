#!/usr/bin/env python3
"""
Extract keystroke measures at time thresholds (5, 10, 15, 20, 25 min).

For each threshold, events are truncated to that point in time, then the same
measures as the full-essay extraction are computed. Text is reconstructed only
from events up to the cutoff. Z-score normalization is done WITHIN each
threshold (so "average at 5 min" is the reference, not "average at 25 min").

Outlier filtering is applied once on the FULL essays, then the same set of
essays is used across all thresholds (so essay sets are comparable).

Output:
  BEA Paper/split_statistics/
    keylog_stats_05min.csv          (raw)
    keylog_stats_05min_normed.csv   (z-scored within 5-min data)
    keylog_stats_10min.csv
    keylog_stats_10min_normed.csv
    ...
    keylog_stats_25min.csv
    keylog_stats_25min_normed.csv
    keylog_stats_full.csv           (no cutoff, for reference)
    keylog_stats_full_normed.csv
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats

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

OUTPUT_DIR = Path("BEA Paper/split_statistics")
HOLISTIC_FILE = "holistic.csv"

# Time thresholds in minutes (None = full essay, no cutoff)
THRESHOLDS_MIN = [5, 10, 15, 20, 25, None]

# Keystroke thresholds
BREAK_THRESHOLD_MS = 2000
MAX_IDLE_TIME_MS = 60000

# Outlier filtering (applied on full essays, then kept consistent)
MIN_CHARS = 10
MAX_CHARS = 10000

# Columns to exclude from z-score normalization
EXCLUDE_COLS = ['filename', 'task', 'chat_used', 'holistic_score']


# ============================================================
# FILE LOADING
# ============================================================

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


# ============================================================
# EVENT TRUNCATION
# ============================================================

def truncate_events(events: list, cutoff_ms: int) -> list:
    """
    Return only events with timestamp_rel <= cutoff_ms.
    Events without timestamps are kept (they may be metadata).
    """
    truncated = []
    for e in events:
        ts = e.get('timestamp_rel')
        if ts is None:
            # Keep non-timestamped events (metadata, etc.)
            truncated.append(e)
        elif ts <= cutoff_ms:
            truncated.append(e)
    return truncated


# ============================================================
# MEASURE EXTRACTION (same logic as original, works on any event list)
# ============================================================

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
            interval = keydown_timestamps[i] - keydown_timestamps[i - 1]
            if interval >= BREAK_THRESHOLD_MS:
                breaks.append(interval)

    measures['break_count'] = len(breaks)
    measures['break_total_time'] = sum(breaks) / 1000 if breaks else 0
    measures['break_mean_duration'] = np.mean(breaks) if breaks else 0
    measures['break_ratio'] = (
        measures['break_total_time'] / measures['total_writing_time']
        if measures['total_writing_time'] > 0 else 0
    )

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
            prev_ts = keydown_events[i - 1]['timestamp_rel']
            if ts - prev_ts >= BREAK_THRESHOLD_MS:
                is_break = True

        is_revision = key in deletion_keys

        if is_break or is_revision:
            if current_burst_chars > 0 and current_burst_start is not None:
                burst_duration = keydown_events[i - 1]['timestamp_rel'] - current_burst_start
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
    measures['burst_mean_duration'] = np.mean([b['duration'] for b in bursts]) if bursts else 0

    # === DELETIONS ===
    deletion_events = [e for e in keydown_events if e.get('key') in deletion_keys and not e.get('IGNORE')]
    measures['deletion_count'] = len(deletion_events)
    measures['deletion_ratio'] = len(deletion_events) / len(keydown_events) if keydown_events else 0

    chars_deleted = 0
    for e in events:
        if e.get('event') == 'TextCut' and not e.get('IGNORE'):
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

    measures['final_text_length_char'] = len(final_text)
    measures['final_text_length_word'] = len(final_text.split()) if final_text.strip() else 0

    active_time_seconds = measures['total_writing_time'] - measures['break_total_time']
    measures['chars_per_minute'] = (
        (measures['final_text_length_char'] / active_time_seconds * 60)
        if active_time_seconds > 0 else 0
    )

    total_chars_typed = sum(
        1 for e in keydown_events if len(e.get('key', '')) == 1 and not e.get('IGNORE')
    )
    measures['process_product_ratio'] = (
        total_chars_typed / measures['final_text_length_char']
        if measures['final_text_length_char'] > 0 else 0
    )

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
        forward_moves = sum(
            1 for i in range(1, len(cursor_positions))
            if cursor_positions[i] >= cursor_positions[i - 1]
        )
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
    """Reconstruct the text from events (works on truncated event lists too)."""
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


def extract_user_key(filename: str) -> str:
    """
    Extract a matching key from filename for merging with holistic scores.
    
    Examples:
        'user-7301760_BIZ#42e6...#1_corr.json' -> '7301760_BIZ'
        '0103500_SLF#791a...#1.txt' -> '0103500_SLF'
    """
    name = filename.replace('user-', '')
    match = re.match(r'^(\d+_[A-Z]+)', name)
    if match:
        return match.group(1)
    return None


def load_holistic_scores(filepath: str) -> dict:
    """
    Load holistic scores and return a dict: user_key -> score.
    Only reads filename and prediction columns (skips large text field).
    """
    df = pd.read_csv(filepath, usecols=['filename', 'prediction'])
    df['_key'] = df['filename'].apply(extract_user_key)
    df = df.dropna(subset=['_key'])
    df = df.drop_duplicates(subset='_key', keep='first')
    return dict(zip(df['_key'], df['prediction']))


# ============================================================
# MAIN PIPELINE
# ============================================================

def load_all_events(base_dir: Path, folders: list) -> list:
    """
    Load all JSON files and return list of dicts:
      {'filename': str, 'task': str, 'chat_used': bool, 'events': list}
    """
    all_files = []
    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            print(f"  Warning: Folder not found: {folder_path}")
            continue

        task, chat_used = parse_folder_info(folder)
        json_files = list(folder_path.glob('*.json'))
        print(f"  Loading {len(json_files)} files from {folder}...")

        for json_file in json_files:
            try:
                events = load_json_events(json_file)
                all_files.append({
                    'filename': json_file.name,
                    'task': task,
                    'chat_used': chat_used,
                    'events': events,
                })
            except Exception as e:
                print(f"  Error loading {json_file}: {e}")

    return all_files


def extract_for_threshold(all_files: list, cutoff_ms: int = None,
                          score_lookup: dict = None) -> pd.DataFrame:
    """
    Extract measures from all files, optionally truncating at cutoff_ms.
    cutoff_ms=None means no truncation (full essay).
    score_lookup: dict of user_key -> holistic_score (optional).
    """
    records = []
    for entry in all_files:
        events = entry['events']

        if cutoff_ms is not None:
            events = truncate_events(events, cutoff_ms)

        measures = extract_measures(events)
        if measures:
            measures['filename'] = entry['filename']
            measures['task'] = entry['task']
            measures['chat_used'] = entry['chat_used']
            if score_lookup is not None:
                key = extract_user_key(entry['filename'])
                measures['holistic_score'] = score_lookup.get(key, np.nan)
            records.append(measures)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Reorder: metadata first, then holistic_score, then measures
    meta_cols = ['filename', 'task', 'chat_used']
    if 'holistic_score' in df.columns:
        meta_cols.append('holistic_score')
    measure_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + measure_cols]


def filter_outliers(df: pd.DataFrame) -> tuple:
    """
    Remove outliers based on full-essay text length.
    Returns (filtered_df, set_of_kept_filenames).
    """
    initial = len(df)
    too_short = (df['final_text_length_char'] < MIN_CHARS).sum()
    too_long = (df['final_text_length_char'] > MAX_CHARS).sum()

    df_filtered = df[
        (df['final_text_length_char'] >= MIN_CHARS)
        & (df['final_text_length_char'] <= MAX_CHARS)
    ].copy()

    removed = initial - len(df_filtered)
    print(f"\nOutlier filtering (on full essays):")
    print(f"  Char range: {df['final_text_length_char'].min():.0f} – {df['final_text_length_char'].max():.0f}")
    print(f"  Removed {too_short} < {MIN_CHARS} chars, {too_long} > {MAX_CHARS} chars")
    print(f"  Kept {len(df_filtered)} of {initial} ({removed} removed, {100 * removed / initial:.1f}%)")

    return df_filtered, set(df_filtered['filename'])


def normalize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize numeric columns within this threshold's data."""
    df_norm = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_normalize = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    for col in cols_to_normalize:
        if df[col].std() > 0:
            df_norm[col] = stats.zscore(df[col], nan_policy='omit')
        else:
            df_norm[col] = 0

    return df_norm


def threshold_label(minutes):
    """Human-readable label for a threshold."""
    if minutes is None:
        return "full"
    return f"{minutes:02d}min"


def main():
    print("=" * 60)
    print("Keystroke Measures – Threshold-Based Extraction")
    print("=" * 60)

    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = DEFAULT_BASE_DIR

    print(f"Base directory: {base_dir.absolute()}")
    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        sys.exit(1)

    # ── Step 1: Load all raw events into memory ────────────────
    print("\nStep 1: Loading all JSON files...")
    all_files = load_all_events(base_dir, FOLDERS)
    print(f"  Total files loaded: {len(all_files)}")

    if not all_files:
        print("No files found. Check folder paths.")
        return

    # ── Step 2: Load holistic scores ─────────────────────────────
    print("\nStep 2: Loading holistic scores...")
    holistic_path = Path(HOLISTIC_FILE)
    if holistic_path.exists():
        score_lookup = load_holistic_scores(HOLISTIC_FILE)
        print(f"  Loaded {len(score_lookup)} score entries from {HOLISTIC_FILE}")
    else:
        print(f"  WARNING: {HOLISTIC_FILE} not found, proceeding without scores")
        score_lookup = None

    # ── Step 3: Extract FULL measures to determine outliers ────
    print("\nStep 3: Extracting full-essay measures for outlier detection...")
    df_full = extract_for_threshold(all_files, cutoff_ms=None, score_lookup=score_lookup)
    print(f"  Full extraction: {len(df_full)} records")
    if score_lookup:
        n_scored = df_full['holistic_score'].notna().sum()
        print(f"  With holistic scores: {n_scored}/{len(df_full)} ({100 * n_scored / len(df_full):.1f}%)")

    df_full_filtered, keep_filenames = filter_outliers(df_full)

    # ── Step 4: Extract at each threshold, filter to same set ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for threshold_min in THRESHOLDS_MIN:
        label = threshold_label(threshold_min)
        cutoff_ms = threshold_min * 60 * 1000 if threshold_min is not None else None

        print(f"\n{'─' * 60}")
        print(f"Threshold: {label} {'(no cutoff)' if cutoff_ms is None else f'(cutoff={cutoff_ms}ms)'}")
        print(f"{'─' * 60}")

        # Extract
        df_thresh = extract_for_threshold(all_files, cutoff_ms=cutoff_ms,
                                          score_lookup=score_lookup)

        # Keep only essays that passed full-essay outlier filter
        df_thresh = df_thresh[df_thresh['filename'].isin(keep_filenames)].copy()
        print(f"  Records after outlier filter: {len(df_thresh)}")

        # Save raw
        raw_path = OUTPUT_DIR / f"keylog_stats_{label}.csv"
        df_thresh.to_csv(raw_path, index=False)
        print(f"  Saved raw:    {raw_path.name}")

        # Normalize WITHIN this threshold
        df_normed = normalize_zscore(df_thresh)
        norm_path = OUTPUT_DIR / f"keylog_stats_{label}_normed.csv"
        df_normed.to_csv(norm_path, index=False)
        print(f"  Saved normed: {norm_path.name}")

        # Summary stats
        summary_rows.append({
            'threshold': label,
            'cutoff_min': threshold_min if threshold_min else 'full',
            'n_records': len(df_thresh),
            'mean_text_chars': df_thresh['final_text_length_char'].mean(),
            'std_text_chars': df_thresh['final_text_length_char'].std(),
            'mean_writing_time_s': df_thresh['total_writing_time'].mean(),
            'mean_keystrokes': df_thresh['total_keystrokes'].mean(),
            'mean_deletions': df_thresh['deletion_count'].mean(),
        })

    # ── Step 5: Save summary comparison ────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "threshold_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY ACROSS THRESHOLDS")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False, float_format='%.1f'))
    print(f"\nAll files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()