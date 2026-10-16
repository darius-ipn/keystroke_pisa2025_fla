#!/usr/bin/env python3
"""
Extract reconstructed essay texts from keystroke JSON files into a TSV lookup.

Output:
  BEA Paper/text_lookup.tsv
    Columns: filename, task, chat_used, text

The text is reconstructed using the same reconstruct_text() logic as the
keystroke extraction pipeline. Newlines within essays are replaced with \\n
literals so the TSV stays single-line-per-record.
"""

import json
import re
from pathlib import Path

# ============================================================
# CONFIGURATION (matches your extraction script)
# ============================================================

DEFAULT_BASE_DIR = Path("Originaldaten/allUser_corr")
FOLDERS = [
    "FLA25MS_advertisement",
    "FLA25MS_chat__advertisement",
    "FLA25MS_chat__teacher",
    "FLA25MS_teacher",
]
OUTPUT_FILE = Path("BEA Paper/text_lookup.tsv")


# ============================================================
# FUNCTIONS (copied from your extraction script for consistency)
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


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Text Extraction from Keystroke Logs")
    print("=" * 60)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    empty = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        # Header
        out.write("filename\ttask\tchat_used\ttext\n")

        for folder in FOLDERS:
            folder_path = DEFAULT_BASE_DIR / folder
            if not folder_path.exists():
                print(f"  Warning: {folder_path} not found, skipping")
                continue

            task, chat_used = parse_folder_info(folder)
            json_files = sorted(folder_path.glob('*.json'))
            print(f"  Processing {len(json_files)} files from {folder}...")

            for json_file in json_files:
                try:
                    events = load_json_events(json_file)
                    text = reconstruct_text(events, 'text')

                    # Escape for TSV: replace actual newlines/tabs with literals
                    text_escaped = (
                        text.replace('\\', '\\\\')  # escape existing backslashes
                            .replace('\n', '\\n')
                            .replace('\r', '\\r')
                            .replace('\t', '\\t')
                    )

                    out.write(f"{json_file.name}\t{task}\t{chat_used}\t{text_escaped}\n")
                    total += 1

                    if len(text.strip()) == 0:
                        empty += 1

                except Exception as e:
                    print(f"    Error: {json_file.name}: {e}")

    print(f"\nDone: {total} texts extracted ({empty} empty)")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()