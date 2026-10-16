#!/usr/bin/env python3
"""
dataset_descriptive_stats.py
────────────────────────────
Extract descriptive dataset statistics for the BEA paper's data section.

Produces a table comparing the "without chat" and "with chat" conditions on:
  - N (total sessions / with valid scores)
  - Final text length (chars): mean ± SD
  - Final text length (words): mean ± SD
  - Sentence count: mean ± SD
  - Holistic score: mean ± SD, median
  - % essays scoring 5.0
  - Total writing time (min): mean ± SD
  - Active writing ratio: mean ± SD

Outlier filtering matches the main pipeline:
  - Exclude < 10 chars or > 10,000 chars (on full essays).

Output:
  BEA Paper/split_statistics/
    dataset_descriptive_stats.csv
    dataset_descriptive_stats.tex
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION  (mirrors main pipeline)
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

# Outlier thresholds (same as main pipeline)
MIN_CHARS = 10
MAX_CHARS = 10000

# Keystroke thresholds (same as main pipeline)
BREAK_THRESHOLD_MS = 2000
MAX_IDLE_TIME_MS = 60000


# ============================================================
# FILE LOADING  (copied from main pipeline for independence)
# ============================================================

def load_json_events(filepath: Path) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            content = content.strip()
            if not content.startswith("["):
                content = "[" + content + "]"
            content = re.sub(r",\s*([}\]])", r"\1", content)
            return json.loads(content)


def get_events_with_timestamps(events: list) -> list:
    timestamped = [e for e in events if "timestamp_rel" in e]
    return sorted(timestamped, key=lambda x: x["timestamp_rel"])


def parse_folder_info(folder_name: str) -> tuple:
    chat_used = "chat__" in folder_name or "_chat_" in folder_name
    parts = folder_name.replace("FLA25MS_", "").replace("chat__", "").split("_")
    task = parts[-1] if parts else "unknown"
    return task, chat_used


def extract_user_key(filename: str) -> str:
    name = filename.replace("user-", "")
    match = re.match(r"^(\d+_[A-Z]+)", name)
    return match.group(1) if match else None


def load_holistic_scores(filepath: str) -> dict:
    df = pd.read_csv(filepath, usecols=["filename", "prediction"])
    df["_key"] = df["filename"].apply(extract_user_key)
    df = df.dropna(subset=["_key"]).drop_duplicates(subset="_key", keep="first")
    return dict(zip(df["_key"], df["prediction"]))


# ============================================================
# TEXT RECONSTRUCTION  (copied from main pipeline)
# ============================================================

def reconstruct_text(events: list, target_field: str = "text") -> str:
    relevant_events = [
        e for e in events
        if (e.get("event") in ["KeyDown", "TextPasted", "TextCut"]
            or e.get("event", "").startswith("cr"))
        and (e.get("target") == target_field or e.get("startElem") == target_field)
        and not e.get("IGNORE")
    ]

    text = ""
    for event in relevant_events:
        event_type = event.get("event", "")
        cursor_start = event.get("cursorStart", 0)
        cursor_end = event.get("cursorEnd", cursor_start)

        cursor_start = max(0, min(cursor_start, len(text)))
        cursor_end = max(0, min(cursor_end, len(text)))
        if cursor_start > cursor_end:
            cursor_start, cursor_end = cursor_end, cursor_start

        pre = text[:cursor_start]
        post = text[cursor_end:]

        if event_type == "TextCut":
            insert = ""
        elif event_type == "KeyDown":
            key = event.get("key", "")
            if key == "Backspace":
                if cursor_start == cursor_end and cursor_start > 0:
                    pre = text[: cursor_start - 1]
                insert = ""
            elif key == "Delete":
                if cursor_start == cursor_end and cursor_end < len(text):
                    post = text[cursor_end + 1 :]
                insert = ""
            elif key == "Enter":
                insert = "\n"
            elif key == "Tab":
                insert = "\t"
            elif len(key) == 1:
                insert = key
            else:
                continue
        else:
            insert = event.get("text", "")

        text = pre + insert + post

    return text


# ============================================================
# MEASURE EXTRACTION  (only the subset we need)
# ============================================================

def extract_descriptive_measures(events: list) -> dict | None:
    """Extract only the measures needed for the descriptive stats table."""

    timestamped_events = get_events_with_timestamps(events)
    if not timestamped_events:
        return None

    measures = {}

    # --- Temporal ---
    first_ts = timestamped_events[0]["timestamp_rel"]
    last_ts = timestamped_events[-1]["timestamp_rel"]
    total_writing_time_s = (last_ts - first_ts) / 1000

    measures["total_writing_time_min"] = total_writing_time_s / 60

    # --- Breaks (for active writing ratio) ---
    keydown_events = [e for e in timestamped_events if e.get("event") == "KeyDown"]
    keydown_timestamps = [e["timestamp_rel"] for e in keydown_events]

    break_total_ms = 0
    if len(keydown_timestamps) > 1:
        for i in range(1, len(keydown_timestamps)):
            interval = keydown_timestamps[i] - keydown_timestamps[i - 1]
            if interval >= BREAK_THRESHOLD_MS:
                break_total_ms += interval

    break_total_s = break_total_ms / 1000
    measures["active_writing_ratio"] = (
        (total_writing_time_s - break_total_s) / total_writing_time_s
        if total_writing_time_s > 0 else 0
    )

    # --- Text properties ---
    final_text = reconstruct_text(events, "text")
    measures["final_text_length_char"] = len(final_text)
    measures["final_text_length_word"] = len(final_text.split()) if final_text.strip() else 0

    # Sentence count: split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", final_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    measures["sentence_count"] = len(sentences)

    # --- Keystrokes ---
    measures["total_keystrokes"] = len(keydown_events)

    return measures


# ============================================================
# MAIN
# ============================================================

def fmt(mean, std):
    """Format mean ± SD with one decimal."""
    return f"{mean:.1f} ± {std:.1f}"


def main():
    print("=" * 60)
    print("Dataset Descriptive Statistics for BEA Paper")
    print("=" * 60)

    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = DEFAULT_BASE_DIR

    print(f"Base directory: {base_dir.absolute()}")
    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        sys.exit(1)

    # ── Load holistic scores ──────────────────────────────────
    holistic_path = Path(HOLISTIC_FILE)
    score_lookup = None
    if holistic_path.exists():
        score_lookup = load_holistic_scores(HOLISTIC_FILE)
        print(f"Loaded {len(score_lookup)} holistic scores from {HOLISTIC_FILE}")
    else:
        print(f"WARNING: {HOLISTIC_FILE} not found — score columns will be empty")

    # ── Load all files ────────────────────────────────────────
    records = []
    for folder in FOLDERS:
        folder_path = base_dir / folder
        if not folder_path.exists():
            print(f"  Warning: folder not found: {folder_path}")
            continue

        task, chat_used = parse_folder_info(folder)
        json_files = list(folder_path.glob("*.json"))
        print(f"  Loading {len(json_files)} files from {folder}...")

        for jf in json_files:
            try:
                events = load_json_events(jf)
                measures = extract_descriptive_measures(events)
                if measures is None:
                    continue
                measures["filename"] = jf.name
                measures["task"] = task
                measures["chat_used"] = chat_used
                if score_lookup:
                    key = extract_user_key(jf.name)
                    measures["holistic_score"] = score_lookup.get(key, np.nan)
                else:
                    measures["holistic_score"] = np.nan
                records.append(measures)
            except Exception as e:
                print(f"  Error loading {jf}: {e}")

    df = pd.DataFrame(records)
    print(f"\nTotal loaded: {len(df)}")

    # ── Outlier filtering ─────────────────────────────────────
    before = len(df)
    n_short = (df["final_text_length_char"] < MIN_CHARS).sum()
    n_long = (df["final_text_length_char"] > MAX_CHARS).sum()
    df = df[
        (df["final_text_length_char"] >= MIN_CHARS)
        & (df["final_text_length_char"] <= MAX_CHARS)
    ].copy()
    print(f"Outlier filtering: removed {n_short} short + {n_long} long → {len(df)} kept ({before - len(df)} removed)")

    # ── Condition label ───────────────────────────────────────
    df["condition"] = df["chat_used"].map({True: "with_chat", False: "wo_chat"})

    # ── Compute statistics per condition ──────────────────────
    rows = []
    for cond in ["wo_chat", "with_chat", "total"]:
        sub = df if cond == "total" else df[df["condition"] == cond]

        n = len(sub)
        n_scored = sub["holistic_score"].notna().sum()
        scored = sub.dropna(subset=["holistic_score"])

        row = {
            "Measure": "",  # placeholder, filled below
            cond: "",
        }

        stats = {
            "N (scored)": f"{n} ({n_scored})",
            "Text length (chars)": fmt(sub["final_text_length_char"].mean(),
                                       sub["final_text_length_char"].std()),
            "Text length (words)": fmt(sub["final_text_length_word"].mean(),
                                       sub["final_text_length_word"].std()),
            "Sentences": fmt(sub["sentence_count"].mean(),
                             sub["sentence_count"].std()),
            "Score mean ± SD": fmt(scored["holistic_score"].mean(),
                                   scored["holistic_score"].std()) if len(scored) else "—",
            "Score median": f"{scored['holistic_score'].median():.1f}" if len(scored) else "—",
            "% scoring 5.0": (
                f"{100 * (scored['holistic_score'] >= 4.75).mean():.1f}"
                if len(scored) else "—"
            ),
            "Writing time (min)": fmt(sub["total_writing_time_min"].mean(),
                                      sub["total_writing_time_min"].std()),
            "Active writing ratio": fmt(sub["active_writing_ratio"].mean(),
                                        sub["active_writing_ratio"].std()),
            "Total keystrokes": fmt(sub["total_keystrokes"].mean(),
                                    sub["total_keystrokes"].std()),
        }
        rows.append((cond, stats))

    # ── Build output dataframe ────────────────────────────────
    measure_names = list(rows[0][1].keys())
    out_data = {"Measure": measure_names}
    for cond, stats in rows:
        label = {"wo_chat": "Without Chat", "with_chat": "With Chat", "total": "Total"}[cond]
        out_data[label] = [stats[m] for m in measure_names]

    out_df = pd.DataFrame(out_data)

    # ── Save CSV ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "dataset_descriptive_stats.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # ── Save LaTeX ────────────────────────────────────────────
    tex_path = OUTPUT_DIR / "dataset_descriptive_stats.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by dataset_descriptive_stats.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{l r r r}\n")
        f.write("\\toprule\n")
        f.write("Measure & Without Chat & With Chat & Total \\\\\n")
        f.write("\\midrule\n")

        for _, row in out_df.iterrows():
            measure = row["Measure"]
            wo = row["Without Chat"]
            wc = row["With Chat"]
            total = row["Total"]
            # Escape % in LaTeX
            wo = str(wo).replace("%", "\\%")
            wc = str(wc).replace("%", "\\%")
            total = str(total).replace("%", "\\%")
            # Use ± symbol
            wo = wo.replace("±", "$\\pm$")
            wc = wc.replace("±", "$\\pm$")
            total = total.replace("±", "$\\pm$")
            f.write(f"{measure} & {wo} & {wc} & {total} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Descriptive statistics of the dataset by condition. "
                "Scores are on a 0--5 holistic scale. "
                "Active writing ratio is computed as 1 minus the proportion of break time "
                "(pauses $\\geq$ 2\\,s) relative to total writing time.}\n")
        f.write("\\label{tab:dataset-descriptive}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX: {tex_path}")

    # ── Print to console ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DESCRIPTIVE STATISTICS")
    print(f"{'=' * 60}")
    print(out_df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()