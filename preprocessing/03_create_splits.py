#!/usr/bin/env python3
"""
Create train/dev/test splits for each time threshold and chat condition.

Input:  BEA Paper/split_statistics/keylog_stats_{threshold}_normed.csv
Output: BEA Paper/splits/{threshold}/wo_chat_train.csv, etc.

- Test set = fixed from stratified_sample_proportional_500.csv
- Train/Dev = remaining data, split 90/10 (stratified by holistic_score)
- Same test set filenames used across all thresholds for comparability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

STATS_DIR = Path("BEA Paper/split_statistics")
TEST_SET_FILE = "stratified_sample_proportional_500.csv"
OUTPUT_BASE = Path("BEA Paper/splits")
SCORE_COL = "holistic_score"
RANDOM_SEED = 42

TRAIN_RATIO = 0.9
DEV_RATIO = 0.1

# Thresholds matching the extraction script output
THRESHOLDS = ['05min', '10min', '15min', '20min', '25min', 'full']


# ============================================================
# HELPERS
# ============================================================

def bin_scores(series):
    """Round scores to nearest 0.5."""
    return (series * 2).round() / 2


def normalise_filename(fn):
    """Normalise full-dataset filenames to match the test CSV format."""
    fn = fn.strip()
    fn = fn.removeprefix('user-')
    fn = fn.replace('_corr.json', '.txt')
    return fn


def show_distribution(df, label):
    """Print binned score distribution."""
    binned = bin_scores(df[SCORE_COL])
    counts = binned.value_counts().sort_index()
    print(f"\n  {label} (n={len(df)})")
    print(f"  {'Score':>8} {'Count':>8} {'Pct':>7}")
    for score, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {score:>8.1f} {count:>8} {pct:>6.1f}%")


def stratified_train_dev_split(df):
    """Create stratified train/dev split using binned scores."""
    df = df.copy()
    df['_score_bin'] = bin_scores(df[SCORE_COL])

    # Merge rare bins (<2 examples) into nearest neighbour
    while df['_score_bin'].value_counts().min() < 2:
        counts = df['_score_bin'].value_counts()
        rare_bin = counts.idxmin()
        all_bins = sorted(counts.index)
        idx = all_bins.index(rare_bin)
        if idx == 0:
            merge_into = all_bins[1]
        elif idx == len(all_bins) - 1:
            merge_into = all_bins[-2]
        else:
            if counts[all_bins[idx - 1]] >= counts[all_bins[idx + 1]]:
                merge_into = all_bins[idx - 1]
            else:
                merge_into = all_bins[idx + 1]
        print(f"      Merging rare bin {rare_bin:.1f} ({counts[rare_bin]} ex) -> {merge_into:.1f}")
        df.loc[df['_score_bin'] == rare_bin, '_score_bin'] = merge_into

    train_df, dev_df = train_test_split(
        df, test_size=DEV_RATIO,
        stratify=df['_score_bin'], random_state=RANDOM_SEED
    )

    train_df = train_df.drop(columns=['_score_bin'])
    dev_df = dev_df.drop(columns=['_score_bin'])

    return train_df, dev_df


# ============================================================
# MAIN
# ============================================================

def main():
    # Load test set filenames (fixed across all thresholds)
    test_csv = pd.read_csv(TEST_SET_FILE)
    test_filenames = set(test_csv['filename'].str.strip())
    print(f"Fixed test set: {len(test_filenames)} filenames from {TEST_SET_FILE}")

    summary_rows = []

    for threshold in THRESHOLDS:
        normed_path = STATS_DIR / f"keylog_stats_{threshold}_normed.csv"

        print(f"\n{'#' * 70}")
        print(f"  THRESHOLD: {threshold}")
        print(f"  Source: {normed_path}")
        print(f"{'#' * 70}")

        if not normed_path.exists():
            print(f"  WARNING: File not found, skipping!")
            continue

        df = pd.read_csv(normed_path)
        df = df.dropna(subset=[SCORE_COL])
        print(f"  Loaded {len(df)} rows with valid {SCORE_COL}")

        # Match test set filenames
        df['_match_key'] = df['filename'].apply(normalise_filename)
        is_test = df['_match_key'].isin(test_filenames)

        test_df = df[is_test].drop(columns=['_match_key']).copy()
        remain_df = df[~is_test].drop(columns=['_match_key']).copy()

        print(f"  Matched test rows: {len(test_df)}")
        print(f"  Remaining for train/dev: {len(remain_df)}")

        if len(test_df) != len(test_filenames):
            missing = len(test_filenames) - len(test_df)
            print(f"  ⚠ {missing} test filenames not found (may have been filtered or missing at this threshold)")

        # Normalise chat_used column
        chat_col = 'chat_used'
        for sub_df in [test_df, remain_df]:
            sub_df[chat_col] = sub_df[chat_col].astype(str).str.strip().map(
                {'True': True, 'False': False, 'true': True, 'false': False}
            )

        # Output directory for this threshold
        threshold_dir = OUTPUT_BASE / threshold
        threshold_dir.mkdir(parents=True, exist_ok=True)

        for chat_val, chat_label, prefix in [
            (False, "Without Chat", "wo_chat"),
            (True, "With Chat", "withchat"),
        ]:
            print(f"\n  --- {chat_label} ---")

            test_sub = test_df[test_df[chat_col] == chat_val].copy()
            remain_sub = remain_df[remain_df[chat_col] == chat_val].copy()

            print(f"    Test: {len(test_sub)}, Remaining: {len(remain_sub)}")

            if len(remain_sub) < 10:
                print(f"    WARNING: Too few remaining samples, skipping split!")
                continue

            train_sub, dev_sub = stratified_train_dev_split(remain_sub)

            print(f"    Train: {len(train_sub)}, Dev: {len(dev_sub)}, Test: {len(test_sub)}")

            # Save
            train_sub.to_csv(threshold_dir / f"{prefix}_train.csv", index=False)
            dev_sub.to_csv(threshold_dir / f"{prefix}_dev.csv", index=False)
            test_sub.to_csv(threshold_dir / f"{prefix}_test.csv", index=False)

            # Quick distribution summary
            for name, split_df in [('Train', train_sub), ('Dev', dev_sub), ('Test', test_sub)]:
                dist = bin_scores(split_df[SCORE_COL]).value_counts().sort_index()
                dist_str = '  '.join(f"{s:.1f}:{c}" for s, c in dist.items())
                print(f"    {name:<6} n={len(split_df):>5}  {dist_str}")

            summary_rows.append({
                'threshold': threshold,
                'condition': chat_label,
                'n_train': len(train_sub),
                'n_dev': len(dev_sub),
                'n_test': len(test_sub),
                'n_total': len(train_sub) + len(dev_sub) + len(test_sub),
            })

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_BASE / "split_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 70}")
        print("SPLIT SUMMARY")
        print(f"{'=' * 70}")
        print(summary_df.to_string(index=False))
        print(f"\nSaved to: {summary_path}")

    print(f"\nAll splits saved under: {OUTPUT_BASE.absolute()}")


if __name__ == '__main__':
    main()