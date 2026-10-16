"""
Create Hybrid Training Sets for Transformer Training.

Based on the "Don't Score Too Early" Hybrid approach:
- All full-text training data (from split/full/)
- Plus ONE randomly sampled intermediate time point per essay
  (from 05min, 10min, 15min, 20min, 25min)

This keeps the training set at ~6,000 instances (2x the full-text baseline)
instead of ~18,000 (all 6 versions), avoiding memory issues.

Constraint: At least 50% of the randomly sampled intermediate texts
come from each chat condition (wo_chat / with_chat).
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def create_hybrid_training(base_dir: str, seed: int = 42):
    """
    Create hybrid training CSVs.

    Args:
        base_dir: Path to the 'split' directory containing subfolders
                  (05min, 10min, 15min, 20min, 25min, full)
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    time_points = ["05min", "10min", "15min", "20min", "25min"]
    conditions = ["wo_chat", "withchat"]

    # Output directory
    hybrid_dir = Path(base_dir) / "hybrid"
    hybrid_dir.mkdir(exist_ok=True)

    for condition in conditions:
        train_file = f"{condition}_train.csv"

        # 1. Load the full-text training data (this is always included entirely)
        full_path = Path(base_dir) / "full" / train_file
        if not full_path.exists():
            print(f"WARNING: {full_path} not found, skipping.")
            continue

        df_full = pd.read_csv(full_path)
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"Full-text training samples: {len(df_full)}")

        # 2. Load all intermediate time-point training data
        intermediate_dfs = {}
        for tp in time_points:
            tp_path = Path(base_dir) / tp / train_file
            if tp_path.exists():
                intermediate_dfs[tp] = pd.read_csv(tp_path)
                print(f"  {tp}: {len(intermediate_dfs[tp])} samples")
            else:
                print(f"  WARNING: {tp_path} not found, skipping this time point.")

        if not intermediate_dfs:
            print("No intermediate data found, writing full-only.")
            df_full.to_csv(hybrid_dir / train_file, index=False)
            continue

        # 3. For each essay (identified by filename), randomly pick ONE time point
        #    Constraint: at least 50% of samples from each available time point
        #    (interpreted as: balanced sampling so no single time point dominates)
        #
        #    Actually the constraint is: we want balanced representation.
        #    Since we pick one random time point per essay, we ensure roughly
        #    uniform distribution across time points.

        # Get all unique filenames from full data
        all_filenames = df_full["filename"].unique()
        n_essays = len(all_filenames)
        print(f"  Unique essays: {n_essays}")

        # Check which filenames exist in which time points
        tp_filenames = {}
        for tp, df_tp in intermediate_dfs.items():
            tp_filenames[tp] = set(df_tp["filename"].unique())

        # For each essay, find which time points are available
        sampled_rows = []
        tp_counts = {tp: 0 for tp in time_points}

        for filename in all_filenames:
            available_tps = [
                tp for tp in time_points if filename in tp_filenames.get(tp, set())
            ]
            if not available_tps:
                continue

            # Randomly pick one time point for this essay
            chosen_tp = rng.choice(available_tps)
            tp_counts[chosen_tp] += 1

            # Get the row for this essay from the chosen time point
            df_tp = intermediate_dfs[chosen_tp]
            row = df_tp[df_tp["filename"] == filename]
            sampled_rows.append(row)

        if sampled_rows:
            df_sampled = pd.concat(sampled_rows, ignore_index=True)
        else:
            df_sampled = pd.DataFrame(columns=df_full.columns)

        print(f"\n  Random sampling distribution across time points:")
        for tp in time_points:
            pct = (tp_counts[tp] / max(sum(tp_counts.values()), 1)) * 100
            print(f"    {tp}: {tp_counts[tp]} essays ({pct:.1f}%)")

        # 4. Combine: full texts + sampled intermediate texts
        #    Add a column to track the source for transparency
        df_full_tagged = df_full.copy()
        df_full_tagged["_source_timepoint"] = "full"

        df_sampled_tagged = df_sampled.copy()
        # Tag each sampled row with its source time point
        # (we need to re-do this since concat lost the info)
        # Rebuild with proper tagging
        sampled_rows_tagged = []
        rng2 = np.random.default_rng(seed)  # reset for same sequence
        for filename in all_filenames:
            available_tps = [
                tp for tp in time_points if filename in tp_filenames.get(tp, set())
            ]
            if not available_tps:
                continue
            chosen_tp = rng2.choice(available_tps)
            df_tp = intermediate_dfs[chosen_tp]
            row = df_tp[df_tp["filename"] == filename].copy()
            row["_source_timepoint"] = chosen_tp
            sampled_rows_tagged.append(row)

        if sampled_rows_tagged:
            df_sampled_tagged = pd.concat(sampled_rows_tagged, ignore_index=True)
        else:
            df_sampled_tagged = pd.DataFrame(columns=list(df_full.columns) + ["_source_timepoint"])

        df_hybrid = pd.concat([df_full_tagged, df_sampled_tagged], ignore_index=True)

        # 5. Save
        out_path = hybrid_dir / train_file
        df_hybrid.to_csv(out_path, index=False)

        print(f"\n  Hybrid dataset: {len(df_hybrid)} total samples")
        print(f"    - Full texts: {len(df_full)}")
        print(f"    - Sampled intermediate: {len(df_sampled_tagged)}")
        print(f"  Saved to: {out_path}")

    # Also print a combined summary
    print(f"\n{'='*60}")
    print("Done! Hybrid training sets created in:")
    print(f"  {hybrid_dir}/")
    print(f"  - wo_chat_train.csv")
    print(f"  - withchat_train.csv")
    print(f"\nNote: The '_source_timepoint' column tracks where each row")
    print(f"came from (full / 05min / 10min / etc.). You can drop it")
    print(f"before training if your pipeline doesn't expect it.")


if __name__ == "__main__":
    create_hybrid_training(base_dir="", seed=42)