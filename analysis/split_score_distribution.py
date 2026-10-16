#!/usr/bin/env python3
"""Label distribution across train/dev/test splits, binned to 0.5 steps."""

import pandas as pd
import numpy as np
from pathlib import Path

SPLITS_DIR = Path("BEA Paper/splits/full")
SCORE_COL = "holistic_score"
BINS = np.arange(0, 5.5, 0.5)

for condition in ["wo_chat", "withchat"]:
    print(f"\n{'=' * 50}")
    print(f"  {condition.upper()}")
    print(f"{'=' * 50}")

    counts = {}
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(SPLITS_DIR / f"{condition}_{split}.csv")
        binned = (df[SCORE_COL] * 2).round() / 2
        dist = binned.value_counts().reindex(BINS, fill_value=0).astype(int)
        counts[split] = dist

    table = pd.DataFrame(counts)
    table.index.name = "score"
    table["total"] = table.sum(axis=1)

    print(table.to_string())
    print(f"\nN:  train={counts['train'].sum()}  dev={counts['dev'].sum()}  test={counts['test'].sum()}")