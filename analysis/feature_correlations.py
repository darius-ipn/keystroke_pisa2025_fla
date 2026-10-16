#!/usr/bin/env python3
"""
Feature correlation analysis for KEYSCORE paper revision.

For all tasks combined (full-text features only):
- Merges keystroke statistics with holistic scores
- Bins scores to nearest 0.5
- Computes Pearson r of each feature vs. text length and vs. score
- Separately for Chat and NoChat conditions
- Outputs one CSV: feature | r_len_nochat | r_score_nochat | r_len_chat | r_score_chat
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
KEYLOG_CSV   = Path("pisa_keylog_statistics.csv")
HOLISTIC_CSV = Path("holistic.csv")
OUTPUT_CSV   = Path("feature_correlations_all_tasks.csv")

# ── Settings ───────────────────────────────────────────────────────────────────
LENGTH_COL   = "final_text_length_char"
SCORE_COL    = "holistic_score_binned"

# Columns that are NOT features (metadata / target)
NON_FEATURE_COLS = {
    "filename", "task", "chat_used", "holistic_score", SCORE_COL,
    # Chat-only area-time features — set to 0 for NoChat, so exclude from NoChat corr
    "time_in_chat_ms", "time_in_chat_prompt_ms",
}


def bin_score(score: float) -> float:
    """Round to nearest 0.5 (e.g. 2.345 -> 2.5, 3.8 -> 4.0)."""
    if pd.isna(score):
        return np.nan
    return round(score * 2) / 2


def strip_keylog_filename(fname: str) -> str:
    """
    'user-2409140_SLF#...#1_corr.json'
    -> '2409140_SLF#...#1'
    """
    fname = Path(fname).stem          # drop .json
    fname = fname.removeprefix("user-")
    if fname.endswith("_corr"):
        fname = fname[:-5]
    return fname


def strip_holistic_filename(fname: str) -> str:
    """
    '0103500_SLF#...#1.txt'
    -> '0103500_SLF#...#1'
    """
    return Path(fname).stem           # drop .txt


def safe_pearsonr(x: pd.Series, y: pd.Series) -> float:
    """Return Pearson r, or NaN if not computable (constant series, too few obs)."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    r, _ = pearsonr(x, y)
    return round(r, 3)


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading keystroke statistics...")
    kl = pd.read_csv(KEYLOG_CSV)
    print(f"  {len(kl)} rows, columns: {list(kl.columns)}")

    print("Loading holistic scores...")
    hol = pd.read_csv(HOLISTIC_CSV, header=0,
                      names=["folder", "filename", "text", "prediction"])
    print(f"  {len(hol)} rows")

    # ── Normalise join keys ────────────────────────────────────────────────────
    kl["join_key"]  = kl["filename"].apply(strip_keylog_filename)
    hol["join_key"] = hol["filename"].apply(strip_holistic_filename)

    # ── Merge ──────────────────────────────────────────────────────────────────
    df = kl.merge(hol[["join_key", "prediction"]], on="join_key", how="inner")
    print(f"\nAfter merge: {len(df)} rows matched out of {len(kl)} keylog rows")

    unmatched = len(kl) - len(df)
    if unmatched > 0:
        print(f"  Warning: {unmatched} keylog rows had no matching holistic score")

    # ── Bin scores ─────────────────────────────────────────────────────────────
    df[SCORE_COL] = df["prediction"].apply(bin_score)
    missing = df[SCORE_COL].isna().sum()
    if missing > 0:
        print(f"  Dropping {missing} rows with missing prediction scores")
        df = df[df[SCORE_COL].notna()].copy()
    print(f"\nScore distribution (binned):\n{df[SCORE_COL].value_counts().sort_index()}")


    # ── Task breakdown (informational) ─────────────────────────────────────────
    if "task" in df.columns:
        print(f"\nTask breakdown:\n{df['task'].value_counts().to_string()}")
    print(f"\nUsing all tasks combined: {len(df)} rows total")

    # ── Split by condition ─────────────────────────────────────────────────────
    nochat = df[df["chat_used"] == False].copy()
    chat   = df[df["chat_used"] == True].copy()
    print(f"  NoChat: {len(nochat)} | Chat: {len(chat)}")

    # ── Identify feature columns ───────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLS
                    and c != "prediction"]

    print(f"\nComputing correlations for {len(feature_cols)} features...")

    # Sanity check
    print("\n── Sanity check ──")
    print(df[["join_key", "prediction", SCORE_COL, LENGTH_COL]].describe())
    print(df[["prediction", SCORE_COL, LENGTH_COL]].corr())
    print(f"Unique scores (raw): {sorted(df['prediction'].unique())[:10]}")
    print(nochat[[LENGTH_COL, SCORE_COL]].corr())
    
    # ── Compute correlations ───────────────────────────────────────────────────
    # For NoChat: exclude chat-only area-time features (they're all 0 -> NaN anyway,
    # but being explicit avoids confusion)
    chat_only_features = {"time_in_chat_ms", "time_in_chat_prompt_ms",
                          "time_in_text_ms", "time_in_task_text_ms",
                          "time_in_none_ms", "target_switches"}

    rows = []
    for feat in feature_cols:
        row = {"feature": feat}

        # NoChat correlations (skip chat-area features)
        if feat in chat_only_features:
            row["r_len_nochat"]   = np.nan
            row["r_score_nochat"] = np.nan
        else:
            row["r_len_nochat"]   = safe_pearsonr(nochat[feat], nochat[LENGTH_COL])
            row["r_score_nochat"] = safe_pearsonr(nochat[feat], nochat[SCORE_COL])

        # Chat correlations (all features valid)
        row["r_len_chat"]   = safe_pearsonr(chat[feat], chat[LENGTH_COL])
        row["r_score_chat"] = safe_pearsonr(chat[feat], chat[SCORE_COL])

        rows.append(row)

    result = pd.DataFrame(rows).set_index("feature")

    # ── Sort by absolute correlation with score (NoChat), descending ───────────
    result["_sort"] = result["r_score_nochat"].abs().fillna(0)
    result = result.sort_values("_sort", ascending=False).drop(columns="_sort")

    # ── Save ───────────────────────────────────────────────────────────────────
    result.to_csv(OUTPUT_CSV)
    print(f"\nSaved to: {OUTPUT_CSV}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n── Top 10 features by |r_score_nochat| ──")
    print(result[["r_score_nochat", "r_len_nochat",
                  "r_score_chat",   "r_len_chat"]].head(10).to_string())

    print("\n── Features most correlated with LENGTH (potential proxies) ──")
    len_sorted = result.reindex(
        result["r_len_nochat"].abs().sort_values(ascending=False).index
    )
    print(len_sorted[["r_len_nochat", "r_score_nochat"]].head(10).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()