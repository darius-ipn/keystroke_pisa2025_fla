"""
Compare annotator holistic scores with ML predictions.
Metrics: QWK, Krippendorff's Alpha, Adjacent (±1) Accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import krippendorff
import warnings
import re

warnings.filterwarnings("ignore")

# --- Configuration ---
CSV_PATH = "holistic.csv"
EXCEL_PATH = "Texts_extracted/iaa_results.xlsx"
SHEET_NAME = "Per-Text Scores"

# --- Load data ---
print("Loading data...")
df_csv = pd.read_csv(CSV_PATH)
df_excel = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# Extract relevant columns from Excel (A=0, B=1, D=3, G=6, I=8, Q=16)
df_iaa = pd.DataFrame({
    "text_id": df_excel.iloc[:, 0],        # A: text id
    "annotator1": df_excel.iloc[:, 1],      # B: annotator 1 name
    "score1": df_excel.iloc[:, 3],          # D: annotator 1 score
    "annotator2": df_excel.iloc[:, 6],      # G: annotator 2 name
    "score2": df_excel.iloc[:, 8],          # I: annotator 2 score
    "avg_score": df_excel.iloc[:, 16],      # Q: average score
})

# --- Extract shared prefix key ---
# CSV filenames:  0103500_SLF#791aa40c290d#...#1.txt  -> prefix: 0103500_SLF
# Excel text_ids: 0100110_SLF_49ca3b03c5df_..._1.txt  -> prefix: 0100110_SLF
#
# Pattern: first two segments are DIGITS_LETTERS (the text ID + task code)

def extract_prefix(name):
    """Extract the text ID prefix like '0103500_SLF' or '9001910_BIZ'."""
    name = str(name).strip()
    # Match: digits, underscore, then uppercase letters (SLF, BIZ, BFS, etc.)
    m = re.match(r"^(\d+_[A-Z]+)", name)
    return m.group(1) if m else None

df_csv["prefix"] = df_csv["filename"].apply(extract_prefix)
df_iaa["prefix"] = df_iaa["text_id"].apply(extract_prefix)

# Check for duplicates in prefix
csv_dupes = df_csv["prefix"].duplicated(keep=False).sum()
iaa_dupes = df_iaa["prefix"].duplicated(keep=False).sum()
print(f"CSV prefix duplicates: {csv_dupes}")
print(f"Excel prefix duplicates: {iaa_dupes}")

if csv_dupes > 0:
    print("\nWARNING: CSV has duplicate prefixes. Examples:")
    print(df_csv[df_csv["prefix"].duplicated(keep=False)]["prefix"].head(10).tolist())

if iaa_dupes > 0:
    print("\nWARNING: Excel has duplicate prefixes. Examples:")
    print(df_iaa[df_iaa["prefix"].duplicated(keep=False)]["prefix"].head(10).tolist())

# Merge
merged = pd.merge(df_iaa, df_csv[["prefix", "prediction"]], on="prefix", how="inner")
print(f"\nMatched {len(merged)} texts out of {len(df_iaa)} Excel / {len(df_csv)} CSV rows.\n")

if len(merged) == 0:
    print("Sample CSV prefixes:", df_csv["prefix"].dropna().head(10).tolist())
    print("Sample Excel prefixes:", df_iaa["prefix"].dropna().head(10).tolist())
    raise SystemExit("Could not match files. Check prefix extraction logic.")

# --- Helper functions ---
def compute_qwk(y1, y2):
    return cohen_kappa_score(y1, y2, weights="quadratic")

def compute_krippendorff(y1, y2):
    reliability_data = np.array([y1, y2])
    return krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="ordinal")

def compute_adjacent_accuracy(y_true, y_pred, tolerance=1):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100

def compute_exact_accuracy(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred) < 0.5) * 100

def evaluate_pair(name, y_ref, y_comp):
    mask = pd.notna(y_ref) & pd.notna(y_comp)
    y_ref = np.array(y_ref[mask], dtype=float)
    y_comp = np.array(y_comp[mask], dtype=float)
    
    y_ref_int = np.round(y_ref).astype(int)
    y_comp_int = np.round(y_comp).astype(int)
    
    n = len(y_ref)
    qwk = compute_qwk(y_ref_int, y_comp_int)
    kalpha = compute_krippendorff(y_ref_int, y_comp_int)
    adj_acc = compute_adjacent_accuracy(y_ref, y_comp, tolerance=1)
    exact_acc = compute_exact_accuracy(y_ref, y_comp)
    
    print(f"  {'N':>25s}: {n}")
    print(f"  {'QWK':>25s}: {qwk:.4f}")
    print(f"  {'Krippendorff Alpha':>25s}: {kalpha:.4f}")
    print(f"  {'Exact Accuracy':>25s}: {exact_acc:.2f}%")
    print(f"  {'Adjacent (±1) Accuracy':>25s}: {adj_acc:.2f}%")
    print()
    return {"name": name, "n": n, "QWK": qwk, "Krippendorff_Alpha": kalpha,
            "Exact_Accuracy": exact_acc, "Adjacent_Accuracy": adj_acc}

# --- Compute metrics ---
pred = merged["prediction"]
s1 = merged["score1"]
s2 = merged["score2"]
avg = merged["avg_score"]

results = []

print("=" * 60)
print("Annotator 1 vs. ML Prediction")
print("=" * 60)
results.append(evaluate_pair("Annotator1 vs ML", s1, pred))

print("=" * 60)
print("Annotator 2 vs. ML Prediction")
print("=" * 60)
results.append(evaluate_pair("Annotator2 vs ML", s2, pred))

print("=" * 60)
print("Average (Annotator 1+2) vs. ML Prediction")
print("=" * 60)
results.append(evaluate_pair("Avg_Annotators vs ML", avg, pred))

print("=" * 60)
print("Annotator 1 vs. Annotator 2 (human baseline)")
print("=" * 60)
results.append(evaluate_pair("Annotator1 vs Annotator2", s1, s2))

# --- Summary table ---
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

output_path = "comparison_results.csv"
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")