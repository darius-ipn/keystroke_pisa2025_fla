"""
Analyze silver standard predictions for the KEYSCORE paper.
Covers holistic + 3 trait scores (content, language_quality, organization).
Produces: score distribution stats, model performance metrics, and figures.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

FILES = {
    "Holistic": "predictions_holistic.csv",
    "Content": "predictions_content.csv",
    "Language Quality": "predictions_language_quality.csv",
    "Organization": "predictions_organization.csv",
}

results = []

for name, path in FILES.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    pred = df["prediction"]
    gold = df["y_test"]
    pred_rounded = pred.round()

    r, r_p = pearsonr(gold, pred)
    rho, rho_p = spearmanr(gold, pred)
    rmse = np.sqrt(mean_squared_error(gold, pred))
    mae = mean_absolute_error(gold, pred)

    # QWK — need integer labels
    try:
        qwk = cohen_kappa_score(
            gold.round().astype(int),
            pred_rounded.astype(int),
            weights="quadratic"
        )
    except Exception:
        qwk = float("nan")

    diff = (pred_rounded - gold).abs()
    exact = (diff == 0).mean() * 100
    adjacent = (diff <= 1).mean() * 100

    row = {
        "Trait": name,
        "N": len(pred),
        "Gold Mean": gold.mean(),
        "Gold SD": gold.std(),
        "Pred Mean": pred.mean(),
        "Pred SD": pred.std(),
        "Pred Min": pred.min(),
        "Pred Max": pred.max(),
        "Pearson r": r,
        "Spearman rho": rho,
        "RMSE": rmse,
        "MAE": mae,
        "QWK": qwk,
        "Exact %": exact,
        "Adjacent +-1 %": adjacent,
    }
    results.append(row)

    # Print detailed per-trait info
    print("=" * 60)
    print(f"  {name.upper()}")
    print("=" * 60)
    print(f"  N = {len(pred)}")
    print(f"  Gold:  M={gold.mean():.3f}, SD={gold.std():.3f}, "
          f"range=[{gold.min():.2f}, {gold.max():.2f}]")
    print(f"  Pred:  M={pred.mean():.3f}, SD={pred.std():.3f}, "
          f"range=[{pred.min():.2f}, {pred.max():.2f}]")
    print(f"  Pearson r = {r:.3f}  (p={r_p:.2e})")
    print(f"  Spearman rho = {rho:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  QWK  = {qwk:.3f}")
    print(f"  Exact agreement: {exact:.1f}%")
    print(f"  Adjacent (+-1):  {adjacent:.1f}%")

    # Gold score distribution
    print(f"\n  Gold score distribution:")
    for score in sorted(gold.unique()):
        count = (gold == score).sum()
        pct = count / len(gold) * 100
        print(f"    {score:.2f}: {count:4d} ({pct:5.1f}%)")

    print()

# --- Summary Table ---
res_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("SUMMARY TABLE (for paper)")
print("=" * 60)
cols_paper = ["Trait", "N", "Gold Mean", "Gold SD", "Pred Mean", "Pred SD",
              "Pearson r", "RMSE", "MAE", "QWK"]
print(res_df[cols_paper].to_string(index=False, float_format="%.3f"))

# Save summary as CSV
res_df.to_csv("silver_standard_summary.csv", index=False, float_format="%.4f")
print("\nSummary saved to: silver_standard_summary.csv")

# --- Figure: 2x4 grid (histogram + scatter per trait) ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (name, path) in enumerate(FILES.items()):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    pred = df["prediction"]
    gold = df["y_test"]
    r = pearsonr(gold, pred)[0]

    # Histogram of predictions
    ax_hist = axes[0, i]
    ax_hist.hist(pred, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax_hist.axvline(pred.mean(), color='red', linestyle='--',
                    label=f'M={pred.mean():.2f}')
    ax_hist.set_title(f"{name}\n(predicted scores)")
    ax_hist.set_xlabel("Score")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(fontsize=8)

    # Scatter: predicted vs gold
    ax_scat = axes[1, i]
    ax_scat.scatter(gold, pred, alpha=0.3, s=10, color='steelblue')
    lims = [min(gold.min(), pred.min()), max(gold.max(), pred.max())]
    ax_scat.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
    ax_scat.set_title(f"{name} (r={r:.3f})")
    ax_scat.set_xlabel("Gold Score")
    ax_scat.set_ylabel("Predicted Score")
    ax_scat.legend(fontsize=8)

plt.tight_layout()
plt.savefig("silver_standard_analysis.png", dpi=150)
print("Figure saved to: silver_standard_analysis.png")