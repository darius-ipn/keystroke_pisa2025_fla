#!/usr/bin/env python3
"""
XGBoost Baseline for Holistic Score Prediction from Keystroke Features.

Runs across all threshold subfolders (05min, 10min, ..., full) and both chat conditions.

Three feature modes:
  1. all        - All keystroke features
  2. length     - Only final_text_length_char
  3. no_length  - All features EXCEPT final_text_length_char

Hyperparameter tuning via grid search on the dev set.
Evaluation: Pearson r, Spearman ρ, MAE, RMSE, QWK (rounded to 0.5)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from scipy import stats
from pathlib import Path
from datetime import datetime
from itertools import product as iter_product
import joblib
import json

# ============================================================
# CONFIGURATION
# ============================================================

SPLITS_DIR = Path("BEA Paper/splits")
OUTPUT_DIR = Path("BEA Paper/models_xgb")
SCORE_COL = "holistic_score"
RANDOM_SEED = 42

# Thresholds matching the split script
THRESHOLDS = ['05min', '10min', '15min', '20min', '25min', 'full']

# Columns to always exclude (metadata, target, redundant)
ALWAYS_EXCLUDE = [
    'filename', 'task', 'chat_used',
    'holistic_score', 'prediction', 'score', 'rating',
    'folder', 'text',
    'final_text_length_word',
]

MODE_CONFIG = {
    'all': {
        'description': 'All keystroke features',
        'exclude': [],
        'include_only': None,
    },
    'length': {
        'description': 'Only final_text_length_char',
        'exclude': [],
        'include_only': ['final_text_length_char'],
    },
    'no_length': {
        'description': 'All features EXCEPT final_text_length_char',
        'exclude': ['final_text_length_char'],
        'include_only': None,
    },
}

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_alpha': [0, 0.1, 1.0],
}


# ============================================================
# HELPERS
# ============================================================

def get_feature_columns(df: pd.DataFrame, mode: str) -> list:
    config = MODE_CONFIG[mode]
    exclude = set(ALWAYS_EXCLUDE + config['exclude'])
    include_only = config['include_only']

    if include_only is not None:
        features = [c for c in include_only if c in df.columns and c not in exclude
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    else:
        features = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
                    and df[c].std() > 0]
    return features


def bin_to_half(values):
    """Round values to nearest 0.5."""
    return (np.array(values) * 2).round() / 2


def evaluate(y_true, y_pred) -> dict:
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_binned = bin_to_half(y_true)
    y_pred_clipped = np.clip(y_pred, 0.0, 5.0)
    y_pred_binned = bin_to_half(y_pred_clipped)

    all_labels = [f"{x:.1f}" for x in np.arange(0, 5.5, 0.5)]
    y_true_str = [f"{x:.1f}" for x in y_true_binned]
    y_pred_str = [f"{x:.1f}" for x in y_pred_binned]
    qwk = cohen_kappa_score(y_true_str, y_pred_str, weights='quadratic', labels=all_labels)

    return {
        'Pearson r': pearson_r,
        'Pearson p': pearson_p,
        'Spearman ρ': spearman_r,
        'Spearman p': spearman_p,
        'MAE': mae,
        'RMSE': rmse,
        'QWK': qwk,
    }


def grid_search(X_train, y_train, X_dev, y_dev, param_grid):
    """
    Grid search over hyperparameters, evaluated on dev set by QWK.
    Returns best params and best QWK.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(iter_product(*values))

    best_qwk = -1
    best_params = None
    best_model = None

    print(f"      Grid search: {len(combos)} combinations")

    for combo in combos:
        params = dict(zip(keys, combo))

        model = XGBRegressor(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_dev)

        metrics = evaluate(y_dev, y_pred)

        if metrics['QWK'] > best_qwk:
            best_qwk = metrics['QWK']
            best_params = params
            best_model = model

    print(f"      Best QWK on dev: {best_qwk:.4f}")
    print(f"      Best params: {best_params}")

    return best_model, best_params, best_qwk


def run_experiment(train_path: str, dev_path: str, eval_path: str, mode: str) -> dict:
    """
    Run a single XGBoost experiment with grid search on dev.

    Args:
        train_path: Path to training CSV
        dev_path: Path to dev CSV (for hyperparameter tuning)
        eval_path: Path to evaluation CSV (dev or test for final metrics)
        mode: Feature mode ('all', 'length', 'no_length')
    """
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    eval_df = pd.read_csv(eval_path)

    features = get_feature_columns(train_df, mode)

    if not features:
        print(f"    ERROR: No features selected for mode '{mode}'!")
        return None

    X_train = train_df[features].fillna(0).values
    y_train = train_df[SCORE_COL].values
    X_dev = dev_df[features].fillna(0).values
    y_dev = dev_df[SCORE_COL].values
    X_eval = eval_df[features].fillna(0).values
    y_eval = eval_df[SCORE_COL].values

    # Special case: length-only mode has 1 feature, limit tree depth
    if mode == 'length':
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [1, 2, 3],
            'learning_rate': [0.01, 0.05, 0.1],
            'reg_alpha': [0, 0.1, 1.0],
        }
    else:
        param_grid = PARAM_GRID

    # Grid search on dev set
    best_model, best_params, dev_qwk = grid_search(
        X_train, y_train, X_dev, y_dev, param_grid
    )

    # Evaluate on the eval set
    y_pred = best_model.predict(X_eval)
    metrics = evaluate(y_eval, y_pred)

    return {
        'mode': mode,
        'description': MODE_CONFIG[mode]['description'],
        'n_features': len(features),
        'features': features,
        'n_train': len(X_train),
        'n_dev': len(X_dev),
        'n_eval': len(X_eval),
        'best_params': best_params,
        'dev_qwk': dev_qwk,
        'metrics': metrics,
        'model': best_model,
        'y_pred': y_pred,
        'y_eval': y_eval,
    }


def save_experiment(result: dict, threshold: str, condition: str, eval_split: str, eval_df: pd.DataFrame):
    exp_dir = OUTPUT_DIR / threshold / condition / result['mode']
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(result['model'], exp_dir / 'model.joblib')

    # Save config
    config = {
        'threshold': threshold,
        'condition': condition,
        'mode': result['mode'],
        'description': result['description'],
        'eval_split': eval_split,
        'n_features': result['n_features'],
        'features': result['features'],
        'n_train': result['n_train'],
        'n_dev': result['n_dev'],
        'n_eval': result['n_eval'],
        'best_params': result['best_params'],
        'dev_qwk': result['dev_qwk'],
        'timestamp': datetime.now().isoformat(),
    }
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'filename': eval_df['filename'].values,
        'true_score': result['y_eval'],
        'predicted_score': result['y_pred'],
        'predicted_binned': bin_to_half(np.clip(result['y_pred'], 0.0, 5.0)),
    })
    pred_df.to_csv(exp_dir / f'predictions_{eval_split}.csv', index=False)

    print(f"    Saved: {exp_dir.relative_to(OUTPUT_DIR)}/")


def print_results(result: dict):
    m = result['metrics']
    print(f"\n    Mode: {result['mode']} — {result['description']}")
    print(f"    Features ({result['n_features']}): {', '.join(result['features'][:10])}{'...' if result['n_features'] > 10 else ''}")
    print(f"    Train: {result['n_train']}, Dev: {result['n_dev']}, Eval: {result['n_eval']}")
    print(f"    Best params: {result['best_params']}")
    print(f"    Dev QWK: {result['dev_qwk']:.4f}")
    print(f"    {'─' * 40}")
    print(f"    Pearson r:  {m['Pearson r']:.4f}  (p={m['Pearson p']:.2e})")
    print(f"    Spearman ρ: {m['Spearman ρ']:.4f}  (p={m['Spearman p']:.2e})")
    print(f"    MAE:        {m['MAE']:.4f}")
    print(f"    RMSE:       {m['RMSE']:.4f}")
    print(f"    QWK:        {m['QWK']:.4f}")


def print_comparison_table(results: list):
    print(f"\n  {'=' * 80}")
    print(f"  {'Mode':<14} {'#Feat':>5} {'Dev QWK':>8} {'Pearson':>8} {'Spearman':>9} {'MAE':>7} {'RMSE':>7} {'QWK':>7}")
    print(f"  {'─' * 80}")
    for r in results:
        m = r['metrics']
        print(f"  {r['mode']:<14} {r['n_features']:>5} {r['dev_qwk']:>8.4f} {m['Pearson r']:>8.4f} {m['Spearman ρ']:>9.4f} "
              f"{m['MAE']:>7.4f} {m['RMSE']:>7.4f} {m['QWK']:>7.4f}")
    print(f"  {'=' * 80}")


def main():
    # ============================================================
    # EXPERIMENT SETTINGS
    # ============================================================
    conditions = ['wo_chat', 'withchat']
    modes = ['all', 'length', 'no_length']

    # Evaluate on dev set first; switch to 'test' for final evaluation
    eval_split = 'test'
    # ============================================================

    all_results = []

    for threshold in THRESHOLDS:
        threshold_dir = SPLITS_DIR / threshold

        if not threshold_dir.exists():
            print(f"\n  WARNING: {threshold_dir} not found, skipping!")
            continue

        print(f"\n{'#' * 80}")
        print(f"  THRESHOLD: {threshold}")
        print(f"{'#' * 80}")

        for condition in conditions:
            train_path = threshold_dir / f"{condition}_train.csv"
            dev_path = threshold_dir / f"{condition}_dev.csv"
            eval_path = threshold_dir / f"{condition}_{eval_split}.csv"

            if not train_path.exists() or not dev_path.exists() or not eval_path.exists():
                print(f"\n  WARNING: Missing files for {condition} at {threshold}, skipping!")
                continue

            label = "WITH CHAT" if 'with' in condition else "WITHOUT CHAT"
            print(f"\n  --- {label} ---")
            print(f"  Train: {train_path.name} | Dev: {dev_path.name} | Eval: {eval_path.name}")

            eval_df = pd.read_csv(str(eval_path))

            results = []
            for mode in modes:
                print(f"\n    [{mode}]")
                result = run_experiment(
                    str(train_path), str(dev_path), str(eval_path), mode
                )
                if result:
                    result['threshold'] = threshold
                    result['condition'] = condition
                    result['eval_split'] = eval_split
                    print_results(result)
                    save_experiment(result, threshold, condition, eval_split, eval_df)
                    results.append(result)
                    all_results.append(result)

            if results:
                print_comparison_table(results)

    # Save combined summary CSV
    if all_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in all_results:
            row = {
                'threshold': r['threshold'],
                'condition': r['condition'],
                'mode': r['mode'],
                'eval_split': r['eval_split'],
                'n_features': r['n_features'],
                'n_train': r['n_train'],
                'n_dev': r['n_dev'],
                'n_eval': r['n_eval'],
                'dev_qwk': r['dev_qwk'],
                'best_n_estimators': r['best_params']['n_estimators'],
                'best_max_depth': r['best_params']['max_depth'],
                'best_learning_rate': r['best_params']['learning_rate'],
                'best_reg_alpha': r['best_params']['reg_alpha'],
                'timestamp': datetime.now().isoformat(),
            }
            row.update(r['metrics'])
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        summary_path = OUTPUT_DIR / 'results_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'=' * 80}")
        print("FULL RESULTS SUMMARY")
        print(f"{'=' * 80}")
        print(summary_df.to_string(index=False))
        print(f"\nSaved to: {summary_path}")

    print(f"\nAll models saved under: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()