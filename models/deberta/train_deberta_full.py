#!/usr/bin/env python3
"""
DeBERTa-v3 Text-Only Baseline for Holistic Essay Score Prediction.

Takes essay text as input, predicts holistic score (regression).
Uses the combined dataset (both chat conditions), with chat_used as context.

Input:
  - BEA Paper/splits/full/{wo_chat,withchat}_{train,dev,test}.csv
  - BEA Paper/text_lookup.tsv

Output:
  - BEA Paper/models_transformer/text_only/
      model/          (saved model + tokenizer)
      config.json     (experiment config)
      predictions_test.csv
      training_log.csv

Usage:
  python train_deberta_text.py                          # train + evaluate
  python train_deberta_text.py --eval_only --checkpoint BEA Paper/models_transformer/text_only/model
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512

SPLITS_DIR = Path("BEA Paper/splits/full")
TEXT_LOOKUP = Path("BEA Paper/text_lookup.tsv")
OUTPUT_DIR = Path("BEA Paper/models_transformer/text_only")

SCORE_COL = "holistic_score"

# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

# Early stopping
EARLY_STOPPING_PATIENCE = 3


# ============================================================
# TEXT LOADING
# ============================================================

def load_text_lookup(path: Path) -> dict:
    """Load text lookup TSV and return dict: filename -> text."""
    lookup = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t', 3)  # split into max 4 parts
            if len(parts) == 4:
                filename, task, chat_used, text_escaped = parts
                # Unescape: reverse the escaping from extraction
                text = (
                    text_escaped.replace('\\n', '\n')
                               .replace('\\r', '\r')
                               .replace('\\t', '\t')
                               .replace('\\\\', '\\')
                )
                lookup[filename] = text
    return lookup


def normalise_filename(fn: str) -> str:
    """Normalise filenames for matching between splits and text lookup."""
    return fn.strip()


# ============================================================
# DATASET
# ============================================================

class EssayDataset(Dataset):
    """Dataset for essay scoring: text -> score."""

    def __init__(self, df: pd.DataFrame, text_lookup: dict, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        self.scores = []
        self.filenames = []

        missing = 0
        for _, row in df.iterrows():
            fn = normalise_filename(row['filename'])
            text = text_lookup.get(fn)

            if text is None:
                missing += 1
                continue

            self.texts.append(text)
            self.scores.append(float(row[SCORE_COL]))
            self.filenames.append(fn)

        if missing > 0:
            print(f"    Warning: {missing}/{len(df)} texts not found in lookup")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.scores[idx], dtype=torch.float32),
        }


# ============================================================
# EVALUATION
# ============================================================

def bin_to_half(values):
    """Round to nearest 0.5."""
    return (np.array(values) * 2).round() / 2


def compute_metrics_from_predictions(y_true, y_pred) -> dict:
    """Compute all evaluation metrics."""
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # QWK with binning to 0.5 steps
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


def compute_metrics_for_trainer(eval_pred):
    """Compute metrics during training (for Trainer callback)."""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    metrics = compute_metrics_from_predictions(labels, predictions)
    # Trainer expects simple metric names
    return {
        'pearson_r': metrics['Pearson r'],
        'spearman_r': metrics['Spearman ρ'],
        'mae': metrics['MAE'],
        'rmse': metrics['RMSE'],
        'qwk': metrics['QWK'],
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_combined_splits(splits_dir: Path):
    """
    Load and combine wo_chat + withchat splits into single train/dev/test.
    """
    splits = {}
    for split_name in ['train', 'dev', 'test']:
        dfs = []
        for prefix in ['wo_chat', 'withchat']:
            path = splits_dir / f"{prefix}_{split_name}.csv"
            if path.exists():
                df = pd.read_csv(path)
                dfs.append(df)
                print(f"  Loaded {path.name}: {len(df)} rows")
            else:
                print(f"  Warning: {path} not found")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.dropna(subset=[SCORE_COL])
            splits[split_name] = combined
            print(f"  Combined {split_name}: {len(combined)} rows")
        else:
            print(f"  ERROR: No data for {split_name}")
            sys.exit(1)

    return splits


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DeBERTa text-only essay scoring")
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, skip training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='HuggingFace model name')
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH, help='Max token length')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    args = parser.parse_args()

    print("=" * 70)
    print("DeBERTa Text-Only Essay Scoring")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load text lookup ──────────────────────────────────────
    print(f"\nLoading text lookup from {TEXT_LOOKUP}...")
    if not TEXT_LOOKUP.exists():
        print(f"ERROR: {TEXT_LOOKUP} not found. Run extract_texts.py first.")
        sys.exit(1)
    text_lookup = load_text_lookup(TEXT_LOOKUP)
    print(f"  Loaded {len(text_lookup)} texts")

    # ── Load splits ───────────────────────────────────────────
    print(f"\nLoading splits from {SPLITS_DIR}...")
    splits = load_combined_splits(SPLITS_DIR)

    # ── Tokenizer ─────────────────────────────────────────────
    model_name = args.checkpoint if args.eval_only and args.checkpoint else args.model_name
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Datasets ──────────────────────────────────────────────
    print("\nCreating datasets...")
    train_dataset = EssayDataset(splits['train'], text_lookup, tokenizer, args.max_length)
    dev_dataset = EssayDataset(splits['dev'], text_lookup, tokenizer, args.max_length)
    test_dataset = EssayDataset(splits['test'], text_lookup, tokenizer, args.max_length)

    print(f"  Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,              # regression
        problem_type="regression",
    )

    # ── Training ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_output = OUTPUT_DIR / "model"

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        overwrite_output_dir=True,

        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=args.lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        seed=args.seed,

        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson_r",
        greater_is_better=True,
        save_total_limit=2,

        # Logging
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        report_to="none",  # set to "wandb" if you use W&B

        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    if not args.eval_only:
        print("\n" + "─" * 70)
        print("TRAINING")
        print("─" * 70)

        train_result = trainer.train()

        # Save best model
        trainer.save_model(str(model_output))
        tokenizer.save_pretrained(str(model_output))
        print(f"\nBest model saved to: {model_output}")

        # Save training log
        log_history = pd.DataFrame(trainer.state.log_history)
        log_history.to_csv(OUTPUT_DIR / "training_log.csv", index=False)

    # ── Evaluation ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("EVALUATION")
    print("─" * 70)

    for split_name, dataset in [('dev', dev_dataset), ('test', test_dataset)]:
        print(f"\n  --- {split_name.upper()} SET ---")

        predictions = trainer.predict(dataset)
        y_pred = predictions.predictions.squeeze()
        y_true = np.array(dataset.scores)

        metrics = compute_metrics_from_predictions(y_true, y_pred)

        print(f"  n = {len(y_true)}")
        print(f"  Pearson r:  {metrics['Pearson r']:.4f}  (p={metrics['Pearson p']:.2e})")
        print(f"  Spearman ρ: {metrics['Spearman ρ']:.4f}  (p={metrics['Spearman p']:.2e})")
        print(f"  MAE:        {metrics['MAE']:.4f}")
        print(f"  RMSE:       {metrics['RMSE']:.4f}")
        print(f"  QWK:        {metrics['QWK']:.4f}")

        # Save predictions
        pred_df = pd.DataFrame({
            'filename': dataset.filenames,
            'true_score': y_true,
            'predicted_score': y_pred,
            'predicted_binned': bin_to_half(np.clip(y_pred, 0.0, 5.0)),
        })
        pred_path = OUTPUT_DIR / f"predictions_{split_name}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved: {pred_path.name}")

    # ── Save experiment config ────────────────────────────────
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'warmup_ratio': WARMUP_RATIO,
        'weight_decay': WEIGHT_DECAY,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'seed': args.seed,
        'n_train': len(train_dataset),
        'n_dev': len(dev_dataset),
        'n_test': len(test_dataset),
        'threshold': 'full',
        'condition': 'combined',
        'timestamp': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()