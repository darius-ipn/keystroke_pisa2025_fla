# KEYSCORE — Keystroke-Enhanced Automated Essay Scoring

Code for the paper *"KEYSCORE — Keystroke-enhanced Automated Essay Scoring"* (BEA 2026),
based on the PISA 2025 FLA writing process dataset (keystroke logs, Chat / NoChat condition).

**Data:** not included. The data may not be published before September 2027 and is available
on request until then (schaller@leibniz-ipn.de). See [`data/README.md`](data/README.md) for the
expected folder structure.

## Repository structure

```
├── preprocessing/     keystroke features, text snapshots at 5/10/15/20/25 min, splits, hybrid set
├── silver_standard/   analysis of the MEWS linear-regression silver standard
├── models/
│   ├── xgboost/       process-feature models (XGBoost ALL / LEN / NOLEN) and OLS baselines
│   ├── deberta/       DeBERTa-v3 FULL and HYBRID (+ evaluation per time point)
│   └── combined/      DeBERTa + keystroke features + time-point one-hot
├── analysis/          descriptive statistics, correlations, agreement, plots
├── results/           small CSV/XLSX summaries behind the paper tables
├── docs/              keystroke variable codebooks / definitions
├── data/              not committed (see data/README.md)
└── COPY_LOG.md        origin of every file, skipped files, hardcoded paths, missing parts
```

## Pipeline (run in this order)

| Step | Script | Output |
|---|---|---|
| 1 | `preprocessing/01_extract_keystroke_features.py` | `pisa_keylog_statistics(_normalised).csv` (full essays) |
| 2 | `preprocessing/02_extract_features_at_timepoints.py` | text reconstruction + features at 5/10/15/20/25 min and full (`keylog_stats_*.csv`) |
| 3 | `preprocessing/03_create_splits.py` | train/dev/test per time point and condition |
| 4 | `preprocessing/04_split_check.py` | overlap check between splits (filename and user level) |
| 5 | `preprocessing/05_combine_timepoints.py` | all time points stacked (`splits/combined/`) |
| 6 | `preprocessing/06_create_hybrid_train.py` | hybrid training set (full text + one random snapshot per essay) |
| 7 | `preprocessing/07_text_lookup_full.py` | `text_lookup.tsv` (full essays) |
| 8 | `preprocessing/08_text_lookup_hybrid.py` | `text_hybrid_lookup.tsv` |
| 9 | `preprocessing/09_text_lookup_all_timepoints.py` | `text_mega_lookup.tsv` (all 6 time points) |
| — | `models/xgboost/xgboost_models.py`, `models/xgboost/ols_baseline.py` | process-feature models |
| — | `models/deberta/train_deberta_full.py`, `models/deberta/train_deberta_hybrid.ipynb` | DeBERTa FULL / HYBRID |
| — | `models/combined/train_deberta_combined.ipynb` | DeBERTa + keystroke features (COMBINED) |
| — | `models/deberta/predict_evaluate.ipynb` | evaluation per time point and condition |

Paths in the scripts are still the original ones (`BEA Paper/...`, `Originaldaten/...`);
they are listed in `COPY_LOG.md`. The DeBERTa notebooks were run on Google Colab.

## Paper table → script

| Paper | Script | Result file |
|---|---|---|
| Table 1: descriptive statistics by condition | `analysis/descriptive_statistics.py` | `results/table1_descriptive_stats.csv` |
| Table 2: score distribution across splits | `analysis/split_score_distribution.py`, `preprocessing/03_create_splits.py` | `results/table2_split_summary.csv` |
| Table 3: hybrid training set | `preprocessing/06_create_hybrid_train.py` (printed only) | TODO |
| 3.3.1 Silver standard (LR on 220 MEWS features) | TODO: training code missing; analysis: `silver_standard/analyse_silver_standard.py` | `results/sec3.3.1_silver_standard_summary.csv`, `silver_standard/mews_lr_model_results.md` |
| 3.3.2 Gold standard (429 essays, α = 0.79, QWK silver vs gold = 0.76) | `analysis/gold_vs_silver_agreement.py` | `results/sec3.3.2_gold_vs_silver_agreement.csv` |
| 3.4 Text snapshots at 5/10/15/20/25 min | `preprocessing/02_extract_features_at_timepoints.py`, `preprocessing/07–09_*` | `results/sec3.4_snapshot_summary.csv` |
| Table 4: XGBoost QWK (ALL / LEN / NOLEN) | `models/xgboost/xgboost_models.py` | `results/table4_xgboost_results.csv`, `results/results_xgboost_ols.xlsx` |
| Table 5: DeBERTa FULL / HYBRID / COMBINED | `models/deberta/*`, `models/combined/train_deberta_combined.ipynb` | `results/table5_*.csv`, `results/results_deberta.xlsx` |
| Figures 2–4 | `analysis/plot_deberta_results.py` | reads `results/results_deberta.xlsx` |
| Table 6: OLS baselines | `models/xgboost/ols_baseline.py` | `results/table6_ols_results.csv` |
| Table 7: keystroke features | `preprocessing/01_extract_keystroke_features.py`, `preprocessing/02_*` | `docs/keystroke_variable_codebook.xlsx`, `docs/Keylog-Definitionen.xlsx`, `docs/keystroke_measures.xlsx` |
| Table 8: Pearson r of features with length and score | `analysis/feature_correlations.py` | TODO: output not available |

## Installation

```
pip install -r requirements.txt
```

---

## Pipeline details

The following description was the original README of this repository. Script names map as follows:
`bea_01b_timesplit_stats.py` → `preprocessing/02_extract_features_at_timepoints.py`,
`bea_02b_create_traindev.py` → `preprocessing/03_create_splits.py`,
`bea_03_linreg.py` → `models/xgboost/ols_baseline.py`,
`bea_00_statistics_for_paper.py` → `analysis/descriptive_statistics.py`.


Scripts used to reproduce the process-feature extraction, data splitting, and
linear regression baselines from the paper *"KEYSCORE — Keystroke-enhanced
Automated Essay Scoring"* (BEA 2026), based on the PISA 2025 FLA writing
process dataset.

### Pipeline overview

```
bea_01b_timesplit_stats.py   →   bea_02b_create_traindev.py   →   bea_03_linreg.py
   (feature extraction)             (train/dev/test splits)         (baseline models)

bea_00_statistics_for_paper.py   →   descriptive statistics (paper Table 1)
   (independent, reads raw JSON directly — not part of the modeling pipeline)
```

Run the three pipeline scripts in order. `bea_00_statistics_for_paper.py` is
standalone and only needed to reproduce the descriptive stats table.

### Input data format

Raw keystroke logs are expected as JSON event lists, one file per writing
session, organized in per-condition folders (e.g. `..._advertisement`,
`..._chat__advertisement`, `..._chat__teacher`, `..._teacher` — folder name
encodes task and whether the AI chat condition was active). Each event
carries at least `timestamp_rel` (ms since session start) and an `event`
type (`KeyDown`, `TextPasted`, `TextCut`, cursor-reposition events, ...).

A `holistic.csv` with `filename` and `prediction` columns provides the
(silver-standard) target scores, keyed by a user ID parsed from the filename.

---

### 1. `bea_01b_timesplit_stats.py` — Feature extraction

Extracts **25 process features** across six behavioral dimensions from the
raw keystroke logs, at multiple points during the writing session.

**Features extracted:**

| Category | Features |
|---|---|
| Temporal | total writing time, initial pause |
| Pauses / breaks | break count, total break time, mean break duration, break ratio (pause threshold: ≥ 2000 ms) |
| Bursts | burst count, mean burst length (chars), mean burst duration |
| Deletions | deletion count, deletion ratio, characters deleted |
| Production | final text length (chars/words), chars/minute, total keystrokes, process-to-product ratio |
| Navigation | cursor repositions, copy-paste count, linearity index, interface-area switches (+ time in editor/task/chat/chat-prompt for the chat condition) |

**Key design points — important when adapting this to a new dataset:**

- **Time-threshold truncation.** Events are cut off at each threshold
  (5/10/15/20/25 min, plus the untruncated full session) via
  `truncate_events()`, and the same extraction logic runs on the truncated
  event stream. This reconstructs the writer's state *as it was* at that
  point in time, not a resampled version of the final text.
- **Outlier filtering is applied once, on the full essay.** Sessions with a
  final text shorter than `MIN_CHARS` (10) or longer than `MAX_CHARS`
  (10,000) are dropped. The resulting filename set is then reused across
  *every* time threshold, so the sample is identical at every stage —
  otherwise per-threshold results would not be comparable.
- **Z-score normalization happens independently within each time
  threshold**, not globally. A 5-minute snapshot is normalized against the
  distribution of all 5-minute snapshots, not against completed essays —
  this keeps the features interpretable as "relative to this writing
  stage" rather than "relative to the finished text."

**Output:** `BEA Paper/split_statistics/keylog_stats_{threshold}.csv` (raw)
and `keylog_stats_{threshold}_normed.csv` (z-scored), for
`threshold ∈ {05min, 10min, 15min, 20min, 25min, full}`.

---

### 2. `bea_02b_create_traindev.py` — Train / dev / test splits

Builds train/dev/test splits per time threshold and per chat condition
(`wo_chat`, `withchat`), using the normalized feature files from step 1.

- **Test set is fixed** from an externally provided stratified sample
  (`stratified_sample_proportional_500.csv`) and reused identically across
  all thresholds, so results stay comparable across the writing timeline.
- Remaining data is split **90/10 into train/dev**, stratified on the
  holistic score binned to the nearest 0.5.
- Score bins with fewer than 2 examples are merged into their nearest
  neighboring bin before stratification (avoids `train_test_split` errors
  on rare bins).

**Output:** `BEA Paper/splits/{threshold}/{wo_chat,withchat}_{train,dev,test}.csv`
plus a `split_summary.csv` with per-split sample counts.

---

### 3. `bea_03_linreg.py` — Linear regression baselines

Trains and evaluates ordinary least squares regression models predicting
the holistic score, across every threshold × condition combination.

- **Features are taken directly from the split CSVs** — every numeric
  column except a fixed metadata exclusion list (`filename`, `task`,
  `chat_used`, `holistic_score`, `final_text_length_word` as redundant with
  the character-length version). No separate feature-selection step exists;
  whatever step 1 extracted becomes the model's feature set.
- Three feature configurations are run per threshold/condition:
  - `all` — every keystroke feature
  - `length` — only `final_text_length_char`
  - `no_length` — every feature except text length

  This isolates how much predictive power comes from raw text length versus
  the remaining behavioral signal.
- **Metrics:** Pearson r, Spearman ρ, MAE, RMSE, and QWK (scores rounded to
  the nearest 0.5 before computing QWK).

**Output:** one model (`model.joblib`), config, and predictions file per
threshold/condition/mode under `BEA Paper/models/`, plus a combined
`results_summary.csv`.

---

### 4. `bea_00_statistics_for_paper.py` — Descriptive statistics (standalone)

Reads the raw JSON logs directly (not the outputs of step 1–3) and produces
the descriptive dataset table (paper Table 1): N, text length in
chars/words, sentence count, holistic score (mean/median/% at max score),
writing time, active writing ratio, and total keystrokes — split by chat
condition. Uses the same outlier thresholds (10–10,000 characters) as the
main pipeline, kept independent so the descriptive numbers don't depend on
the modeling pipeline having been run first.

**Output:** `BEA Paper/split_statistics/dataset_descriptive_stats.csv` and
a ready-to-include `.tex` table.

---

### Adapting to a new dataset

- Update `FOLDERS`, `DEFAULT_BASE_DIR`, and the folder-name parsing in
  `parse_folder_info()` to match your condition/task naming scheme.
- `MIN_CHARS` / `MAX_CHARS` (outlier bounds) and `BREAK_THRESHOLD_MS`
  (pause definition) are dataset-specific constants — recheck them against
  your text-length and typing-speed distributions rather than reusing the
  defaults as-is.
- If your dataset has no time-threshold design, step 1 can be simplified by
  dropping the threshold loop and `truncate_events()` call entirely and
  extracting features once per session.
