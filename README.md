# KEYSCORE — Keystroke-Enhanced Automated Essay Scoring

Scripts used to reproduce the process-feature extraction, data splitting, and
linear regression baselines from the paper *"KEYSCORE — Keystroke-enhanced
Automated Essay Scoring"* (BEA 2026), based on the PISA 2025 FLA writing
process dataset.

## Pipeline overview

```
bea_01b_timesplit_stats.py   →   bea_02b_create_traindev.py   →   bea_03_linreg.py
   (feature extraction)             (train/dev/test splits)         (baseline models)

bea_00_statistics_for_paper.py   →   descriptive statistics (paper Table 1)
   (independent, reads raw JSON directly — not part of the modeling pipeline)
```

Run the three pipeline scripts in order. `bea_00_statistics_for_paper.py` is
standalone and only needed to reproduce the descriptive stats table.

## Input data format

Raw keystroke logs are expected as JSON event lists, one file per writing
session, organized in per-condition folders (e.g. `..._advertisement`,
`..._chat__advertisement`, `..._chat__teacher`, `..._teacher` — folder name
encodes task and whether the AI chat condition was active). Each event
carries at least `timestamp_rel` (ms since session start) and an `event`
type (`KeyDown`, `TextPasted`, `TextCut`, cursor-reposition events, ...).

A `holistic.csv` with `filename` and `prediction` columns provides the
(silver-standard) target scores, keyed by a user ID parsed from the filename.

---

## 1. `bea_01b_timesplit_stats.py` — Feature extraction

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

## 2. `bea_02b_create_traindev.py` — Train / dev / test splits

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

## 3. `bea_03_linreg.py` — Linear regression baselines

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

## 4. `bea_00_statistics_for_paper.py` — Descriptive statistics (standalone)

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

## Adapting to a new dataset

- Update `FOLDERS`, `DEFAULT_BASE_DIR`, and the folder-name parsing in
  `parse_folder_info()` to match your condition/task naming scheme.
- `MIN_CHARS` / `MAX_CHARS` (outlier bounds) and `BREAK_THRESHOLD_MS`
  (pause definition) are dataset-specific constants — recheck them against
  your text-length and typing-speed distributions rather than reusing the
  defaults as-is.
- If your dataset has no time-threshold design, step 1 can be simplified by
  dropping the threshold loop and `truncate_events()` call entirely and
  extracting features once per session.
