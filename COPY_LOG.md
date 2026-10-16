# Copy log

Created 2026-10-16. Source folder: `KEYSCORE/` (left unchanged; files were only copied).
All targets were checked against their sources by SHA-256: **100 of 100 identical** (before clearing notebook outputs).

Afterwards, cell outputs and execution counts were cleared in the 3 notebooks
(`models/combined/train_deberta_combined.ipynb`, `models/deberta/train_deberta_hybrid.ipynb`,
`models/deberta/predict_evaluate.ipynb`), so their hashes now differ from the values below.

## Moved inside the repo (`git mv`)

These files were already in the repo and are identical to local scripts
(`python/bea_01b_timesplit_stats.py`, `python/bea_02b_create_traindev.py`, `python/bea_03_linreg-py.py`,
`python/bea_05_xgboost.py`; the notebook has the same code as `deberta/fusion_modell/deberta_with_features.ipynb`,
only the cell outputs differed). The local copies were not copied again.

| Old repo path | New repo path | Bytes | SHA-256 (first 16) |
|---|---|---|---|
| `data extraction/01_timesplit_stats.py` | `preprocessing/02_extract_features_at_timepoints.py` | 22808 | 013295b13556e56c |
| `data extraction/02_create_traindev.py` | `preprocessing/03_create_splits.py` | 7504 | f0fe20a310b4cfdb |
| `data extraction/03_linreg-py.py` | `models/xgboost/ols_baseline.py` | 10310 | db96a29078b5a2c7 |
| `data extraction/04_xgboost.py` | `models/xgboost/xgboost_models.py` | 13550 | 5b2baece0f32356b |
| `deberta_with_features.ipynb` | `models/combined/train_deberta_combined.ipynb` | 281149 | e5a9da6d0a12d95d |

## Copied (source → target)

| Source (`KEYSCORE/…`) | Target (`git/…`) | Bytes | SHA-256 (first 16) | Committed |
|---|---|---|---|---|
| `python/bea_01_keystats_outliers.py` | `preprocessing/01_extract_keystroke_features.py` | 16700 | b58aa185774c2ae8 | yes |
| `split_check.py` | `preprocessing/04_split_check.py` | 1436 | 9ae2268244f3a05c | yes |
| `splits/join_statistics_into_megafile.py` | `preprocessing/05_combine_timepoints.py` | 1054 | 38e0f7bfb150ebfa | yes |
| `splits/bea_create_hybrid_train.py` | `preprocessing/06_create_hybrid_train.py` | 6620 | 21c754eb159b6e3d | yes |
| `python/bea_06_create_transformer_data.py` | `preprocessing/07_text_lookup_full.py` | 5581 | 12ba838457dcc96b | yes |
| `python/bea_tf_hybrid_text_lookup.py` | `preprocessing/08_text_lookup_hybrid.py` | 9194 | 54b84ed6b080d53d | yes |
| `python/bea_megatsv_lookup.py` | `preprocessing/09_text_lookup_all_timepoints.py` | 7686 | 426de1412429e1d8 | yes |
| `mews_LR/analyse_silver_standard.py` | `silver_standard/analyse_silver_standard.py` | 4675 | f8d2a6f6ac01408c | yes |
| `mews_LR/results.md` | `silver_standard/mews_lr_model_results.md` | 1766 | dea97337cb4e0ef9 | yes |
| `python/bea_07_traindeberta.py` | `models/deberta/train_deberta_full.py` | 15018 | 91c0b7d1f27da54f | yes |
| `deberta/hybrid/deberta_training.ipynb` | `models/deberta/train_deberta_hybrid.ipynb` | 54231 | 7b81656614d687f0 | yes |
| `deberta/predict_evaluate.ipynb` | `models/deberta/predict_evaluate.ipynb` | 19863 | fd069f6895540b46 | yes |
| `deberta/fusion_modell/feature_config.json` | `models/combined/feature_config.json` | 851 | 9697efc3daf80b80 | yes |
| `python/bea_00_statistics_for_paper.py` | `analysis/descriptive_statistics.py` | 15309 | e1a847e26c4e9356 | yes |
| `python/bea_analyse_traindevtest.py` | `analysis/split_score_distribution.py` | 946 | 82f1c0aca66c27c7 | yes |
| `python/bea_review_correlations.py` | `analysis/feature_correlations.py` | 8527 | 434a619bfdcc44f9 | yes |
| `python/bea_compare433withdaniel.py` | `analysis/gold_vs_silver_agreement.py` | 5515 | 7d412c9e8fe04c8e | yes |
| `deberta/plot_updated.py` | `analysis/plot_deberta_results.py` | 9770 | a97fde5415285017 | yes |
| `keystroke_variable_codebook.xlsx` | `docs/keystroke_variable_codebook.xlsx` | 7284 | 9ad6d488f03def7b | yes |
| `Keylog-Definitionen.xlsx` | `docs/Keylog-Definitionen.xlsx` | 10276 | 91a6c8093d4e74c5 | yes |
| `keystroke_measures.xlsx` | `docs/keystroke_measures.xlsx` | 11198 | 393cc19c4e439899 | yes |
| `split_statistics/dataset_descriptive_stats.csv` | `results/table1_descriptive_stats.csv` | 556 | 378d81be7e72f3ea | yes |
| `splits/split_summary.csv` | `results/table2_split_summary.csv` | 486 | 87c3abfeba1428f9 | yes |
| `split_statistics/threshold_summary.csv` | `results/sec3.4_snapshot_summary.csv` | 673 | 974e021d925648c5 | yes |
| `models_xgb/results_summary.csv` | `results/table4_xgboost_results.csv` | 8951 | cf4768f0c4153914 | yes |
| `deberta/full/results/metrics_summary.csv` | `results/table5_deberta_full_metrics.csv` | 4251 | a79ccec6c7d45aa6 | yes |
| `deberta/hybrid/results_test/metrics_summary.csv` | `results/table5_deberta_hybrid_metrics_test.csv` | 4210 | aba209c989cb109f | yes |
| `deberta/hybrid/results_dev/metrics_summary.csv` | `results/table5_deberta_hybrid_metrics_dev.csv` | 4227 | 26148c3f09e4457d | yes |
| `deberta/fusion_modell/results/metrics_summary.csv` | `results/table5_deberta_combined_metrics.csv` | 4175 | ccff03b3902f3332 | yes |
| `deberta/results_deberta.xlsx` | `results/results_deberta.xlsx` | 20573 | 5db384127302dd7d | yes |
| `models/results_summary.csv` | `results/table6_ols_results.csv` | 7513 | 81f7baaaadae3c4b | yes |
| `models/results.xlsx` | `results/results_xgboost_ols.xlsx` | 38211 | a29a6b9aa8c52a0f | yes |
| `mews_LR/silver_standard_summary.csv` | `results/sec3.3.1_silver_standard_summary.csv` | 558 | e9fff534798290dc | yes |
| `comparison_AnnotatorvsDANIEL_results.csv` | `results/sec3.3.2_gold_vs_silver_agreement.csv` | 461 | e4f0b5e904f95e95 | yes |
| `pisa_keylog_statistics.csv` | `data/keystroke_features/pisa_keylog_statistics.csv` | 1235224 | c56c80db924fb7fa | no (data/) |
| `pisa_keylog_statistics_normalised.csv` | `data/keystroke_features/pisa_keylog_statistics_normalised.csv` | 2335186 | 02cd8b50dbe42724 | no (data/) |
| `text_lookup.tsv` | `data/text_lookups/text_lookup.tsv` | 9913083 | 84db5e521b8054ec | no (data/) |
| `text_hybrid_lookup.tsv` | `data/text_lookups/text_hybrid_lookup.tsv` | 8428316 | c3907b3ad10b98e4 | no (data/) |
| `text_mega_lookup.tsv` | `data/text_lookups/text_mega_lookup.tsv` | 44490175 | acafa918699828cd | no (data/) |
| `split_statistics/keylog_stats_05min.csv` | `data/splits/split_statistics/keylog_stats_05min.csv` | 1144948 | 5c17410955eb91e3 | no (data/) |
| `split_statistics/keylog_stats_05min_normed.csv` | `data/splits/split_statistics/keylog_stats_05min_normed.csv` | 2291113 | 0e3fd596658a996d | no (data/) |
| `split_statistics/keylog_stats_10min.csv` | `data/splits/split_statistics/keylog_stats_10min.csv` | 1207751 | 45445133a06dd183 | no (data/) |
| `split_statistics/keylog_stats_10min_normed.csv` | `data/splits/split_statistics/keylog_stats_10min_normed.csv` | 2293311 | efc20db76e4bdd54 | no (data/) |
| `split_statistics/keylog_stats_15min.csv` | `data/splits/split_statistics/keylog_stats_15min.csv` | 1229582 | 0d89bb1c05cf4b52 | no (data/) |
| `split_statistics/keylog_stats_15min_normed.csv` | `data/splits/split_statistics/keylog_stats_15min_normed.csv` | 2281027 | f359fa748f1c0265 | no (data/) |
| `split_statistics/keylog_stats_20min.csv` | `data/splits/split_statistics/keylog_stats_20min.csv` | 1244876 | 831bad5545dc193f | no (data/) |
| `split_statistics/keylog_stats_20min_normed.csv` | `data/splits/split_statistics/keylog_stats_20min_normed.csv` | 2283677 | cb9e6664d11e5d44 | no (data/) |
| `split_statistics/keylog_stats_25min.csv` | `data/splits/split_statistics/keylog_stats_25min.csv` | 1252166 | de9941a9ab498bd2 | no (data/) |
| `split_statistics/keylog_stats_25min_normed.csv` | `data/splits/split_statistics/keylog_stats_25min_normed.csv` | 2285189 | 31374dda4a7d377c | no (data/) |
| `split_statistics/keylog_stats_full.csv` | `data/splits/split_statistics/keylog_stats_full.csv` | 1255701 | bb0083b611edf280 | no (data/) |
| `split_statistics/keylog_stats_full_normed.csv` | `data/splits/split_statistics/keylog_stats_full_normed.csv` | 2288901 | 72b89f7094d3a997 | no (data/) |
| `splits/05min/withchat_dev.csv` | `data/splits/05min/withchat_dev.csv` | 100586 | b0c7cdb672237308 | no (data/) |
| `splits/05min/withchat_test.csv` | `data/splits/05min/withchat_test.csv` | 123915 | 03cd57d0c62a2997 | no (data/) |
| `splits/05min/withchat_train.csv` | `data/splits/05min/withchat_train.csv` | 897524 | b549944f699bf9b8 | no (data/) |
| `splits/05min/wo_chat_dev.csv` | `data/splits/05min/wo_chat_dev.csv` | 98093 | 53baaf990473ec43 | no (data/) |
| `splits/05min/wo_chat_test.csv` | `data/splits/05min/wo_chat_test.csv` | 122166 | ce6156e81ca54712 | no (data/) |
| `splits/05min/wo_chat_train.csv` | `data/splits/05min/wo_chat_train.csv` | 878250 | fb027de0f4a4906a | no (data/) |
| `splits/10min/withchat_dev.csv` | `data/splits/10min/withchat_dev.csv` | 100195 | 0470adc7eb2014b1 | no (data/) |
| `splits/10min/withchat_test.csv` | `data/splits/10min/withchat_test.csv` | 123432 | d79d316e6989b6af | no (data/) |
| `splits/10min/withchat_train.csv` | `data/splits/10min/withchat_train.csv` | 893553 | 9cc0b54425304d69 | no (data/) |
| `splits/10min/wo_chat_dev.csv` | `data/splits/10min/wo_chat_dev.csv` | 97713 | 1f64292055063278 | no (data/) |
| `splits/10min/wo_chat_test.csv` | `data/splits/10min/wo_chat_test.csv` | 121774 | 01fb00d3ed646bf8 | no (data/) |
| `splits/10min/wo_chat_train.csv` | `data/splits/10min/wo_chat_train.csv` | 874899 | 6ef3f634246478b0 | no (data/) |
| `splits/15min/withchat_dev.csv` | `data/splits/15min/withchat_dev.csv` | 100353 | 07a800b7c6690edb | no (data/) |
| `splits/15min/withchat_test.csv` | `data/splits/15min/withchat_test.csv` | 123590 | 8a1e60ca6fcf28fe | no (data/) |
| `splits/15min/withchat_train.csv` | `data/splits/15min/withchat_train.csv` | 895071 | 0770da4a4595bb5f | no (data/) |
| `splits/15min/wo_chat_dev.csv` | `data/splits/15min/wo_chat_dev.csv` | 97911 | 17b9913f8faa2c23 | no (data/) |
| `splits/15min/wo_chat_test.csv` | `data/splits/15min/wo_chat_test.csv` | 121976 | 6cd0429ce976d707 | no (data/) |
| `splits/15min/wo_chat_train.csv` | `data/splits/15min/wo_chat_train.csv` | 876497 | c9dbb3238068b131 | no (data/) |
| `splits/20min/withchat_dev.csv` | `data/splits/20min/withchat_dev.csv` | 100243 | b61cf58ce5772a28 | no (data/) |
| `splits/20min/withchat_test.csv` | `data/splits/20min/withchat_test.csv` | 123619 | b07ee47af803e1f7 | no (data/) |
| `splits/20min/withchat_train.csv` | `data/splits/20min/withchat_train.csv` | 894795 | d33806e7688fee6c | no (data/) |
| `splits/20min/wo_chat_dev.csv` | `data/splits/20min/wo_chat_dev.csv` | 97872 | c42ef53ff3520889 | no (data/) |
| `splits/20min/wo_chat_test.csv` | `data/splits/20min/wo_chat_test.csv` | 121907 | 1f2dff4e98e3a683 | no (data/) |
| `splits/20min/wo_chat_train.csv` | `data/splits/20min/wo_chat_train.csv` | 876121 | 860e1f4e2a047455 | no (data/) |
| `splits/25min/withchat_dev.csv` | `data/splits/25min/withchat_dev.csv` | 100455 | 36e2f6175ab47b78 | no (data/) |
| `splits/25min/withchat_test.csv` | `data/splits/25min/withchat_test.csv` | 123862 | d4a214a998029db2 | no (data/) |
| `splits/25min/withchat_train.csv` | `data/splits/25min/withchat_train.csv` | 896855 | ba2a9ad4ec034b3b | no (data/) |
| `splits/25min/wo_chat_dev.csv` | `data/splits/25min/wo_chat_dev.csv` | 98141 | 8a62d2c9b8ccddb2 | no (data/) |
| `splits/25min/wo_chat_test.csv` | `data/splits/25min/wo_chat_test.csv` | 122240 | 3d5010952b38363b | no (data/) |
| `splits/25min/wo_chat_train.csv` | `data/splits/25min/wo_chat_train.csv` | 878754 | 969f401245623d6a | no (data/) |
| `splits/full/withchat_dev.csv` | `data/splits/full/withchat_dev.csv` | 100224 | 052b94edea90d723 | no (data/) |
| `splits/full/withchat_test.csv` | `data/splits/full/withchat_test.csv` | 123548 | 7e8b927c5c2d3b6b | no (data/) |
| `splits/full/withchat_train.csv` | `data/splits/full/withchat_train.csv` | 895336 | f3aae65ce9da597f | no (data/) |
| `splits/full/wo_chat_dev.csv` | `data/splits/full/wo_chat_dev.csv` | 97949 | 6d59c98410fc02c0 | no (data/) |
| `splits/full/wo_chat_test.csv` | `data/splits/full/wo_chat_test.csv` | 122007 | 1e2f64c809bde962 | no (data/) |
| `splits/full/wo_chat_train.csv` | `data/splits/full/wo_chat_train.csv` | 877167 | 253053b86f75d862 | no (data/) |
| `splits/hybrid/withchat_train.csv` | `data/splits/hybrid/withchat_train.csv` | 1799772 | f36066361b5667b3 | no (data/) |
| `splits/hybrid/wo_chat_train.csv` | `data/splits/hybrid/wo_chat_train.csv` | 1770815 | 89e4ea3e66961194 | no (data/) |
| `splits/combined/withchat_dev.csv` | `data/splits/combined/withchat_dev.csv` | 605698 | 8f6b15bda3508ed2 | no (data/) |
| `splits/combined/withchat_test.csv` | `data/splits/combined/withchat_test.csv` | 747041 | d5e285fd02c9d617 | no (data/) |
| `splits/combined/withchat_train.csv` | `data/splits/combined/withchat_train.csv` | 5425448 | 9039e3ef668da16f | no (data/) |
| `splits/combined/wo_chat_dev.csv` | `data/splits/combined/wo_chat_dev.csv` | 591126 | aff6208d63c0eea1 | no (data/) |
| `splits/combined/wo_chat_test.csv` | `data/splits/combined/wo_chat_test.csv` | 736986 | 289f3854f655e1d2 | no (data/) |
| `splits/combined/wo_chat_train.csv` | `data/splits/combined/wo_chat_train.csv` | 5312485 | be9511d20ab17afd | no (data/) |

## Skipped

### Model checkpoints / weights (and files > 50 MB)

| File | Bytes |
|---|---|
| `deberta/full/model.safetensors` | 737716172 |
| `deberta/full/training_args.bin` | 5265 |
| `deberta/fusion_modell/checkpoint_7750/model.safetensors` | 738123908 |
| `deberta/fusion_modell/checkpoint_7750/optimizer.pt` | 1476372299 |
| `deberta/fusion_modell/checkpoint_7750/rng_state.pth` | 14645 |
| `deberta/fusion_modell/checkpoint_7750/scheduler.pt` | 1465 |
| `deberta/fusion_modell/checkpoint_7750/training_args.bin` | 5265 |
| `deberta/fusion_modell/model.pt` | 738171523 |
| `deberta/fusion_modell/v2/model.safetensors` | 738123940 |
| `deberta/fusion_modell/v2/rng_state.pth` | 14645 |
| `deberta/fusion_modell/v2/scheduler.pt` | 1465 |
| `deberta/fusion_modell/v2/training_args.bin` | 5329 |
| `deberta/hybrid/model.safetensors` | 737716172 |
| `deberta/hybrid/training_args.bin` | 5265 |

Also not copied: the rest of `deberta/fusion_modell/checkpoint_7750/` and `deberta/fusion_modell/v2/`
(`config.json`, `trainer_state.json`), DeBERTa `config.json` / `tokenizer*.json` / `training_log.csv`,
and `model.joblib` / `config.json` of every XGBoost and OLS run (`models_xgb/`, `models/`).

### Excluded by rule

| Item | Reason |
|---|---|
| `PAPERS/` | literature |
| `Sections.docx` | excluded by user |
| `090226.txt`, `10.03.26.txt`, `170226.txt`, `190326.txt`, `scripts.txt` | personal dated notes |
| `text_mega_lookup.7z`, `splits/keystroke_csvs.zip`, `plots/with_chat/correlations_with_chat.zip`, `plots/without_chat/correlations_wo_chat.zip` | archives |
| `user-0103560_BIZ#…#1_corr/` (`1.txt` … `26.txt`) | single student session |
| All per-essay prediction CSVs (`deberta/**/predictions_*.csv`, `models/**/predictions_*.csv`, `models_xgb/**/predictions_*.csv`, `mews_LR/predictions_*.csv`) | per-essay outputs, most with student IDs (`filename` column); user chose not to copy |

### Duplicates (identical by hash, copied once)

| Skipped | Identical to (copied) |
|---|---|
| `splits/forfusion/withchat_train.csv` | `splits/hybrid/withchat_train.csv` |
| `splits/forfusion/wo_chat_train.csv` | `splits/hybrid/wo_chat_train.csv` |
| `splits/forfusion/{withchat,wo_chat}_{dev,test}.csv` | `splits/combined/{withchat,wo_chat}_{dev,test}.csv` |
| `Daniel_vs_Annotator.txt` | same numbers as `comparison_AnnotatorvsDANIEL_results.csv` |
| `python/data extraction/*.py` | the moved repo files |

### Older versions (user decision)

| Skipped | Used instead |
|---|---|
| `python/bea_keylog_stats.py`, `python/bea_remove_outliers.py`, `python/bea_merge_csvs.py` | `bea_01_keystats_outliers.py` |
| `python/bea_create_train_testset.py`, `python/bea_create_traindev_basedon433.py` | `bea_02b_create_traindev.py` |
| `python/bea_linreg_base.py` (+ outputs `models/withchat/`, `models/wo_chat/`) | `bea_03_linreg-py.py` |
| `python/bea_review_correlations_fromcsv.py` | `bea_review_correlations.py` |
| `deberta/plot_results.py` (+ `plot_cool/simple/smart.png`) | `deberta/plot_updated.py` |
| `deberta/keylog_hybrid/train_deberta_with_features.ipynb` | `deberta/fusion_modell/deberta_with_features.ipynb` |
| `splits/{wo_chat,withchat}_{train,dev,test}.csv` (2026-02-11, no score column), `splits/{05..30}min.csv` and `splits/untitled0.py` (2026-07-08) | `splits/<threshold>/` |

### Not part of the paper (user decision)

`weighted_fusion.py`, `fusion_results/`, `plots/`, `plots_experimental/`, `deberta/plots/`, `deberta/results transformerdeberta.txt`,
`split_statistics/threshold summary.xlsx`, `split_statistics/dataset_descriptive_stats.tex`, `models/results_tables.tex`,
`models/plots/`, `mews_LR/silver_standard_analysis.png`, `splits/checkformissingvalues.py`, and in `python/`:
`analyse_copy_paste.py`, `analyse_pisa433.py`, `bea00_analyse_truncation.py`, `bea_01b_error_analysis.py`,
`bea_04_results_to_latex.py`, `bea_confusionmatricex_linreg.py`, `bea_csv_diagnostic.py`, `bea_density_plots.py`,
`bea_experimental_keylog_stats.py`, `bea_extradata_excel.py`, `bea_merge_and_plot_exp.py`, `bea_plots_usersplit.py`,
`bea_text_length.py`, `cluster_v1.py`, `cluster_with_plots.py`, `cluster_without_outlier.py`, `clusters_separate.py`,
`clusters_separate_withcopypaste.py`, `extract_chat_verlauf.py`, `extract_statistics.py`,
`extract_statistics_withcopypaste.py`, `find_optimal_clustersize.py`, `find_outliers_textlength.py`,
`get_plot_for_bea.py`, `keylog_analyser.py`, `keylog_timestamp.py`, `read_jsonfile.py`, `reorder_chat_verlauf.py`,
`timeseries_visual.py`.

## Secret scan

Searched all code for `hf_`, `sk-`, `token`, `api_key`: **no secrets found**, no `.env` files.
`HF_TOKEN` only appeared in Colab warning messages inside notebook outputs (now cleared).

## Hardcoded paths (not changed)

Absolute paths:

| File | Line | Path |
|---|---|---|
| `preprocessing/04_split_check.py` | 4 | `C:\Users\SUNPN861\Desktop\Data\PISA\BEA Paper\splits` (also reads the old top-level split files) |

Relative paths from the original project layout (run from the folder that contains `BEA Paper/` and `Originaldaten/`):

| File | Paths |
|---|---|
| `preprocessing/01_extract_keystroke_features.py` | `Originaldaten/allUser_corr` |
| `preprocessing/02_extract_features_at_timepoints.py` | `Originaldaten/allUser_corr`, `BEA Paper/split_statistics`, `holistic.csv` |
| `preprocessing/03_create_splits.py` | `BEA Paper/split_statistics`, `stratified_sample_proportional_500.csv`, `BEA Paper/splits` |
| `preprocessing/05_combine_timepoints.py` | `.` (run inside `splits/`), `combined/` |
| `preprocessing/06_create_hybrid_train.py` | `<base_dir>/full`, `<base_dir>/<timepoint>`, `<base_dir>/hybrid` |
| `preprocessing/07_text_lookup_full.py` | `Originaldaten/allUser_corr`, `BEA Paper/text_lookup.tsv` |
| `preprocessing/08_text_lookup_hybrid.py` | `Originaldaten/allUser_corr`, `BEA Paper/splits/hybrid`, `BEA Paper/text_hybrid_lookup.tsv` |
| `preprocessing/09_text_lookup_all_timepoints.py` | `Originaldaten/allUser_corr`, `BEA Paper/text_mega_lookup.tsv` |
| `silver_standard/analyse_silver_standard.py` | `predictions_{holistic,content,language_quality,organization}.csv` |
| `models/xgboost/ols_baseline.py` | `BEA Paper/splits`, `BEA Paper/models` |
| `models/xgboost/xgboost_models.py` | `BEA Paper/splits`, `BEA Paper/models_xgb` |
| `models/deberta/train_deberta_full.py` | `BEA Paper/splits/full`, `BEA Paper/models_transformer/text_only` |
| `models/deberta/train_deberta_hybrid.ipynb` | `BEA Paper/hybrid`, `BEA Paper/models_transformer/text_only` |
| `models/deberta/predict_evaluate.ipynb` | `model`, `BEA Paper/text_mega_lookup.tsv`, `BEA Paper/splits/{wo_chat,withchat}_test.csv`, `BEA Paper/predictions` |
| `models/combined/train_deberta_combined.ipynb` | `BEA Paper/combined`, `BEA Paper/models_transformer/text_plus_features` |
| `analysis/descriptive_statistics.py` | `Originaldaten/allUser_corr`, `BEA Paper/split_statistics`, `holistic.csv` |
| `analysis/split_score_distribution.py` | `BEA Paper/splits/full` |
| `analysis/feature_correlations.py` | `pisa_keylog_statistics.csv`, `holistic.csv`, `feature_correlations_all_tasks.csv` |
| `analysis/gold_vs_silver_agreement.py` | `holistic.csv`, `Texts_extracted/iaa_results.xlsx` |
| `analysis/plot_deberta_results.py` | `results_deberta.xlsx`, `plots/` |

## Missing

Needed by the paper but not in the source folder:

- **MEWS linear-regression training code** (Section 3.3.1): only the analysis script and `results.md` exist.
- **`holistic.csv`**: silver-standard scores; input of preprocessing/02 and several analyses.
- **`stratified_sample_proportional_500.csv`**: the fixed test set (gold-standard sample).
- **`Texts_extracted/iaa_results.xlsx`**: gold scores of the two annotators (Section 3.3.2).
- **Raw keystroke logs** `Originaldaten/allUser_corr/` (input of the whole pipeline).
- **Table 3** (hybrid training set): no script writes these statistics (`06_create_hybrid_train.py` only prints them). TODO.
- **Table 8 output** (`feature_correlations_all_tasks.csv`) is not in the folder; it has to be regenerated.
- **COMBINED model**: a newer checkpoint `deberta/fusion_modell/v2/` (2026-07-06) exists without notebook or metrics;
  `results/table5_deberta_combined_metrics.csv` comes from the March run. TODO: confirm which run is in the paper.
- **Silver-standard predictions** `mews_LR/predictions_*.csv` were not copied (user decision), but `analyse_silver_standard.py` needs them.

## Note on the existing repo

`text_mega_lookup.7z` (essay texts + student IDs) is already tracked in the public repo and was left untouched.
The `.gitignore` rule `*.7z` does **not** untrack a file that is already committed.
