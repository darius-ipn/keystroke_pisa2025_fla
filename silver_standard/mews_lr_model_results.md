```
{'description': 'Model trained on MEWS content score prediction (regression).',
'data_split': '80/20 train-test split',
'random_seed': 42,
'target_variable': 'content',
'feature_start_column': 'count_words',
'num_features_full': 220,
'num_features_kept': 110,
'corr_threshold': 0.9,
'evaluation_metrics': {'RMSE': 0.7771,
'R2': 0.2953,
'Pearson_r': 0.5606,
'Pearson_p': 0.0,
'QWK': 0.513},
'min_value': 0.0,
'max_value': 6.0}
```

```
'description': 'Model trained on MEWS holistic score prediction (regression).',
'data_split': '80/20 train-test split',
'random_seed': 42,
'target_variable': 'Label',
'feature_start_column': 'count_words',
'num_features_full': 220,
'num_features_kept': 113,
'corr_threshold': 0.9,
'evaluation_metrics': {'RMSE': 0.4742,
'R2': 0.5962,
'Pearson_r': 0.7729,
'Pearson_p': 0.0,
'QWK': 0.7403},
'min_value': 0.0,
'max_value': 5.0}
```

```
{'description': 'Model trained on MEWS language_quality score prediction (regression).',
'data_split': '80/20 train-test split',
'random_seed': 42,
'target_variable': 'language_quality',
'feature_start_column': 'count_words',
'num_features_full': 220,
'num_features_kept': 112,
'corr_threshold': 0.9,
'evaluation_metrics': {'RMSE': 0.7678,
'R2': 0.5605,
'Pearson_r': 0.7525,
'Pearson_p': 0.0,
'QWK': 0.717},
'min_value': 0.0,
'max_value': 6.0}
```

```
{'description': 'Model trained on MEWS organization score prediction (regression).',
'data_split': '80/20 train-test split',
'random_seed': 42,
'target_variable': 'organization',
'feature_start_column': 'count_words',
'num_features_full': 220,
'num_features_kept': 112,
'corr_threshold': 0.9,
'evaluation_metrics': {'RMSE': 1.2714,
'R2': 0.2668,
'Pearson_r': 0.5402,
'Pearson_p': 0.0,
'QWK': 0.5043},
'min_value': 0.0,
'max_value': 6.0}
```