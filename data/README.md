# Data

Data available on request until September 2027, after which it will be published here.
Contact: schaller@leibniz-ipn.de

Everything in `data/` except this README is excluded from git (see `.gitignore`).
The files contain essay texts and student IDs (in the `filename` column).

## Expected structure

```
data/
├── README.md
├── keystroke_features/
│   ├── pisa_keylog_statistics.csv              # full-essay features (preprocessing/01)
│   └── pisa_keylog_statistics_normalised.csv   # z-scored version
├── text_lookups/
│   ├── text_lookup.tsv          # full essays           (preprocessing/07)
│   ├── text_hybrid_lookup.tsv   # hybrid training texts (preprocessing/08)
│   └── text_mega_lookup.tsv     # all 6 time points     (preprocessing/09)
├── splits/
│   ├── split_statistics/
│   │   ├── keylog_stats_{05min,10min,15min,20min,25min,full}.csv          (preprocessing/02)
│   │   └── keylog_stats_{05min,10min,15min,20min,25min,full}_normed.csv
│   ├── {05min,10min,15min,20min,25min,full}/      (preprocessing/03)
│   │   └── {wo_chat,withchat}_{train,dev,test}.csv
│   ├── hybrid/                                     (preprocessing/06)
│   │   └── {wo_chat,withchat}_train.csv
│   └── combined/                                   (preprocessing/05; all time points stacked)
│       └── {wo_chat,withchat}_{train,dev,test}.csv
└── annotations/
    ├── holistic.csv                               # silver-standard scores (columns: filename, prediction)
    ├── stratified_sample_proportional_500.csv     # fixed test set (gold-standard sample)
    └── iaa_results.xlsx                           # gold scores of the two annotators (429 essays)
```

Not included at all (input of the pipeline): the raw keystroke logs
(`Originaldaten/allUser_corr/<condition folder>/*.json`, one JSON event list per session).

Note: the scripts still use the original relative paths (`BEA Paper/...`,
`Originaldaten/...`); see `COPY_LOG.md` for the list.
The combined/fusion training used a folder `splits/forfusion/`, which is identical to
`hybrid/` (train) plus `combined/` (dev, test).
