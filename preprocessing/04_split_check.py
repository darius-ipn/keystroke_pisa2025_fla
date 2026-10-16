import pandas as pd
from pathlib import Path

splits_dir = Path(r"C:\Users\SUNPN861\Desktop\Data\PISA\BEA Paper\splits")

# Load and combine splits
splits = {}
for split_name in ['train', 'dev', 'test']:
    dfs = []
    for prefix in ['wo_chat', 'withchat']:
        path = splits_dir / f"{prefix}_{split_name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded {path.name}: {len(df)} rows")
    splits[split_name] = pd.concat(dfs, ignore_index=True)

# Filename-level overlap
train_fns = set(splits['train']['filename'])
dev_fns = set(splits['dev']['filename'])
test_fns = set(splits['test']['filename'])

print(f"\nFilename-level overlap:")
print(f"  Train ∩ Dev:  {len(train_fns & dev_fns)}")
print(f"  Train ∩ Test: {len(train_fns & test_fns)}")
print(f"  Dev ∩ Test:   {len(dev_fns & test_fns)}")

# User-level overlap
def get_user(fn): return fn.split('_')[0]
train_users = set(get_user(fn) for fn in train_fns)
dev_users = set(get_user(fn) for fn in dev_fns)
test_users = set(get_user(fn) for fn in test_fns)

print(f"\nUser-level overlap:")
print(f"  Train ∩ Dev:  {len(train_users & dev_users)}")
print(f"  Train ∩ Test: {len(train_users & test_users)}")
print(f"  Dev ∩ Test:   {len(dev_users & dev_users)}")

print(f"\nUnique users: Train={len(train_users)}, Dev={len(dev_users)}, Test={len(test_users)}")