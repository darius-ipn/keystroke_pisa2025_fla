import pandas as pd
from pathlib import Path

# === CONFIG ===
base_dir = Path(".")  # Change to your actual base directory
timepoints = ["05min", "10min", "15min", "20min", "25min", "full"]
files = ["withchat_dev.csv", "withchat_test.csv", "wo_chat_dev.csv", "wo_chat_test.csv", "wo_chat_train.csv", "withchat_train.csv"]
output_dir = Path("combined")
output_dir.mkdir(exist_ok=True)

# === CONCATENATE ===
for file in files:
    frames = []
    for tp in timepoints:
        path = base_dir / tp / file
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        df["source_timepoint"] = tp
        frames.append(df)
        print(f"Loaded {path}: {len(df)} rows")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = output_dir / file
        combined.to_csv(out_path, index=False)
        print(f"Saved {out_path}: {len(combined)} rows\n")
    else:
        print(f"No data found for {file}\n")