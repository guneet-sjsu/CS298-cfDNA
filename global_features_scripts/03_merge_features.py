#!/usr/bin/env python3
import sys, os, pandas as pd

"""
Usage:
    python3 merge_features.py <features_dir> <label_file> <output_csv>

Example:
    python3 merge_features.py \
        /scratch/home/gbhogal/CS298/pipeline/features \
        /scratch/home/gbhogal/CS298/pipeline/train_labels.txt \
        /scratch/home/gbhogal/CS298/pipelinefeatures/features_train_merged.csv
"""

features_dir = sys.argv[1]
label_file   = sys.argv[2]
output_csv   = sys.argv[3]

# Load labels
labels = pd.read_csv(label_file, sep=r"[\t,]", engine="python",
                     names=["sample", "label"])
print(f"Loaded {len(labels)} labels")

# Storage for merged rows
merged_rows = []

# Extract unique sample IDs based on *_global.csv
global_files = sorted([f for f in os.listdir(features_dir)
                       if f.endswith("_global.csv")])

for gf in global_files:
    sample = gf.replace("_global.csv", "")
    sample = sample.replace(".bed", "")
    global_path  = os.path.join(features_dir, f"{sample}.bed_global.csv")
    context_path = os.path.join(features_dir, f"{sample}_features.csv")

    if not os.path.exists(context_path):
        print(f"⚠ Warning: missing context file for {sample}, skipping.")
        continue

    # Load feature tables
    df_global  = pd.read_csv(global_path)
    df_context = pd.read_csv(context_path)

    # Merge horizontally
    df = df_global.merge(df_context, on="sample", how="inner")

    # Add label
    match = labels[labels["sample"] == sample]
    if len(match) == 0:
        print(f"⚠ Warning: sample {sample} has no label, skipping.")
        continue

    df["label"] = match["label"].values[0]

    merged_rows.append(df)

# Combine all rows into one DataFrame
if len(merged_rows) == 0:
    raise ValueError("No feature files merged — check directory or file names.")

final_df = pd.concat(merged_rows, ignore_index=True)

print("Final merged shape:", final_df.shape)

# Drop any rows with remaining NaNs (usually only if annotation was empty)
final_df = final_df.dropna()
print("After dropping NaNs:", final_df.shape)

final_df.to_csv(output_csv, index=False)
print(f"Saved merged feature matrix → {output_csv}")
