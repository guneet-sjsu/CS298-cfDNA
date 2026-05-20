#!/usr/bin/env python3
import sys
import numpy as np
import csv

bed = sys.argv[1]
out_csv = sys.argv[2]
sample = bed.split("/")[-1].replace(".bed", "")

lengths = []

with open(bed) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            start = int(parts[1])
            end = int(parts[2])
            L = end - start
            if 0 < L <= 500:  # enforce clean lengths
                lengths.append(L)
        except:
            continue

lengths = np.array(lengths)

results = {
    "sample": sample,
    "n_global": len(lengths),
    "median_len_global": float(np.median(lengths)),
    "short_frac_global": float(np.mean(lengths < 150)),
    "long_frac_global": float(np.mean(lengths > 220)),
}

# entropy of length distribution
bins = np.linspace(lengths.min(), lengths.max(), 50)
hist, _ = np.histogram(lengths, bins=bins, density=True)
from scipy.stats import entropy
results["entropy_len"] = float(entropy(hist + 1e-9))

with open(out_csv, "w") as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)

print("Saved global ->", out_csv)
