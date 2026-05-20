import sys
import subprocess
import pandas as pd
import numpy as np
import tempfile
import os

bedfile = sys.argv[1]
exonbed = sys.argv[2]
outfile = sys.argv[3]

# Count exon BED columns
with open(exonbed) as f:
    exon_cols = len(f.readline().strip().split())

# bedtools output columns:
# [fragment BED cols] + [exon BED cols]
FRAG_COLS = 3
GENE_COL = FRAG_COLS + exon_cols - 1

# Temporary file
tmp = tempfile.NamedTemporaryFile(delete=False)

# Run bedtools intersect
subprocess.run([
    "bedtools", "intersect",
    "-a", bedfile,
    "-b", exonbed,
    "-wa", "-wb"
], stdout=open(tmp.name, "w"), check=True)

# Load intersection
df = pd.read_csv(tmp.name, sep="\t", header=None)

if df.empty:
    pd.DataFrame(columns=["gene","E1SE","n_fragments"]).to_csv(outfile, index=False)
    os.unlink(tmp.name)
    sys.exit(0)

# Fragment length
df["fraglen"] = df[2] - df[1]

# Gene column (robust)
df["gene"] = df[GENE_COL]

results = []

for gene, g in df.groupby("gene"):
    counts = g["fraglen"].value_counts()
    n = counts.sum()

    if n < 500:
        continue

    p = counts / n
    entropy = -(p * np.log(p)).sum()

    results.append((gene, entropy, n))

out = pd.DataFrame(results, columns=["gene","E1SE","n_fragments"])
out.to_csv(outfile, index=False)

os.unlink(tmp.name)
