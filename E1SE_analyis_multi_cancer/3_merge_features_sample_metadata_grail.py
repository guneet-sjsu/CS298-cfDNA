import pandas as pd

# -----------------------------
# Load E1SE feature matrix
# -----------------------------
X = pd.read_csv("/scratch/home/gbhogal/CS298/cfDNA/matrices/e1se_matrix.csv", index_col=0)
X.index = X.index.astype(str)

# Normalize sample IDs from BED-derived names
X.index = (
    X.index
    .str.replace("collapsed", "", regex=False)
    .str.replace(".bed", "", regex=False)
)

X.index.name = "sample"

# -----------------------------
# Load sample metadata (GRAIL)
# -----------------------------
meta = pd.read_csv("sample_metadata_grail.csv")
meta["sample"] = meta["sample"].astype(str)

# -----------------------------
# Merge
# -----------------------------
df = X.reset_index().merge(meta, on="sample", how="left")

# -----------------------------
# HARD CHECK: labels must exist
# -----------------------------
n_missing = df["label"].isna().sum()

print("Total samples:", len(df))
print("Samples missing labels:", n_missing)

if n_missing > 0:
    missing = df.loc[df["label"].isna(), "sample"].tolist()[:20]
    raise ValueError(
        f"❌ Label merge failed for {n_missing} samples. "
        f"Examples: {missing}"
    )

# -----------------------------
# Save
# -----------------------------
df.to_csv("/scratch/home/gbhogal/CS298/cfDNA/matrices/e1se_with_labels.csv", index=False)
print("✅ Merge successful → /scratch/home/gbhogal/CS298/cfDNA/matrices/e1se_with_labels.csv")
