import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score

# -----------------------------
# Load model
# -----------------------------
obj = joblib.load(
    "/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/models/"
    "multinomial_elasticnet_all_lung_cancer.pkl"
)

model = obj["model"]
genes = obj["genes"]

print("Pipeline steps:", model.named_steps.keys())

lr = model.named_steps["clf"]
print("Classes:", lr.classes_)  # expected: ['healthy', 'cancer'] or similar

# -----------------------------
# Coefficient table (binary)
# -----------------------------
coef = pd.DataFrame(
    lr.coef_[0],
    index=genes,
    columns=["coefficient"]
).sort_values("coefficient", key=np.abs, ascending=False)

# Save coefficients
coef.to_csv(
    "/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/results/"
    "binary_gene_coefficients.csv"
)

# -----------------------------
# Bar plot: Top genes
# -----------------------------
TOP_N = 15
top = coef.head(TOP_N)[::-1]

plt.figure(figsize=(6, 4))
plt.barh(top.index, top["coefficient"])
plt.axvline(0, color="black", lw=0.8)
plt.xlabel("Elastic-net coefficient (Cancer vs Healthy)")
plt.title("Top Discriminative Exon-1 Entropy Genes")
plt.tight_layout()

plt.savefig(
    "/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/results/"
    "top_genes_barplot_binary.png",
    dpi=300
)
plt.close()

print("Saved top_genes_barplot_binary.png")

# -----------------------------
# Load validation data
# -----------------------------
val = pd.read_csv(
    "/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/matrices/val_lung_stage_3_4.csv"
)

X_val = val[genes]
y_val = val["label"]

print("Unique classes in y_true:", np.unique(y_val))
# -----------------------------
# Binary ROC curve
# -----------------------------
# Positive class assumed to be "cancer"
pos_class = lr.classes_[1]

y_score = model.predict_proba(X_val)[:, 1]
y_true = (y_val == pos_class).astype(int)

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"Cancer vs Healthy (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/results/"
    "roc_binary_validation.png",
    dpi=300
)
plt.close()

print("Saved roc_binary_validation.png")

# -----------------------------
# Heatmap (single-row, still useful)
# -----------------------------
TOP_HEATMAP = 30
coef_top = coef.head(TOP_HEATMAP)

plt.figure(figsize=(10, 2.5))

sns.heatmap(
    coef_top.T,
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Elastic-net coefficient (log-odds)"}
)

plt.xlabel("Gene")
plt.ylabel("Model")
plt.title("Top Exon-1 Entropy Gene Coefficients\nBinary Elastic-Net (Cancer vs Healthy)")

plt.tight_layout()
plt.savefig(
    "/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/results/"
    "coef_heatmap_top_genes_binary.png",
    dpi=300
)
plt.close()

print("Saved coef_heatmap_top_genes_binary.png")

# -----------------------------
# Console summary (optional, nice)
# -----------------------------
print("\nTop Cancer-enriched genes:")
print(coef[coef["coefficient"] > 0].head(10))

print("\nTop Healthy-enriched genes:")
print(coef[coef["coefficient"] < 0].head(10))
