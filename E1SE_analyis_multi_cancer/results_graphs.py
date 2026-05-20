import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

obj = joblib.load("/scratch/home/gbhogal/CS298/cfDNA/models/multinomial_elasticnet.pkl")
model = obj["model"]
genes = obj["genes"]

coef = pd.DataFrame(
    model.named_steps["clf"].coef_,
    columns=genes,
    index=model.named_steps["clf"].classes_
)

# Rank genes by total absolute contribution
gene_score = coef.abs().sum(axis=0).sort_values(ascending=False)

# Top genes
top_genes = gene_score.head(30)

# ---- Bar plot (top 15) ----
top = top_genes.head(15)[::-1]

plt.figure(figsize=(6, 4))
plt.barh(top.index, top.values)
plt.xlabel("Sum |Coefficient| (across classes)")
plt.title("Top Discriminative Exon-1 Entropy Genes")
plt.tight_layout()

plt.savefig("/scratch/home/gbhogal/CS298/cfDNA/results/top_genes_barplot.png", dpi=300)
plt.close()

print("Saved results/top_genes_barplot.png")

#plot ROC

# -----------------------------
# Load model and validation data
# -----------------------------
obj = joblib.load("/scratch/home/gbhogal/CS298/cfDNA/models/multinomial_elasticnet.pkl")
model = obj["model"]
genes = obj["genes"]
classes = obj["classes"]

val = pd.read_csv("/scratch/home/gbhogal/CS298/cfDNA/matrices/val.csv")

X_val = val.drop(columns=["sample", "label"])[genes]
y_val = val["label"]

# -----------------------------
# Predict probabilities
# -----------------------------
y_score = model.predict_proba(X_val)
y_bin = label_binarize(y_val, classes=classes)

# -----------------------------
# Plot ROC curves
# -----------------------------

plt.figure(figsize=(7, 6))

for i, cls in enumerate(classes):
    # Per-class ROC
    y_true = y_bin[:, i]
    y_scores = y_score[:, i]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # -----------------------------
    # Bootstrap CI (per class)
    # -----------------------------
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    aucs = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))

        # Skip invalid resamples
        if len(np.unique(y_true[indices])) < 2:
            continue

        aucs.append(roc_auc_score(y_true[indices], y_scores[indices]))

    ci_lower, ci_upper = np.percentile(aucs, [2.5, 97.5])

    print(f"{cls}: AUC = {roc_auc:.3f} (95% CI: {ci_lower:.3f}–{ci_upper:.3f})")

    # Plot ONCE per class (fixed)
    plt.plot(
        fpr, tpr,
        lw=2,
        label=f"{cls} (AUC = {roc_auc:.2f}, CI: {ci_lower:.2f}-{ci_upper:.2f})"
    )

# -----------------------------
# Chance line
# -----------------------------
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curves (Validation)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()

plt.savefig("/scratch/home/gbhogal/CS298/cfDNA/results/roc_multiclass_validation.png", dpi=300)
plt.close()

print("ROC figure saved to results/roc_multiclass_validation.png")


#bar plot for top genes 

top = top_genes.head(15)[::-1]

plt.figure(figsize=(6, 4))
plt.barh(top.index, top.values)
plt.xlabel("Sum |Coefficient| (across classes)")
plt.title("Top Discriminative Exon-1 Entropy Genes")
plt.tight_layout()

plt.savefig("/scratch/home/gbhogal/CS298/cfDNA/results/top_genes_barplot.png", dpi=300)
plt.close()

print("Top genes in Lung Cancer")
print(coef.T.sort_values(by="Cancer_lung", key=np.abs, ascending=False).head(15))

print("Top genes in Breast Cancer")
print(coef.T.sort_values(by="Cancer_breast", key=np.abs, ascending=False).head(15))

print("Top genes in Prostate Cancer")
print(coef.T.sort_values(by="Cancer_prostate", key=np.abs, ascending=False).head(15))

#heatmap

# -----------------------------
# Select top genes by overall importance
# -----------------------------
gene_importance = coef.abs().sum(axis=0).sort_values(ascending=False)

TOP_N = 30
top_genes = gene_importance.head(TOP_N).index

coef_top = coef[top_genes]

# -----------------------------
# Reorder genes by lung contribution (optional but nice)
# -----------------------------
coef_top = coef_top.loc[:, coef_top.loc["Cancer_lung"].abs().sort_values(ascending=False).index]

# -----------------------------
# Plot heatmap
# -----------------------------
#plt.figure(figsize=(10, 6))

sns.heatmap(
    coef_top,
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Elastic-net coefficient (log-odds)"}
)

plt.xlabel("Gene")
plt.ylabel("Cancer Type")
plt.title("Top Discriminative Exon-1 Entropy Genes\nMultinomial Elastic-Net Model")

plt.tight_layout()
plt.savefig("/scratch/home/gbhogal/CS298/cfDNA/results/coef_heatmap_top_genes.png", dpi=300)
plt.close()

print("Saved heatmap to results/coef_heatmap_top_genes.png")

# -----------------------------
# Normalize (symmetric scaling)
# -----------------------------
vmax = np.abs(coef_top.values).max()

plt.figure(figsize=(10, 6))

ax = sns.heatmap(
    coef_top,
    cmap="RdBu_r",
    center=0,
    vmin=-vmax,
    vmax=vmax,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Elastic-net coefficient (log-odds)"}
)

# -----------------------------
# Highlight key genes
# -----------------------------
highlight_genes = ["KCNQ1OT1", "TSIX", "ROCK1P1"]

for gene in highlight_genes:
    if gene in coef_top.columns:
        idx = list(coef_top.columns).index(gene)

        # Draw rectangle around the column
        ax.add_patch(
            plt.Rectangle(
                (idx, 0),                      # (x, y)
                1, coef_top.shape[0],          # width, height
                fill=False,
                edgecolor="black",
                lw=2
            )
        )

# -----------------------------
# Improve readability
# -----------------------------
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=10)

plt.xlabel("Gene")
plt.ylabel("Cancer Type")
plt.title("Top Discriminative Exon-1 Entropy Genes\nMultinomial Elastic-Net Model")

#plt.tight_layout()
#plt.savefig("/scratch/home/gbhogal/CS298/cfDNA/results/coef_heatmap_top_genes_highlighted.png", dpi=300)
#plt.close()

#print("Saved highlighted heatmap")


# -----------------------------
# Get top + bottom genes per cancer
# -----------------------------
top_bottom_genes = {}

for cancer in coef_top.index:
    row = coef_top.loc[cancer]

    top_pos = row.sort_values(ascending=False).head(1).index.tolist()
    top_neg = row.sort_values(ascending=True).head(1).index.tolist()

    top_bottom_genes[cancer] = {
        "positive": top_pos,
        "negative": top_neg
    }

print(top_bottom_genes)

vmax = np.abs(coef_top.values).max()

plt.figure(figsize=(12, 6))

ax = sns.heatmap(
    coef_top,
    cmap="RdBu_r",
    center=0,
    vmin=-vmax,
    vmax=vmax,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Elastic-net coefficient (log-odds)"}
)

# -----------------------------
# Highlight cells
# -----------------------------
for row_idx, cancer in enumerate(coef_top.index):

    # Positive (high entropy → cancer)
    for gene in top_bottom_genes[cancer]["positive"]:
        col_idx = coef_top.columns.get_loc(gene)

        rect = plt.Rectangle(
            (col_idx, row_idx),
            1, 1,
            fill=False,
            edgecolor="red",
            linewidth=2.5,
            zorder=10
        )
        ax.add_patch(rect)

    # Negative (low entropy → healthy)
    for gene in top_bottom_genes[cancer]["negative"]:
        col_idx = coef_top.columns.get_loc(gene)

        rect = plt.Rectangle(
            (col_idx, row_idx),
            1, 1,
            fill=False,
            edgecolor="blue",
            linewidth=2.5,
            zorder=10
        )
        ax.add_patch(rect)


# -----------------------------
# Add annotations
# -----------------------------

for row_idx, cancer in enumerate(coef_top.index):

    gene_pos = top_bottom_genes[cancer]["positive"][0]
    col_idx = coef_top.columns.get_loc(gene_pos)
    value = coef_top.loc[cancer, gene_pos]

    ax.annotate(
        f"{gene_pos}\n({value:.2f})",
        xy=(col_idx + 0.5, row_idx + 0.5),
        xytext=(col_idx + 1.2, row_idx + 0.3),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=9,
        color="red",
        ha="left"
    )

    # Negative (1 gene)
    gene_neg = top_bottom_genes[cancer]["negative"][0]
    col_idx = coef_top.columns.get_loc(gene_neg)
    value = coef_top.loc[cancer, gene_neg]

    ax.annotate(
        f"{gene_neg}\n({value:.2f})",
        xy=(col_idx + 0.5, row_idx + 0.5),
        xytext=(col_idx - 1.2, row_idx + 0.7),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        fontsize=9,
        color="blue",
        ha="right"
    )


plt.xlabel("Gene")
plt.ylabel("Cancer Type")
plt.title("Top Discriminative Exon-1 Entropy Genes\nMultinomial Elastic-Net Model")

#plt.title("Cancer-Specific Fragmentomic Signatures (High vs Low Entropy)", fontsize=14, weight='bold')

plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig("/scratch/home/gbhogal/CS298/cfDNA/results/coef_heatmap_annotated_extremes.png", dpi=300)
plt.close()
