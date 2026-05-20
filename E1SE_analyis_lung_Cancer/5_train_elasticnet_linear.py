import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, auc

# Load training and validation

train = pd.read_csv("/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/matrices/train_lung_stage_3_4.csv")
val   = pd.read_csv("/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/matrices/val_lung_stage_3_4.csv")

X_train = train.drop(columns=["sample", "label"])
y_train = train["label"]

X_val = val.drop(columns=["sample", "label"])
y_val = val["label"]

classes = np.unique(y_train)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Initial gene count:", X_train.shape[1])

# Filter genes by presence

presence = X_train.notna().mean()
keep_genes = presence[presence >= 0.25].index   # ≥25% samples

X_train = X_train[keep_genes]
X_val   = X_val[keep_genes]

print(f"Genes retained after presence filtering: {len(keep_genes)}")

# -----------------------------
# Hyperparameter grid
# -----------------------------
alphas = np.linspace(0.05, 0.95, 10)        # elastic-net mixing
lambdas = np.logspace(-3, 1, 10)             # λ  (C = 1 / λ)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

best_auc = -np.inf
best_params = None

# -----------------------------
# Cross-validated search
# -----------------------------
for alpha in alphas:
    for lam in lambdas:
        C = 1.0 / lam

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=alpha,
                C=C,
                max_iter=5000,
                n_jobs=-1
#		class_weight="balanced"
            ))
        ])

        aucs = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc_ovr"
        )

        mean_auc = aucs.mean()

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = (alpha, lam)

print("\nBest CV AUC:", round(best_auc, 4))
print("Best alpha (l1_ratio):", best_params[0])
print("Best lambda:", best_params[1])

# -----------------------------
# Train final model
# -----------------------------
best_alpha, best_lambda = best_params
best_C = 1.0 / best_lambda

final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=best_alpha,
        C=best_C,
        max_iter=5000,
        n_jobs=-1
#	class_weight="balanced"
    ))
])

final_model.fit(X_train, y_train)

joblib.dump(
    {
        "model": final_model,
        "genes": keep_genes,
        "classes": classes
    },
    "/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/models/multinomial_elasticnet_lung_stage_3_4.pkl"
)

print("\nFinal model trained and saved.")

# -----------------------------
# Validation performance
# -----------------------------
y_val_bin = label_binarize(y_val, classes=classes)
y_val_proba = final_model.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(
    y_val_bin,
    y_val_proba,
    average="macro",
    multi_class="ovr"
)

#PR-AUC Calculation

precision, recall, _ = precision_recall_curve(y_val, y_val_proba, pos_label="cancer")
pr_auc = auc(recall, precision)


print("Validation AUC (macro-OVR):", round(val_auc, 4))
print("Retained genes:", list(keep_genes))
print("Validation PR-AUC:", round(pr_auc, 4))
