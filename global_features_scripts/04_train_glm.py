#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle
import sys

features_csv, model_out = sys.argv[1:]

df = pd.read_csv(features_csv)
X = df.drop(columns=["sample", "label"])
y = (df["label"] == "cancer").astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.3,
    max_iter=5000
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

print(f"Mean AUC = {auc.mean():.4f} ± {auc.std():.4f}")

model.fit(X_scaled, y)

with open(model_out, "wb") as f:
    pickle.dump((model, scaler), f)

print("Saved model →", model_out)
