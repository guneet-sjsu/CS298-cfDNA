#!/usr/bin/env python3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle
import sys

features_csv, model_out = sys.argv[1:]

df = pd.read_csv(features_csv)
X = df.drop(columns=["sample", "label"])
y = (df["label"] == "cancer").astype(int)

model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc"
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

print(f"Mean CV AUC = {auc.mean():.4f} ± {auc.std():.4f}")

model.fit(X, y)

model.save_model(model_out)
print("Saved XGBoost model →", model_out)
