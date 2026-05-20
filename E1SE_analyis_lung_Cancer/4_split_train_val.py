import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/matrices/e1se_with_labels_stage_3_4.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv("/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/matrices/train_lung_stage_3_4.csv", index=False)
val_df.to_csv("/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/matrices/val_lung_stage_3_4.csv", index=False)

print("Train label counts:")
print(train_df["label"].value_counts())

print("\nValidation label counts:")
print(val_df["label"].value_counts())
