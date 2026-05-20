import pandas as pd

X = pd.read_csv("/scratch/home/gbhogal/CS298/cfDNA/matrices/e1se_matrix.csv", index_col=0)
meta = pd.read_csv("sample_metadata_grail.csv").set_index("sample")

final = X.join(meta)
final.to_csv("/scratch/home/gbhogal/CS298/cfDNA/matrices/e1se_with_labels.csv")
