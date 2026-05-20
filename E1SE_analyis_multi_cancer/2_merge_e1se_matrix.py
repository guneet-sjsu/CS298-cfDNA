import pandas as pd
import glob

dfs = []

for f in glob.glob("/scratch/home/gbhogal/CS298/cfDNA/e1se/*_e1se.csv"):
    sample = f.split("/")[-1].replace("_e1se.csv","")
    df = pd.read_csv(f)[["gene","E1SE"]]
    df["sample"] = sample
    dfs.append(df)

long = pd.concat(dfs)

wide = long.pivot(index="sample", columns="gene", values="E1SE")
wide.to_csv("/scratch/home/gbhogal/CS298/cfDNA/matrices/e1se_matrix.csv")
