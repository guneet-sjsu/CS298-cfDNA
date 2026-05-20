import pandas as pd
import glob

dfs = []

for f in glob.glob("/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/e1se_010326/stage_3_4/*_e1se.csv"):
    sample = f.split("/")[-1].replace("_e1se.csv","")
    df = pd.read_csv(f)[["gene","E1SE"]]
    df["sample"] = sample
    dfs.append(df)

long = pd.concat(dfs)

wide = long.pivot(index="sample", columns="gene", values="E1SE")
wide.to_csv("/scratch/home/gbhogal/CS298/cfDNA_lung_cancer/matrices/e1se_matrix_stage_3_4.csv")
