#!/bin/bash
#SBATCH --job-name=ctx_features
#SBATCH --array=1-635
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/ctx_%A_%a.out
#SBATCH --error=logs/ctx_%A_%a.err

BEDDIR="/scratch/home/gbhogal/CS298/bed_train_clean"
ANNOTDIR="/scratch/home/gbhogal/CS298/annotations"
OUTDIR="/scratch/home/gbhogal/CS298/pipeline/features_train"

mkdir -p $OUTDIR

FILE=$(ls $BEDDIR/*.clean.bed | sed -n ${SLURM_ARRAY_TASK_ID}p)
SAMPLE=$(basename "$FILE" .clean.bed)

echo "Processing $SAMPLE"

python3 compute_context_features.py \
    "$FILE" \
    "$ANNOTDIR" \
    "$OUTDIR/${SAMPLE}_features.csv"
