#!/bin/bash
#SBATCH --job-name=clean_beds
#SBATCH --array=1-635
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/clean_%A_%a.out
#SBATCH --error=logs/clean_%A_%a.err

BEDDIR="/scratch/home/gbhogal/CS298/bed_train"
OUTDIR="/scratch/home/gbhogal/CS298/bed_train_clean"

mkdir -p $OUTDIR

FILE=$(ls $BEDDIR/*.bed | sed -n ${SLURM_ARRAY_TASK_ID}p)
BASENAME=$(basename "$FILE" .bed)

echo "Cleaning $FILE → $OUTDIR/${BASENAME}.clean.bed"

awk '($3-$2)<=500' "$FILE" > "$OUTDIR/${BASENAME}.clean.bed"
