#!/bin/bash
#SBATCH --job-name=e1se
#SBATCH --array=1-391
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e1se_%A_%a.out
#SBATCH --error=logs/e1se_%A_%a.err

set -euo pipefail
module purge || true

# Activate conda
source /scratch/home/gbhogal/miniconda3/etc/profile.d/conda.sh
conda activate fragmentomics

# ------------------
# Paths (USER-SET)
# ------------------
BEDDIR=/scratch/home/gbhogal/CS298/cfDNA/grail_dataset_bed_files
EXONBED=/scratch/home/gbhogal/CS298/anno/hg19_first_coding_exons.bed
OUTDIR=/scratch/home/gbhogal/CS298/cfDNA/e1se
SAMPLELIST=/scratch/home/gbhogal/CS298/cfDNA/samples_grail.txt
SCRIPT=/scratch/home/gbhogal/CS298/cfDNA/scripts/compute_e1se_sample.py

mkdir -p "$OUTDIR"

# ------------------
# SAFE sample lookup
# ------------------
mapfile -t SAMPLES < "$SAMPLELIST"
IDX=$((SLURM_ARRAY_TASK_ID - 1))

if (( IDX < 0 || IDX >= ${#SAMPLES[@]} )); then
  echo "Task ${SLURM_ARRAY_TASK_ID} out of range (n=${#SAMPLES[@]}), exiting."
  exit 0
fi

FILENAME="${SAMPLES[$IDX]}"
BEDFILE="$BEDDIR/$FILENAME"
SAMPLE="${FILENAME%.bed}"

# ------------------
# Sanity checks
# ------------------
[[ -f "$BEDFILE" ]] || { echo "Missing BED file: $BEDFILE"; exit 1; }
[[ -f "$EXONBED" ]] || { echo "Missing exon BED file: $EXONBED"; exit 1; }

# ------------------
# Run E1SE
# ------------------
python3 "$SCRIPT" \
  "$BEDFILE" \
  "$EXONBED" \
  "$OUTDIR/${SAMPLE}_e1se.csv"
