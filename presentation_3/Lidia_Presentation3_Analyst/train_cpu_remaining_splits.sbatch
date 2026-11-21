#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --job-name=MIL_P3_CV
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-23%3
#SBATCH --output=/home/dgc2700/STAT390_Presentation3/logs/cv_%A_%a.out
#SBATCH --error=/home/dgc2700/STAT390_Presentation3/logs/cv_%A_%a.err

set -eo pipefail

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

cd /home/dgc2700/STAT390_Presentation3/base_code

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

NAMES=(baseline512 embed256 embed128 embed64 stain30 slice30)
DIMS=(512 256 128 64 512 512)
STAIN=(0 0 0 0 0.30 0)
SLICE=(0 0 0 0 0 0.30)

TASK=${SLURM_ARRAY_TASK_ID}

# Six experiments per split, starting with Split 2
SPLIT=$((TASK / 6 + 2))
IDX=$((TASK % 6))

SPLIT_FILE=$(printf \
  "/home/dgc2700/STAT390_Presentation3/splits_preliminary/data_splits_new_%02d.npz" \
  "$SPLIT")

echo "========================================"
echo "Split: $SPLIT"
echo "Experiment: ${NAMES[$IDX]}"
echo "Embedding dimension: ${DIMS[$IDX]}"
echo "Stain dropout: ${STAIN[$IDX]}"
echo "Slice dropout: ${SLICE[$IDX]}"
echo "Split file: $SPLIT_FILE"
echo "Started: $(date)"
echo "========================================"

python main.py \
  --labels_csv /home/dgc2700/STAT390_Presentation3/base_code/extra_benign_case_grade_match.csv \
  --patches_dir /home/dgc2700/STAT390_Presentation3/patches_with_embeddings \
  --load_splits "$SPLIT_FILE" \
  --embed_dim "${DIMS[$IDX]}" \
  --stain_dropout "${STAIN[$IDX]}" \
  --slice_dropout "${SLICE[$IDX]}" \
  --epochs 20 \
  --batch_size 1 \
  --num_workers 2 \
  --per_slice_cap 500 \
  --max_slices_per_stain 5 \
  --seed 42

echo "Finished: $(date)"
