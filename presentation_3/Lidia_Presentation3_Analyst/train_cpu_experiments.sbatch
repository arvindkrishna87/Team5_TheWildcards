#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --job-name=MIL_P3_CPU
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5%3
#SBATCH --output=/home/dgc2700/STAT390_Presentation3/logs/cpu_exp_%A_%a.out
#SBATCH --error=/home/dgc2700/STAT390_Presentation3/logs/cpu_exp_%A_%a.err

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

IDX=${SLURM_ARRAY_TASK_ID}

echo "Experiment: ${NAMES[$IDX]}"
echo "Embedding dimension: ${DIMS[$IDX]}"
echo "Stain dropout: ${STAIN[$IDX]}"
echo "Slice dropout: ${SLICE[$IDX]}"
echo "Started: $(date)"

python main.py \
  --labels_csv /home/dgc2700/STAT390_Presentation3/base_code/extra_benign_case_grade_match.csv \
  --patches_dir /home/dgc2700/STAT390_Presentation3/patches_with_embeddings \
  --load_splits /home/dgc2700/STAT390_Presentation3/splits_preliminary/data_splits_new_01.npz \
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
