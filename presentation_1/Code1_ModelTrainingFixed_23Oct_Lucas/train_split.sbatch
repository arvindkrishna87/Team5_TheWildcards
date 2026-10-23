#!/bin/bash
# Usage:  sbatch train_split.sbatch <split number 1-5>
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=MIL_split
#SBATCH --output=/home/ele0436/STAT390_coder/logs/%x_%j.out
#SBATCH --error=/home/ele0436/STAT390_coder/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lucasmcdonough2027@u.northwestern

SPLIT=${1:?Give a split number, e.g. sbatch train_split.sbatch 1}
SPLIT_FILE=/home/ele0436/STAT390_coder/splits/data_splits_new_0${SPLIT}.npz

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Split: $SPLIT ($SPLIT_FILE)"
echo "Start time: $(date)"

cd /home/ele0436/STAT390_coder/code

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits "$SPLIT_FILE" \
    --seed 436 \
    "${@:2}"

echo "End time: $(date)"
