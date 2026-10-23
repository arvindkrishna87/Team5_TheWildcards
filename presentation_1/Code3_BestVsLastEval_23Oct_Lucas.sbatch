#!/bin/bash
# Usage: sbatch eval_lastepoch.sbatch <split> <checkpoint>
# Re-scores the test set with a given checkpoint (no training) to compare last-epoch vs best weights.
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --job-name=MIL_lastepoch
#SBATCH --output=/home/ele0436/STAT390_coder/logs/lastepoch/%x_%j.out
#SBATCH --error=/home/ele0436/STAT390_coder/logs/lastepoch/%x_%j.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Split: $1"
echo "Checkpoint: $2"
cd /home/ele0436/STAT390_coder/code
python3 main.py --eval_only --resume "$2" \
    --per_slice_cap 500 --max_slices_per_stain 5 --seed 436 \
    --load_splits /home/ele0436/STAT390_coder/splits/data_splits_new_0$1.npz
