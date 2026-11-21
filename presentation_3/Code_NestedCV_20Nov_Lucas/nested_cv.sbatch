#!/bin/bash
# Nested CV, one outer test group per array task (0..4 = A..E). DECISIONS.md D13.
# Usage: sbatch --array=0-4 nested_cv.sbatch <out_dir_name> <n_trials>
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=P3_nested_cv
#SBATCH --output=/home/ele0436/STAT390_coder/presentation3/logs/%x_%A_%a.out
#SBATCH --error=/home/ele0436/STAT390_coder/presentation3/logs/%x_%A_%a.err
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export OMP_NUM_THREADS=4
P=/home/ele0436/STAT390_coder/presentation3
G=(A B C D E)
cd $P/code
python3 nested_cv.py --outer ${G[$SLURM_ARRAY_TASK_ID]} \
    --cache $P/cache/uni2_native_99cases.pt \
    --groups_csv $P/splits/nested_groups_seed436.csv \
    --out_dir $P/runs/$1 --n_trials $2 --workers 4 --threads_per_worker 4
