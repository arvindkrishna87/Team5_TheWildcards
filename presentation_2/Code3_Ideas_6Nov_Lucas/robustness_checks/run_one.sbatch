#!/bin/bash
# Usage: sbatch --job-name=<name> run_one.sbatch <kimianet|uni2> <split file> <seed> [extra main.py args]
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --output=/home/ele0436/STAT390_coder/robustness_checks/logs/%x_%j.out
#SBATCH --error=/home/ele0436/STAT390_coder/robustness_checks/logs/%x_%j.err

ENC=$1; SPLIT_FILE=$2; SEED=$3; shift 3
case "$ENC" in
    kimianet) EMB=/projects/e32998/patches_varsize_pooled4096; DIM=4096 ;;
    uni2)     EMB=/projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2; DIM=1536 ;;
esac
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
echo "Encoder: $ENC  Splits: $SPLIT_FILE  Seed: $SEED  Extra: $*"
cd /home/ele0436/STAT390_coder/robustness_checks/code
python3 main.py --per_slice_cap 500 --max_slices_per_stain 5 --load_splits "$SPLIT_FILE" \
    --embeddings_dir "$EMB" --pooled_dim "$DIM" --seed "$SEED" "$@"
RUN=$(grep -m1 "Created run directory" "$HOME/STAT390_coder/robustness_checks/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" | awk '{print $NF}')
find "$RUN/checkpoints" -name '*epoch*.pth' -delete   # keep only best.pth to save space
