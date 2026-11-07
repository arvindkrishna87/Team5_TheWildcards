#!/bin/bash
# Usage:  sbatch train_split.sbatch <split 1-5> <kimianet|uni2> [extra main.py args]
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=P2_train
#SBATCH --output=/home/ele0436/STAT390_coder/presentation2/logs/%x_%j.out
#SBATCH --error=/home/ele0436/STAT390_coder/presentation2/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ele0436@u.northwestern.edu

SPLIT=${1:?Give a split number, e.g. sbatch train_split.sbatch 1 kimianet}
ENCODER=${2:?Give an encoder: kimianet or uni2}
SPLIT_FILE=/home/ele0436/STAT390_coder/presentation2/splits/data_splits_new_0${SPLIT}.npz

case "$ENCODER" in
    kimianet) EMB=/projects/e32998/patches_varsize_pooled4096; DIM=4096 ;;
    uni2)     EMB=/projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2; DIM=1536 ;;
    *) echo "Unknown encoder: $ENCODER"; exit 1 ;;
esac

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Split: $SPLIT ($SPLIT_FILE)  Encoder: $ENCODER ($EMB, dim $DIM)"
echo "Start time: $(date)"

cd /home/ele0436/STAT390_coder/presentation2/code

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits "$SPLIT_FILE" \
    --embeddings_dir "$EMB" \
    --pooled_dim "$DIM" \
    --val_every_updates 10 \
    --seed 436 \
    "${@:3}"

echo "End time: $(date)"
