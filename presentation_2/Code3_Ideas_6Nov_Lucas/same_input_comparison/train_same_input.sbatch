#!/bin/bash
# Usage:  sbatch --job-name=SAME_INPUT_<A|B>_s<split> train_same_input.sbatch <split> <A|B>
# Same split files, seed and training settings as presentation2/train_split.sbatch.
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --output=/home/ele0436/STAT390_coder/same_input_comparison/logs/%x_%j.out
#SBATCH --error=/home/ele0436/STAT390_coder/same_input_comparison/logs/%x_%j.err

SPLIT=${1:?split}
TEST=${2:?A or B}
X=/home/ele0436/STAT390_coder/same_input_comparison
case "$TEST" in
    A) EMB=$X/embeddings_kimianet_224; DIM=4096 ;;
    B) EMB=$X/embeddings_uni2_original_size; DIM=1536 ;;
    *) echo "Unknown test: $TEST"; exit 1 ;;
esac

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Split: $SPLIT  Test: $TEST ($EMB, dim $DIM)"
cd $X/code
python3 main.py \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /home/ele0436/STAT390_coder/presentation2/splits/data_splits_new_0${SPLIT}.npz \
    --embeddings_dir "$EMB" \
    --pooled_dim "$DIM" \
    --val_every_updates 10 \
    --seed 436
