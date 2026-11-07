#!/bin/bash
# Test A: KimiaNet features from patches resized to 224x224. Each array task encodes every 4th patch.
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --job-name=SAME_INPUT_A_embed
#SBATCH --output=/home/ele0436/STAT390_coder/same_input_comparison/logs/%x_%A_%a.out
#SBATCH --error=/home/ele0436/STAT390_coder/same_input_comparison/logs/%x_%A_%a.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

X=/home/ele0436/STAT390_coder/same_input_comparison
cd $X/code
python3 precompute_pooled_features.py \
    --encoder kimianet --resize_224 --kimianet_ckpt KimiaNet.pth \
    --patches_dir /projects/e32998/patches_benign_split \
    --embeddings_dir $X/embeddings_kimianet_224 \
    --device cpu --batch_size 64 --num_workers 6 \
    --num_shards 4 --shard_index "$SLURM_ARRAY_TASK_ID" --log_every 20
