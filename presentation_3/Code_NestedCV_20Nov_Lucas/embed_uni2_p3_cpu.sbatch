#!/bin/bash
# P3: UNI2 native-size embeddings for the 5,372 new-case patches, on CPUs (DECISIONS.md D10).
# GPU queue was days long (D9); CPU embeddings match A100 ones to rounding level (D9 results).
# Same instructor script and output folder as the GPU job; only --device cpu differs.
# Existing .pt files are skipped, so this is safe to rerun.
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=P3_embed_uni2_cpu
#SBATCH --output=/home/ele0436/STAT390_coder/presentation3/logs/%x_%A_%a.out
#SBATCH --error=/home/ele0436/STAT390_coder/presentation3/logs/%x_%A_%a.err
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
P=/home/ele0436/STAT390_coder/presentation3
cd $P/code
python3 precompute_pooled_features.py \
    --encoder uni2 --uni2_native_size --device cpu \
    --patches_dir $P/new_case_patch_links \
    --embeddings_dir $P/embeddings_uni2_native_99 \
    --batch_size 8 --num_workers 2 \
    --num_shards 4 --shard_index "$SLURM_ARRAY_TASK_ID" --log_every 20
