#!/bin/bash
# Build the 99-case in-memory embedding cache (DECISIONS.md D11). Reads 86,719 .pt files once.
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=P3_build_cache
#SBATCH --output=/home/ele0436/STAT390_coder/presentation3/logs/%x_%j.out
#SBATCH --error=/home/ele0436/STAT390_coder/presentation3/logs/%x_%j.err
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
P=/home/ele0436/STAT390_coder/presentation3
cd $P/code
python3 build_embedding_cache.py --embeddings_dir $P/embeddings_uni2_native_99 \
    --out $P/cache/uni2_native_99cases.pt --threads 32
