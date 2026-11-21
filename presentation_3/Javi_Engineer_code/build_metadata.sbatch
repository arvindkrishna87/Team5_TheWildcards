#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --job-name=build_metadata
#SBATCH --output=build_metadata_%j.out
#SBATCH --error=build_metadata_%j.err

# Builds the metadata tables (patches, cases, case features), then validates them.
# Submit from inside Presentation2_base_code:  sbatch build_metadata.sbatch

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
echo "python: $(which python)"

cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from
echo "running in: $(pwd)"

VERSION=v1
OUT="$SLURM_SUBMIT_DIR/metadata"  # tables are saved inside Presentation2_base_code/metadata

# Original data folders (read-only use)
PATCHES=/projects/e32998/patches_benign_split
EMB=/projects/e32998/patches_varsize_pooled4096

python build_metadata.py \
    --patches_dir "$PATCHES" \
    --emb_dir "$EMB" \
    --labels_csv extra_benign_case_grade_match.csv \
    --splits_dir no_train_val_leakage_splits \
    --out_dir "$OUT" \
    --version "$VERSION" \
    --overwrite \
    --size_workers 8 || exit 1

python validate_metadata.py \
    --patches "$OUT/patches_${VERSION}.parquet" \
    --splits_dir no_train_val_leakage_splits \
    --check_pt 200 \
    --allow_missing_emb