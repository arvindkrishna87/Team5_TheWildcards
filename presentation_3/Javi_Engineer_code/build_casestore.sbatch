#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=build_case_store
#SBATCH --output=build_case_store_%j.out
#SBATCH --error=build_case_store_%j.err

# Builds one file per case from the metadata tables, then checks them against the table.
# Reads the embedding .pt files once; never changes them.
# Run build_metadata.sbatch first. Submit from inside Presentation2_base_code:
#     sbatch build_case_store.sbatch

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
echo "python: $(which python)"

cd "$SLURM_SUBMIT_DIR"
echo "running in: $(pwd)"

VERSION=v1
META="$SLURM_SUBMIT_DIR/metadata"
STORE="$SLURM_SUBMIT_DIR/case_store/$VERSION"   # ~1.3 GB; keep case_store/ in .gitignore

# ORDER must match how data_utils.py sorts patches and slices:
#   text    = sorted(filenames)          -> patch0, patch1, patch10, patch100 ... / match_1, match_10, match_2
#   numeric = sorted by the number       -> patch0, patch1, patch2 ...           / match_1, match_2, match_10
ORDER=text

python build_case_store.py \
    --patches "$META/patches_${VERSION}.parquet" \
    --cases_table "$META/cases_${VERSION}.parquet" \
    --out_dir "$STORE" \
    --order "$ORDER" \
    --project_root /projects/e32998 || exit 1

python check_case_store.py \
    --patches "$META/patches_${VERSION}.parquet" \
    --store "$STORE" \
    --sample 300