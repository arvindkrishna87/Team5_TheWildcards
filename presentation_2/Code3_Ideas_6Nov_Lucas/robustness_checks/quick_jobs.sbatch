#!/bin/bash
# Usage: sbatch --job-name=<name> quick_jobs.sbatch <script.py>
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --output=/home/ele0436/STAT390_coder/robustness_checks/logs/%x_%j.out
#SBATCH --error=/home/ele0436/STAT390_coder/robustness_checks/logs/%x_%j.err
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
cd /home/ele0436/STAT390_coder/robustness_checks
python3 "$@"
