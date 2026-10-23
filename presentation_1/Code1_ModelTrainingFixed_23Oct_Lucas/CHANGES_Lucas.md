# Changes from the Presentation 1 base code (Lucas McDonough, Coders track)

This folder is a copy of `Team6_InstructionalTeam/Code/Presentation1_base_code` with four changes.
Everything else (model, hyperparameters, data loading) is unchanged.

| File | Line | Change | Why |
|---|---|---|---|
| `make_splits.py` | 156 | `StratifiedGroupKFold(n_splits=..., shuffle=True, random_state=seed)` | The `--seed` argument was never used, so every seed produced the same 5 splits. |
| `utils.py` | 86–90 | Run folder is named `run_<timestamp>_job<SLURM_JOB_ID>` | Folders were named by time to the second, so jobs that started in the same second wrote into the same folder. |
| `main.py` | 295–299 | Reload `best.pth` after training, before testing | The test previously used the last epoch's weights even though the best-validation checkpoint was saved. (Measured afterwards: this did not improve test accuracy; see `Code3`.) |
| `config.py` | 12 | `runs_dir` points to my own folder | Keeps my results separate from the shared class folder. |

## How I ran it

```bash
# Stage 3: make 5 splits with my seed (output included here in splits_seed436/)
python make_splits.py --seed 436 --save_dir splits_seed436

# Stage 4: one SLURM job per split
for i in 1 2 3 4 5; do sbatch train_split.sbatch $i; done
```

`train_split.sbatch` uses absolute paths under `/home/ele0436/STAT390_coder/`; change them to run it elsewhere.
The patch features it reads are on Quest at `/projects/e32998/patches_varsize_pooled4096`.
