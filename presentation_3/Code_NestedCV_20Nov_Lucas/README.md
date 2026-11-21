# Presentation 3 — Coders: nested cross-validation with UNI2 (Lucas)

Leak-free estimate of the UNI2 attention-MIL model: 5 outer groups over all 99 gradable cases, a
separate Optuna study per outer group (40 trials, 16 scrambled Sobol startup points, then TPE), each
trial scored by 4 inner models, then one refit on the 4 training groups and one prediction for the test
group. In total 5 × (4 × 40 + 1) = 805 models. Test labels are never used to choose hyperparameters
or epochs.

**Result:** 93 of 99 correct (accuracy 0.939). Benign recall 0.857 (30/35), high-grade recall 0.984 (63/64).
Groups range from 0.80 (A) to 1.00. See `../Results_NestedCV_20Nov_Lucas/`.


## Files

| File | What it does |
|---|---|
| `nested_cv.py` | The whole nested CV for one outer group: Optuna study, inner models, selection, refit, test |
| `nested_cv.sbatch` | Slurm array, one task per outer group A–E (CPU) |
| `make_nested_groups.py` | Builds the 5 outer groups (seed 436, patient groups kept together) → `splits/` |
| `precompute_pooled_features.py`, `embed_uni2_p3_cpu.sbatch` | UNI2 patch embeddings (instructor script, unmodified) |
| `build_embedding_cache.py`, `build_cache.sbatch`, `cached_dataset.py` | Packs all 99 cases' embeddings into one in-memory cache so 805 models train fast |
| `models.py`, `config.py`, `data_utils.py` | Model and config from the Presentation 3 base code |
| `make_results.py` | Per-group table, pooled table, selected hyperparameters, refit plots |
| `make_slide_figures.py` | The figures used on the slides |
| `extra_benign_case_grade_match.csv` | Case labels |

## Run order
1. `sbatch embed_uni2_p3_cpu.sbatch` → embeddings
2. `sbatch build_cache.sbatch` → `cache/uni2_native_99cases.pt`
3. `python make_nested_groups.py` → `splits/nested_groups_seed436.csv`
4. `sbatch nested_cv.sbatch` → `runs/full_40trials/outer_{A..E}/`
5. `python make_results.py`, then `python make_slide_figures.py`

Paths in the sbatch files are Quest paths. The embeddings (1.4 GB) and cache (260 MB) are not in the
repo; rebuild them with steps 1–2.
