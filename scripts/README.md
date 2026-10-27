# Scripts

| Folder | What it holds |
|---|---|
| `utils/` | `bbo_utils.py` (GP tuning, UCB, calibration, duplicate check, TuRBO wrapper) and `turbo_np.py` (NumPy/SciPy TuRBO). These are what the main notebook uses. `build_*.py` and `turbo_F5_data.py` are small helpers from earlier weeks. |
| `pipeline/` | The weekly pipelines as I ran them (`week4_pipeline.py` ... `week12_pipeline.py`) and the tuning-grid scripts they grew out of. Kept as a record of how the method developed. |
| `diagnostics/` | One-off tests: extrapolation trust (F4), sensitivity analysis, linear / SVM / neural-network surrogate comparisons, the F1 hybrid strategy, and the small-step "box pick" for F4 and F7. |

**Running the historical scripts.** Run them from the repository root, for example
`python scripts/pipeline/week12_pipeline.py`. They read `data/initial_data/` and write their CSV outputs to the folder you run them from. Each weekly script holds that week's results inside the file, so it reproduces that week's state, not the final data. For the final data use `data/queries.csv` and the notebook.
