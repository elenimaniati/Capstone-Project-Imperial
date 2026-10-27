"""
Exploit step for hard functions (F4, F7): ask the GP for its highest predicted
point inside a small box around the best observed point.

Steps
  1. Fit the same GP the pipeline uses (Matern nu=1.5, alpha=1e-6, length-scale
     bounds 0.05-2, standardised outputs) on ALL observations so far.
  2. Find the best observed point.
  3. Draw 20,000 random points with every coordinate within +-r of that point
     (clipped to [0, 1]).
  4. Predict the GP mean at each point and take the highest.

Needs week12_pipeline.py (or any later weekN_pipeline.py) in the same folder,
plus the initial_data/ folder. The pipeline's data dictionaries and
load_function_data() are reused, so add new weeks there (week12_X / week12_Y,
and extend load_function_data) and re-run this script.
"""
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'pipeline'))
from week12_pipeline import load_function_data   # importing does NOT re-run the pipeline

warnings.filterwarnings("ignore")

FUNCTIONS = (4, 7)
RADII = (0.03, 0.05, 0.10)      # box half-widths to compare; +-0.05 is the one I recommended
N_CANDIDATES = 20_000
ALPHA = 1e-6                    # near-noise-free, as in the pipeline for F4 and F7
SEED = 0

for fid in FUNCTIONS:
    X, Y = load_function_data(fid)

    gp = GaussianProcessRegressor(
        kernel=Matern(length_scale=0.1, length_scale_bounds=(0.05, 2.0), nu=1.5),
        alpha=ALPHA, normalize_y=True, n_restarts_optimizer=10, random_state=42,
    ).fit(X, Y)

    x_best = X[Y.argmax()]
    print(f"\nF{fid}: {len(Y)} observations; fitted kernel {gp.kernel_}")
    print(f"   best so far {Y.max():.4f} at {x_best.round(4).tolist()}")

    rng = np.random.default_rng(SEED)
    for r in RADII:
        cand = np.clip(x_best + rng.uniform(-r, r, (N_CANDIDATES, len(x_best))), 0.0, 1.0)
        mean, std = gp.predict(cand, return_std=True)
        i = mean.argmax()
        d_prior = np.linalg.norm(X - cand[i], axis=1).min()
        flag = "  <-- DUPLICATE" if d_prior < 0.02 else ""
        print(f"   box +-{r:<4}: pick {cand[i].round(4).tolist()}  "
              f"GP mean {mean[i]:.3f} +- {std[i]:.3f}  nearest prior {d_prior:.3f}{flag}")
