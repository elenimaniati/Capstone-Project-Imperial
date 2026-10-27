"""
Structured exploration of the three main GP-UCB tuning points:
  1. kernel TYPE          (RBF vs Matern vs RationalQuadratic -- shape of correlation)
  2. length_scale LOWER BOUND  (the actual free choice -- the starting guess
                                  doesn't matter, the optimizer searches within
                                  whatever bounds you give it; see note below)
  3. beta                 (exploration/exploitation trade-off)

Each is a genuinely separate axis:
- kernel TYPE and length_scale BOUND both require a fresh model.fit() call,
  since they change what the GP is even allowed to learn.
- beta does NOT require refitting -- it only recombines an already-fitted
  model's mean/std -- so it's kept as the cheap inner loop.

Note on "rbf_lengthscale": the STARTING GUESS you pass to RBF(length_scale=...)
barely matters, because with n_restarts_optimizer>0 the optimizer searches
from several random starting points anyway and (per our own testing) reliably
converges to the same answer regardless of the initial guess. What actually
matters, and what we sweep here, is length_scale_bounds -- the fence around
that search. That's the genuinely free tuning choice.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

warnings.filterwarnings('ignore')

RNG_SEED = 42

KERNEL_BUILDERS = {
    'RBF': lambda ls, bounds: RBF(length_scale=ls, length_scale_bounds=bounds),
    'Matern_nu1.5': lambda ls, bounds: Matern(length_scale=ls, length_scale_bounds=bounds, nu=1.5),
    'Matern_nu2.5': lambda ls, bounds: Matern(length_scale=ls, length_scale_bounds=bounds, nu=2.5),
    'RationalQuadratic': lambda ls, bounds: RationalQuadratic(length_scale=ls, length_scale_bounds=bounds),
}


def _build_candidates(n_dims, n_candidates_highdim=20000, n_grid_2d=50):
    rng = np.random.RandomState(RNG_SEED)
    if n_dims == 2:
        g = np.linspace(0, 1, n_grid_2d)
        Xg, Yg = np.meshgrid(g, g)
        return np.column_stack([Xg.ravel(), Yg.ravel()])
    return rng.uniform(0, 1, size=(n_candidates_highdim, n_dims))


def explore_tuning_grid(X, Y,
                         kernel_types=('RBF', 'Matern_nu1.5', 'Matern_nu2.5'),
                         length_scale_lower_bounds=(0.05, 0.1, 0.2, 0.5),
                         length_scale_upper_bound=2.0,
                         betas=(1.96, 3, 5),
                         noise_assumption=1e-6, initial_guess=0.1):
    """
    Fits one GP per (kernel_type x length_scale_lower_bound) combination,
    then evaluates every beta against each fit.

    Returns two DataFrames:
      fit_summary : one row per (kernel_type, lower_bound) -- the expensive axis.
                    Sorted by log_marginal_likelihood, best first. This is your
                    primary read: which kernel/bound choice does the data
                    actually support?
      full_grid   : one row per (kernel_type, lower_bound, beta) -- the full
                    picture, for once you've narrowed down which fits to trust.
    """
    n_dims = X.shape[1]
    candidates = _build_candidates(n_dims)

    fit_rows, full_rows = [], []

    for kname in kernel_types:
        for lb in length_scale_lower_bounds:
            kernel = KERNEL_BUILDERS[kname](initial_guess, (lb, length_scale_upper_bound))
            model = GaussianProcessRegressor(kernel=kernel, alpha=noise_assumption,
                                              normalize_y=True, n_restarts_optimizer=10,
                                              random_state=RNG_SEED)
            model.fit(X, Y)

            lml = model.log_marginal_likelihood(model.kernel_.theta)
            post_mean, post_std = model.predict(candidates, return_std=True)
            std_span = post_std.max() - post_std.min()
            fitted_ls = model.kernel_.length_scale
            pinned = abs(fitted_ls - lb) < 1e-3

            fit_rows.append({
                'kernel_type': kname,
                'length_scale_lower_bound': lb,
                'fitted_length_scale': round(fitted_ls, 4),
                'pinned_at_bound': pinned,   # True = optimizer wanted to go
                                              # even lower -> collapse risk
                'log_marginal_likelihood': round(lml, 2),
                'post_std_span': round(std_span, 4),  # near 0 = flat/collapsed posterior
            })

            for beta in betas:
                ucb_scores = post_mean + beta * post_std
                idx = np.argmax(ucb_scores)
                full_rows.append({
                    'kernel_type': kname,
                    'length_scale_lower_bound': lb,
                    'beta': beta,
                    'next_x': candidates[idx].round(4).tolist(),
                    'pred_mean': round(post_mean[idx], 4),
                    'pred_std': round(post_std[idx], 4),
                })

    fit_summary = pd.DataFrame(fit_rows).sort_values(
        'log_marginal_likelihood', ascending=False).reset_index(drop=True)
    full_grid = pd.DataFrame(full_rows)
    return fit_summary, full_grid


def print_report(fit_summary, full_grid, top_n=3):
    pd.set_option('display.width', 160)
    print("=" * 100)
    print("STEP 1 -- which (kernel, length_scale bound) does the data actually support?")
    print("          (ranked by log marginal likelihood, best first)")
    print("=" * 100)
    print(fit_summary.to_string(index=False))

    flagged = fit_summary[fit_summary['pinned_at_bound']]
    if len(flagged):
        print(f"\n  Note: {len(flagged)} config(s) are pinned at their lower bound -- "
              f"the optimizer wanted to go even lower. Treat these with the same "
              f"skepticism we applied to Functions 1/2/7's degenerate fits.")

    print("\n" + "=" * 100)
    print(f"STEP 2 -- beta sensitivity for the top {top_n} best-supported configs")
    print("=" * 100)
    top_configs = fit_summary.head(top_n)[['kernel_type', 'length_scale_lower_bound']]
    for _, row in top_configs.iterrows():
        subset = full_grid[
            (full_grid.kernel_type == row.kernel_type) &
            (full_grid.length_scale_lower_bound == row.length_scale_lower_bound)
        ]
        print(f"\n-- {row.kernel_type}, lower_bound={row.length_scale_lower_bound} --")
        print(subset[['beta', 'next_x', 'pred_mean', 'pred_std']].to_string(index=False))


if __name__ == '__main__':
    # ---- example: Function 2, with week1/week2 folded in ----
    week1_X2 = [0.816326, 0.959183]; week1_Y2 = -0.06913022970475102
    week2_X2 = [0.673469, 0.897959]; week2_Y2 = 0.4900359752987052
    X0 = np.load('data/initial_data/function_2/initial_inputs.npy')
    Y0 = np.load('data/initial_data/function_2/initial_outputs.npy')
    X = np.vstack([X0, week1_X2, week2_X2])
    Y = np.concatenate([Y0, [week1_Y2, week2_Y2]])

    fit_summary, full_grid = explore_tuning_grid(X, Y)
    print_report(fit_summary, full_grid)

    fit_summary.to_csv('fit_summary_function2.csv', index=False)
    full_grid.to_csv('full_grid_function2.csv', index=False)
    print("\nSaved fit_summary_function2.csv and full_grid_function2.csv")
