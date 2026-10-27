"""
Structured exploration of the three main GP-UCB tuning points, run across
ALL 8 functions:
  1. kernel TYPE               (RBF vs Matern vs RationalQuadratic)
  2. length_scale LOWER BOUND  (the genuinely free choice -- see note below)
  3. beta                      (exploration/exploitation trade-off)

Each is a genuinely separate axis:
- kernel TYPE and length_scale BOUND both require a fresh model.fit() call,
  since they change what the GP is even allowed to learn.
- beta does NOT require refitting -- it only recombines an already-fitted
  model's mean/std -- so it's kept as the cheap inner loop.

Note on "rbf_lengthscale": the STARTING GUESS you pass to RBF(length_scale=...)
barely matters, because with n_restarts_optimizer>0 the optimizer searches
from several random starting points anyway and reliably converges to the
same answer regardless of the initial guess. What actually matters, and
what we sweep here, is length_scale_bounds -- the fence around that search.
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

# --- round 1 and round 2 query points + results, for every function ---
week1_X = {
    1: [0.500000, 0.500000],
    2: [0.816326, 0.959183],
    3: [0.406027, 0.410592, 0.490850],
    4: [0.394868, 0.425405, 0.352448, 0.464929],
    5: [0.323763, 0.780786, 0.952277, 0.952084],
    6: [0.345264, 0.328436, 0.412802, 0.727459, 0.096332],
    7: [0.057185, 0.424604, 0.394749, 0.207966, 0.379147, 0.653218],
    8: [0.158359, 0.089191, 0.006755, 0.068470, 0.779613, 0.339703, 0.109925, 0.073427],
}
week1_Y = {
    1: 2.6752879910742468e-9, 2: -0.06913022970475102, 3: -0.027253680430408482,
    4: -0.3932940148544044, 5: 1703.1512205236875, 6: -0.41036378008423247,
    7: 2.0679435112701356, 8: 9.8674177220056,
}
week2_X = {
    1: [0.600000, 0.300000],
    2: [0.673469, 0.897959],
    3: [0.377732, 0.089220, 0.423517],
    4: [0.451289, 0.457435, 0.214355, 0.422482],
    5: [0.532623, 0.594318, 0.990109, 0.985016],
    6: [0.313541, 0.259869, 0.535241, 0.987761, 0.214623],
    7: [0.142611, 0.367475, 0.491490, 0.152728, 0.361910, 0.685650],
    8: [0.192332, 0.148385, 0.293081, 0.032111, 0.743423, 0.258716, 0.124238, 0.699129],
}
week2_Y = {
    1: 3.7759561298146587e-34, 2: 0.4900359752987052, 3: -0.0665391852481089,
    4: -3.2314120937555297, 5: 1880.1864969021228, 6: -0.6644176392227166,
    7: 2.3848190050793328, 8: 9.8169820770504,
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
    """Fits one GP per (kernel_type x length_scale_lower_bound), evaluates every
    beta against each fit. Returns (fit_summary, full_grid) DataFrames."""
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
                'pinned_at_bound': pinned,
                'log_marginal_likelihood': round(lml, 2),
                'post_std_span': round(std_span, 4),
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


def print_report(function_id, fit_summary, full_grid, top_n=3):
    pd.set_option('display.width', 160)
    print("=" * 100)
    print(f"FUNCTION {function_id}")
    print("=" * 100)
    print("STEP 1 -- which (kernel, length_scale bound) does the data actually support?")
    print(fit_summary.to_string(index=False))

    flagged = fit_summary[fit_summary['pinned_at_bound']]
    if len(flagged):
        print(f"\n  Note: {len(flagged)} config(s) pinned at their lower bound -- "
              f"optimizer wanted to go even lower. Treat with extra skepticism.")

    print(f"\nSTEP 2 -- beta sensitivity for the top {top_n} best-supported configs")
    top_configs = fit_summary.head(top_n)[['kernel_type', 'length_scale_lower_bound']]
    for _, row in top_configs.iterrows():
        subset = full_grid[
            (full_grid.kernel_type == row.kernel_type) &
            (full_grid.length_scale_lower_bound == row.length_scale_lower_bound)
        ]
        print(f"\n-- {row.kernel_type}, lower_bound={row.length_scale_lower_bound} --")
        print(subset[['beta', 'next_x', 'pred_mean', 'pred_std']].to_string(index=False))
    print()


def load_function_data(function_id):
    """Loads a function's initial data and folds in week1 + week2 results."""
    X0 = np.load(f'data/initial_data/function_{function_id}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{function_id}/initial_outputs.npy')
    X1 = np.array([week1_X[function_id]])
    Y1 = np.array([week1_Y[function_id]])
    X2 = np.array([week2_X[function_id]])
    Y2 = np.array([week2_Y[function_id]])
    X = np.vstack([X0, X1, X2])
    Y = np.concatenate([Y0, Y1, Y2])
    return X, Y


if __name__ == '__main__':
    best_per_function = []  # one-row-per-function summary across the whole project

    for function_id in range(1, 9):
        X, Y = load_function_data(function_id)
        fit_summary, full_grid = explore_tuning_grid(X, Y)
        print_report(function_id, fit_summary, full_grid)

        fit_summary.to_csv(f'fit_summary_function{function_id}.csv', index=False)
        full_grid.to_csv(f'full_grid_function{function_id}.csv', index=False)

        best_row = fit_summary.iloc[0]
        best_per_function.append({
            'function': function_id,
            'n_obs': len(Y),
            'best_kernel_type': best_row['kernel_type'],
            'best_length_scale_lower_bound': best_row['length_scale_lower_bound'],
            'fitted_length_scale': best_row['fitted_length_scale'],
            'pinned_at_bound': best_row['pinned_at_bound'],
            'log_marginal_likelihood': best_row['log_marginal_likelihood'],
        })

    overview = pd.DataFrame(best_per_function)
    overview.to_csv('best_config_overview_all_functions.csv', index=False)

    print("=" * 100)
    print("OVERVIEW -- best-supported kernel/bound choice per function")
    print("=" * 100)
    print(overview.to_string(index=False))
    print("\nSaved per-function CSVs (fit_summary_functionN.csv, full_grid_functionN.csv)")
    print("and best_config_overview_all_functions.csv")
