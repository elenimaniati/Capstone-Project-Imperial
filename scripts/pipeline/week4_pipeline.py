"""
Week 4 pipeline: runs the structured (kernel x length_scale bound x beta)
tuning exploration with all 3 rounds of prior queries folded in, then builds
a comparison table of the beta=1.0 (more exploitation-leaning) pick vs. the
standard beta=1.96 pick for each function, using each function's
best-supported (kernel, bound) combo by log marginal likelihood.

Run from the directory containing initial_data/. Produces:
  fit_summary_functionN.csv, full_grid_functionN.csv  (one pair per function)
  best_config_overview_all_functions.csv
  week4_candidates.csv   <- the comparison table
"""

import ast
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

# --- all 3 rounds of query points + results, for every function ---
week1_X = {1: [0.5, 0.5], 2: [0.816326, 0.959183], 3: [0.406027, 0.410592, 0.490850],
           4: [0.394868, 0.425405, 0.352448, 0.464929], 5: [0.323763, 0.780786, 0.952277, 0.952084],
           6: [0.345264, 0.328436, 0.412802, 0.727459, 0.096332],
           7: [0.057185, 0.424604, 0.394749, 0.207966, 0.379147, 0.653218],
           8: [0.158359, 0.089191, 0.006755, 0.068470, 0.779613, 0.339703, 0.109925, 0.073427]}
week1_Y = {1: 2.6752879910742468e-9, 2: -0.06913022970475102, 3: -0.027253680430408482,
           4: -0.3932940148544044, 5: 1703.1512205236875, 6: -0.41036378008423247,
           7: 2.0679435112701356, 8: 9.8674177220056}

week2_X = {1: [0.6, 0.3], 2: [0.673469, 0.897959], 3: [0.377732, 0.089220, 0.423517],
           4: [0.451289, 0.457435, 0.214355, 0.422482], 5: [0.532623, 0.594318, 0.990109, 0.985016],
           6: [0.313541, 0.259869, 0.535241, 0.987761, 0.214623],
           7: [0.142611, 0.367475, 0.491490, 0.152728, 0.361910, 0.685650],
           8: [0.192332, 0.148385, 0.293081, 0.032111, 0.743423, 0.258716, 0.124238, 0.699129]}
week2_Y = {1: 3.7759561298146587e-34, 2: 0.4900359752987052, 3: -0.0665391852481089,
           4: -3.2314120937555297, 5: 1880.1864969021228, 6: -0.6644176392227166,
           7: 2.3848190050793328, 8: 9.8169820770504}

week3_X = {1: [0.7755, 0.7959], 2: [0.6735, 0.9592], 3: [0.6491, 0.9696, 0.0045],
           4: [0.3682, 0.4922, 0.4132, 0.4558], 5: [0.5956, 0.9937, 0.9873, 0.9817],
           6: [0.2832, 0.5708, 0.0595, 0.9600, 0.1099],
           7: [0.1018, 0.2910, 0.5756, 0.0160, 0.4824, 0.6257],
           8: [0.0955, 0.2689, 0.0107, 0.4009, 0.9253, 0.8020, 0.0575, 0.7128]}
week3_Y = {1: 5.706193265431438e-35, 2: 0.5556031562372118, 3: -0.12661734906119,
           4: -1.4732579244500381, 5: 4397.233847397341, 6: -1.141864063971705,
           7: 1.443148014381498, 8: 9.739235081}


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
                         betas=(1.0, 1.96, 3, 5),   # <-- 1.0 added this week, for
                                                     #     an exploitation-leaning comparison
                         noise_assumption=1e-6, initial_guess=0.1):
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
            fitted_ls = model.kernel_.length_scale
            pinned = abs(fitted_ls - lb) < 1e-3

            fit_rows.append({
                'kernel_type': kname, 'length_scale_lower_bound': lb,
                'fitted_length_scale': round(fitted_ls, 4), 'pinned_at_bound': pinned,
                'log_marginal_likelihood': round(lml, 2),
            })

            for beta in betas:
                idx = np.argmax(post_mean + beta * post_std)
                full_rows.append({
                    'kernel_type': kname, 'length_scale_lower_bound': lb, 'beta': beta,
                    'next_x': candidates[idx].round(4).tolist(),
                    'pred_mean': round(post_mean[idx], 4),
                    'pred_std': round(post_std[idx], 4),
                })

    fit_summary = pd.DataFrame(fit_rows).sort_values(
        'log_marginal_likelihood', ascending=False).reset_index(drop=True)
    return fit_summary, pd.DataFrame(full_rows)


def load_function_data(function_id):
    X0 = np.load(f'data/initial_data/function_{function_id}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{function_id}/initial_outputs.npy')
    X = np.vstack([X0, week1_X[function_id], week2_X[function_id], week3_X[function_id]])
    Y = np.concatenate([Y0, [week1_Y[function_id], week2_Y[function_id], week3_Y[function_id]]])
    return X, Y


if __name__ == '__main__':
    best_per_function = []

    # --- step 1: fit each function, save the full tuning grid per function ---
    for function_id in range(1, 9):
        X, Y = load_function_data(function_id)
        fit_summary, full_grid = explore_tuning_grid(X, Y)
        fit_summary.to_csv(f'fit_summary_function{function_id}.csv', index=False)
        full_grid.to_csv(f'full_grid_function{function_id}.csv', index=False)

        best_row = fit_summary.iloc[0]
        best_per_function.append({
            'function': function_id, 'n_obs': len(Y),
            'best_kernel_type': best_row['kernel_type'],
            'best_length_scale_lower_bound': best_row['length_scale_lower_bound'],
            'fitted_length_scale': best_row['fitted_length_scale'],
            'pinned_at_bound': best_row['pinned_at_bound'],
            'log_marginal_likelihood': best_row['log_marginal_likelihood'],
        })

    overview = pd.DataFrame(best_per_function)
    overview.to_csv('best_config_overview_all_functions.csv', index=False)

    # --- step 2: build the beta=1.0 vs beta=1.96 comparison table ---
    rows = []
    for i in range(1, 9):
        fit = pd.read_csv(f'fit_summary_function{i}.csv')
        full = pd.read_csv(f'full_grid_function{i}.csv')
        best = fit.iloc[0]  # top-LML (kernel, bound) combo for this function

        Y0 = np.load(f'data/initial_data/function_{i}/initial_outputs.npy')
        current_best = np.append(Y0, [week1_Y[i], week2_Y[i], week3_Y[i]]).max()

        row = {'function': i, 'kernel': best.kernel_type,
               'pinned': best.pinned_at_bound, 'current_best': round(current_best, 4)}

        for beta in [1.0, 1.96]:
            sub = full[
                (full.kernel_type == best.kernel_type) &
                (full.length_scale_lower_bound == best.length_scale_lower_bound) &
                (full.beta == beta)
            ]
            nx = ast.literal_eval(sub.iloc[0]['next_x'])
            row[f'beta{beta}_next_x'] = [round(v, 4) for v in nx]
            row[f'beta{beta}_pred_mean'] = sub.iloc[0]['pred_mean']
            row[f'beta{beta}_pred_std'] = sub.iloc[0]['pred_std']

        rows.append(row)

    candidates_df = pd.DataFrame(rows)
    candidates_df.to_csv('week4_candidates.csv', index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 50)
    print(candidates_df[['function', 'kernel', 'pinned', 'current_best',
                          'beta1.0_next_x', 'beta1.0_pred_mean',
                          'beta1.96_next_x', 'beta1.96_pred_mean', 'beta1.96_pred_std']
                         ].to_string(index=False))
    print("\nSaved week4_candidates.csv (and fit_summary_functionN.csv / full_grid_functionN.csv per function)")
