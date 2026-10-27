"""
Systematic sensitivity analysis for a GP-UCB Bayesian optimisation pipeline.

Sweeps the three main tuning points -- kernel choice, length_scale_bounds,
and beta -- and reports how the fitted model and the proposed next query
point change under each combination.

Design principle (kept from the beta sweep earlier): anything that requires
REFITTING the GP (kernel type, length_scale_bounds) is looped on the outside,
since it's expensive. Anything that only recombines an already-fitted
model's predictions (beta) is looped on the inside, since it's cheap --
each kernel config gets exactly one .fit() and one .predict() call, no
matter how many beta values you try against it.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

warnings.filterwarnings('ignore')

RNG_SEED = 42


def sensitivity_analysis(X, Y, kernel_configs=None, betas=None,
                          n_candidates_highdim=20000, n_grid_2d=50,
                          noise_assumption=1e-6):
    """
    Parameters
    ----------
    X, Y : your observed inputs/outputs for one function (already includes
           any prior rounds you've folded in)
    kernel_configs : list of dicts, each describing one kernel to try.
        Each dict needs:
          'name'   : label for the output table
          'kernel' : a callable that RETURNS a fresh sklearn kernel object
                     (must be a callable, not an already-built kernel,
                     since kernels get mutated in place during fit)
        If None, a sensible default set is used (see below).
    betas : list of beta values to try against EACH kernel fit.
        If None, defaults to [1.0, 1.96, 3, 5, 8].

    Returns
    -------
    pandas.DataFrame, one row per (kernel_config x beta) combination, with
    columns for the fitted hyperparameters, the log marginal likelihood
    (higher = kernel explains the data better -- lets you compare kernel
    choices on the same footing), and the resulting next_x.
    """
    n_dims = X.shape[1]

    if kernel_configs is None:
        kernel_configs = [
            {'name': 'RBF (tight bound 0.05)',
             'kernel': lambda: RBF(length_scale=0.1, length_scale_bounds=(0.05, 2.0))},
            {'name': 'RBF (loose bound 0.2)',
             'kernel': lambda: RBF(length_scale=0.1, length_scale_bounds=(0.2, 2.0))},
            {'name': 'RBF (loose bound 0.5)',
             'kernel': lambda: RBF(length_scale=0.1, length_scale_bounds=(0.5, 2.0))},
            {'name': 'Matern nu=1.5',
             'kernel': lambda: Matern(length_scale=0.1, length_scale_bounds=(0.05, 2.0), nu=1.5)},
            {'name': 'Matern nu=2.5',
             'kernel': lambda: Matern(length_scale=0.1, length_scale_bounds=(0.05, 2.0), nu=2.5)},
            {'name': 'RationalQuadratic',
             'kernel': lambda: RationalQuadratic(length_scale=0.1, length_scale_bounds=(0.05, 2.0))},
        ]

    if betas is None:
        betas = [1.0, 1.96, 3, 5, 8]

    # candidate set is built once, identically for every kernel config,
    # so comparisons are apples-to-apples
    rng = np.random.RandomState(RNG_SEED)
    if n_dims == 2:
        g = np.linspace(0, 1, n_grid_2d)
        Xg, Yg = np.meshgrid(g, g)
        candidates = np.column_stack([Xg.ravel(), Yg.ravel()])
    else:
        candidates = rng.uniform(0, 1, size=(n_candidates_highdim, n_dims))

    rows = []
    for cfg in kernel_configs:
        kernel = cfg['kernel']()
        model = GaussianProcessRegressor(kernel=kernel, alpha=noise_assumption,
                                          normalize_y=True, n_restarts_optimizer=10,
                                          random_state=RNG_SEED)
        model.fit(X, Y)

        lml = model.log_marginal_likelihood(model.kernel_.theta)
        post_mean, post_std = model.predict(candidates, return_std=True)
        std_span = post_std.max() - post_std.min()

        for beta in betas:
            ucb_scores = post_mean + beta * post_std
            best_idx = np.argmax(ucb_scores)
            next_x = candidates[best_idx]

            rows.append({
                'kernel_config': cfg['name'],
                'fitted_kernel': str(model.kernel_),
                'log_marginal_likelihood': round(lml, 3),
                'post_std_span': round(std_span, 4),  # near 0 => posterior is
                                                        # basically flat -> a
                                                        # collapse warning sign
                'beta': beta,
                'next_x': next_x.round(4).tolist(),
                'pred_mean_at_next_x': round(post_mean[best_idx], 4),
                'pred_std_at_next_x': round(post_std[best_idx], 4),
            })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    # ---- example usage on Function 2 (edit for whichever function you're analysing) ----
    week1_X2 = [0.816326, 0.959183]; week1_Y2 = -0.06913022970475102
    week2_X2 = [0.673469, 0.897959]; week2_Y2 = 0.4900359752987052

    X0 = np.load('data/initial_data/function_2/initial_inputs.npy')
    Y0 = np.load('data/initial_data/function_2/initial_outputs.npy')
    X = np.vstack([X0, week1_X2, week2_X2])
    Y = np.concatenate([Y0, [week1_Y2, week2_Y2]])

    df = sensitivity_analysis(X, Y)

    pd.set_option('display.width', 160)
    pd.set_option('display.max_colwidth', 40)

    # log_marginal_likelihood only depends on kernel_config, not beta -- so
    # show ONE row per kernel config first, ranked by how well each kernel
    # explains the data, before drilling into the full beta breakdown
    print("=== kernel comparison (best-explaining kernel first) ===")
    print(df.drop_duplicates('kernel_config')[
        ['kernel_config', 'fitted_kernel', 'log_marginal_likelihood', 'post_std_span']
    ].sort_values('log_marginal_likelihood', ascending=False).to_string(index=False))

    print("\n=== full kernel x beta grid ===")
    print(df[['kernel_config', 'beta', 'next_x', 'pred_mean_at_next_x', 'pred_std_at_next_x']]
          .to_string(index=False))

    df.to_csv('sensitivity_analysis_function2.csv', index=False)
    print("\nSaved full results to sensitivity_analysis_function2.csv")
