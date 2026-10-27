"""
Structured exploration of the FOUR main GP-UCB tuning points:
  1. kernel TYPE           (RBF vs Matern vs RationalQuadratic)
  2. length_scale LOWER BOUND
  3. alpha (noise assumption)   <-- NEW this week
  4. beta                  (exploration/exploitation trade-off)

Cost tiers, same principle as before:
- kernel TYPE, length_scale BOUND, and alpha all require a fresh model.fit()
  call, since they change what the GP is even allowed to learn.
- beta does NOT require refitting -- it only recombines an already-fitted
  model's mean/std -- so it stays the cheap inner loop.

Why alpha was added: for Function 4, the default alpha=1e-6 (assume
near-zero noise) forced the GP to explain a lot of local up-and-down
bouncing in the data as real spatial structure, which produced an
unsupported positive-mean extrapolation right next to the current best.
Letting the model assume a small amount of noise (found via log marginal
likelihood, same as the other axes) gave a much better-supported fit and
a completely different, more conservative prediction at the same point.
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
                         alphas=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1),   # <-- NEW axis
                         betas=(1.0, 1.96, 3, 5),
                         initial_guess=0.1):
    """Fits one GP per (kernel_type x length_scale_lower_bound x alpha),
    evaluates every beta against each fit. Returns (fit_summary, full_grid)."""
    n_dims = X.shape[1]
    candidates = _build_candidates(n_dims)
    fit_rows, full_rows = [], []

    for kname in kernel_types:
        for lb in length_scale_lower_bounds:
            for alpha in alphas:
                kernel = KERNEL_BUILDERS[kname](initial_guess, (lb, length_scale_upper_bound))
                model = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                                  normalize_y=True, n_restarts_optimizer=10,
                                                  random_state=RNG_SEED)
                model.fit(X, Y)

                lml = model.log_marginal_likelihood(model.kernel_.theta)
                post_mean, post_std = model.predict(candidates, return_std=True)
                fitted_ls = model.kernel_.length_scale
                pinned = abs(fitted_ls - lb) < 1e-3

                fit_rows.append({
                    'kernel_type': kname, 'length_scale_lower_bound': lb, 'alpha': alpha,
                    'fitted_length_scale': round(fitted_ls, 4), 'pinned_at_bound': pinned,
                    'log_marginal_likelihood': round(lml, 2),
                })

                for beta in betas:
                    idx = np.argmax(post_mean + beta * post_std)
                    full_rows.append({
                        'kernel_type': kname, 'length_scale_lower_bound': lb, 'alpha': alpha,
                        'beta': beta,
                        'next_x': candidates[idx].round(4).tolist(),
                        'pred_mean': round(post_mean[idx], 4),
                        'pred_std': round(post_std[idx], 4),
                    })

    fit_summary = pd.DataFrame(fit_rows).sort_values(
        'log_marginal_likelihood', ascending=False).reset_index(drop=True)
    return fit_summary, pd.DataFrame(full_rows)


def print_report(function_id, fit_summary, full_grid, top_n=3):
    pd.set_option('display.width', 160)
    print("=" * 100)
    print(f"FUNCTION {function_id}")
    print("=" * 100)
    print("STEP 1 -- which (kernel, length_scale bound, alpha) does the data actually support?")
    print(fit_summary.head(10).to_string(index=False))

    flagged = fit_summary[fit_summary['pinned_at_bound']]
    if len(flagged):
        print(f"\n  Note: {len(flagged)} config(s) pinned at their length_scale lower bound.")

    default_alpha_rank = fit_summary.reset_index().query('alpha == 1e-6')['index'].min()
    print(f"\n  The project-default alpha=1e-6 ranks #{default_alpha_rank + 1} of "
          f"{len(fit_summary)} configs by LML for this function.")

    print(f"\nSTEP 2 -- beta sensitivity for the top {top_n} best-supported configs")
    top_configs = fit_summary.head(top_n)[['kernel_type', 'length_scale_lower_bound', 'alpha']]
    for _, row in top_configs.iterrows():
        subset = full_grid[
            (full_grid.kernel_type == row.kernel_type) &
            (full_grid.length_scale_lower_bound == row.length_scale_lower_bound) &
            (full_grid.alpha == row.alpha)
        ]
        print(f"\n-- {row.kernel_type}, lower_bound={row.length_scale_lower_bound}, alpha={row.alpha} --")
        print(subset[['beta', 'next_x', 'pred_mean', 'pred_std']].to_string(index=False))
    print()


if __name__ == '__main__':
    # ---- example: Function 4, with all 5 rounds folded in ----
    X0 = np.load('data/initial_data/function_4/initial_inputs.npy')
    Y0 = np.load('data/initial_data/function_4/initial_outputs.npy')
    rounds_x = [
        [0.394868, 0.425405, 0.352448, 0.464929],
        [0.451289, 0.457435, 0.214355, 0.422482],
        [0.3682, 0.4922, 0.4132, 0.4558],
        [0.4488, 0.3474, 0.4204, 0.4541],
        [0.4974, 0.437, 0.3929, 0.4997],
    ]
    rounds_y = [-0.3932940148544044, -3.2314120937555297, -1.4732579244500381,
                -0.27656674599950604, -2.337100677643711]
    X = np.vstack([X0] + rounds_x)
    Y = np.append(Y0, rounds_y)

    fit_summary, full_grid = explore_tuning_grid(X, Y)
    print_report(function_id=4, fit_summary=fit_summary, full_grid=full_grid)

    fit_summary.to_csv('fit_summary_function4_with_alpha.csv', index=False)
    full_grid.to_csv('full_grid_function4_with_alpha.csv', index=False)
    print("\nSaved fit_summary_function4_with_alpha.csv and full_grid_function4_with_alpha.csv")
