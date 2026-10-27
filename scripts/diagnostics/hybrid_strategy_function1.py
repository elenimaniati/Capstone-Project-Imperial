"""
Hybrid surrogate strategy for Function 1, inspired by Wu et al. (2020)
"Higher Performance for AutoML: The Benefit of Various Ensemble Bayesian
Optimization Strategy" (NeurIPS 2020 BBO Challenge).

Rather than trusting the GP alone (which has been pinned at its degenerate
length_scale bound for Function 1 since week 1), this runs TWO independent
strategies against the same shared observation history each round:

  Strategy A: GP + UCB (the existing pipeline)
  Strategy B: RBF interpolation + a distance-based exploration bonus
              (the "stochastic RBF" idea behind pysot, one of the two
              surrogates in Wu et al.'s best-performing pairing)

Plan: alternate which strategy's suggestion actually gets submitted each
round. BOTH strategies are refit on the complete history every round,
regardless of which one's suggestion was actually queried -- this is the
key mechanism from the paper: every surrogate learns from every result,
not just the ones it proposed itself.

Run from the directory containing initial_data/.
"""

import numpy as np
import warnings
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

warnings.filterwarnings('ignore')
RNG_SEED = 42


def strategy_a_gp_ucb(X, Y, candidates, beta=1.96,
                       kernel_name='Matern_nu1.5', length_scale_bound=(0.05, 2.0), alpha=0.1):
    """The existing pipeline: GP fit by LML, scored by UCB."""
    kernel = Matern(length_scale=0.1, length_scale_bounds=length_scale_bound, nu=1.5)
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                      n_restarts_optimizer=10, random_state=RNG_SEED)
    model.fit(X, Y)
    post_mean, post_std = model.predict(candidates, return_std=True)
    ucb_scores = post_mean + beta * post_std
    idx = np.argmax(ucb_scores)
    return candidates[idx], post_mean[idx], post_std[idx], model.kernel_


def strategy_b_rbf_distance(X, Y, candidates, exploration_weight=0.5):
    """
    A second, independent surrogate: RBF interpolation for predicted value,
    combined with a distance-to-nearest-point exploration bonus -- the core
    idea behind pysot's stochastic RBF method, one of the two surrogates
    in Wu et al.'s best-performing ensemble pairing (pysot + turbo).

    exploration_weight in [0, 1]: 0 = pure exploitation of the RBF surface,
    1 = pure distance-based exploration (ignores predicted value entirely).
    Regis & Shoemaker's original method cycles this weight over rounds
    (e.g. 0.3, 0.5, 0.7, 0.95, 1.0) rather than fixing it -- worth doing
    the same here across future rounds rather than using one fixed value.
    """
    rbf = RBFInterpolator(X, Y, kernel='thin_plate_spline', smoothing=0.01)
    predicted_value = rbf(candidates)

    dists_to_nearest = np.min(
        np.linalg.norm(candidates[:, None, :] - X[None, :, :], axis=2), axis=1
    )

    # normalize both to [0,1] so they're comparable regardless of Y's scale
    v_norm = (predicted_value - predicted_value.min()) / (np.ptp(predicted_value) + 1e-12)
    d_norm = (dists_to_nearest - dists_to_nearest.min()) / (np.ptp(dists_to_nearest) + 1e-12)

    score = (1 - exploration_weight) * v_norm + exploration_weight * d_norm
    idx = np.argmax(score)
    return candidates[idx], predicted_value[idx], dists_to_nearest[idx]


def build_candidates_2d(n_grid=50):
    g = np.linspace(0, 1, n_grid)
    Xg, Yg = np.meshgrid(g, g)
    return np.column_stack([Xg.ravel(), Yg.ravel()])


if __name__ == '__main__':
    # ---- Function 1's full history through week 6 ----
    X0 = np.load('data/initial_data/function_1/initial_inputs.npy')
    Y0 = np.load('data/initial_data/function_1/initial_outputs.npy')
    rounds_x = [
        [0.5, 0.5], [0.6, 0.3], [0.7755, 0.7959], [0.0, 1.0],
        [0.84, 0.265], [0.66, 0.69],
    ]
    rounds_y = [
        2.6752879910742468e-9, 3.7759561298146587e-34, 5.706193265431438e-35,
        0.0, 5.547267530712984e-124, -0.00023213423561594152,
    ]
    X = np.vstack([X0] + rounds_x)
    Y = np.append(Y0, rounds_y)
    print(f"Function 1: {len(Y)} total observations\n")

    candidates = build_candidates_2d()

    print("=== Strategy A: GP + UCB (existing pipeline) ===")
    next_x_a, mean_a, std_a, fitted_kernel = strategy_a_gp_ucb(X, Y, candidates)
    print(f"  fitted kernel: {fitted_kernel}")
    print(f"  next_x: {next_x_a.round(4)}  pred_mean={mean_a:.5f}  pred_std={std_a:.5f}\n")

    print("=== Strategy B: RBF interpolation + distance exploration ===")
    for w in [0.3, 0.5, 0.7, 0.95]:
        next_x_b, pred_b, dist_b = strategy_b_rbf_distance(X, Y, candidates, exploration_weight=w)
        print(f"  exploration_weight={w}: next_x={next_x_b.round(4)}  "
              f"RBF_pred={pred_b:.5f}  distance_to_nearest={dist_b:.4f}")

    print("\nPlan: alternate which strategy's point gets submitted each round.")
    print("Both strategies refit on the FULL shared history every round, regardless")
    print("of which strategy's point was actually queried -- this is what lets each")
    print("model benefit from data the other strategy generated.")
