"""
Checks whether a GP's prediction at a proposed candidate point is
trustworthy, using two independent signals:

1. Nearest-neighbor distance -- is this candidate close to existing data,
   or a genuine leap into unexplored territory?
2. Leave-one-out calibration -- refit the GP with each point held out in
   turn, and check whether the true value actually falls within the
   predicted uncertainty. If most held-out points fall within ~2 standard
   deviations of their prediction, the model's uncertainty is trustworthy;
   if not (or if recent points are poorly calibrated specifically), that's
   a warning sign for trusting a nearby extrapolation.

Run from the directory containing initial_data/.
"""

import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

warnings.filterwarnings('ignore')

KERNEL_BUILDERS = {
    'RBF': lambda ls, bounds: RBF(length_scale=ls, length_scale_bounds=bounds),
    'Matern_nu1.5': lambda ls, bounds: Matern(length_scale=ls, length_scale_bounds=bounds, nu=1.5),
    'Matern_nu2.5': lambda ls, bounds: Matern(length_scale=ls, length_scale_bounds=bounds, nu=2.5),
}


def check_extrapolation_trust(X, Y, proposed_point, kernel_name='RBF',
                               length_scale_bound=(0.05, 2.0), initial_guess=0.1,
                               noise_assumption=1e-6, loo_restarts=5):
    proposed_point = np.array(proposed_point)

    # --- 1. nearest-neighbor distance ---
    dists = np.linalg.norm(X - proposed_point, axis=1)
    nearest_idx = np.argmin(dists)
    print(f"Proposed point: {proposed_point}")
    print(f"Nearest existing point: {X[nearest_idx].round(4)}  "
          f"y={Y[nearest_idx]:.4f}  distance={dists[nearest_idx]:.4f}\n")

    # --- 2. fit on all data, predict at the proposed point ---
    kernel = KERNEL_BUILDERS[kernel_name](initial_guess, length_scale_bound)
    model = GaussianProcessRegressor(kernel=kernel, alpha=noise_assumption,
                                      normalize_y=True, n_restarts_optimizer=10,
                                      random_state=42)
    model.fit(X, Y)
    pred_mean, pred_std = model.predict([proposed_point], return_std=True)
    print(f"Fitted kernel: {model.kernel_}")
    print(f"Predicted mean: {pred_mean[0]:.4f}  predicted std: {pred_std[0]:.4f}")
    print(f"-> {(pred_mean[0] - Y.max()) / pred_std[0]:.2f} standard deviations "
          f"above current best ({Y.max():.4f})\n")

    # --- 3. leave-one-out calibration check ---
    print("=== Leave-one-out calibration check ===")
    errors_in_std_units = []
    for i in range(len(Y)):
        mask = np.ones(len(Y), dtype=bool)
        mask[i] = False
        k = KERNEL_BUILDERS[kernel_name](initial_guess, length_scale_bound)
        m = GaussianProcessRegressor(kernel=k, alpha=noise_assumption, normalize_y=True,
                                      n_restarts_optimizer=loo_restarts, random_state=42)
        m.fit(X[mask], Y[mask])
        pred, std = m.predict([X[i]], return_std=True)
        err = (Y[i] - pred[0]) / std[0] if std[0] > 1e-9 else np.nan
        errors_in_std_units.append(err)
        print(f"  held-out point {i}: true={Y[i]:.3f}  "
              f"predicted={pred[0]:.3f}+-{std[0]:.3f}  error={err:.2f} std-devs")

    errs = np.array(errors_in_std_units)
    frac_bad = np.mean(np.abs(errs) > 2)
    print(f"\nFraction of held-out points with |error| > 2 std-devs: {frac_bad:.2f} "
          f"(should be ~0.05 if well-calibrated)")

    # flag if the most recent points specifically are poorly calibrated --
    # more relevant to trusting a NEW nearby prediction than overall average calibration
    n_recent = min(3, len(errs))
    recent_errs = errs[-n_recent:]
    print(f"Most recent {n_recent} points' calibration errors: {recent_errs.round(2)}")
    if np.any(np.abs(recent_errs) > 2):
        print("  WARNING: at least one recent point was poorly calibrated -- "
              "treat nearby extrapolations with extra caution.")

    return pred_mean[0], pred_std[0], errs


if __name__ == '__main__':
    # ---- example: Function 4, checking the week 6 candidate point ----
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

    check_extrapolation_trust(X, Y, proposed_point=[0.3771, 0.3295, 0.347, 0.4157],
                               kernel_name='Matern_nu2.5')
