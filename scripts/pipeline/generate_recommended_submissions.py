"""
Builds the "recommended submission" table: one row per function, using each
function's BEST-SUPPORTED (kernel, length_scale_bound) combo -- i.e. the
top row of its fit_summary_functionN.csv, ranked by log marginal likelihood
-- evaluated at the standard beta=1.96.

Requires structured_tuning_all_functions.py to have already been run, since
it reads the fit_summary_functionN.csv / full_grid_functionN.csv files that
script produces.
"""

import ast
import numpy as np
import pandas as pd

week1_Y = {1: 2.6752879910742468e-9, 2: -0.06913022970475102, 3: -0.027253680430408482,
           4: -0.3932940148544044, 5: 1703.1512205236875, 6: -0.41036378008423247,
           7: 2.0679435112701356, 8: 9.8674177220056}
week2_Y = {1: 3.7759561298146587e-34, 2: 0.4900359752987052, 3: -0.0665391852481089,
           4: -3.2314120937555297, 5: 1880.1864969021228, 6: -0.6644176392227166,
           7: 2.3848190050793328, 8: 9.8169820770504}

BETA_STANDARD = 1.96

rows = []
for i in range(1, 9):
    fit = pd.read_csv(f'fit_summary_function{i}.csv')
    full = pd.read_csv(f'full_grid_function{i}.csv')

    # top row of fit_summary = best (kernel, bound) by log marginal likelihood
    best = fit.iloc[0]

    # pull out the beta=1.96 row for that same (kernel, bound) combo
    sub = full[
        (full.kernel_type == best.kernel_type) &
        (full.length_scale_lower_bound == best.length_scale_lower_bound) &
        (full.beta == BETA_STANDARD)
    ]
    next_x = ast.literal_eval(sub.iloc[0]['next_x'])  # stored as a string in the CSV

    X0 = np.load(f'data/initial_data/function_{i}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{i}/initial_outputs.npy')
    Y_all = np.append(Y0, [week1_Y[i], week2_Y[i]])
    current_best_y = Y_all.max()

    rows.append({
        'function': i,
        'kernel_used': best.kernel_type,
        'length_scale_bound': best.length_scale_lower_bound,
        'pinned_at_bound': best.pinned_at_bound,     # True -> treat with extra caution
        'current_best_y': round(current_best_y, 6),
        'recommended_next_x': [round(v, 6) for v in next_x],
        'predicted_mean': sub.iloc[0]['pred_mean'],
        'predicted_std': sub.iloc[0]['pred_std'],
    })

df = pd.DataFrame(rows)
df.to_csv('week3_recommended_submissions.csv', index=False)

pd.set_option('display.width', 160)
pd.set_option('display.max_colwidth', 60)
print(df.to_string(index=False))
print("\nSaved week3_recommended_submissions.csv")
