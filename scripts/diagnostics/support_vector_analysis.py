"""
Fits an SVM classifier separating "high-performance" (top third of observed
Y) from "low-performance" points, and reports which points the SVM actually
used as support vectors -- i.e. points sitting near the boundary between
the two classes. A high fraction of support vectors relative to total
points is itself informative: it means the classes are not cleanly
separated, so the "boundary" is fuzzy rather than sharp.

Run from the directory containing initial_data/.
"""

import numpy as np
import warnings
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# round 1-3 query points + results (edit/extend these each week)
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


def load_function_data(function_id):
    X0 = np.load(f'data/initial_data/function_{function_id}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{function_id}/initial_outputs.npy')
    X = np.vstack([X0, week1_X[function_id], week2_X[function_id], week3_X[function_id]])
    Y = np.concatenate([Y0, [week1_Y[function_id], week2_Y[function_id], week3_Y[function_id]]])
    return X, Y


def find_support_vectors(function_id, top_fraction_percentile=67, C=1.0):
    X, Y = load_function_data(function_id)

    threshold = np.percentile(Y, top_fraction_percentile)
    labels = (Y >= threshold).astype(int)

    svc = SVC(kernel='rbf', C=C)
    svc.fit(X, labels)
    sv_idx = svc.support_

    print(f"Function {function_id}: {len(sv_idx)} of {len(Y)} points are support vectors "
          f"({100*len(sv_idx)/len(Y):.0f}%)")
    print(f"high-performance threshold: y >= {threshold:.4f}\n")

    order = np.argsort(Y[sv_idx])
    for idx in sv_idx[order]:
        print(f"  x={X[idx].round(3)}  y={Y[idx]:.3f}  label={labels[idx]}")

    return sv_idx, threshold


if __name__ == '__main__':
    find_support_vectors(function_id=8)
