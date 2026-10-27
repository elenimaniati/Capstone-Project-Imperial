"""
Compares 4 classifiers on the same "high vs low performance" task (top
third of observed Y = 'high') across all 8 functions, using leave-one-out
cross-validation (necessary given how few observations each function has).

Run from the directory containing initial_data/.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

warnings.filterwarnings('ignore')

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


def compare_classifiers(top_fraction_percentile=67):
    loo = LeaveOneOut()
    rows = []

    for i in range(1, 9):
        X, Y = load_function_data(i)
        labels = (Y >= np.percentile(Y, top_fraction_percentile)).astype(int)

        log_acc = cross_val_score(LogisticRegression(max_iter=1000), X, labels, cv=loo).mean()
        linsvm_acc = cross_val_score(SVC(kernel='linear', C=1), X, labels, cv=loo).mean()
        rbfsvm_acc = cross_val_score(SVC(kernel='rbf', C=1), X, labels, cv=loo).mean()
        mlp_acc = cross_val_score(
            MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=42),
            X, labels, cv=loo
        ).mean()

        rows.append({
            'function': i, 'n_obs': len(Y),
            'logistic_acc': round(log_acc, 3),
            'linear_svm_acc': round(linsvm_acc, 3),
            'rbf_svm_acc': round(rbfsvm_acc, 3),
            'mlp_acc': round(mlp_acc, 3),
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = compare_classifiers()
    df.to_csv('classifier_comparison.csv', index=False)
    pd.set_option('display.width', 160)
    print(df.to_string(index=False))
    print("\nSaved classifier_comparison.csv")
