import numpy as np
from scipy.sparse import csr_matrix

def load_libsvm_multilabel(filepath, num_features, num_labels):
    data, rows, cols = [], [], []
    labels = []

    with open(filepath, "r") as f:
        for row_idx, line in enumerate(f):
            parts = line.strip().split()

            # Labels
            y = np.zeros(num_labels)
            for l in parts[0].split(","):
                y[int(l)] = 1
            labels.append(y)

            # Features
            for item in parts[1:]:
                col, val = item.split(":")
                rows.append(row_idx)
                cols.append(int(col) - 1)
                data.append(float(val))

    X = csr_matrix((data, (rows, cols)),
                   shape=(row_idx + 1, num_features))
    Y = np.array(labels)

    return X, Y
