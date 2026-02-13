from sklearn.preprocessing import normalize

def normalize_features(X):
    return normalize(X, norm="l2")
