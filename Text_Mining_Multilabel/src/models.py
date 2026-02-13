from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def svm_binary_relevance():
    return OneVsRestClassifier(LinearSVC())

def logistic_binary_relevance():
    return OneVsRestClassifier(
        LogisticRegression(
            max_iter=300,
            solver="liblinear"
        )
    )
