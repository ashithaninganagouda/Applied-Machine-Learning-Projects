from src.data_loader import load_libsvm_multilabel
from src.preprocessing import normalize_features
from src.models import svm_binary_relevance
from src.train import train
from src.evaluate import evaluate

NUM_FEATURES = 30438
NUM_LABELS = 22

print("Loading data...")
X_train, Y_train = load_libsvm_multilabel(
    "data/raw/tmc2007_train.svm",
    NUM_FEATURES,
    NUM_LABELS
)

X_test, Y_test = load_libsvm_multilabel(
    "data/raw/tmc2007_test.svm",
    NUM_FEATURES,
    NUM_LABELS
)

X_train = normalize_features(X_train)
X_test = normalize_features(X_test)

model = svm_binary_relevance()

print("Training...")
model = train(model, X_train, Y_train)

print("Evaluating...")
evaluate(model, X_test, Y_test)
