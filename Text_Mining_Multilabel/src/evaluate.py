from sklearn.metrics import hamming_loss, f1_score

def evaluate(model, X_test, Y_test):
    preds = model.predict(X_test)

    print("Hamming Loss:", hamming_loss(Y_test, preds))
    print("Micro F1:", f1_score(Y_test, preds, average="micro"))
    print("Macro F1:", f1_score(Y_test, preds, average="macro"))
