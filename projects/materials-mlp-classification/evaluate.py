from sklearn.metrics import accuracy_score, log_loss


def evaluate_binary_classifier(model, x_test, y_test) -> dict:
    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities)),
    }
