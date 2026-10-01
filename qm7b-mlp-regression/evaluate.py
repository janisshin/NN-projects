import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_regression(model, x_test, y_test) -> dict[str, float]:
    predictions = model.predict(x_test, verbose=0).ravel()
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))

    return {
        "mse": float(mse),
        "rmse": rmse,
    }
