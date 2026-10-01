from pathlib import Path

from data import load_qm7b, prepare_regression_data
from evaluate import evaluate_regression
from model import MLPConfig
from train import fit_model


DATA_PATH = Path("qm7b.csv")


def main() -> None:
    data = load_qm7b(DATA_PATH)
    x_train, x_test, y_train, y_test, _ = prepare_regression_data(data)

    model, _ = fit_model(
        x_train,
        y_train,
        MLPConfig(),
        epochs=20,
        batch_size=32,
    )

    metrics = evaluate_regression(model, x_test, y_test)
    print(metrics)


if __name__ == "__main__":
    main()
