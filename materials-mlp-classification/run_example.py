from pathlib import Path

from data import load_granta, prepare_binary_classification
from evaluate import evaluate_binary_classifier
from model import MLPConfig
from train import fit_model, randomized_cv_search


DATA_PATH = Path("grantadata.p")


def main() -> None:
    data = load_granta(DATA_PATH)
    x_train, x_test, y_train, y_test, _ = prepare_binary_classification(data)

    best_params, cv_accuracy = randomized_cv_search(
        x_train,
        y_train,
        n_iter=20,
        cv=3,
        epochs=100,
        batch_size=45,
    )

    print("Best CV parameters:", best_params)
    print(f"Mean CV accuracy: {cv_accuracy:.3f}")

    model, _ = fit_model(
        x_train,
        y_train,
        MLPConfig(**best_params),
        epochs=1000,
        batch_size=45,
    )

    metrics = evaluate_binary_classifier(model, x_test, y_test)
    print(metrics)


if __name__ == "__main__":
    main()
