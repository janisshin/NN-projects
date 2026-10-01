import numpy as np
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.metrics import accuracy_score

from model import MLPConfig, build_binary_mlp


SEARCH_SPACE = {
    "hidden_units_1": list(range(4, 20)),
    "hidden_units_2": list(range(4, 20)),
    "activation_1": ["relu", "tanh"],
    "activation_2": ["relu", "tanh"],
    "learning_rate": [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
}


def fit_model(
    x_train,
    y_train,
    config: MLPConfig,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.33,
    verbose: int = 0,
):
    model = build_binary_mlp(x_train.shape[1], config)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    return model, history


def randomized_cv_search(
    x_train,
    y_train,
    n_iter: int = 20,
    cv: int = 3,
    epochs: int = 100,
    batch_size: int = 45,
    random_state: int = 111,
):
    rng = np.random.RandomState(random_state)
    candidates = list(
        ParameterSampler(
            SEARCH_SPACE,
            n_iter=n_iter,
            random_state=rng,
        )
    )

    splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    best_config = None
    best_score = -np.inf

    for params in candidates:
        fold_scores = []

        for train_index, val_index in splitter.split(x_train, y_train):
            config = MLPConfig(**params)
            model = build_binary_mlp(x_train.shape[1], config)
            model.fit(
                x_train[train_index],
                y_train.iloc[train_index],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

            predictions = (
                model.predict(x_train[val_index], verbose=0).ravel() >= 0.5
            ).astype(int)
            fold_scores.append(
                accuracy_score(y_train.iloc[val_index], predictions)
            )

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_config = params

    return best_config, best_score
