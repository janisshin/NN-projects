from model import MLPConfig, build_regression_mlp


def fit_model(
    x_train,
    y_train,
    config: MLPConfig,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.33,
    verbose: int = 0,
):
    model = build_regression_mlp(x_train.shape[1], config)
    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    return model, history
