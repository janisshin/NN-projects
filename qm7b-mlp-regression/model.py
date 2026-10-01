from dataclasses import dataclass

from tensorflow import keras


@dataclass
class MLPConfig:
    first_hidden_units: int = 2
    second_hidden_units: int = 20
    first_activation: str = "relu"
    second_activation: str = "relu"
    learning_rate: float = 0.001
    extra_hidden_layer: bool = False


def build_regression_mlp(input_dim: int, config: MLPConfig) -> keras.Model:
    layers = [
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(config.first_hidden_units, activation=config.first_activation),
        keras.layers.Dense(config.second_hidden_units, activation=config.second_activation),
    ]

    if config.extra_hidden_layer:
        layers.append(
            keras.layers.Dense(
                config.second_hidden_units,
                activation=config.second_activation,
            )
        )

    layers.append(keras.layers.Dense(1))

    model = keras.Sequential(layers)
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model
