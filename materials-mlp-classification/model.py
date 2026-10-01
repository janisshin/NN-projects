from dataclasses import dataclass

from tensorflow import keras


@dataclass
class MLPConfig:
    hidden_units_1: int = 8
    hidden_units_2: int = 8
    activation_1: str = "tanh"
    activation_2: str = "tanh"
    learning_rate: float = 0.001


def build_binary_mlp(input_dim: int, config: MLPConfig) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(config.hidden_units_1, activation=config.activation_1),
            keras.layers.Dense(config.hidden_units_2, activation=config.activation_2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
