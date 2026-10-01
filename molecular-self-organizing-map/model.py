from dataclasses import dataclass

import numpy as np
from minisom import MiniSom


@dataclass
class SOMConfig:
    width: int = 50
    height: int = 50
    sigma: float = 1.0
    learning_rate: float = 1.0
    iterations: int = 5000
    random_seed: int = 0


def train_som(data: np.ndarray, config: SOMConfig) -> MiniSom:
    som = MiniSom(
        config.width,
        config.height,
        data.shape[1],
        sigma=config.sigma,
        learning_rate=config.learning_rate,
        neighborhood_function="gaussian",
        random_seed=config.random_seed,
    )
    som.pca_weights_init(data)
    som.train(data, config.iterations)
    return som


def som_errors(som: MiniSom, data: np.ndarray) -> dict[str, float]:
    return {
        "topographic_error": float(som.topographic_error(data)),
        "quantization_error": float(som.quantization_error(data)),
    }
