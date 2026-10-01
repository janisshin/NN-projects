from dataclasses import dataclass

import optuna
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from model import SequenceCNN


@dataclass
class TrainConfig:
    batch_size: int
    kernel_size: int
    num_filters: int
    optimizer: str
    learning_rate: float
    epochs: int = 120


def make_optimizer(
    model: torch.nn.Module,
    name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def fit_model(
    model: torch.nn.Module,
    dataset: Dataset,
    config: TrainConfig,
    device: torch.device,
) -> list[float]:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = make_optimizer(model, config.optimizer, config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    losses: list[float] = []

    model.to(device)

    for _ in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(features)
            sample_count += len(features)

        losses.append(epoch_loss / max(sample_count, 1))

    return losses


def validation_accuracy(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return correct / max(total, 1)


def tune_hyperparameters(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
    timeout_seconds: int = 900,
) -> optuna.Study:
    """Replicate the original Optuna search with one epoch per trial."""

    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8])
        kernel_size = trial.suggest_categorical("kernel_size", [6, 12, 18, 24])
        num_filters = trial.suggest_categorical(
            "num_filters", [16, 64, 128, 256, 512]
        )
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-2, log=True)

        model = SequenceCNN(
            kernel_size=kernel_size,
            num_filters=num_filters,
        ).to(device)

        config = TrainConfig(
            batch_size=batch_size,
            kernel_size=kernel_size,
            num_filters=num_filters,
            optimizer=optimizer_name,
            learning_rate=learning_rate,
            epochs=1,
        )
        fit_model(model, train_dataset, config, device)

        return validation_accuracy(
            model,
            validation_dataset,
            batch_size=batch_size,
            device=device,
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=timeout_seconds)
    return study


def train_selected_model(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    best_params: dict,
    device: torch.device,
    epochs: int = 120,
) -> tuple[SequenceCNN, list[float]]:
    """Retrain a fresh model on train + validation using selected hyperparameters."""
    combined_dataset = ConcatDataset([train_dataset, validation_dataset])

    model = SequenceCNN(
        kernel_size=best_params["kernel_size"],
        num_filters=best_params["num_filters"],
    )

    config = TrainConfig(
        batch_size=best_params["batch_size"],
        kernel_size=best_params["kernel_size"],
        num_filters=best_params["num_filters"],
        optimizer=best_params["optimizer"],
        learning_rate=best_params["lr"],
        epochs=epochs,
    )

    losses = fit_model(model, combined_dataset, config, device)
    return model, losses
