from pathlib import Path

import torch

from data import SequenceDataset, discover_dataset_files, load_features, load_labels
from evaluate import auroc
from train import train_selected_model, tune_hyperparameters


DATA_DIR = Path("data/encode-chip")
DATASET = "CREB3L1"


def load_split(files: dict, split: str) -> SequenceDataset:
    features = load_features(files[f"{split}-fasta"])
    labels = load_labels(files[f"{split}-label"])
    return SequenceDataset(features, labels)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_files = discover_dataset_files(DATA_DIR, [DATASET])[DATASET]

    train_dataset = load_split(dataset_files, "train")
    validation_dataset = load_split(dataset_files, "val")
    test_dataset = load_split(dataset_files, "test")

    study = tune_hyperparameters(
        train_dataset,
        validation_dataset,
        device=device,
    )
    print("Best hyperparameters:", study.best_trial.params)

    model, _ = train_selected_model(
        train_dataset,
        validation_dataset,
        best_params=study.best_trial.params,
        device=device,
        epochs=120,
    )

    score = auroc(
        model,
        test_dataset,
        batch_size=study.best_trial.params["batch_size"],
        device=device,
    )
    print(f"Test AUROC: {score:.4f}")


if __name__ == "__main__":
    main()
