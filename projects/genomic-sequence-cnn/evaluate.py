import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


def predict_probabilities(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return probability of class 1 for each sample."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probabilities = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                features = batch[0]
            else:
                features = batch

            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(probs.cpu())

    return torch.cat(probabilities).numpy()


def auroc(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> float:
    labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])
    probabilities = predict_probabilities(model, dataset, batch_size, device)
    return float(roc_auc_score(labels, probabilities))


def write_predictions(probabilities: np.ndarray, path: str) -> None:
    np.savetxt(path, probabilities, fmt="%.8f")
