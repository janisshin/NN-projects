import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


def predict_scores(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    scoring: str = "legacy_sigmoid",
) -> np.ndarray:
    """Return class-1 scores.

    scoring="legacy_sigmoid" reproduces the original notebook:
    sigmoid(logit_for_class_1).

    scoring="softmax" returns the class-1 probability implied by the
    two-logit cross-entropy model.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                features = batch[0]
            else:
                features = batch

            features = features.to(device)
            logits = model(features)

            if scoring == "legacy_sigmoid":
                batch_scores = torch.sigmoid(logits[:, 1])
            elif scoring == "softmax":
                batch_scores = torch.softmax(logits, dim=1)[:, 1]
            else:
                raise ValueError(
                    "scoring must be 'legacy_sigmoid' or 'softmax'."
                )

            scores.append(batch_scores.cpu())

    return torch.cat(scores).numpy()


def auroc(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    scoring: str = "legacy_sigmoid",
) -> float:
    labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])
    scores = predict_scores(model, dataset, batch_size, device, scoring=scoring)
    return float(roc_auc_score(labels, scores))


def write_predictions(scores: np.ndarray, path: str) -> None:
    np.savetxt(path, scores, fmt="%.8f")
