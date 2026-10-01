from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def read_fasta_sequences(path: str | Path) -> list[str]:
    """Read sequences from a simple two-line-per-record FASTA file."""
    lines = Path(path).read_text().splitlines()
    return [line.strip().upper() for i, line in enumerate(lines, start=1) if i % 2 == 0]


def one_hot_encode_sequences(
    sequences: Iterable[str],
    sequence_length: int = 150,
) -> torch.Tensor:
    """Convert DNA sequences to tensors with shape (N, 4, sequence_length)."""
    encoded = []

    for sequence in sequences:
        if len(sequence) != sequence_length:
            raise ValueError(
                f"Expected sequence length {sequence_length}, got {len(sequence)}."
            )
        try:
            indices = torch.tensor([BASES[base] for base in sequence], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"Unsupported nucleotide: {exc.args[0]}") from exc

        encoded.append(
            torch.nn.functional.one_hot(indices, num_classes=4).float().T
        )

    if not encoded:
        return torch.empty((0, 4, sequence_length), dtype=torch.float32)

    return torch.stack(encoded)


def load_features(path: str | Path, sequence_length: int = 150) -> torch.Tensor:
    return one_hot_encode_sequences(
        read_fasta_sequences(path),
        sequence_length=sequence_length,
    )


def load_labels(path: str | Path) -> torch.Tensor:
    """Load integer class labels as a one-dimensional LongTensor."""
    labels = np.loadtxt(path, dtype=int)
    labels = np.atleast_1d(labels)
    return torch.from_numpy(labels).long()


class SequenceDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        if len(features) != len(labels):
            raise ValueError("Features and labels must contain the same number of samples.")
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def discover_dataset_files(
    data_dir: str | Path,
    dataset_names: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Path]]:
    """Map each transcription factor to its train/val/test FASTA and label files."""
    data_dir = Path(data_dir)
    allowed = set(dataset_names) if dataset_names is not None else None
    result: Dict[str, Dict[str, Path]] = {}

    for path in data_dir.iterdir():
        if not path.is_file() or "_" not in path.name:
            continue

        dataset = path.name.split("_")[0]
        if allowed is not None and dataset not in allowed:
            continue

        suffix = ".".join(path.name.split(".")[-2:])
        key = suffix.replace(".", "-")
        result.setdefault(dataset, {})[key] = path

    return result
