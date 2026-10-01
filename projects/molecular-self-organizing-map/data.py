from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["nAtom", "nBonds", "bpol", "apol", "TopoPSA"]


def load_descriptors(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def select_and_standardize(
    data: pd.DataFrame,
    features: list[str] = FEATURES,
) -> tuple[pd.DataFrame, np.ndarray]:
    subset = data[features].copy()
    values = subset.to_numpy(dtype=float)

    means = values.mean(axis=0)
    stds = values.std(axis=0)

    if np.any(stds == 0):
        raise ValueError("At least one selected descriptor has zero variance.")

    standardized = (values - means) / stds
    return subset, standardized
