from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = ["Young's modulus (10^6 psi)", "Melting point (°F)"]
TARGET = "Oxidation at 500C"


def load_granta(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def make_binary_target(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: 0 if value == "Excellent" else 1
    ).astype(int)


def prepare_binary_classification(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 111,
):
    x = data[FEATURES].to_numpy()
    y = make_binary_target(data[TARGET])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler
