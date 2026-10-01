from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = ["homo_zindo", "lumo_zindo"]
TARGET = "e1_zindo"


def load_qm7b(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_regression_data(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 111,
):
    x = data[FEATURES].to_numpy()
    y = data[TARGET].to_numpy()

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
