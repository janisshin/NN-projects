import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom


def plot_feature_correlation(subset: pd.DataFrame) -> None:
    correlation = subset.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(correlation, vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()


def plot_best_matching_units(som: MiniSom, data: np.ndarray) -> None:
    coordinates = np.array([som.winner(row) for row in data])
    plt.figure(figsize=(8, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=8)
    plt.xlabel("SOM x")
    plt.ylabel("SOM y")
    plt.title("Best-matching units")


def plot_feature_maps(
    som: MiniSom,
    feature_names: list[str],
) -> None:
    weights = som.get_weights()

    for index, feature in enumerate(feature_names):
        plt.figure(figsize=(6, 5))
        plt.pcolor(weights[:, :, index])
        plt.title(feature)
        plt.xlabel("SOM x")
        plt.ylabel("SOM y")
        plt.colorbar()


def plot_cluster_map(cluster_labels: np.ndarray) -> None:
    plt.figure(figsize=(7, 6))
    plt.pcolor(cluster_labels)
    plt.xlabel("SOM x")
    plt.ylabel("SOM y")
    plt.title("K-means clusters of SOM nodes")
    plt.colorbar()
