import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans


def cluster_som_nodes(
    som: MiniSom,
    n_clusters: int = 4,
    random_state: int = 55,
) -> tuple[np.ndarray, KMeans]:
    weights = som.get_weights()
    flat_weights = weights.reshape(-1, weights.shape[-1])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    cluster_labels = kmeans.fit_predict(flat_weights)

    return cluster_labels.reshape(weights.shape[:2]), kmeans
