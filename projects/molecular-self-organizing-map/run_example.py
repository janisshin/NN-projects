from pathlib import Path

from cluster import cluster_som_nodes
from data import FEATURES, load_descriptors, select_and_standardize
from model import SOMConfig, som_errors, train_som
from visualize import plot_best_matching_units, plot_cluster_map, plot_feature_maps


DATA_PATH = Path("mordred_df.csv")


def main() -> None:
    data = load_descriptors(DATA_PATH)
    _, standardized = select_and_standardize(data)

    config = SOMConfig()
    som = train_som(standardized, config)

    print(som_errors(som, standardized))

    cluster_labels, _ = cluster_som_nodes(
        som,
        n_clusters=4,
        random_state=55,
    )

    plot_best_matching_units(som, standardized)
    plot_feature_maps(som, FEATURES)
    plot_cluster_map(cluster_labels)


if __name__ == "__main__":
    main()
