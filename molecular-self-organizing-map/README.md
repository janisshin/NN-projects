# Molecular Self-Organizing Map

An unsupervised neural-network workflow for organizing molecular descriptor space with a self-organizing map (SOM), then clustering the learned SOM node representations with K-means.

## Project overview

This project was completed for MSE 542 coursework. The analysis used 2,500 molecules represented by Mordred molecular descriptors.

Five descriptors were selected as SOM inputs:

- nAtom — number of atoms
- nBonds — number of bonds
- bpol — bond polarizability
- apol — atomic polarizability
- TopoPSA — topological polar surface area

Each feature was standardized before training.

## SOM configuration

The original notebook used:

- grid size: 50 × 50
- input dimension: 5
- sigma: 1.0
- learning rate: 1.0
- neighborhood function: Gaussian
- PCA weight initialization
- training iterations: 5,000
- random seed: 0

The trained SOM maps molecules from five-dimensional descriptor space onto a two-dimensional grid while attempting to preserve neighborhood structure.

## Evaluation

The saved notebook reports:

| Metric | Value |
| --- | ---: |
| Topographic error | 0.0148 |
| Quantization error | 0.0630 |

Topographic error measures how often the first and second best-matching units are not adjacent on the SOM. Quantization error measures the average distance between each input sample and its best-matching unit.

These values are historical outputs from the saved coursework run.

## Representation analysis

The original analysis also:

- visualized the best-matching SOM node for each molecule
- inspected descriptor-specific SOM weight maps
- compared weight-map structure with pairwise feature correlations
- observed similar maps for highly correlated descriptors such as atom and bond counts

## K-means clustering over SOM nodes

After SOM training, the 50 × 50 × 5 weight tensor was reshaped into 2,500 five-dimensional node vectors.

K-means clustering was then applied to those learned node representations with:

- number of clusters: 4
- random state: 55

This separates the SOM representation into coarse molecular-property regions.

## Clean implementation

The cleaned implementation separates the notebook workflow into reusable modules:

- `data.py` — descriptor loading, feature selection, and standardization
- `model.py` — SOM configuration, training, and evaluation metrics
- `cluster.py` — K-means clustering of learned SOM node weights
- `visualize.py` — best-matching-unit, feature-map, correlation, and cluster visualizations
- `run_example.py` — end-to-end example
- `training_experiments.ipynb` — original coursework notebook preserved unchanged

The cleaned code preserves the original SOM and K-means configuration while removing assignment scaffolding.

## Data

The project expects a Mordred descriptor table named:

```
mordred_df.csv
```

The original notebook states that the dataset contains 2,500 molecules and was supplied through the course. It is not duplicated in this portfolio repository.

## Provenance

Original coursework repository: https://github.com/janisshin/MSE541

The original source repository remains the provenance record. This portfolio version preserves the submitted notebook while presenting the SOM workflow in a cleaner, reusable form.
