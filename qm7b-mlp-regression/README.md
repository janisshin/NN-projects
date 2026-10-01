# QM7b MLP Regression

A TensorFlow/Keras multilayer perceptron for predicting molecular first excitation energy from HOMO and LUMO energies in the QM7b dataset.

This project originated as an MSE 542 course exercise. The original notebook is preserved as `training_experiments.ipynb`; the cleaned implementation below removes assignment scaffolding and separates the workflow into reusable modules.

## Task

Predict:

- `e1_zindo` — first excitation energy

from:

- `homo_zindo`
- `lumo_zindo`

The data are split into training and test sets with `random_state=111`. Input features are standardized using statistics fit on the training split.

## Baseline model

The baseline network in the original notebook contains:

1. two-unit dense hidden layer with ReLU
2. 20-unit dense hidden layer with ReLU
3. linear scalar output

Training uses Adam and mean squared error.

The saved notebook reports a held-out test MSE of approximately **0.80** for the baseline run.

## Model experiments

The notebook also explores how several modeling choices affect test error:

| Experiment | Recorded test MSE |
| --- | ---: |
| Baseline ReLU model | 0.80 |
| Exponential activation experiment | 0.78 |
| Adam learning rate = 0.1 | 1.33 |
| Adam learning rate = 1.0 | 3.48 |
| Extra hidden layer with learning rate = 0.1 | 1.18 |

These are historical outputs from the saved run, not guaranteed reproduction targets.

The experiments demonstrate the effect of activation functions, optimization hyperparameters, and network depth on regression performance.

## Clean implementation

- `data.py` — QM7b loading, train/test splitting, and feature standardization
- `model.py` — configurable regression MLP
- `train.py` — model fitting
- `evaluate.py` — MSE and RMSE evaluation
- `run_example.py` — minimal end-to-end example
- `training_experiments.ipynb` — original notebook and experiment outputs
- `requirements.txt` — Python dependencies

## Data

The original notebook downloaded `qm7b.csv` from a University of Washington course resource. The portfolio implementation expects a local file:

```
qm7b.csv
```

The dataset itself is not duplicated here.

## Provenance

Original source repository: https://github.com/janisshin/MSE541
