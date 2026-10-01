# Materials MLP Classification

A TensorFlow/Keras multilayer perceptron for predicting high-temperature oxidation tolerance from tabular materials properties.

## Project overview

This project was completed for MSE 542 in December 2020. The binary classification task predicts whether a material has **Excellent** oxidation tolerance at 500°C or falls into a combined **Other** category.

The model uses two input features:

- Young's modulus
- melting point

Inputs are standardized using statistics fit on the training split.

## Model

The cleaned implementation uses a two-hidden-layer multilayer perceptron:

1. input layer
2. dense hidden layer
3. dense hidden layer
4. sigmoid output

The original notebook used `tanh` hidden activations in its baseline model and explored `relu` and `tanh` during hyperparameter tuning.

## Training and tuning

The original notebook:

- split the data into training and test sets with `random_state=111`
- standardized features using the training split
- trained a baseline two-hidden-layer Keras classifier
- used randomized 3-fold cross-validation to tune optimizer, batch size, and learning rate
- performed a second randomized search over hidden-layer widths and activation functions
- evaluated the selected model on the held-out test set

The saved notebook records:

- baseline test accuracy: **0.79**
- one tuned model test accuracy: **0.81**

Because neural-network training is stochastic, these values are historical results from the saved run rather than guaranteed reproduction targets.

## Clean implementation

The cleaned code separates the original workflow into reusable modules:

- `data.py` — dataset loading, binary target construction, train/test split, and standardization
- `model.py` — configurable two-hidden-layer Keras MLP
- `train.py` — model fitting and randomized cross-validated hyperparameter search
- `evaluate.py` — test accuracy and log-loss calculation
- `run_example.py` — end-to-end example
- `training_experiments.ipynb` — original coursework notebook preserved unchanged

The cleaned search code uses current TensorFlow/Keras APIs rather than the deprecated `KerasClassifier` wrapper that appeared in the original notebook.

## Historical multiclass section

The final multiclass exercise in the original notebook is preserved for provenance but is **not included in the cleaned implementation**.

The submitted code used a single-output network with `tanh` activation and binary cross-entropy, which does not implement the multiclass formulation described by the assignment. A correct mutually exclusive multiclass classifier would use one output per class with softmax and an appropriate categorical cross-entropy loss.

The portfolio implementation therefore focuses only on the valid binary-classification work rather than silently rewriting the historical result.

## Data

The notebook uses `grantadata.p`, a materials-property dataset supplied with the course. The portfolio repository does not duplicate that course data.

The expected local path for the example is:

```
grantadata.p
```

## Provenance

Original coursework repository: https://github.com/janisshin/MSE541

The original source repository remains the provenance record. This portfolio version preserves the submitted notebook while presenting the valid binary neural-network workflow in a cleaner, reusable form.
