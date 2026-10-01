# NN-projects

A curated portfolio of neural-network work spanning genomic sequence modeling, tabular prediction, molecular regression, and unsupervised representation learning.

## What this repo demonstrates

- building and training neural networks in PyTorch and TensorFlow/Keras
- convolutional modeling of biological sequences
- hyperparameter optimization and cross-validation
- held-out evaluation with AUROC, accuracy, MSE, and RMSE
- model architecture experiments and failure analysis
- unsupervised neural representations with self-organizing maps

## Projects

| Project | Problem | Methods |
| --- | --- | --- |
| [Genomic sequence CNN](genomic-sequence-cnn/) | Predict transcription-factor binding from 150-bp DNA sequences | PyTorch, 1D CNNs, Optuna, AUROC |
| [Materials MLP classification](materials-mlp-classification/) | Predict high-temperature oxidation tolerance from materials properties | TensorFlow/Keras, MLP classification, randomized cross-validation |
| [QM7b MLP regression](qm7b-mlp-regression/) | Predict first excitation energy from HOMO/LUMO energies | TensorFlow/Keras, MLP regression, architecture and optimizer experiments |
| [Molecular self-organizing map](molecular-self-organizing-map/) | Organize molecular descriptor space and identify property regions | Self-organizing maps, Mordred descriptors, K-means |

## Repository structure

Each project contains:

- a concise project README
- the original experimental notebook
- cleaned, reusable implementation files where appropriate
- dependency and data/provenance notes

The original source repositories remain the provenance record for historical coursework and experiments. This repository reorganizes that work for readability without rewriting the original results.
