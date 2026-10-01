# Genomic Sequence CNN for Transcription-Factor Binding

A PyTorch convolutional neural network for classifying 150-bp genomic sequences as transcription-factor ChIP-seq peaks or matched flanking non-peak regions.

## Project overview

This project was completed for GENOME 541 in April 2025. The goal was to build and tune sequence classifiers that distinguish experimentally observed ChIP-seq binding regions from nearby genomic background.

The input to each model is a 150-bp DNA sequence represented as a 4 × 150 one-hot encoded tensor over A, C, G, and T.

## Model

The effective network used in the forward pass contains:

1. 1D convolution
2. Batch normalization
3. ReLU
4. 1D convolution
5. Batch normalization
6. ReLU
7. Dropout
8. Max pooling
9. Fully connected layer
10. ReLU + dropout
11. Two-logit classification output

The historical notebook also contains unused experimental convolutional layers from architecture exploration. Those are not part of the executed forward pass. The cleaned `model.py` exposes only the effective architecture.

## Training and model selection

Models were developed separately for transcription-factor datasets.

Optuna was used to search over:

- batch size: 1, 2, 4, 8
- convolution kernel size: 6, 12, 18, 24
- number of first-layer filters: 16, 64, 128, 256, 512
- optimizer: Adam or SGD
- learning rate

Hyperparameter trials trained on the provided training split and selected configurations using validation accuracy. The selected model was then reinitialized and retrained on the combined training + validation data before test evaluation or prediction generation.

Loss: cross-entropy on two output logits.

## Data

The source repository contains 14 K562 ChIP-seq datasets.

Ten datasets include labeled train/validation/test splits used for model development:

- CREB3L1
- eGFP-CUX1
- eGFP-ELK1
- eGFP-ETV1
- eGFP-FOXJ2
- eGFP-KLF13
- eGFP-NR2C2
- MAX
- NR2F1
- PKNOX1

Four additional datasets were treated as evaluation sets for prediction generation:

- CTCF
- CEBPB
- NRF1
- MGA

The large sequence files are not duplicated in this portfolio repository. See [DATA.md](DATA.md) for provenance and expected paths.

## Recorded test AUROC

The original notebook records the following test-set AUROC values for the ten labeled development datasets:

| Dataset | AUROC |
| --- | ---: |
| CREB3L1 | 0.6494 |
| eGFP-CUX1 | 0.5893 |
| eGFP-ELK1 | 0.7590 |
| eGFP-ETV1 | 0.5841 |
| eGFP-FOXJ2 | 0.6384 |
| eGFP-KLF13 | 0.6027 |
| eGFP-NR2C2 | 0.6377 |
| MAX | 0.9330 |
| NR2F1 | 0.8263 |
| PKNOX1 | 0.9197 |

These values are reported from the saved notebook outputs and reflect separate model-training runs for each transcription factor.

## Clean implementation

The repeated notebook workflow has been factored into reusable modules:

- `data.py` — FASTA parsing, one-hot encoding, dataset construction, and file discovery
- `model.py` — the effective two-layer 1D CNN
- `train.py` — model fitting, validation accuracy, Optuna search, and retraining on train + validation
- `evaluate.py` — class-1 probability prediction, AUROC calculation, and prediction export
- `run_example.py` — end-to-end example for one transcription factor

The refactor is intended to preserve the original experimental workflow while making the implementation easier to inspect and reuse. The saved notebook remains the record of the original runs and reported results.

## Files

- `training_experiments.ipynb` — original training, hyperparameter search, evaluation, and held-out prediction workflow
- `data.py` — cleaned data pipeline
- `model.py` — cleaned CNN definition
- `train.py` — cleaned training and tuning workflow
- `evaluate.py` — cleaned evaluation helpers
- `run_example.py` — minimal end-to-end usage example
- `util.py` — original FASTA and one-hot encoding helpers retained for comparison
- `DATA.md` — dataset provenance and local directory expectations
- `requirements.txt` — Python dependencies

## Provenance

Original coursework repository: https://github.com/janisshin/GENOME-541

The original repository remains the provenance record. This portfolio version separates the historical notebook from a cleaned, reusable implementation.
