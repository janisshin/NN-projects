import torch
import numpy as np



def make_ohe_features_tensor(features_file):
    nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Open the file in read mode
    with open(features_file, "r") as file:
        # Read all lines
        lines = file.readlines()

    # Get even-numbered lines (indexing starts from 0, so even lines have odd indices)
    even_lines = [line.strip() for i, line in enumerate(lines, start=1) if i % 2 == 0]

    one_hot_sequences = []

    # Print or process the even-numbered lines
    for seq in even_lines:
        indices = torch.tensor([nt_to_idx[nt] for nt in seq], dtype=torch.long)

        # One-hot encode with num_classes=4 (A, C, G, T)
        one_hot_encoded = torch.nn.functional.one_hot(indices, num_classes=4).float()
        one_hot_sequences.append(one_hot_encoded.T)

    # this reshape may pose a problem, maybe in the axis
    train_features = torch.concatenate(one_hot_sequences, axis=0)
    train_features = train_features.reshape((len(one_hot_sequences),4,150))

    return train_features

def make_ohe_labels_tensor(labels_file):
    with open(labels_file, "r") as file:
        lines = file.readlines()
    train_labels = np.array(lines, dtype=int)
    train_labels = torch.from_numpy(train_labels)

    # Convert labels into one-hot-encoding
    train_labels_ohe = torch.nn.functional.one_hot(train_labels, num_classes=2) 

    return train_labels_ohe.float()