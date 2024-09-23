import numpy as np
from collections import defaultdict
import torch


def get_train_samples(X, y, m, n_samples):
    # Ensure n_samples is divisible by 10 (since there are 10 labels)
    if n_samples % 10 != 0:
        raise ValueError("n_samples must be divisible by 10.")

    # Get the unique domains and labels
    unique_domains = torch.unique(m)
    unique_labels = torch.arange(10)  # Assumes labels are 0-9 (10 labels)

    # Number of samples per label (uniform distribution across labels)
    samples_per_label = n_samples // 10

    # Prepare the output tensor (number_of_domains, n_samples, 10, 4096)
    num_domains = unique_domains.size(0)
    X_train = torch.zeros((num_domains, n_samples, 10, 4096), dtype=torch.float32)
    y_train = torch.zeros((num_domains, n_samples * 10), dtype=y.dtype)

    # To store the indices of selected samples
    selected_indices = []

    # Iterate over each domain
    for i, domain in enumerate(unique_domains):
        # Get indices corresponding to the current domain
        domain_indices = torch.where(m == int(domain))[0]

        # For each label, select samples uniformly
        for j, label in enumerate(unique_labels):
            # Get indices for the current label within this domain
            label_indices = domain_indices[
                torch.where(y[domain_indices] == int(label + 1))[0]
            ]

            if len(label_indices) < samples_per_label:
                raise ValueError(
                    f"Not enough samples for label {label.item() + 1} in domain {domain.item()}. Only {len(label_indices)} were found"
                )

            # Randomly sample from the available label indices
            selected_label_indices = label_indices[
                torch.randperm(len(label_indices))[:samples_per_label]
            ]

            # Store the selected samples in the output tensor
            X_train[i, :samples_per_label, j] = X[selected_label_indices]
            y_train[i, j * samples_per_label : (j + 1) * samples_per_label] = y[
                selected_label_indices
            ]

            # Append selected indices to the list
            selected_indices.append(selected_label_indices)

    # Concatenate all selected indices and flatten the list
    selected_indices = torch.cat(selected_indices)

    # Mask for remaining samples (samples not in selected_indices)
    mask = torch.ones(X.size(0), dtype=torch.bool)
    mask[selected_indices] = False

    # Split the original array into remaining samples and the selected samples
    X_test = X[mask]
    y_test = y[mask]
    m_test = m[mask]

    return X_train, y_train, X_test, y_test, m_test
