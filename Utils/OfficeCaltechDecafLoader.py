import os
import numpy as np
import json
import matplotlib.pyplot as plt


class OfficeCaltechDecafDataset:

    def __init__(self, data_path: str, standardize=False, scale=False):
        self.X, self.y, self.m, self.fold_dict = self._load_data(
            data_path=data_path, standardize=standardize, scale=scale
        )
        self.domains = np.unique(self.m).astype(int)
        self.domain_names = ["Webcam", "Amazon", "dslr", "Caltech"]

    def _load_data(self, data_path: str, standardize=False, scale=False):
        dataset = np.load(os.path.join(data_path, "Objects_Decaf.npy"))
        with open(os.path.join(data_path, "Objects_crossval_index.json"), "r") as f:
            fold_dict = json.loads(f.read())
        X = dataset[:, :-2]
        if standardize:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif scale:
            # Feature scaling
            X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

        y = np.array(dataset[:, -2], dtype=int)
        m = np.array(dataset[:, -1], dtype=int)
        return X, y, m, fold_dict

    def plot_label_occurences(self):
        label_domain_occurence = np.zeros((4, 10), dtype=int)
        for label in range(10):
            for i, domain in enumerate(self.domains):
                domain_indices = np.where(self.m == domain)
                domain_labels = self.y[domain_indices]
                label_indices = np.where(domain_labels == label + 1)
                label_domain_occurence[i][label] = len(label_indices[0])

        fig, ax = plt.subplots()
        cax = ax.imshow(label_domain_occurence, cmap="viridis", aspect="auto")
        for (i, j), val in np.ndenumerate(label_domain_occurence):
            ax.text(j, i, f"{val}", ha="center", va="center", color="white")
        fig.colorbar(cax)
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(str(i) for i in range(10))
        ax.set_yticklabels(self.domain_names)
        plt.title("Domain label occurences")
        plt.show()
