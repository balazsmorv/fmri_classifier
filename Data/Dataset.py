import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class LatentFMRIDataset(Dataset):

    def __init__(self, data_dir: str, reduced_dim = None):
        super().__init__()
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(filepath_or_buffer=os.path.join(self.data_dir, 'labels.csv'))
        self.reduced_dim = reduced_dim
        if self.reduced_dim is not None:
            self.pca = PCA(n_components=reduced_dim)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, item: int):
        row = self.labels_df.iloc[item]
        return {
            'data': np.load(file=os.path.join(self.data_dir, f'{row["FILE_ID"]}.npy')),
            'label': row['DX_GROUP'],
            'time_slice': row['TIME_SLICE']
        }

    def get_all_items(self):
        all_data = np.ndarray(shape=(self.__len__(), 4, 16, 18))
        for i, ind in enumerate(self.labels_df.index):
            file_id = self.labels_df['FILE_ID'][ind]
            timestep = self.labels_df['TIME_SLICE'][ind]
            all_data[i] = np.load(file=os.path.join(self.data_dir, f'{file_id}-{timestep}.npy'))
        all_labels = self.labels_df['DX_GROUP'].to_numpy(dtype=int)
        normalized_data = (all_data - all_data.min()) / (all_data.max() - all_data.min())
        return {
            'X': normalized_data if self.reduced_dim is None else self.pca.fit_transform(normalized_data.reshape((normalized_data.shape[0], -1))),
            'y': all_labels
        }

    def get_items_from_class(self, cls: int):
        all_items = self.get_all_items()
        all_x = all_items['X']
        all_y = all_items['y']
        return {
            'X': all_x[all_y == cls],
            'y': all_y[all_y == cls]
        }
