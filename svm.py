import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from Data.Dataset import LatentFMRIDataset

def train_svm(c: float):

    dataset = LatentFMRIDataset(data_dir='TODO')
    all_data_items = dataset.get_all_items()
    X = all_data_items['X']
    y = all_data_items['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    model = SVC(kernel='rbf', C=c, class_weight='balanced', random_state=42, verbose=True)
    model.fit(X=X, y=y)
