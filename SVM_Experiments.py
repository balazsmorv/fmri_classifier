import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Data.Dataset import LatentFMRIDataset
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from scipy.linalg import orthogonal_procrustes

from SVMLogger import SVMLogger

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


class SVMExperiment:

    def __init__(self, train_site_name: str,
                 test_site_name: str, c_values: [int], kernel: str = 'rbf', test_ratio: int = 0.15,
                 logger = SVMLogger(), experiment_description='', data_shape = (4, 16, 18), X_train = None,
                 y_train = None, X_test = None, y_test = None, train_dir: str = None, test_dir: str = None):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_site_name = train_site_name
        self.test_site_name = test_site_name
        self.kernel = kernel
        self.test_ratio = test_ratio
        self.c_values = c_values
        self.logger = logger
        self.experiment_description = experiment_description
        self.data_shape = data_shape
        self.models = []
        self.metrics = []

        if X_train is None:
            self.X_train, self.y_train, self.X_test, self.y_test = self.setup_data()
        else:
            self.X_train, self.y_train, self.X_test, self.y_test = self.transform_data(X_train, y_train, X_test, y_test)

    def setup_data(self):
        train_data = LatentFMRIDataset(data_dir=self.train_dir, data_shape=self.data_shape).get_all_items()
        test_data = LatentFMRIDataset(data_dir=self.test_dir, data_shape=self.data_shape).get_all_items()
        X_train, y_train = train_data['X'], train_data['y']
        X_test, y_test = test_data['X'], test_data['y']
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
        return X_train, y_train, X_test, y_test

    def transform_data(self, X_train, y_train, X_test, y_test):
        return X_train, y_train, X_test, y_test

    def perform_experiment(self):
        self.logger.before_experiment(train_site_name=self.train_site_name, test_site_name=self.test_site_name,
                                   experiment_description=self.experiment_description)

        for c in self.c_values:
            self.train_and_test(self.X_train, self.y_train, self.X_test, self.y_test, c=c)

        self.logger.after_experiment()

    def train_and_test(self, X_train, y_train, X_test, y_test, c):
        self.logger.before_train_and_test(c=c, kernel=self.kernel)
        trained_svm = self.train_svm(X_train=X_train, y_train=y_train, c=c)
        self.models.append(trained_svm)
        metrics = self.test_svm(model=trained_svm, X_test=X_test, y_test=y_test)
        self.metrics.append(metrics)
        self.logger.after_train_and_test(c=c, kernel=self.kernel, models=self.models, metrics=self.metrics)

    def train_svm(self, X_train, y_train, c: int):
        self.logger.before_train(train_site_name=self.train_site_name, test_site_name=self.test_site_name,
                                 train_dir=self.train_dir, test_dir=self.test_dir, test_ratio=self.test_ratio,
                                 train_labels_mean=np.mean(y_train), c=c, kernel=self.kernel, experiment_description=self.experiment_description)
        model = SVC(kernel=self.kernel, C=c, class_weight='balanced', random_state=42, verbose=True)
        model.fit(X=X_train, y=y_train)
        return model

    def test_svm(self, model, X_test, y_test):
        self.logger.before_test(test_labels_mean=np.mean(y_test))
        test_predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, test_predictions, labels=model.classes_)
        self.logger.make_confusion_matrix_image(cm=cm, model=model)
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            'accuracy': ((tp + tn) / (tp + tn + fp + fn + 1e-6)),
            'recall': (tp / (tp + fn + 1e-6)),
            'precision': (tp / (tp + fp + 1e-6))
        }
        return metrics


class SVMAffine(SVMExperiment):
    def __init__(self, train_site_name: str, test_site_name: str, c_values: [int],
                 A1, b1, A2, b2, kernel: str = 'rbf', test_ratio: int = 0.15, logger = SVMLogger(),
                 experiment_description="A@nyu + b affine", data_shape=(4, 16, 18), X_train = None,
                 y_train = None, X_test = None, y_test = None, train_dir: str = None, test_dir: str = None):
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        super().__init__(train_site_name, test_site_name, c_values, kernel, test_ratio, logger,
                         experiment_description=experiment_description, data_shape=data_shape, train_dir=train_dir,
                         test_dir=test_dir, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    def transform_data(self, X_train, y_train, X_test, y_test):
        X_train_ones = X_train[y_train == 1]
        X_train_twos = X_train[y_train == 2]
        # Transform the train set, do not transform the test set
        X_train_ones = self.A1.dot(X_train_ones.transpose()) + self.b1
        X_train_twos = self.A2.dot(X_train_twos.transpose()) + self.b2
        X_train = np.concatenate([X_train_ones.T, X_train_twos.T])

        y_train_ones = np.ones(shape=X_train_ones.shape[1])
        y_train_twos = np.ones(shape=X_train_twos.shape[1]) + 1
        y_train = np.concatenate([y_train_ones, y_train_twos])

        return X_train, y_train, X_test, y_test

class SVMAffineWithoutClass(SVMExperiment):
    def __init__(self, train_site_name: str, test_site_name: str, c_values: [int],
                 A, b, kernel: str = 'rbf', test_ratio: int = 0.15, logger = SVMLogger(),
                 experiment_description="A@nyu + b affine", data_shape=(4, 16, 18), X_train = None,
                 y_train = None, X_test = None, y_test = None, train_dir: str = None, test_dir: str = None):
        self.A = A
        self.b = b
        super().__init__(train_site_name, test_site_name, c_values, kernel, test_ratio, logger,
                         experiment_description=experiment_description, data_shape=data_shape, train_dir=train_dir,
                         test_dir=test_dir, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    def transform_data(self, X_train, y_train, X_test, y_test):
        return (self.A.dot(X_train.T) + self.b).T, y_train, X_test, y_test


class SVMProcrustes(SVMExperiment):
    number_of_train_data_points = None
    experiment_description = 'Orthogonal Procrustes'

    def __init__(self, train_dir: str, test_dir: str, train_site_name: str, test_site_name: str, c_values: [int],
                 kernel: str = 'rbf',
                 test_ratio: int = 0.15):
        super().__init__(train_dir, test_dir, train_site_name, test_site_name, c_values, kernel, test_ratio)

    def transform_data(self, train_data, test_data):
        X_train, y_train, X_test, y_test = super().transform_data(train_data, test_data)

        X_train_ones = X_train[y_train == 1]
        X_train_twos = X_train[y_train == 2]
        X_test_ones = X_test[y_test == 1]
        X_test_twos = X_test[y_test == 2]
        self.number_of_train_data_points = np.min([X_train_ones.shape[0],
                                                   X_test_ones.shape[0],
                                                   X_train_twos.shape[0],
                                                   X_test_twos.shape[0]])
        print(f'{self.number_of_train_data_points} data points from each class are used for procrustes')
        Target_matrix = np.concatenate(
            [X_train_ones[0:self.number_of_train_data_points], X_train_twos[0:self.number_of_train_data_points]])
        Matrix_to_be_mapped = np.concatenate(
            [X_test_ones[0:self.number_of_train_data_points], X_test_twos[0:self.number_of_train_data_points]])
        R, scale = orthogonal_procrustes(A=Matrix_to_be_mapped, B=Target_matrix)

        return X_train, y_train, X_test @ R, y_test

