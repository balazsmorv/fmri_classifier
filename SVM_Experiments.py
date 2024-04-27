import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Data.Dataset import LatentFMRIDataset
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from scipy.linalg import orthogonal_procrustes
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


class SVMExperiment:

    def __init__(self, train_dir: str, test_dir: str, train_site_name: str, test_site_name: str, c_values: [int], kernel: str = 'rbf',
                 test_ratio: int = 0.15, log_path = '/Users/balazsmorvay/PycharmProjects/fmri_classifier/Experiments'):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_site_name = train_site_name
        self.test_site_name = test_site_name
        self.kernel = kernel
        self.test_ratio = test_ratio
        self.c_values = c_values
        self.log_path = log_path
        self.models = []
        self.metrics = []

        self.X_train, self.y_train, self.X_test, self.y_test = self.setup_data()

    def setup_data(self):
        train_data = LatentFMRIDataset(data_dir=self.train_dir).get_all_items()
        test_data = LatentFMRIDataset(data_dir=self.test_dir).get_all_items()
        return self.transform_data(train_data, test_data)

    def transform_data(self, train_data, test_data):
        X_train, y_train = train_data['X'], train_data['y']
        X_test, y_test = test_data['X'], test_data['y']
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
        return X_train, y_train, X_test, y_test

    def perform_experiment(self):
        for c in self.c_values:
            self.train_and_test(self.X_train, self.y_train, self.X_test, self.y_test, c=c)

    def train_and_test(self, X_train, y_train, X_test, y_test, c):
        trained_svm = self.train_svm(X_train=X_train, y_train=y_train, c=c)
        self.models.append(trained_svm)
        metrics = self.test_svm(model=trained_svm, X_test=X_test, y_test=y_test)
        self.metrics.append(metrics)

    def train_svm(self, X_train, y_train, c: int):
        model = SVC(kernel=self.kernel, C=c, class_weight='balanced', random_state=42, verbose=True)
        model.fit(X=X_train, y=y_train)
        return model

    def test_svm(self, model, X_test, y_test):
        test_predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, test_predictions, labels=model.classes_)
        self.make_confusion_matrix_image(cm, model)
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            'accuracy': ((tp + tn) / (tp + tn + fp + fn + 1e-6)),
            'recall': (tp / (tp + fn + 1e-6)),
            'precision': (tp / (tp + fp + 1e-6))
        }
        return metrics

    def make_confusion_matrix_image(self, cm, model):
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        display.plot()





class SVMExperimentWithLogging(SVMExperiment):

    experiment_description = ''

    def __init__(self, train_dir: str, test_dir: str, train_site_name: str, test_site_name: str, c_values: [int],
                 kernel: str = 'rbf',
                 test_ratio: int = 0.15, log_path='/Users/balazsmorvay/PycharmProjects/fmri_classifier/Experiments'):
        super().__init__(train_dir, test_dir, train_site_name, test_site_name, c_values, kernel, test_ratio, log_path)
        self.current_experiment_path = None

    def perform_experiment(self):
        experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_experiment(
            f"{experiment_time};Train:{self.train_site_name},test:{self.test_site_name},extra:{self.experiment_description}"
        )
        super().perform_experiment()

    def train_and_test(self, X_train, y_train, X_test, y_test, c):
        with mlflow.start_run():
            experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            experiment_name = f"{experiment_time};c={c},ker={self.kernel}"
            mlflow.set_tag("mlflow.runName", experiment_name)
            experiment_path = os.path.join(self.log_path, experiment_name)
            os.mkdir(path=experiment_path)
            self.current_experiment_path = experiment_path

            super().train_and_test(X_train, y_train, X_test, y_test, c)

            mlflow.sklearn.log_model(self.models[-1], experiment_name)
            mlflow.log_metrics(self.metrics[-1], synchronous=True)

    def train_svm(self, X_train, y_train, c: int):
        mlflow_tracking_params = {
            'Train site': self.train_site_name,
            'Train directory': self.train_dir,
            'Test site': self.test_site_name,
            'Test directory': self.test_dir,
            'Test_ratio': self.test_ratio,
            'Train_labels_mean': np.mean(y_train),
            'C': c,
            'Kernel': self.kernel,
            'extra info': self.experiment_description
        }
        mlflow.log_params(params=mlflow_tracking_params, synchronous=True)
        return super().train_svm(X_train, y_train, c)

    def test_svm(self, model, X_test, y_test) -> dict:
        mlflow.log_param(key='Test_labels_mean', value=np.mean(y_test))
        return super().test_svm(model, X_test, y_test)

    def make_confusion_matrix_image(self, cm, model):
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        display.plot().figure_.savefig(os.path.join(self.current_experiment_path, 'confusion_matrix.png'))
        plt.close('all')
        mlflow.log_artifact(os.path.join(self.current_experiment_path, 'confusion_matrix.png'))




class SVMRidgeRegression(SVMExperimentWithLogging):

    experiment_description = 'Ridge Regression Transform'

    def __init__(self, train_dir: str, test_dir: str, train_site_name: str, test_site_name: str, c_values: [int],
                 lambda_: float, number_of_train_data_points=None, kernel: str = 'rbf',
                 test_ratio: int = 0.15, log_path='/Users/balazsmorvay/PycharmProjects/fmri_classifier/Experiments'):
        self.lambda_ = lambda_
        self.number_of_train_data_points = number_of_train_data_points
        super().__init__(train_dir, test_dir, train_site_name, test_site_name, c_values, kernel, test_ratio, log_path)

    def transform_data(self, train_data, test_data):
        X_train, y_train, X_test, y_test = super().transform_data(train_data, test_data)
        W = self.train_regression(X_train, X_test, y_train, y_test)
        return X_train, y_train, X_test @ W, y_test

    def train_regression(self, X_train, X_test, y_train, y_test):
        X_train_ones = X_train[y_train == 1]
        X_train_twos = X_train[y_train == 2]
        X_test_ones = X_test[y_test == 1]
        X_test_twos = X_test[y_test == 2]
        if self.number_of_train_data_points is None: self.number_of_train_data_points = np.min([X_train_ones.shape[0],
                                                                                                X_test_ones.shape[0],
                                                                                                X_train_twos.shape[0],
                                                                                                X_test_twos.shape[0]])
        print(f'{self.number_of_train_data_points} data points from each class are used for regression')
        Y = np.concatenate(
            [X_train_ones[0:self.number_of_train_data_points], X_train_twos[0:self.number_of_train_data_points]])
        X = np.concatenate(
            [X_test_ones[0:self.number_of_train_data_points], X_test_twos[0:self.number_of_train_data_points]])

        I = np.ones(1152)
        W = np.linalg.solve(a=X.T @ X + self.lambda_ * I, b=X.T @ Y) # p*p size matrix
        return W

    def train_svm(self, X_train, y_train, c: int):
        mlflow.log_param(key='Regression num. of training data point from each class',
                         value=self.number_of_train_data_points)
        mlflow.log_param(key='lambda', value=self.lambda_)
        return super().train_svm(X_train, y_train, c)





class SVMProcrustes(SVMExperiment):

    number_of_train_data_points = None
    experiment_description = 'Orthogonal Procrustes'

    def __init__(self, train_dir: str, test_dir: str, train_site_name: str, test_site_name: str, c_values: [int],
                 kernel: str = 'rbf',
                 test_ratio: int = 0.15, log_path='/Users/balazsmorvay/PycharmProjects/fmri_classifier/Experiments'):
        super().__init__(train_dir, test_dir, train_site_name, test_site_name, c_values, kernel, test_ratio, log_path)

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

        
class SVMAffine(SVMExperiment):
    experiment_description = "A@nyu + b affine"
    def __init__(self, train_dir: str, test_dir: str, train_site_name: str, test_site_name: str, c_values: [int],
                 A1, b1, A2, b2, kernel: str = 'rbf',
                 test_ratio: int = 0.15, log_path='/Users/balazsmorvay/PycharmProjects/fmri_classifier/Experiments'):
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        super().__init__(train_dir, test_dir, train_site_name, test_site_name, c_values, kernel, test_ratio, log_path)

    def transform_data(self, train_data, test_data):
        X_train, y_train, X_test, y_test = super().transform_data(train_data, test_data)
        X_train_ones = X_train[y_train == 1]
        X_train_twos = X_train[y_train == 2]
        # Transform the train set, do not transform the test set
        X_train_ones = self.A1.dot(X_train_ones.transpose()) + self.b1
        X_train_twos = self.A2.dot(X_train_twos.transpose()) + self.b2
        X_train = np.concatenate([X_train_ones.T, X_train_twos.T])

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        return X_train, y_train, X_test, y_test




