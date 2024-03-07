import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Data.Dataset import LatentFMRIDataset
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


def perform_experiment_train_on_whole_test_on_UM1(kernel: str = 'rbf'):
    nyu_dataset_directory = '/Users/balazsmorvay/PycharmProjects/VAE/Assets/NYU_all'
    um1_dataset_directory = '/Users/balazsmorvay/PycharmProjects/VAE/Assets/UM_1_all'
    test_ratio = 0.15

    nyu_data = LatentFMRIDataset(data_dir=nyu_dataset_directory).get_all_items()
    nyu_X_train, nyu_y_train = nyu_data['X'], nyu_data['y']

    um1_data = LatentFMRIDataset(data_dir=um1_dataset_directory).get_all_items()
    um1_X_train, X_test, um1_y_train, y_test = train_test_split(um1_data['X'], um1_data['y'], test_size=test_ratio)

    X_train = np.concatenate((nyu_X_train, um1_X_train))
    y_train = np.concatenate((nyu_y_train, um1_y_train))

    mlflow.set_experiment(f"({datetime.now()})Train:NYU+UM1,test:UM1")
    for c in np.arange(start=0.01, stop=5, step=0.5):
        with mlflow.start_run():
            train_and_test_svm(train_set='NYU+UM1', train_dir=f'{nyu_dataset_directory} + {um1_dataset_directory}',
                               test_set='UM1', test_dir=um1_dataset_directory, test_ratio=test_ratio, X_train=X_train,
                               X_test=X_test, y_train=y_train, y_test=y_test, kernel=kernel, c=c)

def grid_search_all_datasets(kernel: str = 'rbf'):
    data_directories = ['/Users/balazsmorvay/PycharmProjects/VAE/Assets/UM_1_all',
                        '/Users/balazsmorvay/PycharmProjects/VAE/Assets/NYU_all',
                        '/Users/balazsmorvay/PycharmProjects/VAE/Assets/NYU_UM1_merged']
    dataset_names = ['UM_1', 'NYU', 'UM1+NYU']
    test_ratio = 0.15
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mlflow.set_experiment(f"({experiment_time}) grid search")

    for dataset_name, data_directory in zip(dataset_names, data_directories):
        dataset = LatentFMRIDataset(data_dir=data_directory)
        all_data_items = dataset.get_all_items()
        X = all_data_items['X']
        y = all_data_items['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

        for c in np.arange(start=0.01, stop=5, step=0.5):
            with mlflow.start_run():
                train_and_test_svm(train_set=dataset_name, train_dir=data_directory, test_set=dataset_name,
                                   test_dir=data_directory, test_ratio=test_ratio, X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test, kernel=kernel, c=c)

def train_and_test_svm(train_set: str, train_dir: str, test_set: str, test_dir: str, test_ratio: float,
                       X_train, X_test, y_train, y_test, kernel, c):
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_name = f"({experiment_time})Train:{train_set},test:{test_set},c={c},ker={kernel}"
    mlflow.set_tag("mlflow.runName", experiment_name)
    experiment_path = os.path.join('/Users/balazsmorvay/PycharmProjects/fmri_classifier/Experiments',
                                   experiment_name)
    os.mkdir(path=experiment_path)

    mlflow_tracking_params = {
        'Train site': train_set,
        'Train directory': train_dir,
        'Test site': test_set,
        'Test directory': test_dir,
        'Test_ratio': test_ratio,
        'Train_labels_mean': np.mean(y_train),
        'Test_labels_mean': np.mean(y_test),
        'C': c,
        'Kernel': kernel
    }
    mlflow.log_params(params=mlflow_tracking_params, synchronous=True)

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    model = SVC(kernel=kernel, C=c, class_weight='balanced', random_state=42, verbose=True)
    model.fit(X=X_train, y=y_train)
    mlflow.sklearn.log_model(model, experiment_name)

    test_predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, test_predictions, labels=model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    display.plot().figure_.savefig(os.path.join(experiment_path, 'confusion_matrix.png'))
    plt.close('all')
    mlflow.log_artifact(os.path.join(experiment_path, 'confusion_matrix.png'))

    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy': ((tp + tn) / (tp + tn + fp + fn + 1e-6)),
        'recall': (tp / (tp + fn + 1e-6)),
        'precision': (tp / (tp + fp + 1e-6))
    }
    mlflow.log_metrics(metrics, synchronous=True)


if __name__ == '__main__':
    #perform_experiment_train_on_whole_test_on_UM1()
    grid_search_all_datasets(kernel='linear')
